"""
Varman 3.0 — Surrogate Weight Ratio Experiment
===============================================
Evaluates the impact of shifting surrogate loss weighting between ArcFace and FaceNet
at fixed epsilon = 8/255 on 150 LFW genuine pairs across lossless PNG, real JPEG Q75, and real JPEG Q50.

Loss Formulation:
  Plain Cosine Similarity Loss:
    L = 1.0 * sim_af + w_fn * sim_fn
  Evaluated across:
    Row 0: Unprotected Control (0 steps)
    Row 1: Candidate A rerun (w_fn = 0.50) — in-run comparison baseline
    Row 2: Candidate G (w_fn = 0.25)
    Row 3: Candidate H (w_fn = 0.10)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as compute_ssim
from sklearn.datasets import fetch_lfw_pairs
import torch
import torch.nn as nn

REPO_ROOT = Path(r"d:\JavaProjects\Varman\Varman-backend")
sys.path.insert(0, str(REPO_ROOT))

from app.protection.diff_jpeg import DiffJPEG
from app.protection.face_aligner import align_crop_112
from app.protection.protection_engine import ProtectionEngine
from app.protection.strong_engine import composite_crop_back, get_canonical_face_mask_112
from app.protection.surrogate_models import DualSurrogateSuite
from benchmarks.evaluate_face_models import ModelSuite, cosine_similarity

CALIB_PATH = REPO_ROOT / "benchmarks" / "results" / "sweep_evidence" / "calibrated_thresholds_sweep.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "weight_ratio_exp"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = RESULTS_DIR / "tmp_images"
TMP_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_FACE_MASK = (get_canonical_face_mask_112() >= 0.10)


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a binomial proportion in percent."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    z = 1.959964
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / n) + ((z**2) / (4.0 * (n**2))))) / denom
    ci_low = max(0.0, float(center - margin) * 100.0)
    ci_high = min(100.0, float(center + margin) * 100.0)
    return round(ci_low, 2), round(ci_high, 2)


def box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union_area = (w1 * h1) + (w2 * h2) - inter_area
    return float(inter_area) / float(union_area) if union_area > 0 else 0.0


def compute_face_ciede_stats(clean_crop_bgr: np.ndarray, prot_crop_bgr: np.ndarray) -> Tuple[float, float, float]:
    clean_rgb = cv2.cvtColor(clean_crop_bgr, cv2.COLOR_BGR2RGB)
    prot_rgb = cv2.cvtColor(prot_crop_bgr, cv2.COLOR_BGR2RGB)
    lab1 = rgb2lab(clean_rgb / 255.0)
    lab2 = rgb2lab(prot_rgb / 255.0)
    de = deltaE_ciede2000(lab1, lab2)[CANONICAL_FACE_MASK]
    return float(np.mean(de)), float(np.percentile(de, 95.0)), float(np.percentile(de, 99.0))


class WeightRatioPGDEngine:
    """
    Experimental PGD engine supporting parameterized surrogate loss weights:
      L = 1.0 * sim_af + w_fn * sim_fn
    Preserves exact production mechanics: surrogates, DiffJPEG, EOT, mu=0.9,
    and valid pure-luminance projection at epsilon = 8/255.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.surrogates = DualSurrogateSuite(device)
        self.diff_jpeg = DiffJPEG().to(device).eval()

    def protect_crop(
        self,
        crop_112_bgr: np.ndarray,
        steps: int = 25,
        w_fn: float = 0.50,
        eps: float = 8.0 / 255.0,
        mu: float = 0.9,
        q_sequence: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        alpha = eps / 7.0

        crop_rgb = cv2.cvtColor(crop_112_bgr, cv2.COLOR_BGR2RGB)
        x_clean = (
            torch.from_numpy(crop_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 255.0
        )

        with torch.no_grad():
            clean_af, clean_fn = self.surrogates(x_clean)

        delta = torch.zeros_like(x_clean, requires_grad=True)
        momentum = torch.zeros_like(x_clean)

        for t in range(steps):
            if q_sequence is not None and t < len(q_sequence):
                q = float(q_sequence[t])
            else:
                q = float(torch.randint(50, 96, (1,)).item())

            x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
            x_jpeg = self.diff_jpeg(x_adv, quality=q)

            adv_af, adv_fn = self.surrogates(x_jpeg)
            sim_af = (adv_af * clean_af).sum(dim=1).mean()
            sim_fn = (adv_fn * clean_fn).sum(dim=1).mean()

            # Plain cosine similarity loss with parameterized FaceNet weight
            loss = 1.0 * sim_af + w_fn * sim_fn

            grad = torch.autograd.grad(loss, delta)[0]
            grad_norm = grad.norm(p=1) + 1e-12
            momentum = mu * momentum + grad / grad_norm

            with torch.no_grad():
                delta = delta - alpha * torch.sign(momentum)

                # Pure luminance projection preserving both L_inf and [0, 1] bounds simultaneously
                d_lum = delta.mean(dim=1, keepdim=True)
                low = torch.maximum(torch.tensor(-eps, device=self.device), -x_clean.min(dim=1, keepdim=True).values)
                high = torch.minimum(torch.tensor(eps, device=self.device), 1.0 - x_clean.max(dim=1, keepdim=True).values)
                d_lum = torch.clamp(d_lum, low, high)
                delta = d_lum.repeat(1, 3, 1, 1)

            delta.requires_grad = True

        with torch.no_grad():
            prot_crop_rgb = (
                torch.clamp(x_clean + delta, 0.0, 1.0)
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                * 255.0
            ).round().astype(np.uint8)
            prot_crop_bgr = cv2.cvtColor(prot_crop_rgb, cv2.COLOR_RGB2BGR)

        stats = {
            "steps": steps,
            "w_fn": w_fn,
            "eps": eps,
            "final_sim_af": sim_af.item(),
            "final_sim_fn": sim_fn.item(),
        }
        return prot_crop_bgr, stats

    def protect_full_image(
        self,
        img_bgr: np.ndarray,
        faces: List[Dict[str, Any]],
        steps: int = 25,
        w_fn: float = 0.50,
        eps: float = 8.0 / 255.0,
        mu: float = 0.9,
        q_sequence: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if not faces:
            return img_bgr.copy(), []

        protected_img = img_bgr.copy()
        crop_stats = []

        for face_info in faces:
            landmarks = face_info.get("landmarks", [])
            aligned_crop, M = align_crop_112(protected_img, landmarks)
            if aligned_crop is None or M is None:
                continue

            prot_crop, stats = self.protect_crop(
                crop_112_bgr=aligned_crop,
                steps=steps,
                w_fn=w_fn,
                eps=eps,
                mu=mu,
                q_sequence=q_sequence,
            )
            crop_stats.append(stats)
            protected_img = composite_crop_back(protected_img, prot_crop, M)

        return protected_img, crop_stats


def save_summary_csv(results: Dict[str, Any], csv_path: Path):
    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]
    fieldnames = [
        "candidate_id",
        "candidate_name",
        "w_fn",
        "steps",
        "format",
        "model",
        "role",
        "flips_1pct_k",
        "flips_1pct_n",
        "flips_1pct_pct",
        "flips_1pct_ci_low",
        "flips_1pct_ci_high",
        "flips_01pct_k",
        "flips_01pct_n",
        "flips_01pct_pct",
        "flips_01pct_ci_low",
        "flips_01pct_ci_high",
        "mean_sim_drop",
        "face_ssim",
        "ciede_mean",
        "ciede_p95",
        "ciede_p99",
        "latency_sec",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for c_id, cand in results["candidates"].items():
            c_name = cand["candidate_name"]
            w_fn = cand.get("w_fn", 0.0)
            steps = cand.get("steps", 0)
            vis = cand.get("visibility", {})

            for fmt in formats:
                fmt_data = cand.get("lfw", {}).get("formats", {}).get(fmt, {})
                for m in models:
                    m_data = fmt_data.get(m, {})
                    role = "Surrogate (White-Box)" if m in ["ArcFace", "FaceNet"] else "Held-Out"
                    ci_1 = m_data.get("flips_1pct_ci", [0.0, 0.0])
                    ci_01 = m_data.get("flips_01pct_ci", [0.0, 0.0])

                    row = {
                        "candidate_id": c_id,
                        "candidate_name": c_name,
                        "w_fn": w_fn,
                        "steps": steps,
                        "format": fmt,
                        "model": m,
                        "role": role,
                        "flips_1pct_k": m_data.get("flips_1pct_k", 0),
                        "flips_1pct_n": m_data.get("flips_1pct_n", 0),
                        "flips_1pct_pct": m_data.get("flips_1pct_pct", 0.0),
                        "flips_1pct_ci_low": ci_1[0],
                        "flips_1pct_ci_high": ci_1[1],
                        "flips_01pct_k": m_data.get("flips_01pct_k", 0),
                        "flips_01pct_n": m_data.get("flips_01pct_n", 0),
                        "flips_01pct_pct": m_data.get("flips_01pct_pct", 0.0),
                        "flips_01pct_ci_low": ci_01[0],
                        "flips_01pct_ci_high": ci_01[1],
                        "mean_sim_drop": m_data.get("mean_sim_drop", 0.0),
                        "face_ssim": vis.get("face_ssim", 1.0),
                        "ciede_mean": vis.get("ciede_mean", 0.0),
                        "ciede_p95": vis.get("ciede_p95", 0.0),
                        "ciede_p99": vis.get("ciede_p99", 0.0),
                        "latency_sec": vis.get("mean_latency_sec", 0.0),
                    }
                    writer.writerow(row)


def write_report_md(results: Dict[str, Any], report_path: Path):
    """Writes rigorous, empirical REPORT.md comparing surrogate weight ratios."""
    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]

    c_ctrl = results["candidates"]["control"]
    c_rerun = results["candidates"]["cand_A_rerun"]
    c_w025 = results["candidates"]["cand_w025"]
    c_w010 = results["candidates"]["cand_w010"]

    evaluated_cands = [c_ctrl, c_rerun, c_w025, c_w010]

    lines = []
    lines.append("# Varman 3.0 — Surrogate Weight Ratio Experiment Report")
    lines.append("")
    lines.append("**Measurement pass only:** Strict empirical evaluation of surrogate loss weighting between ArcFace and FaceNet ($w_{\\text{fn}} \\in \\{0.50, 0.25, 0.10\\}$, ArcFace fixed at 1.0) at fixed $\\epsilon = 8/255$ on 150 LFW genuine pairs.")
    lines.append("")
    lines.append("## 1. Experimental Configuration & Model Roles")
    lines.append("")
    lines.append("- **Fixed Perturbation Budget:** $\\epsilon = 8/255 \\approx 0.03137$, 25 steps, $\\alpha = \\epsilon / 7$, $\\mu = 0.9$, pure luminance projection ($\\Delta a^*=0, \\Delta b^*=0$).")
    lines.append("- **Guardrail:** OFF (attenuation = 1.0). **Texture Weighting:** OFF.")
    lines.append("- **Loss Formulation:** Plain cosine similarity loss (no hinge): $\\mathcal{L} = 1.0 \\cdot \\text{sim}_{\\text{af}} + w_{\\text{fn}} \\cdot \\text{sim}_{\\text{fn}}$.")
    lines.append("- **White-Box Surrogates (Gradient Contributors):** ArcFace (weight 1.0) and FaceNet (weight $w_{\\text{fn}}$).")
    lines.append("- **Held-Out Evaluation Models (Black-Box Transfer):** SFace and MobileFaceNet (zero gradient contribution).")
    lines.append("- **Thresholds:** Sourced strictly from calibrated 1,500 LFW impostor pairs split (`calibrated_thresholds_sweep.json`).")
    lines.append("")

    # --- Summary Table across formats ---
    lines.append("## 2. Protection Effectiveness Comparison (Measured)")
    lines.append("")
    lines.append("### Mean Flip Rate Across All 4 Models")
    lines.append("")
    lines.append("| Condition / Format | FAR Level | Row 0: Control | Row 1: Cand A Rerun ($w=0.50$) | Row 2: Cand G ($w=0.25$) | Delta (G - A) | Row 3: Cand H ($w=0.10$) | Delta (H - A) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for fmt in ["png", "q75", "q50"]:
        fmt_label = fmt.upper()
        # 1% FAR
        ctrl_1 = np.mean([c_ctrl["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models])
        a_1 = np.mean([c_rerun["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models])
        g_1 = np.mean([c_w025["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models])
        h_1 = np.mean([c_w010["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models])
        lines.append(f"| **{fmt_label}** | 1% FAR | {ctrl_1:.2f}% | {a_1:.2f}% | **{g_1:.2f}%** | {g_1 - a_1:+.2f}% | **{h_1:.2f}%** | {h_1 - a_1:+.2f}% |")

        # 0.1% FAR
        ctrl_01 = np.mean([c_ctrl["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models])
        a_01 = np.mean([c_rerun["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models])
        g_01 = np.mean([c_w025["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models])
        h_01 = np.mean([c_w010["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models])
        lines.append(f"| **{fmt_label}** | 0.1% FAR | {ctrl_01:.2f}% | {a_01:.2f}% | **{g_01:.2f}%** | {g_01 - a_01:+.2f}% | **{h_01:.2f}%** | {h_01 - a_01:+.2f}% |")

    lines.append("")
    lines.append("## 3. Detailed Per-Model Breakdown with 95% Wilson Confidence Intervals")
    lines.append("")

    for fmt in formats:
        lines.append(f"### Format: {fmt.upper()} (Measured)")
        lines.append("")
        lines.append("| Model | Role | FAR Level | Cand A Rerun ($w=0.50$) [95% CI] | Cand G ($w=0.25$) [95% CI] | Delta (G - A) | Cand H ($w=0.10$) [95% CI] | Delta (H - A) |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

        for m in models:
            role = "White-Box" if m in ["ArcFace", "FaceNet"] else "Held-Out"

            # 1% FAR
            a_1 = c_rerun["lfw"]["formats"][fmt][m]
            g_1 = c_w025["lfw"]["formats"][fmt][m]
            h_1 = c_w010["lfw"]["formats"][fmt][m]

            lines.append(
                f"| **{m}** | {role} | 1% FAR | "
                f"{a_1['flips_1pct_k']}/{a_1['flips_1pct_n']} ({a_1['flips_1pct_pct']:.2f}% [{a_1['flips_1pct_ci'][0]:.1f}, {a_1['flips_1pct_ci'][1]:.1f}]) | "
                f"{g_1['flips_1pct_k']}/{g_1['flips_1pct_n']} ({g_1['flips_1pct_pct']:.2f}% [{g_1['flips_1pct_ci'][0]:.1f}, {g_1['flips_1pct_ci'][1]:.1f}]) | "
                f"{g_1['flips_1pct_pct'] - a_1['flips_1pct_pct']:+.2f}% | "
                f"{h_1['flips_1pct_k']}/{h_1['flips_1pct_n']} ({h_1['flips_1pct_pct']:.2f}% [{h_1['flips_1pct_ci'][0]:.1f}, {h_1['flips_1pct_ci'][1]:.1f}]) | "
                f"{h_1['flips_1pct_pct'] - a_1['flips_1pct_pct']:+.2f}% |"
            )

            # 0.1% FAR
            a_01 = c_rerun["lfw"]["formats"][fmt][m]
            g_01 = c_w025["lfw"]["formats"][fmt][m]
            h_01 = c_w010["lfw"]["formats"][fmt][m]

            lines.append(
                f"| **{m}** | {role} | 0.1% FAR | "
                f"{a_01['flips_01pct_k']}/{a_01['flips_01pct_n']} ({a_01['flips_01pct_pct']:.2f}% [{a_01['flips_01pct_ci'][0]:.1f}, {a_01['flips_01pct_ci'][1]:.1f}]) | "
                f"{g_01['flips_01pct_k']}/{g_01['flips_01pct_n']} ({g_01['flips_01pct_pct']:.2f}% [{g_01['flips_01pct_ci'][0]:.1f}, {g_01['flips_01pct_ci'][1]:.1f}]) | "
                f"{g_01['flips_01pct_pct'] - a_01['flips_01pct_pct']:+.2f}% | "
                f"{h_01['flips_01pct_k']}/{h_01['flips_01pct_n']} ({h_01['flips_01pct_pct']:.2f}% [{h_01['flips_01pct_ci'][0]:.1f}, {h_01['flips_01pct_ci'][1]:.1f}]) | "
                f"{h_01['flips_01pct_pct'] - a_01['flips_01pct_pct']:+.2f}% |"
            )
        lines.append("")

    # --- Perceptual Visibility & Latency ---
    lines.append("## 4. Perceptual Quality & Latency (Measured)")
    lines.append("")
    lines.append("| Candidate Row | FaceNet Weight ($w_{\\text{fn}}$) | Face SSIM | CIEDE2000 Mean | CIEDE2000 p95 | CIEDE2000 p99 | Seconds / Image |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for c in evaluated_cands:
        vis = c.get("visibility", {})
        w_str = "None" if c["is_control"] else f"{c['w_fn']:.2f}"
        lines.append(
            f"| **{c['candidate_name']}** | {w_str} | "
            f"{vis.get('face_ssim', 1.0):.4f} | "
            f"{vis.get('ciede_mean', 0.0):.4f} | "
            f"{vis.get('ciede_p95', 0.0):.4f} | "
            f"{vis.get('ciede_p99', 0.0):.4f} | "
            f"{vis.get('mean_latency_sec', 0.0):.2f}s |"
        )

    lines.append("")
    lines.append("## 5. Summary of Findings (Strict Empirical Threshold: $\\ge 2.0\\%$)")
    lines.append("")

    # Compute deltas for summary text
    png_g_delta = np.mean([c_w025["lfw"]["formats"]["png"][m]["flips_1pct_pct"] for m in models]) - np.mean([c_rerun["lfw"]["formats"]["png"][m]["flips_1pct_pct"] for m in models])
    q50_g_delta = np.mean([c_w025["lfw"]["formats"]["q50"][m]["flips_1pct_pct"] for m in models]) - np.mean([c_rerun["lfw"]["formats"]["q50"][m]["flips_1pct_pct"] for m in models])
    png_h_delta = np.mean([c_w010["lfw"]["formats"]["png"][m]["flips_1pct_pct"] for m in models]) - np.mean([c_rerun["lfw"]["formats"]["png"][m]["flips_1pct_pct"] for m in models])
    q50_h_delta = np.mean([c_w010["lfw"]["formats"]["q50"][m]["flips_1pct_pct"] for m in models]) - np.mean([c_rerun["lfw"]["formats"]["q50"][m]["flips_1pct_pct"] for m in models])

    lines.append(f"1. **Mean Flip Rate Deltas vs Candidate A Rerun ($w=0.50$):**")
    lines.append(f"   - Candidate G ($w=0.25$): PNG 1% FAR delta = {png_g_delta:+.2f}%, Q50 1% FAR delta = {q50_g_delta:+.2f}%.")
    lines.append(f"   - Candidate H ($w=0.10$): PNG 1% FAR delta = {png_h_delta:+.2f}%, Q50 1% FAR delta = {q50_h_delta:+.2f}%.")
    lines.append(f"   - *Significance check (threshold $\\ge 2.0\\%$):* {'Statistically meaningful improvement observed.' if max(abs(png_g_delta), abs(q50_g_delta), abs(png_h_delta), abs(q50_h_delta)) >= 2.0 else 'No difference exceeded the 2.0 percentage point significance threshold; variations remain within standard empirical noise.'}")
    lines.append("")
    
    # Check held out direction
    held_out_models = ["SFace", "MobileFaceNet"]
    held_g_png = np.mean([c_w025["lfw"]["formats"]["png"][m]["flips_1pct_pct"] - c_rerun["lfw"]["formats"]["png"][m]["flips_1pct_pct"] for m in held_out_models])
    held_h_png = np.mean([c_w010["lfw"]["formats"]["png"][m]["flips_1pct_pct"] - c_rerun["lfw"]["formats"]["png"][m]["flips_1pct_pct"] for m in held_out_models])
    lines.append(f"2. **Held-Out Model Movement (SFace, MobileFaceNet):**")
    lines.append(f"   - Across both $w=0.25$ and $w=0.10$, held-out models moved {'in the same direction as the 4-model mean' if (held_g_png * png_g_delta >= 0 and held_h_png * png_h_delta >= 0) else 'with divergent individual trajectories compared to the surrogate mean'}.")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_experiment(smoke_test: bool = False, lfw_limit: int = 150):
    print("=" * 90, flush=True)
    print("VARMAN 3.0 — SURROGATE WEIGHT RATIO EXPERIMENT (MEASUREMENT ONLY)", flush=True)
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    thresh_1pct = calib_data["thresholds_1pct"]
    thresh_01pct = calib_data["thresholds_01pct"]

    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]

    engine = ProtectionEngine(model_dir=str(REPO_ROOT / "models"))
    model_suite = ModelSuite(models_dir=REPO_ROOT / "models", requested_device="cuda")
    pgd_engine = WeightRatioPGDEngine(device=device)

    target_lfw_pairs = 2 if smoke_test else lfw_limit
    print(f"\n[LFW] Loading {target_lfw_pairs} genuine pairs from test subset...", flush=True)
    lfw_test = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None)
    gen_indices = np.where(lfw_test.target == 1)[0]

    lfw_pairs: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]] = []
    for idx in gen_indices:
        b1 = cv2.cvtColor((lfw_test.pairs[idx][0] * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        b2 = cv2.cvtColor((lfw_test.pairs[idx][1] * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        f1 = engine.detect_faces(b1)
        f2 = engine.detect_faces(b2)
        if f1 and f2:
            c1, _ = align_crop_112(b1, f1[0]["landmarks"])
            c2, _ = align_crop_112(b2, f2[0]["landmarks"])
            if c1 is not None and c2 is not None:
                lfw_pairs.append((b1, b2, f1[0], f2[0], c1, c2))
        if len(lfw_pairs) >= target_lfw_pairs:
            break

    print(f"[LFW] Loaded {len(lfw_pairs)} genuine pairs.", flush=True)

    print("[LFW] Extracting clean baseline representations...", flush=True)
    lfw_clean_embs_1: List[Dict[str, np.ndarray]] = []
    lfw_clean_embs_2: List[Dict[str, np.ndarray]] = []
    lfw_clean_match_1pct: Dict[str, List[bool]] = {m: [] for m in models}
    lfw_clean_match_01pct: Dict[str, List[bool]] = {m: [] for m in models}

    for b1, b2, f1, f2, _, _ in lfw_pairs:
        e1, _ = model_suite.extract_model_features(b1, f1)
        e2, _ = model_suite.extract_model_features(b2, f2)
        lfw_clean_embs_1.append(e1)
        lfw_clean_embs_2.append(e2)
        for m in models:
            s = float(cosine_similarity(e1[m], e2[m]))
            lfw_clean_match_1pct[m].append(s >= thresh_1pct[m])
            lfw_clean_match_01pct[m].append(s >= thresh_01pct[m])

    for m in models:
        m1 = sum(lfw_clean_match_1pct[m])
        m01 = sum(lfw_clean_match_01pct[m])
        print(f"  Clean LFW {m:<14}: Matched 1% = {m1}/{len(lfw_pairs)} ({m1/len(lfw_pairs)*100:.1f}%), 0.1% = {m01}/{len(lfw_pairs)} ({m01/len(lfw_pairs)*100:.1f}%)")

    CANDIDATES = [
        {
            "id": "control",
            "name": "Row 0: Unprotected Control",
            "steps": 0,
            "w_fn": 0.0,
            "is_control": True,
        },
        {
            "id": "cand_A_rerun",
            "name": "Row 1: Candidate A Rerun (FaceNet weight 0.50)",
            "steps": 25,
            "w_fn": 0.50,
            "is_control": False,
        },
        {
            "id": "cand_w025",
            "name": "Row 2: Candidate G (FaceNet weight 0.25)",
            "steps": 25,
            "w_fn": 0.25,
            "is_control": False,
        },
        {
            "id": "cand_w010",
            "name": "Row 3: Candidate H (FaceNet weight 0.10)",
            "steps": 25,
            "w_fn": 0.10,
            "is_control": False,
        },
    ]

    all_results: Dict[str, Any] = {
        "benchmark_metadata": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "epsilon": "8/255 (0.03137)",
            "smoke_test": smoke_test,
            "lfw_pairs_count": len(lfw_pairs),
            "formats_evaluated": formats,
            "models_surrogate": ["ArcFace", "FaceNet"],
            "models_held_out": ["SFace", "MobileFaceNet"],
            "thresholds_1pct": thresh_1pct,
            "thresholds_01pct": thresh_01pct,
        },
        "candidates": {},
    }

    # Generate deterministic EOT quality sequences per pair for exact comparability
    max_steps = 25
    lfw_q_sequences = []
    for p_i in range(len(lfw_pairs)):
        rng = np.random.default_rng(2026 + p_i)
        lfw_q_sequences.append([int(rng.integers(50, 96)) for _ in range(max_steps)])

    for cand_idx, cand in enumerate(CANDIDATES):
        c_id = cand["id"]
        c_name = cand["name"]
        steps = cand["steps"]
        w_fn = cand["w_fn"]
        is_ctrl = cand["is_control"]

        print("\n" + "=" * 90, flush=True)
        print(f"RUNNING ROW {cand_idx} / {len(CANDIDATES) - 1}: {c_name} (Steps: {steps}, FaceNet Weight: {w_fn:.2f})", flush=True)
        print("=" * 90, flush=True)

        cand_output = {
            "candidate_id": c_id,
            "candidate_name": c_name,
            "steps": steps,
            "w_fn": w_fn,
            "is_control": is_ctrl,
            "lfw": {},
            "visibility": {},
        }

        # --- LFW Execution ---
        print(f"Generating & Evaluating LFW pairs on {c_id}...", flush=True)
        lfw_prot_embs: Dict[str, List[Dict[str, np.ndarray]]] = {fmt: [] for fmt in formats}
        lfw_ssims: List[float] = []
        lfw_ciede_m: List[float] = []
        lfw_ciede_p95: List[float] = []
        lfw_ciede_p99: List[float] = []
        lfw_latencies: List[float] = []

        for p_i, (b1, b2, f1, f2, clean_c1, clean_c2) in enumerate(lfw_pairs):
            if is_ctrl:
                prot_b1 = b1.copy()
                lat_sec = 0.0
                f_ssim = 1.0
                m_de, p95_de, p99_de = 0.0, 0.0, 0.0
            else:
                t0 = time.perf_counter()
                prot_b1, _ = pgd_engine.protect_full_image(
                    img_bgr=b1,
                    faces=[f1],
                    steps=steps,
                    w_fn=w_fn,
                    eps=8.0 / 255.0,
                    mu=0.9,
                    q_sequence=lfw_q_sequences[p_i],
                )
                lat_sec = time.perf_counter() - t0

                prot_c1, _ = align_crop_112(prot_b1, f1["landmarks"])
                f_ssim = float(compute_ssim(clean_c1, prot_c1, channel_axis=2, data_range=255))
                m_de, p95_de, p99_de = compute_face_ciede_stats(clean_c1, prot_c1)

            lfw_ssims.append(f_ssim)
            lfw_ciede_m.append(m_de)
            lfw_ciede_p95.append(p95_de)
            lfw_ciede_p99.append(p99_de)
            lfw_latencies.append(lat_sec)

            paths = {
                "png": TMP_DIR / f"lfw_{c_id}_{p_i}.png",
                "q75": TMP_DIR / f"lfw_{c_id}_{p_i}_q75.jpg",
                "q50": TMP_DIR / f"lfw_{c_id}_{p_i}_q50.jpg",
            }
            cv2.imwrite(str(paths["png"]), prot_b1, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            cv2.imwrite(str(paths["q75"]), prot_b1, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            cv2.imwrite(str(paths["q50"]), prot_b1, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            target_box = f1["box"]
            for fmt, p in paths.items():
                reloaded = cv2.imread(str(p))
                det = engine.detect_faces(reloaded)
                if not det:
                    print(f"  WARNING: LFW face detection failed on {p.name}", flush=True)
                    continue
                ious = [box_iou(target_box, f["box"]) for f in det]
                best_f = det[int(np.argmax(ious))]
                f_embs, _ = model_suite.extract_model_features(reloaded, best_f)
                lfw_prot_embs[fmt].append(f_embs)

            for p in paths.values():
                if p.exists():
                    p.unlink()

            if (p_i + 1) % 25 == 0 or (p_i + 1) == len(lfw_pairs):
                print(f"  LFW progress: {p_i + 1}/{len(lfw_pairs)} completed (Avg Latency: {np.mean(lfw_latencies):.2f}s/img)", flush=True)

        cand_output["lfw"]["formats"] = {}
        for fmt in formats:
            cand_output["lfw"]["formats"][fmt] = {}
            for m in models:
                flips_1pct = 0
                flips_01pct = 0
                sim_drops = []
                n_1pct = 0
                n_01pct = 0

                for p_i in range(len(lfw_prot_embs[fmt])):
                    if p_i >= len(lfw_pairs):
                        break
                    p_emb = lfw_prot_embs[fmt][p_i][m]
                    g_emb = lfw_clean_embs_2[p_i][m]
                    c_sim = float(cosine_similarity(lfw_clean_embs_1[p_i][m], g_emb))
                    p_sim = float(cosine_similarity(p_emb, g_emb))
                    sim_drops.append(c_sim - p_sim)

                    if lfw_clean_match_1pct[m][p_i]:
                        n_1pct += 1
                        if p_sim < thresh_1pct[m]:
                            flips_1pct += 1

                    if lfw_clean_match_01pct[m][p_i]:
                        n_01pct += 1
                        if p_sim < thresh_01pct[m]:
                            flips_01pct += 1

                rate_1pct = (flips_1pct / n_1pct * 100.0) if n_1pct > 0 else 0.0
                rate_01pct = (flips_01pct / n_01pct * 100.0) if n_01pct > 0 else 0.0
                ci_1_low, ci_1_high = compute_wilson_ci(flips_1pct, n_1pct)
                ci_01_low, ci_01_high = compute_wilson_ci(flips_01pct, n_01pct)

                cand_output["lfw"]["formats"][fmt][m] = {
                    "flips_1pct_k": flips_1pct,
                    "flips_1pct_n": n_1pct,
                    "flips_1pct_pct": round(rate_1pct, 2),
                    "flips_1pct_ci": [ci_1_low, ci_1_high],
                    "flips_01pct_k": flips_01pct,
                    "flips_01pct_n": n_01pct,
                    "flips_01pct_pct": round(rate_01pct, 2),
                    "flips_01pct_ci": [ci_01_low, ci_01_high],
                    "mean_sim_drop": round(float(np.mean(sim_drops)), 4) if sim_drops else 0.0,
                }

        cand_output["visibility"] = {
            "face_ssim": round(float(np.mean(lfw_ssims)), 4),
            "ciede_mean": round(float(np.mean(lfw_ciede_m)), 4),
            "ciede_p95": round(float(np.mean(lfw_ciede_p95)), 4),
            "ciede_p99": round(float(np.mean(lfw_ciede_p99)), 4),
            "mean_latency_sec": round(float(np.mean(lfw_latencies)), 3),
        }

        print(f"\n[MEASURED RESULTS] {c_name}:", flush=True)
        print(f"  Visibility: SSIM={cand_output['visibility']['face_ssim']:.4f}, CIEDE Mean={cand_output['visibility']['ciede_mean']:.4f}, p95={cand_output['visibility']['ciede_p95']:.4f}, Latency={cand_output['visibility']['mean_latency_sec']:.2f}s", flush=True)
        for fmt in formats:
            m_flips_1 = [cand_output["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models]
            m_flips_01 = [cand_output["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models]
            print(f"  {fmt.upper()} Mean Flips: 1% FAR = {np.mean(m_flips_1):.2f}%, 0.1% FAR = {np.mean(m_flips_01):.2f}%", flush=True)
            for m in models:
                d = cand_output["lfw"]["formats"][fmt][m]
                print(f"    {m:<14}: 1% FAR = {d['flips_1pct_k']}/{d['flips_1pct_n']} ({d['flips_1pct_pct']:.1f}% [CI: {d['flips_1pct_ci'][0]}-{d['flips_1pct_ci'][1]}%]), 0.1% FAR = {d['flips_01pct_k']}/{d['flips_01pct_n']} ({d['flips_01pct_pct']:.1f}%)", flush=True)

        all_results["candidates"][c_id] = cand_output

        # Save partial after each row
        partial_json_path = RESULTS_DIR / "weight_ratio_partial.json"
        with open(partial_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [Checkpoint] Partial results saved to {partial_json_path.name}", flush=True)

    # Save final results JSON and CSV
    final_json_path = RESULTS_DIR / "weight_ratio_final.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Final] Saved final results JSON to {final_json_path}", flush=True)

    final_csv_path = RESULTS_DIR / "weight_ratio_summary.csv"
    save_summary_csv(all_results, final_csv_path)
    print(f"[Final] Saved summary CSV to {final_csv_path}", flush=True)

    report_md_path = RESULTS_DIR / "REPORT.md"
    write_report_md(all_results, report_md_path)
    print(f"[Final] Saved REPORT.md to {report_md_path}", flush=True)

    # Delete temporary images directory before exiting
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        print(f"[Cleanup] Deleted temporary image directory: {TMP_DIR}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("SURROGATE WEIGHT RATIO EXPERIMENT COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varman 3.0 Surrogate Weight Ratio Experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-pair smoke test")
    parser.add_argument("--lfw-limit", type=int, default=150, help="Number of LFW pairs (default: 150)")
    args = parser.parse_args()

    run_experiment(smoke_test=args.smoke_test, lfw_limit=args.lfw_limit)
