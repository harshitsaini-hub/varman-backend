"""
Varman 3.0 — Threshold-Aimed Loss Experiment
=============================================
Evaluates Candidate F (Threshold-Aimed Hinge Loss) against Candidate A (Baseline Cosine Loss)
at fixed epsilon = 8/255 on 150 LFW genuine pairs across lossless PNG, real JPEG Q75, and real JPEG Q50.

Hinge Loss Formulation:
  Margin rule: 15% of the gap between calibrated 1% FAR threshold and mean impostor similarity
  from the 1,500 LFW calibration split.
    - ArcFace:
        tau_1pct = 0.1476, mu_impostor = 0.0039, gap = 0.1437
        margin = 0.15 * 0.1437 = 0.021555
        target = 0.1476 - 0.021555 = 0.126045
    - FaceNet:
        tau_1pct = 0.3754, mu_impostor = 0.0264, gap = 0.3490
        margin = 0.15 * 0.3490 = 0.052350
        target = 0.3754 - 0.052350 = 0.323050

  Hinge loss:
    L = 1.0 * max(0, sim_af - 0.126045) + 0.5 * max(0, sim_fn - 0.323050)
  When similarity falls below the target, loss and gradient drop to zero, conserving
  the perturbation budget across the multi-surrogate landscape.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "threshold_loss_exp"
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


class ThresholdLossPGDEngine:
    """
    Experimental PGD engine supporting:
      1. Baseline Cosine Loss: L = sim_af + 0.5 * sim_fn
      2. Threshold-Aimed Hinge Loss: L = 1.0 * relu(sim_af - T_af) + 0.5 * relu(sim_fn - T_fn)
    Preserves exact production mechanics: surrogates, DiffJPEG, EOT, mu=0.9,
    and valid pure-luminance projection at epsilon = 8/255.
    """
    def __init__(self, device: torch.device, target_af: float = 0.126045, target_fn: float = 0.323050):
        self.device = device
        self.surrogates = DualSurrogateSuite(device)
        self.diff_jpeg = DiffJPEG().to(device).eval()
        self.target_af = target_af
        self.target_fn = target_fn

    def protect_crop(
        self,
        crop_112_bgr: np.ndarray,
        steps: int = 25,
        loss_mode: str = "cosine",
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

            if loss_mode == "hinge":
                loss_af = torch.relu(sim_af - self.target_af)
                loss_fn = torch.relu(sim_fn - self.target_fn)
                loss = 1.0 * loss_af + 0.5 * loss_fn
            else:
                # Baseline raw cosine loss
                loss = sim_af + 0.5 * sim_fn

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
            "loss_mode": loss_mode,
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
        loss_mode: str = "cosine",
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
                loss_mode=loss_mode,
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
        "loss_mode",
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
            l_mode = cand.get("loss_mode", "none")
            steps = cand.get("steps", 0)
            vis = cand.get("visibility", {})

            for fmt in formats:
                fmt_data = cand.get("lfw", {}).get("formats", {}).get(fmt, {})
                for m in models:
                    m_data = fmt_data.get(m, {})
                    role = "Surrogate" if m in ["ArcFace", "FaceNet"] else "Held-Out"
                    ci_1 = m_data.get("flips_1pct_ci", [0.0, 0.0])
                    ci_01 = m_data.get("flips_01pct_ci", [0.0, 0.0])

                    row = {
                        "candidate_id": c_id,
                        "candidate_name": c_name,
                        "loss_mode": l_mode,
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
    """Writes rigorous, empirical REPORT.md comparing Candidate F vs Candidate A rerun."""
    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]

    c_ctrl = results["candidates"]["control"]
    c_rerun = results["candidates"]["cand_A_rerun"]
    c_hinge = results["candidates"]["cand_F_hinge"]

    # Try to load earlier committed Candidate A for run-to-run variation check
    earlier_cand_a = None
    baseline_json_path = REPO_ROOT / "benchmarks" / "results" / "pgd_convergence_q50_exp" / "pgd_convergence_final.json"
    if baseline_json_path.exists():
        try:
            with open(baseline_json_path, "r", encoding="utf-8") as f:
                base_data = json.load(f)
                earlier_cand_a = base_data.get("candidates", {}).get("cand_A")
        except Exception:
            pass

    lines = []
    lines.append("# Varman 3.0 — Threshold-Aimed Loss Experiment Report")
    lines.append("")
    lines.append("**Measurement pass only:** Strict empirical evaluation of Candidate F (Threshold-Aimed Hinge Loss) versus Candidate A rerun (Baseline Cosine Loss) at fixed $\\epsilon = 8/255$ on 150 LFW genuine pairs.")
    lines.append("")
    lines.append("## 1. Experimental Configuration & Margin Rule")
    lines.append("")
    lines.append("- **Fixed Perturbation Budget:** $\\epsilon = 8/255 \\approx 0.03137$, 25 steps, $\\alpha = \\epsilon / 7$, $\\mu = 0.9$, pure luminance projection ($\\Delta a^*=0, \\Delta b^*=0$).")
    lines.append("- **Guardrail:** OFF (attenuation = 1.0). Texture weighting: OFF.")
    lines.append("- **Evaluation Harness:** Same 150 LFW genuine pairs, real saved and reloaded files (PNG, Q75, Q50).")
    lines.append("- **Surrogates (Gradient Contributors):** ArcFace, FaceNet.")
    lines.append("- **Held-Out Models (Black-box Transfer):** SFace, MobileFaceNet.")
    lines.append("")
    lines.append("### Margin Rule and Hinge Targets")
    lines.append("For each surrogate model, the margin is strictly **15% of the gap** between that model's calibrated 1% FAR threshold and its mean impostor similarity, computed strictly from the 1,500 LFW calibration split (never from the test pairs):")
    lines.append("")
    lines.append("| Surrogate Model | Calibrated 1% FAR Threshold | Calibration Impostor Mean | Gap | 15% Margin | Hinge Target ($T$) | Surrogate Weight |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    lines.append(f"| **ArcFace** | 0.1476 | 0.0039 | 0.1437 | 0.021555 | **{c_hinge.get('target_af', 0.126045):.6f}** | 1.0 |")
    lines.append(f"| **FaceNet** | 0.3754 | 0.0264 | 0.3490 | 0.052350 | **{c_hinge.get('target_fn', 0.323050):.6f}** | 0.5 |")
    lines.append("")
    lines.append("**Loss function:**")
    lines.append("$$\\mathcal{L}_{\\text{hinge}} = 1.0 \\cdot \\max(0, \\text{sim}_{\\text{af}} - 0.126045) + 0.5 \\cdot \\max(0, \\text{sim}_{\\text{fn}} - 0.323050)$$")
    lines.append("")
    lines.append("**Rationale for 15% Margin:**")
    lines.append("- *Below 10%:* The target sits too close to the decision boundary; subsequent JPEG quantization pushes similarity back over the threshold at Q50.")
    lines.append("- *Above 20%:* The optimizer spends its limited $\\epsilon=8$ budget pursuing impossible pairs that were never going to flip, exhausting perturbation capacity before boundary-proximal pairs cross the threshold.")
    lines.append("- *15%:* Sits in the optimal middle of this range and provides a defensible, fixed target.")
    lines.append("")

    # --- Summary Table across formats ---
    lines.append("## 2. Protection Effectiveness Comparison (Measured)")
    lines.append("")
    lines.append("### Mean Flip Rate Across All 4 Models")
    lines.append("")
    lines.append("| Condition / Format | FAR Level | Candidate A Rerun (Cosine Loss) | Candidate F (Threshold Hinge) | Delta (F - A) | Earlier Committed Cand A |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for fmt in ["png", "q75", "q50"]:
        fmt_label = fmt.upper()
        # 1% FAR
        a_1_vals = [c_rerun["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models]
        f_1_vals = [c_hinge["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models]
        mean_a_1 = np.mean(a_1_vals)
        mean_f_1 = np.mean(f_1_vals)
        delta_1 = mean_f_1 - mean_a_1

        earlier_1_str = "N/A"
        if earlier_cand_a:
            e_1_vals = [earlier_cand_a["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models]
            earlier_1_str = f"{np.mean(e_1_vals):.2f}%"

        lines.append(f"| **{fmt_label}** | 1% FAR | {mean_a_1:.2f}% | **{mean_f_1:.2f}%** | {delta_1:+.2f}% | {earlier_1_str} |")

        # 0.1% FAR
        a_01_vals = [c_rerun["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models]
        f_01_vals = [c_hinge["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models]
        mean_a_01 = np.mean(a_01_vals)
        mean_f_01 = np.mean(f_01_vals)
        delta_01 = mean_f_01 - mean_a_01

        earlier_01_str = "N/A"
        if earlier_cand_a:
            e_01_vals = [earlier_cand_a["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models]
            earlier_01_str = f"{np.mean(e_01_vals):.2f}%"

        lines.append(f"| **{fmt_label}** | 0.1% FAR | {mean_a_01:.2f}% | **{mean_f_01:.2f}%** | {delta_01:+.2f}% | {earlier_01_str} |")

    lines.append("")
    lines.append("## 3. Detailed Per-Model Flip Rates with 95% Wilson Confidence Intervals")
    lines.append("")

    for fmt in formats:
        lines.append(f"### Format: {fmt.upper()} (Measured)")
        lines.append("")
        lines.append("| Model | Role | FAR Level | Control Flips | Candidate A Rerun Flips ($k/n$) | Cand A Flip % [95% CI] | Candidate F Flips ($k/n$) | Cand F Flip % [95% CI] | Delta (F - A) |")
        lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

        for m in models:
            role = "Surrogate" if m in ["ArcFace", "FaceNet"] else "Held-Out"

            # 1% FAR
            ctrl_1 = c_ctrl["lfw"]["formats"][fmt][m]
            a_1 = c_rerun["lfw"]["formats"][fmt][m]
            f_1 = c_hinge["lfw"]["formats"][fmt][m]
            delta_1 = f_1["flips_1pct_pct"] - a_1["flips_1pct_pct"]

            lines.append(
                f"| **{m}** | {role} | 1% FAR | {ctrl_1['flips_1pct_k']}/{ctrl_1['flips_1pct_n']} ({ctrl_1['flips_1pct_pct']:.1f}%) | "
                f"{a_1['flips_1pct_k']}/{a_1['flips_1pct_n']} | {a_1['flips_1pct_pct']:.2f}% [{a_1['flips_1pct_ci'][0]:.1f}, {a_1['flips_1pct_ci'][1]:.1f}] | "
                f"{f_1['flips_1pct_k']}/{f_1['flips_1pct_n']} | **{f_1['flips_1pct_pct']:.2f}%** [{f_1['flips_1pct_ci'][0]:.1f}, {f_1['flips_1pct_ci'][1]:.1f}] | "
                f"{delta_1:+.2f}% |"
            )

            # 0.1% FAR
            ctrl_01 = c_ctrl["lfw"]["formats"][fmt][m]
            a_01 = c_rerun["lfw"]["formats"][fmt][m]
            f_01 = c_hinge["lfw"]["formats"][fmt][m]
            delta_01 = f_01["flips_01pct_pct"] - a_01["flips_01pct_pct"]

            lines.append(
                f"| **{m}** | {role} | 0.1% FAR | {ctrl_01['flips_01pct_k']}/{ctrl_01['flips_01pct_n']} ({ctrl_01['flips_01pct_pct']:.1f}%) | "
                f"{a_01['flips_01pct_k']}/{a_01['flips_01pct_n']} | {a_01['flips_01pct_pct']:.2f}% [{a_01['flips_01pct_ci'][0]:.1f}, {a_01['flips_01pct_ci'][1]:.1f}] | "
                f"{f_01['flips_01pct_k']}/{f_01['flips_01pct_n']} | **{f_01['flips_01pct_pct']:.2f}%** [{f_01['flips_01pct_ci'][0]:.1f}, {f_01['flips_01pct_ci'][1]:.1f}] | "
                f"{delta_01:+.2f}% |"
            )
        lines.append("")

    # --- Perceptual Visibility & Latency ---
    lines.append("## 4. Perceptual Quality & Latency (Measured)")
    lines.append("")
    lines.append("| Candidate | Loss Function | Face SSIM | CIEDE2000 Mean | CIEDE2000 p95 | CIEDE2000 p99 | Seconds / Image |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for c in [c_ctrl, c_rerun, c_hinge]:
        vis = c.get("visibility", {})
        l_name = "None" if c["is_control"] else ("Cosine Loss" if c["loss_mode"] == "cosine" else "Threshold Hinge Loss")
        lines.append(
            f"| **{c['candidate_name']}** | {l_name} | "
            f"{vis.get('face_ssim', 1.0):.4f} | "
            f"{vis.get('ciede_mean', 0.0):.4f} | "
            f"{vis.get('ciede_p95', 0.0):.4f} | "
            f"{vis.get('ciede_p99', 0.0):.4f} | "
            f"{vis.get('mean_latency_sec', 0.0):.2f}s |"
        )

    lines.append("")
    lines.append("## 5. Run-to-Run Variation (Candidate A Rerun vs Previously Committed Candidate A)")
    lines.append("")
    if earlier_cand_a:
        lines.append("| Format | FAR Level | Candidate A Rerun (This Run) | Earlier Committed Cand A | Run-to-Run Difference |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for fmt in ["png", "q75", "q50"]:
            a1 = np.mean([c_rerun["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models])
            e1 = np.mean([earlier_cand_a["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models])
            lines.append(f"| {fmt.upper()} | 1% FAR | {a1:.2f}% | {e1:.2f}% | {a1 - e1:+.2f}% |")

            a01 = np.mean([c_rerun["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models])
            e01 = np.mean([earlier_cand_a["lfw"]["formats"][fmt][m]["flips_01pct_pct"] for m in models])
            lines.append(f"| {fmt.upper()} | 0.1% FAR | {a01:.2f}% | {e01:.2f}% | {a01 - e01:+.2f}% |")
    else:
        lines.append("Earlier committed Candidate A file not found for comparison.")

    lines.append("")
    lines.append("## 6. Summary of Findings")
    lines.append("")
    lines.append("1. **Surrogate Efficiency vs Transferability:** State whether the threshold hinge loss preserved or shifted gradient capacity toward the secondary surrogate, and its effect on black-box transferability (SFace and MobileFaceNet).")
    lines.append("2. **JPEG Q50 Robustness:** Observe whether the 15% margin prevented post-compression similarity rebound above the calibrated threshold.")
    lines.append("3. **Multi-Threshold Consistency:** Verify whether protection gains at 1% FAR were preserved at 0.1% FAR.")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_experiment(smoke_test: bool = False, lfw_limit: int = 150):
    print("=" * 90, flush=True)
    print("VARMAN 3.0 — THRESHOLD-AIMED LOSS EXPERIMENT (MEASUREMENT ONLY)", flush=True)
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    thresh_1pct = calib_data["thresholds_1pct"]
    thresh_01pct = calib_data["thresholds_01pct"]
    impostor_stats = calib_data["impostor_stats"]

    # ArcFace Margin Calculation
    tau_af = thresh_1pct["ArcFace"]
    mu_af = impostor_stats["ArcFace"]["mean"]
    gap_af = tau_af - mu_af
    margin_af = 0.15 * gap_af
    target_af = tau_af - margin_af

    # FaceNet Margin Calculation
    tau_fn = thresh_1pct["FaceNet"]
    mu_fn = impostor_stats["FaceNet"]["mean"]
    gap_fn = tau_fn - mu_fn
    margin_fn = 0.15 * gap_fn
    target_fn = tau_fn - margin_fn

    print(f"\n[Calibration Setup] Loaded thresholds from {CALIB_PATH.name}:", flush=True)
    print(f"  ArcFace 1% FAR Threshold: {tau_af:.4f}, Impostor Mean: {mu_af:.4f}, Gap: {gap_af:.4f}, Margin (15%): {margin_af:.6f} -> Hinge Target: {target_af:.6f}", flush=True)
    print(f"  FaceNet 1% FAR Threshold: {tau_fn:.4f}, Impostor Mean: {mu_fn:.4f}, Gap: {gap_fn:.4f}, Margin (15%): {margin_fn:.6f} -> Hinge Target: {target_fn:.6f}", flush=True)

    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]

    engine = ProtectionEngine(model_dir=str(REPO_ROOT / "models"))
    model_suite = ModelSuite(models_dir=REPO_ROOT / "models", requested_device="cuda")
    pgd_engine = ThresholdLossPGDEngine(device=device, target_af=target_af, target_fn=target_fn)

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
            "name": "Unprotected Control",
            "steps": 0,
            "loss_mode": "none",
            "is_control": True,
        },
        {
            "id": "cand_A_rerun",
            "name": "Candidate A Rerun (Baseline Cosine Loss)",
            "steps": 25,
            "loss_mode": "cosine",
            "is_control": False,
        },
        {
            "id": "cand_F_hinge",
            "name": "Candidate F (Threshold-Aimed Hinge Loss)",
            "steps": 25,
            "loss_mode": "hinge",
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
            "target_af": target_af,
            "target_fn": target_fn,
        },
        "candidates": {},
    }

    # Generate fixed deterministic EOT quality sequences per pair for comparability
    max_steps = 25
    lfw_q_sequences = []
    for p_i in range(len(lfw_pairs)):
        rng = np.random.default_rng(2026 + p_i)
        lfw_q_sequences.append([int(rng.integers(50, 96)) for _ in range(max_steps)])

    for cand_idx, cand in enumerate(CANDIDATES):
        c_id = cand["id"]
        c_name = cand["name"]
        steps = cand["steps"]
        l_mode = cand["loss_mode"]
        is_ctrl = cand["is_control"]

        print("\n" + "=" * 90, flush=True)
        print(f"RUNNING ROW {cand_idx} / {len(CANDIDATES) - 1}: {c_name} (Steps: {steps}, Loss: {l_mode})", flush=True)
        print("=" * 90, flush=True)

        cand_output = {
            "candidate_id": c_id,
            "candidate_name": c_name,
            "steps": steps,
            "loss_mode": l_mode,
            "is_control": is_ctrl,
            "target_af": target_af if l_mode == "hinge" else None,
            "target_fn": target_fn if l_mode == "hinge" else None,
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
                    loss_mode=l_mode,
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
        partial_json_path = RESULTS_DIR / "threshold_loss_partial.json"
        with open(partial_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [Checkpoint] Partial results saved to {partial_json_path.name}", flush=True)

    # Save final results JSON and CSV
    final_json_path = RESULTS_DIR / "threshold_loss_final.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Final] Saved final results JSON to {final_json_path}", flush=True)

    final_csv_path = RESULTS_DIR / "threshold_loss_summary.csv"
    save_summary_csv(all_results, final_csv_path)
    print(f"[Final] Saved summary CSV to {final_csv_path}", flush=True)

    report_md_path = RESULTS_DIR / "REPORT.md"
    write_report_md(all_results, report_md_path)
    print(f"[Final] Saved REPORT.md to {report_md_path}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("EXPERIMENT COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varman 3.0 Threshold-Aimed Loss Experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 2-pair smoke test")
    parser.add_argument("--lfw-limit", type=int, default=150, help="Number of LFW pairs (default: 150)")
    args = parser.parse_args()

    run_experiment(smoke_test=args.smoke_test, lfw_limit=args.lfw_limit)
