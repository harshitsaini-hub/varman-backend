"""
Varman 3.0 — Controlled PGD Convergence and Real JPEG Q50 Robustness Experiment
================================================================================
Compares five PGD convergence candidates at fixed epsilon = 8/255:
  Candidate A (Baseline):  25 steps, fixed step size (alpha = eps / 7)
  Candidate B:             50 steps, fixed step size (alpha = eps / 7)
  Candidate C:             75 steps, fixed step size (alpha = eps / 7)
  Candidate D:             50 steps, cosine schedule (alpha_max=1.5*alpha, alpha_min=0.1*alpha)
  Candidate E:             75 steps, cosine schedule (alpha_max=1.5*alpha, alpha_min=0.1*alpha)

Evaluated under:
  - Lossless PNG
  - Actual disk-reloaded JPEG Q75
  - Actual disk-reloaded JPEG Q50
Across:
  - LFW Genuine Pairs (150 pairs)
  - Real-photo Genuine Pairs (26 directed pairs across Clusters A & B)
On:
  - Surrogates: ArcFace, FaceNet
  - Held-out: SFace, MobileFaceNet
At:
  - 1% FAR (primary)
  - 0.1% FAR (secondary)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

TESTPICS_DIR = Path(r"C:\Users\ASUS F17\OneDrive\Desktop\Testpics")
LABELS_CSV_PATH = Path(r"C:\Users\ASUS F17\OneDrive\Desktop\real_photo_labels\photo_labels.csv")
CALIB_PATH = REPO_ROOT / "benchmarks" / "results" / "sweep_evidence" / "calibrated_thresholds_sweep.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "pgd_convergence_q50_exp"
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


class ExperimentalPGDEngine:
    """
    Experimental PGD engine implementing the five convergence candidates.
    Preserves all production mechanics: surrogates, DiffJPEG, EOT, mu=0.9,
    and applies strictly valid pure-luminance projection.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.surrogates = DualSurrogateSuite(device)
        self.diff_jpeg = DiffJPEG().to(device).eval()

    def protect_crop(
        self,
        crop_112_bgr: np.ndarray,
        steps: int = 25,
        schedule: str = "fixed",
        eps: float = 8.0 / 255.0,
        mu: float = 0.9,
        q_sequence: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        base_alpha = eps / 7.0
        alpha_max = 1.5 * base_alpha
        alpha_min = 0.1 * base_alpha

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
            if schedule == "cosine":
                alpha_t = alpha_min + 0.5 * (alpha_max - alpha_min) * (1.0 + math.cos(math.pi * t / steps))
            else:
                alpha_t = base_alpha

            if q_sequence is not None and t < len(q_sequence):
                q = float(q_sequence[t])
            else:
                q = float(torch.randint(50, 96, (1,)).item())

            x_adv = torch.clamp(x_clean + delta, 0.0, 1.0)
            x_jpeg = self.diff_jpeg(x_adv, quality=q)

            adv_af, adv_fn = self.surrogates(x_jpeg)
            sim_af = (adv_af * clean_af).sum(dim=1).mean()
            sim_fn = (adv_fn * clean_fn).sum(dim=1).mean()
            loss = sim_af + 0.5 * sim_fn

            grad = torch.autograd.grad(loss, delta)[0]
            grad_norm = grad.norm(p=1) + 1e-12
            momentum = mu * momentum + grad / grad_norm

            with torch.no_grad():
                delta = delta - alpha_t * torch.sign(momentum)

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
            "schedule": schedule,
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
        schedule: str = "fixed",
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
                schedule=schedule,
                eps=eps,
                mu=mu,
                q_sequence=q_sequence,
            )
            crop_stats.append(stats)
            protected_img = composite_crop_back(protected_img, prot_crop, M)

        return protected_img, crop_stats


def generate_diagnostic_plots(results_dir: Path, data: Dict[str, Any]):
    """Generates auditable plots summarizing Q50 robustness, retention, and visibility."""
    cands = [c for c in data["candidates"].values() if not c["is_control"]]
    if not cands:
        return

    labels = [c["candidate_id"].replace("cand_", "Cand ") for c in cands]
    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]

    # 1. Flip Rates by Format (PNG vs Q75 vs Q50) on LFW (1% FAR)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cands))
    width = 0.25

    png_means = [np.mean([c["lfw"]["formats"]["png"][m]["flips_1pct_pct"] for m in models]) for c in cands]
    q75_means = [np.mean([c["lfw"]["formats"]["q75"][m]["flips_1pct_pct"] for m in models]) for c in cands]
    q50_means = [np.mean([c["lfw"]["formats"]["q50"][m]["flips_1pct_pct"] for m in models]) for c in cands]

    rects1 = ax.bar(x - width, png_means, width, label="Protected PNG", color="#2b5c8f")
    rects2 = ax.bar(x, q75_means, width, label="Protected JPEG Q75", color="#e28743")
    rects3 = ax.bar(x + width, q50_means, width, label="Protected JPEG Q50", color="#c93b2b")

    ax.set_ylabel("Mean Flip Rate (%) [1% FAR]")
    ax.set_title("LFW Protection Effectiveness Across Compression Conditions (Epsilon = 8/255)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(results_dir / "lfw_flip_rates_by_condition.png", dpi=150)
    plt.close()

    # 2. Per-Model Q50 Robustness Breakdown
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(cands))
    width = 0.20

    m_colors = {"ArcFace": "#1f77b4", "FaceNet": "#aec7e8", "SFace": "#ff7f0e", "MobileFaceNet": "#2ca02c"}
    for idx, m in enumerate(models):
        role = "Surrogate" if m in ["ArcFace", "FaceNet"] else "Held-Out"
        rates = [c["lfw"]["formats"]["q50"][m]["flips_1pct_pct"] for c in cands]
        rects = ax.bar(x + (idx - 1.5) * width, rates, width, label=f"{m} ({role})", color=m_colors[m])
        for rect in rects:
            h = rect.get_height()
            ax.annotate(f"{h:.0f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Q50 Flip Rate (%) [1% FAR]")
    ax.set_title("Real JPEG Q50 Robustness: Surrogates vs Held-Out Models (LFW)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "q50_model_breakdown.png", dpi=150)
    plt.close()

    # 3. Trade-off: Q50 Mean Flip vs Mean CIEDE2000
    fig, ax = plt.subplots(figsize=(8, 5))
    ciede_means = [c["visibility"]["lfw"]["ciede_mean"] for c in cands]
    latencies = [c["visibility"]["lfw"]["mean_latency_sec"] for c in cands]

    scatter = ax.scatter(ciede_means, q50_means, s=[l * 40 for l in latencies], c=q50_means, cmap="viridis", edgecolors="black", zorder=3)
    for i, txt in enumerate(labels):
        ax.annotate(f"{txt}\n({latencies[i]:.1f}s)", (ciede_means[i], q50_means[i]),
                    textcoords="offset points", xytext=(8, -4), fontsize=9)

    ax.set_xlabel("Face Region CIEDE2000 Mean (lower is better visual quality)")
    ax.set_ylabel("JPEG Q50 Mean Flip Rate (%) [1% FAR]")
    ax.set_title("Trade-off: Real JPEG Q50 Protection vs Perceptual Visibility (Bubble Size = GPU Latency)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(results_dir / "q50_protection_vs_visibility.png", dpi=150)
    plt.close()


def run_experiment(smoke_test: bool = False, lfw_limit: int = 150):
    print("=" * 90, flush=True)
    print("VARMAN 3.0 - CONTROLLED PGD CONVERGENCE AND REAL JPEG Q50 ROBUSTNESS EXPERIMENT", flush=True)
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    thresh_1pct = calib_data["thresholds_1pct"]
    thresh_01pct = calib_data["thresholds_01pct"]
    print(f"Loaded calibrated thresholds from {CALIB_PATH.name}:", flush=True)
    print(f"  1% FAR:   {thresh_1pct}", flush=True)
    print(f"  0.1% FAR: {thresh_01pct}", flush=True)

    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]

    engine = ProtectionEngine(model_dir=str(REPO_ROOT / "models"))
    model_suite = ModelSuite(models_dir=REPO_ROOT / "models", requested_device="cuda")
    pgd_engine = ExperimentalPGDEngine(device=device)

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

    print(f"\n[Real Photos] Loading ground truth labels from {LABELS_CSV_PATH}...", flush=True)
    real_photo_labels: Dict[str, Dict[str, Any]] = {}
    with open(LABELS_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            real_photo_labels[row["label"]] = row

    real_large_photos = [lbl for lbl, r in real_photo_labels.items() if r["cluster"] in ["A", "B"]]
    if smoke_test:
        real_large_photos = real_large_photos[:2]

    real_clusters: Dict[str, List[str]] = {"A": [], "B": []}
    for lbl in real_large_photos:
        real_clusters[real_photo_labels[lbl]["cluster"]].append(lbl)

    real_pairs: List[Tuple[str, str]] = []
    for c_name, members in real_clusters.items():
        for p_i in members:
            for p_j in members:
                if p_i != p_j:
                    real_pairs.append((p_i, p_j))

    print(f"[Real Photos] Selected photos: {real_large_photos} -> {len(real_pairs)} directed genuine pairs.", flush=True)

    real_clean_imgs: Dict[str, np.ndarray] = {}
    real_clean_faces: Dict[str, Dict[str, Any]] = {}
    real_clean_crops: Dict[str, np.ndarray] = {}
    real_clean_embs: Dict[str, Dict[str, np.ndarray]] = {}

    for lbl in real_large_photos:
        p_path = TESTPICS_DIR / real_photo_labels[lbl]["filename"]
        img_bgr = cv2.imread(str(p_path))
        real_clean_imgs[lbl] = img_bgr

        all_faces = engine.detect_faces(img_bgr)
        if lbl == "P01":
            subj_f = all_faces[0]
        else:
            subj_f = sorted(all_faces, key=lambda f: f["box"][2] * f["box"][3], reverse=True)[0]

        real_clean_faces[lbl] = subj_f
        c112, _ = align_crop_112(img_bgr, subj_f["landmarks"])
        real_clean_crops[lbl] = c112
        embs, _ = model_suite.extract_model_features(img_bgr, subj_f)
        real_clean_embs[lbl] = embs

    real_clean_match_1pct: Dict[str, Dict[Tuple[str, str], bool]] = {m: {} for m in models}
    real_clean_match_01pct: Dict[str, Dict[Tuple[str, str], bool]] = {m: {} for m in models}
    for p_i, p_j in real_pairs:
        for m in models:
            s = float(cosine_similarity(real_clean_embs[p_i][m], real_clean_embs[p_j][m]))
            real_clean_match_1pct[m][(p_i, p_j)] = (s >= thresh_1pct[m])
            real_clean_match_01pct[m][(p_i, p_j)] = (s >= thresh_01pct[m])

    CANDIDATES = [
        {
            "id": "control",
            "name": "Unprotected Control",
            "steps": 0,
            "schedule": "none",
            "is_control": True,
        },
        {
            "id": "cand_A",
            "name": "Candidate A - Baseline (25 steps, fixed)",
            "steps": 25,
            "schedule": "fixed",
            "is_control": False,
        },
        {
            "id": "cand_B",
            "name": "Candidate B - Longer horizon (50 steps, fixed)",
            "steps": 50,
            "schedule": "fixed",
            "is_control": False,
        },
        {
            "id": "cand_C",
            "name": "Candidate C - Longer horizon (75 steps, fixed)",
            "steps": 75,
            "schedule": "fixed",
            "is_control": False,
        },
        {
            "id": "cand_D",
            "name": "Candidate D - Cosine schedule (50 steps)",
            "steps": 50,
            "schedule": "cosine",
            "is_control": False,
        },
        {
            "id": "cand_E",
            "name": "Candidate E - Cosine schedule (75 steps)",
            "steps": 75,
            "schedule": "cosine",
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
            "real_pairs_count": len(real_pairs),
            "formats_evaluated": formats,
            "models_surrogate": ["ArcFace", "FaceNet"],
            "models_held_out": ["SFace", "MobileFaceNet"],
            "thresholds_1pct": thresh_1pct,
            "thresholds_01pct": thresh_01pct,
        },
        "candidates": {},
    }

    max_steps = 75
    lfw_q_sequences = []
    for p_i in range(len(lfw_pairs)):
        rng = np.random.default_rng(2026 + p_i)
        lfw_q_sequences.append([int(rng.integers(50, 96)) for _ in range(max_steps)])

    real_q_sequences = {}
    for p_idx, lbl in enumerate(real_large_photos):
        rng = np.random.default_rng(5000 + p_idx)
        real_q_sequences[lbl] = [int(rng.integers(50, 96)) for _ in range(max_steps)]

    for cand in CANDIDATES:
        c_id = cand["id"]
        c_name = cand["name"]
        steps = cand["steps"]
        schedule = cand["schedule"]
        is_ctrl = cand["is_control"]

        print("\n" + "=" * 90, flush=True)
        print(f"RUNNING: {c_name} (Steps: {steps}, Schedule: {schedule})", flush=True)
        print("=" * 90, flush=True)

        cand_output = {
            "candidate_id": c_id,
            "candidate_name": c_name,
            "steps": steps,
            "schedule": schedule,
            "is_control": is_ctrl,
            "lfw": {},
            "real": {},
            "visibility": {},
        }

        # --- A. LFW Execution ---
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
                    schedule=schedule,
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

        # --- B. Real Photos Execution ---
        print(f"Generating & Evaluating Real Photos on {c_id}...", flush=True)
        real_prot_imgs: Dict[str, np.ndarray] = {}
        real_prot_embs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {fmt: {} for fmt in formats}
        real_ssims: List[float] = []
        real_ciede_m: List[float] = []
        real_ciede_p95: List[float] = []
        real_ciede_p99: List[float] = []

        for lbl in real_large_photos:
            clean_bgr = real_clean_imgs[lbl]
            subj_f = real_clean_faces[lbl]
            clean_crop = real_clean_crops[lbl]

            if is_ctrl:
                prot_bgr = clean_bgr.copy()
                f_ssim = 1.0
                m_de, p95_de, p99_de = 0.0, 0.0, 0.0
            else:
                prot_bgr, _ = pgd_engine.protect_full_image(
                    img_bgr=clean_bgr,
                    faces=[subj_f],
                    steps=steps,
                    schedule=schedule,
                    eps=8.0 / 255.0,
                    mu=0.9,
                    q_sequence=real_q_sequences[lbl],
                )
                prot_crop, _ = align_crop_112(prot_bgr, subj_f["landmarks"])
                f_ssim = float(compute_ssim(clean_crop, prot_crop, channel_axis=2, data_range=255))
                m_de, p95_de, p99_de = compute_face_ciede_stats(clean_crop, prot_crop)

            real_prot_imgs[lbl] = prot_bgr
            real_ssims.append(f_ssim)
            real_ciede_m.append(m_de)
            real_ciede_p95.append(p95_de)
            real_ciede_p99.append(p99_de)

            paths = {
                "png": TMP_DIR / f"real_{c_id}_{lbl}.png",
                "q75": TMP_DIR / f"real_{c_id}_{lbl}_q75.jpg",
                "q50": TMP_DIR / f"real_{c_id}_{lbl}_q50.jpg",
            }
            cv2.imwrite(str(paths["png"]), prot_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
            cv2.imwrite(str(paths["q75"]), prot_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            cv2.imwrite(str(paths["q50"]), prot_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            target_box = subj_f["box"]
            for fmt, p in paths.items():
                reloaded = cv2.imread(str(p))
                det = engine.detect_faces(reloaded)
                if not det:
                    print(f"  WARNING: Real photo face detection failed on {p.name}", flush=True)
                    continue
                if lbl == "P01":
                    best_f = det[0]
                else:
                    ious = [box_iou(target_box, f["box"]) for f in det]
                    best_f = det[int(np.argmax(ious))]
                f_embs, _ = model_suite.extract_model_features(reloaded, best_f)
                real_prot_embs[fmt][lbl] = f_embs

            for p in paths.values():
                if p.exists():
                    p.unlink()

        cand_output["real"]["formats"] = {}
        for fmt in formats:
            cand_output["real"]["formats"][fmt] = {}
            for m in models:
                flips_1pct = 0
                flips_01pct = 0
                sim_drops = []
                n_1pct = 0
                n_01pct = 0

                for p_i, p_j in real_pairs:
                    if p_i not in real_prot_embs[fmt] or p_j not in real_clean_embs:
                        continue
                    p_emb = real_prot_embs[fmt][p_i][m]
                    g_emb = real_clean_embs[p_j][m]
                    c_sim = float(cosine_similarity(real_clean_embs[p_i][m], g_emb))
                    p_sim = float(cosine_similarity(p_emb, g_emb))
                    sim_drops.append(c_sim - p_sim)

                    if real_clean_match_1pct[m][(p_i, p_j)]:
                        n_1pct += 1
                        if p_sim < thresh_1pct[m]:
                            flips_1pct += 1

                    if real_clean_match_01pct[m][(p_i, p_j)]:
                        n_01pct += 1
                        if p_sim < thresh_01pct[m]:
                            flips_01pct += 1

                rate_1pct = (flips_1pct / n_1pct * 100.0) if n_1pct > 0 else 0.0
                rate_01pct = (flips_01pct / n_01pct * 100.0) if n_01pct > 0 else 0.0
                ci_1_low, ci_1_high = compute_wilson_ci(flips_1pct, n_1pct)
                ci_01_low, ci_01_high = compute_wilson_ci(flips_01pct, n_01pct)

                cand_output["real"]["formats"][fmt][m] = {
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
            "lfw": {
                "face_ssim": round(float(np.mean(lfw_ssims)), 4),
                "ciede_mean": round(float(np.mean(lfw_ciede_m)), 4),
                "ciede_std_of_means": round(float(np.std(lfw_ciede_m)), 4),
                "ciede_p95": round(float(np.mean(lfw_ciede_p95)), 4),
                "ciede_p99": round(float(np.mean(lfw_ciede_p99)), 4),
                "mean_latency_sec": round(float(np.mean(lfw_latencies)), 3),
            },
            "real": {
                "face_ssim": round(float(np.mean(real_ssims)), 4),
                "ciede_mean": round(float(np.mean(real_ciede_m)), 4),
                "ciede_std_of_means": round(float(np.std(real_ciede_m)), 4),
                "ciede_p95": round(float(np.mean(real_ciede_p95)), 4),
                "ciede_p99": round(float(np.mean(real_ciede_p99)), 4),
            },
        }

        print(f"\n--- {c_name} Results Summary ---", flush=True)
        for fmt in formats:
            m_flips = [cand_output["lfw"]["formats"][fmt][m]["flips_1pct_pct"] for m in models]
            mean_flip = np.mean(m_flips)
            print(f"  LFW {fmt.upper():<4} 1% FAR Mean Flip: {mean_flip:5.2f}% | "
                  f"ArcFace: {cand_output['lfw']['formats'][fmt]['ArcFace']['flips_1pct_pct']:5.2f}% | "
                  f"FaceNet: {cand_output['lfw']['formats'][fmt]['FaceNet']['flips_1pct_pct']:5.2f}% | "
                  f"SFace: {cand_output['lfw']['formats'][fmt]['SFace']['flips_1pct_pct']:5.2f}% | "
                  f"MobileFaceNet: {cand_output['lfw']['formats'][fmt]['MobileFaceNet']['flips_1pct_pct']:5.2f}%", flush=True)

        if not is_ctrl:
            png_mean = np.mean([cand_output["lfw"]["formats"]["png"][m]["flips_1pct_pct"] for m in models])
            q50_mean = np.mean([cand_output["lfw"]["formats"]["q50"][m]["flips_1pct_pct"] for m in models])
            retention_pct = (q50_mean / png_mean * 100.0) if png_mean > 0 else 0.0
            print(f"  Q50 vs PNG Retention: {q50_mean:.2f}% / {png_mean:.2f}% = {retention_pct:.1f}%", flush=True)
            print(f"  LFW Visibility: Face SSIM={cand_output['visibility']['lfw']['face_ssim']:.4f}, "
                  f"CIEDE2000 Mean={cand_output['visibility']['lfw']['ciede_mean']:.4f}", flush=True)

        all_results["candidates"][c_id] = cand_output

        # Save partial checkpoint after each candidate
        with open(RESULTS_DIR / "pgd_convergence_partial.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    print("\n" + "=" * 90, flush=True)
    print("SAVING FINAL AUDITABLE ARTIFACTS AND PLOTS...", flush=True)
    print("=" * 90, flush=True)

    with open(RESULTS_DIR / "pgd_convergence_final.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    csv_rows = []
    fieldnames = [
        "dataset", "candidate_id", "candidate_name", "steps", "schedule", "format",
        "model", "role", "flips_1pct_k", "flips_1pct_n", "flips_1pct_pct",
        "ci_1pct_low", "ci_1pct_high", "flips_01pct_k", "flips_01pct_n",
        "flips_01pct_pct", "ci_01pct_low", "ci_01pct_high", "mean_sim_drop",
        "face_ssim", "ciede_mean", "ciede_p95", "latency_sec"
    ]

    for d_name in ["lfw", "real"]:
        for c_id, c_data in all_results["candidates"].items():
            for fmt in formats:
                for m in models:
                    m_data = c_data[d_name]["formats"][fmt][m]
                    role = "surrogate" if m in ["ArcFace", "FaceNet"] else "held_out"
                    vis = c_data["visibility"][d_name]
                    lat = c_data["visibility"]["lfw"]["mean_latency_sec"] if d_name == "lfw" else 0.0
                    csv_rows.append({
                        "dataset": d_name,
                        "candidate_id": c_id,
                        "candidate_name": c_data["candidate_name"],
                        "steps": c_data["steps"],
                        "schedule": c_data["schedule"],
                        "format": fmt,
                        "model": m,
                        "role": role,
                        "flips_1pct_k": m_data["flips_1pct_k"],
                        "flips_1pct_n": m_data["flips_1pct_n"],
                        "flips_1pct_pct": m_data["flips_1pct_pct"],
                        "ci_1pct_low": m_data["flips_1pct_ci"][0],
                        "ci_1pct_high": m_data["flips_1pct_ci"][1],
                        "flips_01pct_k": m_data["flips_01pct_k"],
                        "flips_01pct_n": m_data["flips_01pct_n"],
                        "flips_01pct_pct": m_data["flips_01pct_pct"],
                        "ci_01pct_low": m_data["flips_01pct_ci"][0],
                        "ci_01pct_high": m_data["flips_01pct_ci"][1],
                        "mean_sim_drop": m_data["mean_sim_drop"],
                        "face_ssim": vis["face_ssim"],
                        "ciede_mean": vis["ciede_mean"],
                        "ciede_p95": vis["ciede_p95"],
                        "latency_sec": lat,
                    })

    csv_path = RESULTS_DIR / "pgd_convergence_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved machine-readable CSV summary to {csv_path}", flush=True)

    # Generate diagnostic plots
    try:
        generate_diagnostic_plots(RESULTS_DIR, all_results)
        print(f"Generated diagnostic plots in {RESULTS_DIR}", flush=True)
    except Exception as e:
        print(f"Plot generation notice: {e}", flush=True)

    # Clean up tmp folder
    if TMP_DIR.exists():
        for p in TMP_DIR.glob("*.*"):
            p.unlink()
        TMP_DIR.rmdir()

    print("\nEXPERIMENT RUN COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varman 3.0 PGD Convergence Experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 pairs smoke test")
    parser.add_argument("--lfw-limit", type=int, default=150, help="Number of LFW pairs to evaluate")
    args = parser.parse_args()

    run_experiment(smoke_test=args.smoke_test, lfw_limit=args.lfw_limit)
