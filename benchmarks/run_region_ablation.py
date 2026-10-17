"""
Varman Region-Size Ablation Experiment
Evaluates the effect of mask region size (scale 0.7, 1.0, 1.3) and lower epsilon (eps 6 vs 8)
on biometric evasion, visibility, and paired statistical significance across 150 LFW genuine pairs.
Includes Step 3: Phase-randomized matched-grain measurement on the first 50 pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import scipy.stats
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as compute_ssim
from sklearn.datasets import fetch_lfw_pairs
import torch
import torch.nn.functional as F

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.protection.diff_jpeg import DiffJPEG
from app.protection.face_aligner import (
    align_crop_112,
    ARCFACE_CANONICAL_112,
)
from app.protection.protection_engine import ProtectionEngine
from app.protection.strong_engine import get_canonical_face_mask_112
from app.protection.surrogate_models import DualSurrogateSuite
from benchmarks.evaluate_face_models import ModelSuite, cosine_similarity
from benchmarks.run_paste_aware_experiment import (
    build_paste_grid,
    build_realign_grid,
    compute_face_ciede_stats,
    compute_wilson_ci,
)

CALIB_PATH = REPO_ROOT / "benchmarks" / "results" / "sweep_evidence" / "calibrated_thresholds_sweep.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "region_ablation"
IMAGE_STORE_DIR = Path(r"D:\JavaProjects\Varman-research\region_ablation_2026-10-15")

CANONICAL_FACE_MASK_BASE = (get_canonical_face_mask_112() >= 0.10)


def build_scaled_mask(mask_scale: float, feather_sigma: float = 4.0) -> np.ndarray:
    """
    Builds canonical 112x112 elliptical blending mask scaled by mask_scale.
    Base ellipse semi-axes: rx=42, ry=50.
    OpenCV automatically clamps rasterization to canvas boundaries [0, 112) x [0, 112).
    """
    mask_112 = np.zeros((112, 112), dtype=np.float32)
    rx = int(round(42.0 * mask_scale))
    ry = int(round(50.0 * mask_scale))
    cv2.ellipse(mask_112, (56, 56), (rx, ry), 0, 0, 360, 1.0, -1)
    if feather_sigma > 0:
        mask_112 = cv2.GaussianBlur(mask_112, (15, 15), feather_sigma)
    return mask_112


def composite_crop_back_scaled(
    original_bgr: np.ndarray,
    protected_crop_112: np.ndarray,
    affine_matrix_M: np.ndarray,
    mask_scale: float = 1.0,
    feather_sigma: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Composites crop back to canvas resolution using inverse affine transform
    and scaled elliptical feathered mask. Returns (composited_bgr, warped_mask).
    """
    h, w = original_bgr.shape[:2]
    inv_M = cv2.invertAffineTransform(affine_matrix_M)
    mask_112 = build_scaled_mask(mask_scale, feather_sigma=feather_sigma)

    warped_prot = cv2.warpAffine(
        protected_crop_112.astype(np.float32), inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    warped_mask = cv2.warpAffine(
        mask_112, inv_M, (w, h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    warped_mask_3ch = np.repeat(warped_mask[:, :, np.newaxis], 3, axis=2)

    composited = (
        warped_mask_3ch * warped_prot.astype(np.float32)
        + (1.0 - warped_mask_3ch) * original_bgr.astype(np.float32)
    )
    clipped = np.clip(composited, 0.0, 255.0).astype(np.uint8)
    return clipped, warped_mask


class RegionAblationEngine:
    """
    Paste-aware PGD engine parameterized by mask_scale.
    Operates in luminance space with DiffJPEG, straight-through rounding,
    and online landmark jitter.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.surrogates = DualSurrogateSuite(device)
        self.diff_jpeg = DiffJPEG().to(device).eval()

    def protect(
        self,
        img_bgr: np.ndarray,
        landmarks: List[Tuple[int, int]],
        crop_112_bgr: np.ndarray,
        M: np.ndarray,
        mask_scale: float = 1.0,
        steps: int = 25,
        w_fn: float = 1.0,
        eps: float = 8.0 / 255.0,
        mu: float = 0.9,
        q_sequence: Optional[List[int]] = None,
        jitter_params: Optional[List[Tuple[float, float, float]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], torch.Tensor]:
        alpha = eps / 7.0
        h, w = img_bgr.shape[:2]
        inv_M = cv2.invertAffineTransform(M).astype(np.float32)

        crop_rgb = cv2.cvtColor(crop_112_bgr, cv2.COLOR_BGR2RGB)
        x_clean = (
            torch.from_numpy(crop_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 255.0
        )

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x_full_orig = (
            torch.from_numpy(img_rgb)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
            / 255.0
        )

        with torch.no_grad():
            clean_af, clean_fn = self.surrogates(x_clean)

        # Scaled mask tensor in 112x112 space
        mask_np = build_scaled_mask(mask_scale, feather_sigma=4.0)
        t_mask_112 = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # Precompute paste grid and warped blending mask
        grid_paste = build_paste_grid(M, dst_w=w, dst_h=h, src_w=112, src_h=112, device=self.device)
        warped_mask = F.grid_sample(t_mask_112, grid_paste, mode="bilinear", padding_mode="zeros", align_corners=False)

        grid_realign_base = build_realign_grid(inv_M, dst_w=112, dst_h=112, src_w=w, src_h=h, device=self.device)

        delta = torch.zeros_like(x_clean, requires_grad=True)
        momentum = torch.zeros_like(x_clean)

        src_pts = np.array(landmarks[:5], dtype=np.float32)
        cx, cy = float(np.mean(src_pts[:, 0])), float(np.mean(src_pts[:, 1]))

        for t in range(steps):
            q = float(q_sequence[t]) if (q_sequence is not None and t < len(q_sequence)) else 75.0

            # 1. Perturbation clamp
            x_crop_adv = torch.clamp(x_clean + delta, 0.0, 1.0)

            # 2. Differentiable paste
            warped_prot = F.grid_sample(x_crop_adv, grid_paste, mode="bilinear", padding_mode="reflection", align_corners=False)
            x_comp = warped_mask * warped_prot + (1.0 - warped_mask) * x_full_orig

            # 3. 8-bit straight-through rounding
            x_scaled = x_comp * 255.0
            x_round = x_scaled + (torch.round(x_scaled) - x_scaled).detach()
            x_round_01 = torch.clamp(x_round / 255.0, 0.0, 1.0)

            # 4. DiffJPEG
            x_jpeg = self.diff_jpeg(x_round_01, quality=q)

            # 5. Differentiable re-align with paired jitter
            if jitter_params is not None and t < len(jitter_params):
                s, tx, ty = jitter_params[t]
                jittered_pts = np.zeros_like(src_pts)
                jittered_pts[:, 0] = cx + s * (src_pts[:, 0] - cx) + tx
                jittered_pts[:, 1] = cy + s * (src_pts[:, 1] - cy) + ty
                M_jit, _ = cv2.estimateAffinePartial2D(jittered_pts, ARCFACE_CANONICAL_112)
                if M_jit is not None:
                    inv_M_jit = cv2.invertAffineTransform(M_jit).astype(np.float32)
                    grid_realign = build_realign_grid(inv_M_jit, dst_w=112, dst_h=112, src_w=w, src_h=h, device=self.device)
                else:
                    grid_realign = grid_realign_base
            else:
                grid_realign = grid_realign_base

            x_realigned = F.grid_sample(x_jpeg, grid_realign, mode="bilinear", padding_mode="reflection", align_corners=False)

            # 6. Surrogate loss
            adv_af, adv_fn = self.surrogates(x_realigned)
            sim_af = (adv_af * clean_af).sum(dim=1).mean()
            sim_fn = (adv_fn * clean_fn).sum(dim=1).mean()
            loss = 1.0 * sim_af + w_fn * sim_fn

            grad = torch.autograd.grad(loss, delta)[0]
            grad_norm = grad.norm(p=1) + 1e-12
            momentum = mu * momentum + grad / grad_norm

            with torch.no_grad():
                delta = delta - alpha * torch.sign(momentum)
                d_lum = delta.mean(dim=1, keepdim=True)
                low = torch.maximum(torch.tensor(-eps, device=self.device), -x_clean.min(dim=1, keepdim=True).values)
                high = torch.minimum(torch.tensor(eps, device=self.device), 1.0 - x_clean.max(dim=1, keepdim=True).values)
                d_lum = torch.clamp(d_lum, low, high)
                delta = d_lum.repeat(1, 3, 1, 1)

            delta.requires_grad = True

        with torch.no_grad():
            x_adv_final = torch.clamp(x_clean + delta, 0.0, 1.0)
            prot_crop_rgb = (
                x_adv_final.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            ).round().astype(np.uint8)
            prot_crop_bgr = cv2.cvtColor(prot_crop_rgb, cv2.COLOR_RGB2BGR)

        stats = {
            "steps": steps,
            "mask_scale": mask_scale,
            "eps": eps,
            "final_sim_af": sim_af.item(),
            "final_sim_fn": sim_fn.item(),
        }
        return prot_crop_bgr, stats, x_adv_final.detach()


def run_mcnemar_exact(
    flips_base: List[bool],
    flips_cand: List[bool],
) -> Dict[str, Any]:
    """
    Computes exact paired McNemar test between baseline and candidate.
    b: baseline flipped (True), candidate failed (False)
    c: candidate flipped (True), baseline failed (False)
    """
    assert len(flips_base) == len(flips_cand)
    b = 0
    c = 0
    a = 0  # both True
    d = 0  # both False
    for fb, fc in zip(flips_base, flips_cand):
        if fb and not fc:
            b += 1
        elif not fb and fc:
            c += 1
        elif fb and fc:
            a += 1
        else:
            d += 1

    n_discordant = b + c
    if n_discordant == 0:
        p_val = 1.0
    else:
        # Two-sided binomial test under H0: p = 0.5
        res = scipy.stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
        p_val = float(res.pvalue)

    return {
        "contingency": {"a": a, "b": b, "c": c, "d": d},
        "b": b,
        "c": c,
        "n_discordant": n_discordant,
        "exact_p": p_val,
    }


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies step-down Holm-Bonferroni correction to a list of p-values."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p_adj = p_values[idx] * (m - rank)
        cum_max = max(cum_max, p_adj)
        adjusted[idx] = min(1.0, cum_max)
    return adjusted


# ==============================================================================
# MAIN EXPERIMENT EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Varman Region-Size Ablation Experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Quick run on 2 pairs")
    parser.add_argument("--lfw-limit", type=int, default=150, help="Number of LFW pairs (default: 150)")
    args = parser.parse_args()

    print("=" * 90, flush=True)
    print("VARMAN — REGION-SIZE ABLATION EXPERIMENT (150 LFW Genuine Pairs)", flush=True)
    print(f"Start Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # Free space checks
    disk_c = shutil.disk_usage("C:\\")
    disk_d = shutil.disk_usage("D:\\")
    c_free_gb = disk_c.free / (1024 ** 3)
    d_free_gb = disk_d.free / (1024 ** 3)
    print(f"[Preflight] Free Space: C: {c_free_gb:.2f} GB, D: {d_free_gb:.2f} GB", flush=True)
    if c_free_gb < 10.0:
        raise RuntimeError(f"STOP CONDITION: C: drive free space ({c_free_gb:.2f} GB) is below 10 GB limit.")

    IMAGE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load thresholds
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    thresh_1pct = calib_data["thresholds_1pct"]
    thresh_01pct = calib_data["thresholds_01pct"]
    print(f"[Thresholds] 1% FAR: {thresh_1pct}", flush=True)
    print(f"[Thresholds] 0.1% FAR: {thresh_01pct}", flush=True)

    engine = ProtectionEngine(model_dir=str(REPO_ROOT / "models"))
    model_suite = ModelSuite(models_dir=REPO_ROOT / "models", requested_device="cuda")
    pgd_engine = RegionAblationEngine(device=device)

    # Load LFW pairs
    target_count = 2 if args.smoke_test else args.lfw_limit
    print(f"\n[LFW] Fetching LFW test split and locating {target_count} genuine pairs...", flush=True)
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
        if len(lfw_pairs) >= target_count:
            break

    n_pairs = len(lfw_pairs)
    print(f"[LFW] Loaded {n_pairs} genuine pairs with verified 112x112 canonical crops.\n", flush=True)

    # Pre-extract clean embeddings for all pairs
    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    clean_embs_1: List[Dict[str, np.ndarray]] = []
    clean_embs_2: List[Dict[str, np.ndarray]] = []
    clean_match_1pct: Dict[str, List[bool]] = {m: [] for m in models}
    clean_match_01pct: Dict[str, List[bool]] = {m: [] for m in models}

    print("[Clean Baseline] Extracting embeddings and verifying clean pair matches...", flush=True)
    for p_i, (b1, b2, f1, f2, _, _) in enumerate(lfw_pairs):
        e1, _ = model_suite.extract_model_features(b1, f1)
        e2, _ = model_suite.extract_model_features(b2, f2)
        clean_embs_1.append(e1)
        clean_embs_2.append(e2)
        for m in models:
            s = float(cosine_similarity(e1[m], e2[m])) if (e1[m] is not None and e2[m] is not None) else 0.0
            clean_match_1pct[m].append(bool(s >= thresh_1pct[m]))
            clean_match_01pct[m].append(bool(s >= thresh_01pct[m]))

    clean_denoms_1pct = {m: int(sum(clean_match_1pct[m])) for m in models}
    clean_denoms_01pct = {m: int(sum(clean_match_01pct[m])) for m in models}
    print(f"[Clean Baseline] Denominators at 1% FAR: {clean_denoms_1pct}", flush=True)
    print(f"[Clean Baseline] Denominators at 0.1% FAR: {clean_denoms_01pct}", flush=True)

    # Pre-generate paired random schedules per pair
    # Same seed 2026 + p_i ensures R1, R2, R3, R4 get identical JPEG qualities and jitter transformations
    paired_q_sequences: List[List[int]] = []
    paired_jitter_params: List[List[Tuple[float, float, float]]] = []
    for p_i in range(n_pairs):
        rng = np.random.default_rng(2026 + p_i)
        q_seq = [int(rng.integers(50, 96)) for _ in range(25)]
        jit = [(float(rng.uniform(0.97, 1.03)), float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0))) for _ in range(25)]
        paired_q_sequences.append(q_seq)
        paired_jitter_params.append(jit)

    # Define Rows
    # Report effective changed-area fraction for each scale
    area_stats = {}
    for sc in [0.7, 1.0, 1.3]:
        m_tmp = build_scaled_mask(sc, 4.0)
        area_stats[sc] = {
            "rx": int(round(42.0 * sc)),
            "ry": int(round(50.0 * sc)),
            "strict_frac": float(np.mean(m_tmp >= 0.10)),
            "feathered_frac": float(np.mean(m_tmp >= 0.001)),
        }
    print("[Mask Scale Analysis]")
    for sc, ast in area_stats.items():
        print(f"  Scale {sc:3.1f}: radii=({ast['rx']}, {ast['ry']}) | strict area = {ast['strict_frac']*100:.2f}% | feathered area = {ast['feathered_frac']*100:.2f}%", flush=True)

    ROWS = [
        {
            "id": "R0_control",
            "name": "Row 0: Unprotected Control",
            "steps": 0,
            "eps": 8.0 / 255.0,
            "mask_scale": 1.0,
            "jitter": False,
        },
        {
            "id": "R1_base_scale10",
            "name": "Row 1: Baseline (eps=8/255, mask_scale=1.0)",
            "steps": 25,
            "eps": 8.0 / 255.0,
            "mask_scale": 1.0,
            "jitter": True,
        },
        {
            "id": "R2_scale07",
            "name": "Row 2: Smaller Mask (eps=8/255, mask_scale=0.7)",
            "steps": 25,
            "eps": 8.0 / 255.0,
            "mask_scale": 0.7,
            "jitter": True,
        },
        {
            "id": "R3_scale13",
            "name": "Row 3: Larger Mask (eps=8/255, mask_scale=1.3)",
            "steps": 25,
            "eps": 8.0 / 255.0,
            "mask_scale": 1.3,
            "jitter": True,
        },
        {
            "id": "R4_eps6_scale10",
            "name": "Row 4: Lower-Visibility Frontier (eps=6/255, mask_scale=1.0)",
            "steps": 25,
            "eps": 6.0 / 255.0,
            "mask_scale": 1.0,
            "jitter": True,
        },
    ]

    formats = ["png", "q75", "q50"]
    all_results: Dict[str, Any] = {
        "benchmark_metadata": {
            "name": "Varman Region-Size Ablation Experiment",
            "date": time.strftime("%Y-%m-%d"),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "n_pairs": n_pairs,
            "thresholds_1pct": thresh_1pct,
            "thresholds_01pct": thresh_01pct,
            "clean_denoms_1pct": clean_denoms_1pct,
            "clean_denoms_01pct": clean_denoms_01pct,
            "mask_area_stats": area_stats,
            "image_store_dir": str(IMAGE_STORE_DIR),
        },
        "candidates": {},
        "pair_flips": {},
        "pair_sims": {},
    }

    recent_latencies: List[float] = []

    # Execute Rows
    for row_idx, row_cfg in enumerate(ROWS):
        r_id = row_cfg["id"]
        r_name = row_cfg["name"]
        steps = row_cfg["steps"]
        eps = row_cfg["eps"]
        m_scale = row_cfg["mask_scale"]
        use_jitter = row_cfg["jitter"]

        row_dir = IMAGE_STORE_DIR / r_id
        row_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90, flush=True)
        print(f"RUNNING {r_name} (Row {row_idx + 1} of {len(ROWS)})", flush=True)
        print(f"  steps={steps}, eps={eps:.5f}, mask_scale={m_scale}, jitter={use_jitter}", flush=True)
        print("=" * 90, flush=True)

        cand_output: Dict[str, Any] = {
            "config": row_cfg,
            "results_by_format": {},
        }

        # Per-pair flip and sim storage
        cand_pair_flips: Dict[str, Dict[str, List[bool]]] = {fmt: {m: [] for m in models} for fmt in formats}
        cand_pair_sims: Dict[str, Dict[str, List[float]]] = {fmt: {m: [] for m in models} for fmt in formats}

        lfw_ssims: List[float] = []
        lfw_full_ssims: List[float] = []
        lfw_ciede_m: List[float] = []
        lfw_ciede_p95: List[float] = []
        lfw_ciede_p99: List[float] = []
        lfw_full_ciede_m: List[float] = []
        lfw_full_ciede_max: List[float] = []
        lfw_max_pixel_changes: List[float] = []
        lfw_mean_pixel_changes: List[float] = []
        lfw_changed_inside_fracs: List[float] = []
        lfw_changed_outside_fracs: List[float] = []
        lfw_latencies: List[float] = []

        row_t0 = time.perf_counter()

        for p_i, (b1, b2, f1, f2, c1, c2) in enumerate(lfw_pairs):
            t_pair0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

            # Seed for pair reproducibility across rows
            torch.manual_seed(2026 + p_i)

            if steps > 0:
                prot_crop_bgr, _, _ = pgd_engine.protect(
                    img_bgr=b1,
                    landmarks=f1["landmarks"],
                    crop_112_bgr=c1,
                    M=f1.get("M", cv2.estimateAffinePartial2D(np.array(f1["landmarks"][:5], dtype=np.float32), ARCFACE_CANONICAL_112)[0]),
                    mask_scale=m_scale,
                    steps=steps,
                    w_fn=1.0,
                    eps=eps,
                    mu=0.9,
                    q_sequence=paired_q_sequences[p_i],
                    jitter_params=paired_jitter_params[p_i] if use_jitter else None,
                )
                M1 = cv2.estimateAffinePartial2D(np.array(f1["landmarks"][:5], dtype=np.float32), ARCFACE_CANONICAL_112)[0]
                prot_full_bgr, warped_mask = composite_crop_back_scaled(
                    original_bgr=b1,
                    protected_crop_112=prot_crop_bgr,
                    affine_matrix_M=M1,
                    mask_scale=m_scale,
                    feather_sigma=4.0,
                )
            else:
                prot_crop_bgr = c1.copy()
                prot_full_bgr = b1.copy()
                warped_mask = np.zeros(b1.shape[:2], dtype=np.float32)

            t_pair1 = time.perf_counter()
            lat_sec = t_pair1 - t_pair0
            lfw_latencies.append(lat_sec)
            recent_latencies.append(lat_sec)
            if len(recent_latencies) > 5:
                recent_latencies.pop(0)

            # Check stop conditions
            if len(recent_latencies) == 5 and (sum(recent_latencies) / 5.0) > 6.0:
                raise RuntimeError(f"STOP CONDITION: Sustained latency ({sum(recent_latencies)/5.0:.2f}s) exceeded 6.0s over 5 pairs.")
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
            if peak_vram > 3.8:
                raise RuntimeError(f"STOP CONDITION: Peak VRAM ({peak_vram:.3f} GB) exceeded 3.8 GB limit.")

            # Save real images to disk
            png_path = row_dir / f"pair_{p_i:04d}_png.png"
            q75_path = row_dir / f"pair_{p_i:04d}_q75.jpg"
            q50_path = row_dir / f"pair_{p_i:04d}_q50.jpg"

            cv2.imwrite(str(png_path), prot_full_bgr)
            cv2.imwrite(str(q75_path), prot_full_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            cv2.imwrite(str(q50_path), prot_full_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            # Reload from disk (strictly real saved-and-reloaded files)
            reloads = {
                "png": cv2.imread(str(png_path)),
                "q75": cv2.imread(str(q75_path)),
                "q50": cv2.imread(str(q50_path)),
            }

            # Evaluate each format
            for fmt in formats:
                rel_img = reloads[fmt]
                rel_faces = engine.detect_faces(rel_img)
                rel_face = rel_faces[0] if rel_faces else f1
                embs_prot, _ = model_suite.extract_model_features(rel_img, rel_face)
                e2 = clean_embs_2[p_i]

                for m in models:
                    ep = embs_prot.get(m)
                    e_clean2 = e2.get(m)
                    if ep is not None and e_clean2 is not None:
                        sim = float(cosine_similarity(ep, e_clean2))
                    else:
                        sim = 1.0
                    cand_pair_sims[fmt][m].append(sim)
                    is_flip = bool(sim < thresh_1pct[m])
                    cand_pair_flips[fmt][m].append(is_flip)

            # Compute Visibility Metrics (PNG vs Clean b1)
            diff_full = np.abs(prot_full_bgr.astype(np.float32) - b1.astype(np.float32))
            max_px_diff = float(np.max(diff_full))
            mean_px_diff = float(np.mean(diff_full))

            # Changed fraction inside mask (>= 0.001) and outside mask (< 0.001)
            in_mask = (warped_mask >= 0.001)
            out_mask = (warped_mask < 0.001)
            n_in = int(np.sum(in_mask))
            n_out = int(np.sum(out_mask))
            c_in = int(np.sum((diff_full > 0).any(axis=2) & in_mask))
            c_out = int(np.sum((diff_full > 0).any(axis=2) & out_mask))
            ch_in_frac = float(c_in) / float(n_in) if n_in > 0 else 0.0
            ch_out_frac = float(c_out) / float(n_out) if n_out > 0 else 0.0

            # Canonical face crop visibility
            f_ssim = float(compute_ssim(c1, prot_crop_bgr, channel_axis=2, data_range=255))
            full_ssim = float(compute_ssim(b1, prot_full_bgr, channel_axis=2, data_range=255))
            m_de, p95_de, p99_de = compute_face_ciede_stats(c1, prot_crop_bgr)

            lab_clean = rgb2lab(cv2.cvtColor(b1, cv2.COLOR_BGR2RGB) / 255.0)
            lab_prot = rgb2lab(cv2.cvtColor(prot_full_bgr, cv2.COLOR_BGR2RGB) / 255.0)
            de_full = deltaE_ciede2000(lab_clean, lab_prot)
            full_ciede_m = float(np.mean(de_full))
            full_ciede_mx = float(np.max(de_full))

            lfw_ssims.append(f_ssim)
            lfw_full_ssims.append(full_ssim)
            lfw_ciede_m.append(m_de)
            lfw_ciede_p95.append(p95_de)
            lfw_ciede_p99.append(p99_de)
            lfw_full_ciede_m.append(full_ciede_m)
            lfw_full_ciede_max.append(full_ciede_mx)
            lfw_max_pixel_changes.append(max_px_diff)
            lfw_mean_pixel_changes.append(mean_px_diff)
            lfw_changed_inside_fracs.append(ch_in_frac)
            lfw_changed_outside_fracs.append(ch_out_frac)

            if (p_i + 1) % 15 == 0 or (p_i + 1) == n_pairs:
                print(f"  [{r_id}] Completed {p_i + 1:3d} / {n_pairs} pairs | Pair latency: {lat_sec:.2f}s | Peak VRAM: {peak_vram:.3f} GB", flush=True)

        row_time = time.perf_counter() - row_t0

        # Compile row statistics across formats
        for fmt in formats:
            fmt_res: Dict[str, Any] = {}
            for m in models:
                flips_list = cand_pair_flips[fmt][m]
                valid_mask_1 = clean_match_1pct[m]
                valid_flips_1 = [flips_list[i] for i in range(n_pairs) if valid_mask_1[i]]
                k1 = sum(valid_flips_1)
                n1 = len(valid_flips_1)
                pct1 = (k1 / n1 * 100.0) if n1 > 0 else 0.0
                ci1 = compute_wilson_ci(k1, n1)

                valid_mask_01 = clean_match_01pct[m]
                valid_flips_01 = [bool(cand_pair_sims[fmt][m][i] < thresh_01pct[m]) for i in range(n_pairs) if valid_mask_01[i]]
                k01 = sum(valid_flips_01)
                n01 = len(valid_flips_01)
                pct01 = (k01 / n01 * 100.0) if n01 > 0 else 0.0
                ci01 = compute_wilson_ci(k01, n01)

                sim_drops = [float(clean_embs_1[i][m] @ clean_embs_2[i][m] - cand_pair_sims[fmt][m][i]) for i in range(n_pairs) if clean_embs_1[i][m] is not None]
                mean_drop = float(np.mean(sim_drops)) if sim_drops else 0.0

                fmt_res[m] = {
                    "flips_1pct_k": k1,
                    "flips_1pct_n": n1,
                    "flips_1pct_pct": round(pct1, 2),
                    "flips_1pct_ci_low": round(ci1[0], 2),
                    "flips_1pct_ci_high": round(ci1[1], 2),
                    "flips_01pct_k": k01,
                    "flips_01pct_n": n01,
                    "flips_01pct_pct": round(pct01, 2),
                    "flips_01pct_ci_low": round(ci01[0], 2),
                    "flips_01pct_ci_high": round(ci01[1], 2),
                    "mean_sim_drop": round(mean_drop, 4),
                }

            mean_4m_1pct = float(np.mean([fmt_res[m]["flips_1pct_pct"] for m in models]))
            heldout_1pct = float(np.mean([fmt_res["SFace"]["flips_1pct_pct"], fmt_res["MobileFaceNet"]["flips_1pct_pct"]]))
            mean_4m_01pct = float(np.mean([fmt_res[m]["flips_01pct_pct"] for m in models]))
            heldout_01pct = float(np.mean([fmt_res["SFace"]["flips_01pct_pct"], fmt_res["MobileFaceNet"]["flips_01pct_pct"]]))

            fmt_res["summary"] = {
                "mean_flip_1pct": round(mean_4m_1pct, 2),
                "heldout_flip_1pct": round(heldout_1pct, 2),
                "mean_flip_01pct": round(mean_4m_01pct, 2),
                "heldout_flip_01pct": round(heldout_01pct, 2),
            }
            cand_output["results_by_format"][fmt] = fmt_res

        cand_output["visibility"] = {
            "face_ssim": round(float(np.mean(lfw_ssims)), 4),
            "full_ssim": round(float(np.mean(lfw_full_ssims)), 4),
            "ciede_mean": round(float(np.mean(lfw_ciede_m)), 4),
            "ciede_p95": round(float(np.mean(lfw_ciede_p95)), 4),
            "ciede_p99": round(float(np.mean(lfw_ciede_p99)), 4),
            "full_ciede_mean": round(float(np.mean(lfw_full_ciede_m)), 4),
            "full_ciede_max": round(float(np.mean(lfw_full_ciede_max)), 4),
            "mean_pixel_change": round(float(np.mean(lfw_mean_pixel_changes)), 4),
            "max_pixel_change_max": round(float(np.max(lfw_max_pixel_changes)), 2),
            "changed_inside_frac": round(float(np.mean(lfw_changed_inside_fracs)), 4),
            "changed_outside_frac": round(float(np.mean(lfw_changed_outside_fracs)), 4),
            "mean_latency_sec": round(float(np.mean(lfw_latencies)), 2),
            "total_row_time_sec": round(row_time, 2),
            "peak_vram_gb": round(peak_vram, 3),
        }

        all_results["candidates"][r_id] = cand_output
        all_results["pair_flips"][r_id] = cand_pair_flips
        all_results["pair_sims"][r_id] = cand_pair_sims

        print(f"  --> Completed {r_id} in {row_time:.1f}s")
        print(f"  --> Q50 1% FAR: 4-model mean = {cand_output['results_by_format']['q50']['summary']['mean_flip_1pct']}%, held-out = {cand_output['results_by_format']['q50']['summary']['heldout_flip_1pct']}%")
        print(f"  --> Visibility: Face SSIM={cand_output['visibility']['face_ssim']}, CIEDE Mean={cand_output['visibility']['ciede_mean']}, p95={cand_output['visibility']['ciede_p95']}", flush=True)

        # Write partial results after every row
        partial_json_path = RESULTS_DIR / "region_ablation_partial.json"
        with open(partial_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  --> Saved partial results to {partial_json_path}", flush=True)

    # ==========================================================================
    # PAIRED STATISTICAL TESTS (McNemar with Holm-Bonferroni correction)
    # ==========================================================================
    print("\n" + "=" * 90, flush=True)
    print("COMPUTING PAIRED EXACT MCNEMAR TESTS (vs Row 1 Baseline at Q50, 1% FAR)", flush=True)
    print("=" * 90, flush=True)

    r1_flips = all_results["pair_flips"]["R1_base_scale10"]["q50"]
    mcnemar_results: Dict[str, Any] = {}

    for cand_id in ["R2_scale07", "R3_scale13", "R4_eps6_scale10"]:
        cand_flips = all_results["pair_flips"][cand_id]["q50"]
        cand_mcnemar: Dict[str, Any] = {}
        raw_p_values: List[float] = []

        for m in models:
            valid_idx = [i for i in range(n_pairs) if clean_match_1pct[m][i]]
            f_r1_valid = [r1_flips[m][i] for i in valid_idx]
            f_cand_valid = [cand_flips[m][i] for i in valid_idx]

            m_test = run_mcnemar_exact(f_r1_valid, f_cand_valid)
            cand_mcnemar[m] = m_test
            raw_p_values.append(m_test["exact_p"])

        # Holm-Bonferroni correction over the 4 models
        adj_p_values = holm_bonferroni(raw_p_values)
        for idx, m in enumerate(models):
            cand_mcnemar[m]["holm_adj_p"] = round(adj_p_values[idx], 5)
            cand_mcnemar[m]["significant_05"] = bool(adj_p_values[idx] < 0.05)

        mcnemar_results[cand_id] = cand_mcnemar
        print(f"\nPaired Tests for {cand_id} vs R1_base_scale10:")
        for m in models:
            t = cand_mcnemar[m]
            print(f"  {m:14s}: b={t['b']:2d} (R1 win), c={t['c']:2d} (Cand win) | raw p={t['exact_p']:.4f} | Holm adj p={t['holm_adj_p']:.4f} | Sig: {t['significant_05']}", flush=True)

    all_results["mcnemar_tests_q50_1pct"] = mcnemar_results

    # ==========================================================================
    # STEP 3: MATCHED-GRAIN MEASUREMENT (H2) on first 50 pairs
    # ==========================================================================
    print("\n" + "=" * 90, flush=True)
    print("STEP 3: MATCHED-GRAIN MEASUREMENT (H2) on First 50 Pairs (No PGD)", flush=True)
    print("=" * 90, flush=True)

    grain_dir = IMAGE_STORE_DIR / "grain_extended"
    grain_dir.mkdir(parents=True, exist_ok=True)

    grain_pairs_count = min(50, n_pairs)
    r1_dir = IMAGE_STORE_DIR / "R1_base_scale10"

    grain_results = {
        "pairs_evaluated": grain_pairs_count,
        "models": models,
        "eval_by_format": {"png": {}, "q50": {}},
        "yunet_detection_rate": 0.0,
        "mean_landmark_displacement_px": 0.0,
        "whole_image_ssim": 0.0,
        "whole_image_ciede": 0.0,
        "changed_pixel_fraction": 0.0,
        "conspicuity_proxy": {
            "r1_conspicuity_ratio": 0.0,
            "grain_conspicuity_ratio": 0.0,
            "r1_rms_inside": 0.0,
            "r1_rms_ring": 0.0,
            "grain_rms_inside": 0.0,
            "grain_rms_ring": 0.0,
            "note": "Proxy metric: high-pass RMS inside face mask vs 15px ring. NOT human evidence.",
        }
    }

    grain_flips_png = {m: [] for m in models}
    grain_flips_q50 = {m: [] for m in models}
    r1_50_flips_png = {m: [] for m in models}
    r1_50_flips_q50 = {m: [] for m in models}

    landmark_displacements: List[float] = []
    detections_count = 0
    whole_ssims: List[float] = []
    whole_ciedes: List[float] = []
    changed_fractions: List[float] = []

    r1_rms_insides: List[float] = []
    r1_rms_rings: List[float] = []
    grain_rms_insides: List[float] = []
    grain_rms_rings: List[float] = []

    kernel_ring = np.ones((31, 31), np.uint8)

    for p_i in range(grain_pairs_count):
        b1, b2, f1, f2, c1, c2 = lfw_pairs[p_i]
        r1_png_path = r1_dir / f"pair_{p_i:04d}_png.png"
        r1_img = cv2.imread(str(r1_png_path))
        if r1_img is None:
            continue

        h, w = b1.shape[:2]
        delta_r1 = r1_img.astype(np.float32) - b1.astype(np.float32)

        M1 = cv2.estimateAffinePartial2D(np.array(f1["landmarks"][:5], dtype=np.float32), ARCFACE_CANONICAL_112)[0]
        mask_112 = build_scaled_mask(1.0, 4.0)
        inv_M = cv2.invertAffineTransform(M1)
        warped_mask = cv2.warpAffine(mask_112, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        # Synthesize matched grain via phase-randomized 2D FFT per channel
        rng_grain = np.random.default_rng(2026 + p_i)
        grain = np.zeros_like(delta_r1)
        for ch in range(3):
            d_ch = delta_r1[:, :, ch]
            F_ch = np.fft.rfft2(d_ch)
            amp = np.abs(F_ch)
            phase = rng_grain.uniform(-np.pi, np.pi, size=F_ch.shape)
            phase[0, 0] = 0.0
            F_rand = amp * np.exp(1j * phase)
            syn = np.fft.irfft2(F_rand, s=(h, w))

            # Match variance of active face perturbation
            active_std = np.std(d_ch[warped_mask > 0.5]) if np.sum(warped_mask > 0.5) > 0 else 1.0
            syn_std = np.std(syn)
            if syn_std > 1e-6:
                syn = syn * (active_std / syn_std)
            grain[:, :, ch] = syn

        # Seamless blend: inside warped mask is identical to R1; outside is matched grain
        w_mask_3ch = warped_mask[:, :, None]
        grain_comp = w_mask_3ch * r1_img.astype(np.float32) + (1.0 - w_mask_3ch) * (b1.astype(np.float32) + grain)
        grain_img = np.clip(grain_comp, 0.0, 255.0).astype(np.uint8)

        # Save grain extended variants
        g_png_path = grain_dir / f"pair_{p_i:04d}_png.png"
        g_q50_path = grain_dir / f"pair_{p_i:04d}_q50.jpg"
        cv2.imwrite(str(g_png_path), grain_img)
        cv2.imwrite(str(g_q50_path), grain_img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

        g_png_rel = cv2.imread(str(g_png_path))
        g_q50_rel = cv2.imread(str(g_q50_path))

        # Re-detect face on grain image
        faces_g = engine.detect_faces(g_png_rel)
        if faces_g:
            detections_count += 1
            fg = faces_g[0]
            # Landmark displacement vs f1 (YuNet on clean)
            lm_b1 = np.array(f1["landmarks"][:5], dtype=np.float32)
            lm_g = np.array(fg["landmarks"][:5], dtype=np.float32)
            disp = float(np.mean(np.linalg.norm(lm_g - lm_b1, axis=1)))
            landmark_displacements.append(disp)
        else:
            fg = f1

        # Extract embeddings and evaluate flips for grain variants
        embs_g_png, _ = model_suite.extract_model_features(g_png_rel, fg)
        embs_g_q50, _ = model_suite.extract_model_features(g_q50_rel, fg)
        e2 = clean_embs_2[p_i]

        for m in models:
            s_png = float(cosine_similarity(embs_g_png[m], e2[m])) if (embs_g_png[m] is not None and e2[m] is not None) else 1.0
            s_q50 = float(cosine_similarity(embs_g_q50[m], e2[m])) if (embs_g_q50[m] is not None and e2[m] is not None) else 1.0
            grain_flips_png[m].append(bool(s_png < thresh_1pct[m]))
            grain_flips_q50[m].append(bool(s_q50 < thresh_1pct[m]))

            # Collect R1 flips for exact same 50 pairs
            r1_flips_png_val = all_results["pair_flips"]["R1_base_scale10"]["png"][m][p_i]
            r1_flips_q50_val = all_results["pair_flips"]["R1_base_scale10"]["q50"][m][p_i]
            r1_50_flips_png[m].append(r1_flips_png_val)
            r1_50_flips_q50[m].append(r1_flips_q50_val)

        # Whole-image visibility
        w_ssim = float(compute_ssim(b1, grain_img, channel_axis=2, data_range=255))
        lab_b1 = rgb2lab(cv2.cvtColor(b1, cv2.COLOR_BGR2RGB) / 255.0)
        lab_g = rgb2lab(cv2.cvtColor(grain_img, cv2.COLOR_BGR2RGB) / 255.0)
        w_ciede = float(np.mean(deltaE_ciede2000(lab_b1, lab_g)))
        whole_ssims.append(w_ssim)
        whole_ciedes.append(w_ciede)

        diff_g = np.abs(grain_img.astype(np.float32) - b1.astype(np.float32))
        changed_fractions.append(float(np.mean((diff_g > 0).any(axis=2))))

        # Conspicuity proxy: high-pass RMS inside mask vs 15px ring around it
        # High pass: delta minus Gaussian blur (sigma=2.0)
        def compute_conspicuity(delta_arr: np.ndarray, mask_2d: np.ndarray):
            blur = cv2.GaussianBlur(delta_arr, (9, 9), 2.0)
            hp = delta_arr - blur
            hp_lum = np.mean(hp, axis=2)

            core_mask = (mask_2d >= 0.5)
            ring_outer = cv2.dilate((mask_2d >= 0.05).astype(np.uint8), kernel_ring, iterations=1)
            ring_mask = (ring_outer > 0) & (mask_2d < 0.05)

            rms_in = float(np.sqrt(np.mean(hp_lum[core_mask] ** 2))) if np.sum(core_mask) > 0 else 0.0
            rms_rg = float(np.sqrt(np.mean(hp_lum[ring_mask] ** 2))) if np.sum(ring_mask) > 0 else 0.0
            ratio = (rms_in / (rms_rg + 1e-6))
            return rms_in, rms_rg, ratio

        r_in_r1, r_rg_r1, _ = compute_conspicuity(delta_r1, warped_mask)
        r_in_g, r_rg_g, _ = compute_conspicuity(grain_comp - b1.astype(np.float32), warped_mask)

        r1_rms_insides.append(r_in_r1)
        r1_rms_rings.append(r_rg_r1)
        grain_rms_insides.append(r_in_g)
        grain_rms_rings.append(r_rg_g)

    # Summarize Step 3
    grain_results["yunet_detection_rate"] = round(float(detections_count / grain_pairs_count * 100.0), 2)
    grain_results["mean_landmark_displacement_px"] = round(float(np.mean(landmark_displacements)), 3)
    grain_results["whole_image_ssim"] = round(float(np.mean(whole_ssims)), 4)
    grain_results["whole_image_ciede"] = round(float(np.mean(whole_ciedes)), 4)
    grain_results["changed_pixel_fraction"] = round(float(np.mean(changed_fractions)), 4)

    mean_r1_in = float(np.mean(r1_rms_insides))
    mean_r1_rg = float(np.mean(r1_rms_rings))
    mean_g_in = float(np.mean(grain_rms_insides))
    mean_g_rg = float(np.mean(grain_rms_rings))

    grain_results["conspicuity_proxy"]["r1_rms_inside"] = round(mean_r1_in, 4)
    grain_results["conspicuity_proxy"]["r1_rms_ring"] = round(mean_r1_rg, 4)
    grain_results["conspicuity_proxy"]["r1_conspicuity_ratio"] = round(mean_r1_in / (mean_r1_rg + 1e-6), 2)
    grain_results["conspicuity_proxy"]["grain_rms_inside"] = round(mean_g_in, 4)
    grain_results["conspicuity_proxy"]["grain_rms_ring"] = round(mean_g_rg, 4)
    grain_results["conspicuity_proxy"]["grain_conspicuity_ratio"] = round(mean_g_in / (mean_g_rg + 1e-6), 2)

    # Flips comparison on same 50 pairs
    for fmt_name, g_flips, r_flips in [("png", grain_flips_png, r1_50_flips_png), ("q50", grain_flips_q50, r1_50_flips_q50)]:
        f_dict = {}
        for m in models:
            valid_50 = [i for i in range(grain_pairs_count) if clean_match_1pct[m][i]]
            k_g = sum([g_flips[m][i] for i in valid_50])
            k_r = sum([r_flips[m][i] for i in valid_50])
            n_v = len(valid_50)
            f_dict[m] = {
                "grain_flips": f"{k_g}/{n_v} ({(k_g/n_v*100.0) if n_v > 0 else 0.0:.1f}%)",
                "r1_flips": f"{k_r}/{n_v} ({(k_r/n_v*100.0) if n_v > 0 else 0.0:.1f}%)",
            }
        f_dict["mean_4m_grain"] = round(float(np.mean([(sum([g_flips[m][i] for i in [idx for idx in range(grain_pairs_count) if clean_match_1pct[m][idx]]]) / len([idx for idx in range(grain_pairs_count) if clean_match_1pct[m][idx]]) * 100.0) for m in models])), 2)
        f_dict["mean_4m_r1"] = round(float(np.mean([(sum([r_flips[m][i] for i in [idx for idx in range(grain_pairs_count) if clean_match_1pct[m][idx]]]) / len([idx for idx in range(grain_pairs_count) if clean_match_1pct[m][idx]]) * 100.0) for m in models])), 2)
        grain_results["eval_by_format"][fmt_name] = f_dict

    all_results["step3_matched_grain_results"] = grain_results

    print(f"Step 3 Results Summary (N={grain_pairs_count}):")
    print(f"  YuNet detection rate: {grain_results['yunet_detection_rate']}% | Landmark displacement: {grain_results['mean_landmark_displacement_px']} px")
    print(f"  Whole Image SSIM: {grain_results['whole_image_ssim']} | Whole Image CIEDE: {grain_results['whole_image_ciede']} | Changed Px: {grain_results['changed_pixel_fraction']*100:.1f}%")
    print(f"  PNG 1% FAR 4-model mean: Grain={grain_results['eval_by_format']['png']['mean_4m_grain']}% vs R1={grain_results['eval_by_format']['png']['mean_4m_r1']}%")
    print(f"  Q50 1% FAR 4-model mean: Grain={grain_results['eval_by_format']['q50']['mean_4m_grain']}% vs R1={grain_results['eval_by_format']['q50']['mean_4m_r1']}%")
    print(f"  Conspicuity Ratio: R1 = {grain_results['conspicuity_proxy']['r1_conspicuity_ratio']} -> Grain = {grain_results['conspicuity_proxy']['grain_conspicuity_ratio']}", flush=True)

    # Save final JSON and pair flips JSON
    final_json_path = RESULTS_DIR / "region_ablation_final.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Final JSON results: {final_json_path}", flush=True)

    pair_flips_path = RESULTS_DIR / "region_ablation_pair_flips.json"
    with open(pair_flips_path, "w", encoding="utf-8") as f:
        json.dump({"flips": all_results["pair_flips"], "sims": all_results["pair_sims"]}, f)
    print(f"[Saved] Per-pair flip vectors: {pair_flips_path}", flush=True)

    # Export Summary CSV
    csv_path = RESULTS_DIR / "region_ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_id", "row_name", "format",
            "mean_flip_1pct", "heldout_flip_1pct",
            "FaceNet_1pct", "ArcFace_1pct", "SFace_1pct", "MobileFaceNet_1pct",
            "face_ssim", "full_ssim", "ciede_mean", "ciede_p95", "changed_inside_frac", "changed_outside_frac",
            "latency_sec", "peak_vram_gb"
        ])
        for r_id, cand in all_results["candidates"].items():
            vis = cand["visibility"]
            for fmt in formats:
                f_data = cand["results_by_format"][fmt]
                writer.writerow([
                    r_id, cand["config"]["name"], fmt,
                    f_data["summary"]["mean_flip_1pct"], f_data["summary"]["heldout_flip_1pct"],
                    f_data["FaceNet"]["flips_1pct_pct"], f_data["ArcFace"]["flips_1pct_pct"],
                    f_data["SFace"]["flips_1pct_pct"], f_data["MobileFaceNet"]["flips_1pct_pct"],
                    vis["face_ssim"], vis["full_ssim"], vis["ciede_mean"], vis["ciede_p95"],
                    vis["changed_inside_frac"], vis["changed_outside_frac"],
                    vis["mean_latency_sec"], vis["peak_vram_gb"]
                ])
    print(f"[Saved] Summary CSV: {csv_path}", flush=True)

    # Generate compact REPORT.md in benchmarks/results/region_ablation/
    report_md_path = RESULTS_DIR / "REPORT.md"
    frontier_lines = [
        "# Varman Region-Size Ablation Benchmark Results",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Pairs:** {n_pairs} LFW genuine pairs  ",
        f"**Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  ",
        "",
        "## Visibility vs. Protection Frontier Table (Real JPEG Q50, 1% FAR)",
        "",
        "| Row ID | Configuration | Face SSIM | CIEDE Mean | CIEDE p95 | Changed Area Frac | 4-Model Mean | Held-Out Mean |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r_id in ["R0_control", "R1_base_scale10", "R2_scale07", "R3_scale13", "R4_eps6_scale10"]:
        c_obj = all_results["candidates"][r_id]
        v = c_obj["visibility"]
        q50_sum = c_obj["results_by_format"]["q50"]["summary"]
        m_sc = c_obj["config"]["mask_scale"]
        c_frac = area_stats[m_sc]["strict_frac"] if m_sc in area_stats else 0.6452
        frontier_lines.append(
            f"| **{r_id}** | {c_obj['config']['name']} | {v['face_ssim']:.4f} | {v['ciede_mean']:.4f} | {v['ciede_p95']:.4f} | {c_frac*100:.1f}% | **{q50_sum['mean_flip_1pct']:.2f}%** | **{q50_sum['heldout_flip_1pct']:.2f}%** |"
        )
    frontier_lines.append("")
    frontier_lines.append("## Paired Exact McNemar Tests vs R1 Baseline (JPEG Q50, 1% FAR)")
    frontier_lines.append("")
    frontier_lines.append("| Candidate vs R1 | Model | R1 Flips | Cand Flips | Discordant (b / c) | Exact p | Holm Adj p | Significant (< 0.05) |")
    frontier_lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cand_id in ["R2_scale07", "R3_scale13", "R4_eps6_scale10"]:
        m_tests = all_results["mcnemar_tests_q50_1pct"][cand_id]
        for m in models:
            t = m_tests[m]
            frontier_lines.append(
                f"| **{cand_id}** | {m} | {all_results['candidates']['R1_base_scale10']['results_by_format']['q50'][m]['flips_1pct_k']} | {all_results['candidates'][cand_id]['results_by_format']['q50'][m]['flips_1pct_k']} | {t['b']} / {t['c']} | {t['exact_p']:.4f} | {t['holm_adj_p']:.4f} | {'**YES**' if t['significant_05'] else 'No'} |"
            )
    frontier_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(frontier_lines))
    print(f"[Saved] Compact REPORT.md: {report_md_path}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("ALL ABLATION RUNS AND MEASUREMENTS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
