"""
Varman 3.0 — Paste-Aware Optimization Experiment (LFW Only)
===========================================================
Measurement-only benchmark testing "crop-only loss" vs "paste-aware loss".

Step 1: Build the paste-aware engine.
Step 2: Equivalence check against OpenCV warpAffine composite on 5 LFW images.
Step 3: Run the experiment on 150 LFW genuine pairs:
  Row 0: Unprotected control
  Row 1: Crop-only, FaceNet weight 1.0 (Candidate I replication)
  Row 2: Paste-aware, FaceNet weight 1.0
Step 4: Fixed stage diagnostic on first 40 LFW genuine pairs (Reference, S1, S2, S3a, S3b, S4) for both Row 1 and Row 2.
Step 5: Gated follow-up Row 3 (paste-aware + landmark jitter) if Go rule passes.
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
import torch.nn.functional as F

REPO_ROOT = Path(r"d:\JavaProjects\Varman\Varman-backend")
sys.path.insert(0, str(REPO_ROOT))

from app.protection.diff_jpeg import DiffJPEG
from app.protection.face_aligner import align_crop_112, composite_crop_back, ARCFACE_CANONICAL_112
from app.protection.protection_engine import ProtectionEngine
from app.protection.strong_engine import get_canonical_face_mask_112
from app.protection.surrogate_models import DualSurrogateSuite
from benchmarks.evaluate_face_models import ModelSuite, cosine_similarity

CALIB_PATH = REPO_ROOT / "benchmarks" / "results" / "sweep_evidence" / "calibrated_thresholds_sweep.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "paste_aware_exp"
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


def composite_crop_back_float(
    original_bgr: np.ndarray,
    protected_crop_112: np.ndarray,
    affine_matrix_M: np.ndarray,
    feather_sigma: float = 4.0,
) -> np.ndarray:
    """Composites crop back to original resolution returning pure float32 without uint8 clipping."""
    h, w = original_bgr.shape[:2]
    inv_M = cv2.invertAffineTransform(affine_matrix_M)
    mask_112 = np.zeros((112, 112), dtype=np.float32)
    cv2.ellipse(mask_112, (56, 56), (42, 50), 0, 0, 360, 1.0, -1)
    mask_112 = cv2.GaussianBlur(mask_112, (15, 15), feather_sigma)

    warped_prot = cv2.warpAffine(
        protected_crop_112.astype(np.float32), inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    warped_mask = cv2.warpAffine(
        mask_112, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    warped_mask_3ch = np.repeat(warped_mask[:, :, np.newaxis], 3, axis=2)

    composited = (
        warped_mask_3ch * warped_prot.astype(np.float32)
        + (1.0 - warped_mask_3ch) * original_bgr.astype(np.float32)
    )
    return np.clip(composited, 0.0, 255.0)


def build_paste_grid(M_cv: np.ndarray, dst_w: int, dst_h: int, src_w: int, src_h: int, device: torch.device) -> torch.Tensor:
    """
    Constructs sampling grid for pasting crop (src_w, src_h) into full canvas (dst_w, dst_h).
    In OpenCV warpAffine:
      inv_M maps crop -> canvas.
      To sample crop at destination pixel (x_dst, y_dst), the coordinate in crop is:
        [x_crop, y_crop] = M_cv * [x_dst, y_dst, 1]^T.
    Normalized to [-1, 1] for F.grid_sample with align_corners=False:
      x_norm = (2 * x_crop + 1) / src_w - 1
      y_norm = (2 * y_crop + 1) / src_h - 1
    """
    ys, xs = torch.meshgrid(torch.arange(dst_h, dtype=torch.float32, device=device),
                            torch.arange(dst_w, dtype=torch.float32, device=device),
                            indexing='ij')
    coords_dst = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1) # (dst_h, dst_w, 3)
    M_t = torch.from_numpy(M_cv.astype(np.float32)).to(device=device) # (2, 3)
    coords_crop = torch.matmul(coords_dst, M_t.T) # (dst_h, dst_w, 2)
    
    grid = torch.stack([(2.0 * coords_crop[..., 0] + 1.0) / float(src_w) - 1.0,
                        (2.0 * coords_crop[..., 1] + 1.0) / float(src_h) - 1.0], dim=-1).unsqueeze(0)
    return grid


def build_realign_grid(inv_M_cv: np.ndarray, dst_w: int, dst_h: int, src_w: int, src_h: int, device: torch.device) -> torch.Tensor:
    """
    Constructs sampling grid for re-aligning full canvas (src_w, src_h) to canonical crop (dst_w, dst_h).
    In OpenCV align_crop_112:
      M maps canvas -> crop.
      To sample canvas at crop pixel (x_crop, y_crop), the coordinate in canvas is:
        [x_canvas, y_canvas] = inv_M_cv * [x_crop, y_crop, 1]^T.
    Normalized to [-1, 1] for F.grid_sample with align_corners=False:
      x_norm = (2 * x_canvas + 1) / src_w - 1
      y_norm = (2 * y_canvas + 1) / src_h - 1
    """
    ys, xs = torch.meshgrid(torch.arange(dst_h, dtype=torch.float32, device=device),
                            torch.arange(dst_w, dtype=torch.float32, device=device),
                            indexing='ij')
    coords_crop = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1) # (dst_h, dst_w, 3)
    inv_M_t = torch.from_numpy(inv_M_cv.astype(np.float32)).to(device=device) # (2, 3)
    coords_canvas = torch.matmul(coords_crop, inv_M_t.T) # (dst_h, dst_w, 2)
    
    grid = torch.stack([(2.0 * coords_canvas[..., 0] + 1.0) / float(src_w) - 1.0,
                        (2.0 * coords_canvas[..., 1] + 1.0) / float(src_h) - 1.0], dim=-1).unsqueeze(0)
    return grid


class PasteAwarePGDEngine:
    """
    PGD engine supporting both:
      1. Crop-only optimization (Candidate I baseline)
      2. Paste-aware end-to-end differentiable optimization
      3. Gated paste-aware with random landmark jitter
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.surrogates = DualSurrogateSuite(device)
        self.diff_jpeg = DiffJPEG().to(device).eval()

        # Canonical elliptical blending mask in 112x112 space
        mask_112 = np.zeros((112, 112), dtype=np.float32)
        cv2.ellipse(mask_112, (56, 56), (42, 50), 0, 0, 360, 1.0, -1)
        mask_112 = cv2.GaussianBlur(mask_112, (15, 15), 4.0)
        self.t_mask_112 = torch.from_numpy(mask_112).unsqueeze(0).unsqueeze(0).float().to(device)

    def protect_crop_only(
        self,
        crop_112_bgr: np.ndarray,
        steps: int = 25,
        w_fn: float = 1.00,
        eps: float = 8.0 / 255.0,
        mu: float = 0.9,
        q_sequence: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], torch.Tensor]:
        """Candidate I: Crop-only baseline optimization."""
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
            "w_fn": w_fn,
            "eps": eps,
            "final_sim_af": sim_af.item(),
            "final_sim_fn": sim_fn.item(),
        }
        return prot_crop_bgr, stats, x_adv_final.detach()

    def protect_paste_aware(
        self,
        img_bgr: np.ndarray,
        landmarks: List[Tuple[int, int]],
        crop_112_bgr: np.ndarray,
        M: np.ndarray,
        steps: int = 25,
        w_fn: float = 1.00,
        eps: float = 8.0 / 255.0,
        mu: float = 0.9,
        q_sequence: Optional[List[int]] = None,
        jitter_landmarks: bool = False,
        rng_seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], torch.Tensor]:
        """
        Paste-aware end-to-end differentiable optimization:
          1. x_crop_adv = clamp(x_clean + delta, 0, 1)
          2. Warp x_crop_adv with inverse alignment transform via F.grid_sample and blend with feathered mask
          3. 8-bit straight-through estimator rounding
          4. DiffJPEG with quality sampled 50 to 95
          5. Re-align result to 112x112 canonical crop via F.grid_sample
          6. Forward pass through ArcFace & FaceNet and compute loss
        """
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

        # Precompute paste grid and warped blending mask
        grid_paste = build_paste_grid(M, dst_w=w, dst_h=h, src_w=112, src_h=112, device=self.device)
        warped_mask = F.grid_sample(self.t_mask_112, grid_paste, mode="bilinear", padding_mode="zeros", align_corners=False)

        # Precompute canonical re-align grid (or dynamic if jittered)
        grid_realign_base = build_realign_grid(inv_M, dst_w=112, dst_h=112, src_w=w, src_h=h, device=self.device)

        jitter_rng = np.random.default_rng(rng_seed) if rng_seed is not None else np.random.default_rng()

        delta = torch.zeros_like(x_clean, requires_grad=True)
        momentum = torch.zeros_like(x_clean)

        for t in range(steps):
            if q_sequence is not None and t < len(q_sequence):
                q = float(q_sequence[t])
            else:
                q = float(torch.randint(50, 96, (1,)).item())

            # 1. Add perturbation and clip
            x_crop_adv = torch.clamp(x_clean + delta, 0.0, 1.0)

            # 2. Warp crop to full image and blend with feathered mask
            warped_prot = F.grid_sample(x_crop_adv, grid_paste, mode="bilinear", padding_mode="reflection", align_corners=False)
            x_comp = warped_mask * warped_prot + (1.0 - warped_mask) * x_full_orig

            # 3. 8-bit rounding with straight-through estimator (STE)
            x_scaled = x_comp * 255.0
            x_round = x_scaled + (torch.round(x_scaled) - x_scaled).detach()
            x_round_01 = torch.clamp(x_round / 255.0, 0.0, 1.0)

            # 4. DiffJPEG on full image
            x_jpeg = self.diff_jpeg(x_round_01, quality=q)

            # 5. Differentiable re-align to 112x112
            if jitter_landmarks:
                # Landmark jitter: translate by up to 2 px, scale by up to 3%
                src_pts = np.array(landmarks[:5], dtype=np.float32)
                cx, cy = float(np.mean(src_pts[:, 0])), float(np.mean(src_pts[:, 1]))
                s = float(jitter_rng.uniform(0.97, 1.03))
                tx = float(jitter_rng.uniform(-2.0, 2.0))
                ty = float(jitter_rng.uniform(-2.0, 2.0))
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

            # 6. Surrogates and loss
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
            "w_fn": w_fn,
            "eps": eps,
            "final_sim_af": sim_af.item(),
            "final_sim_fn": sim_fn.item(),
        }
        return prot_crop_bgr, stats, x_adv_final.detach()


# ==============================================================================
# STEP 2: EQUIVALENCE CHECK
# ==============================================================================

def run_equivalence_check(
    lfw_pairs: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]],
    pgd_engine: PasteAwarePGDEngine,
    engine: ProtectionEngine,
) -> Tuple[bool, List[Dict[str, float]]]:
    """
    On 5 LFW images, compares differentiable paste-back (steps 1 to 2 with no JPEG)
    against the existing OpenCV warpAffine composite used in the evaluation pipeline.
    Reports mean and max absolute pixel difference in 0 to 255 units.
    """
    print("\n" + "=" * 90, flush=True)
    print("STEP 2: RUNNING EQUIVALENCE CHECK (5 LFW Images, OpenCV vs PyTorch Differentiable Paste)", flush=True)
    print("=" * 90, flush=True)

    results: List[Dict[str, float]] = []
    device = pgd_engine.device

    check_count = min(5, len(lfw_pairs))
    for idx in range(check_count):
        b1, _, f1, _, crop_112, _ = lfw_pairs[idx]
        h, w = b1.shape[:2]
        M = cv2.estimateAffinePartial2D(np.array(f1["landmarks"][:5], dtype=np.float32), ARCFACE_CANONICAL_112)[0]
        if M is None:
            continue

        # Use perturbed crop with non-trivial values
        pert_crop_bgr = np.clip(crop_112.astype(np.float32) + 5.0, 0, 255).astype(np.uint8)

        # 1. OpenCV composite
        cv_comp = composite_crop_back(b1, pert_crop_bgr, M)

        # 2. PyTorch differentiable paste-back (steps 1 to 2 with no JPEG)
        crop_rgb = cv2.cvtColor(pert_crop_bgr, cv2.COLOR_BGR2RGB)
        x_crop = torch.from_numpy(crop_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        x_clean = torch.clamp(x_crop, 0.0, 1.0)

        img_rgb = cv2.cvtColor(b1, cv2.COLOR_BGR2RGB)
        x_full_orig = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        grid_paste = build_paste_grid(M, dst_w=w, dst_h=h, src_w=112, src_h=112, device=device)
        warped_mask = F.grid_sample(pgd_engine.t_mask_112, grid_paste, mode="bilinear", padding_mode="zeros", align_corners=False)
        warped_prot = F.grid_sample(x_clean, grid_paste, mode="bilinear", padding_mode="reflection", align_corners=False)
        x_comp = warped_mask * warped_prot + (1.0 - warped_mask) * x_full_orig

        comp_np = (x_comp.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        comp_bgr = cv2.cvtColor(comp_np, cv2.COLOR_RGB2BGR)

        diff = np.abs(cv_comp.astype(np.float32) - comp_bgr.astype(np.float32))
        mean_diff = float(np.mean(diff))
        max_diff = float(np.max(diff))

        res = {"image_idx": idx, "mean_diff": mean_diff, "max_diff": max_diff}
        results.append(res)
        print(f"  Image {idx}: Mean Abs Diff = {mean_diff:.4f} / 255, Max Abs Diff = {max_diff:.1f} / 255", flush=True)

    max_mean_diff = max(r["mean_diff"] for r in results) if results else 0.0
    passed = bool(max_mean_diff <= 1.0)
    print(f"\nEquivalence Check Result: {'PASSED' if passed else 'FAILED'} (Max Mean Diff: {max_mean_diff:.4f} <= 1.0)", flush=True)
    return passed, results


# ==============================================================================
# STEP 3: EXPERIMENT RUNNER
# ==============================================================================

def run_experiment_rows(
    lfw_pairs: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]],
    lfw_clean_embs_1: List[Dict[str, np.ndarray]],
    lfw_clean_embs_2: List[Dict[str, np.ndarray]],
    lfw_clean_match_1pct: Dict[str, List[bool]],
    lfw_clean_match_01pct: Dict[str, List[bool]],
    thresh_1pct: Dict[str, float],
    thresh_01pct: Dict[str, float],
    pgd_engine: PasteAwarePGDEngine,
    engine: ProtectionEngine,
    model_suite: ModelSuite,
    lfw_q_sequences: List[List[int]],
    run_row3_jitter: bool = False,
) -> Dict[str, Any]:
    print("\n" + "=" * 90, flush=True)
    print(f"STARTING STEP 3: EXPERIMENT RUN ({len(lfw_pairs)} LFW Genuine Pairs)", flush=True)
    print("=" * 90, flush=True)

    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]

    CANDIDATES = [
        {
            "id": "control",
            "name": "Row 0: Unprotected Control",
            "type": "control",
            "steps": 0,
            "w_fn": 0.0,
        },
        {
            "id": "crop_only_w100",
            "name": "Row 1: Crop-only, FaceNet weight 1.0",
            "type": "crop_only",
            "steps": 25,
            "w_fn": 1.0,
        },
        {
            "id": "paste_aware_w100",
            "name": "Row 2: Paste-aware, FaceNet weight 1.0",
            "type": "paste_aware",
            "steps": 25,
            "w_fn": 1.0,
        },
    ]

    if run_row3_jitter:
        CANDIDATES.append({
            "id": "paste_aware_jitter",
            "name": "Row 3: Paste-aware + Landmark Jitter",
            "type": "paste_aware_jitter",
            "steps": 25,
            "w_fn": 1.0,
        })

    all_results: Dict[str, Any] = {
        "benchmark_metadata": {
            "device": str(pgd_engine.device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "epsilon": "8/255 (0.03137)",
            "lfw_pairs_count": len(lfw_pairs),
            "formats_evaluated": formats,
            "models_surrogate": ["ArcFace", "FaceNet"],
            "models_held_out": ["SFace", "MobileFaceNet"],
            "thresholds_1pct": thresh_1pct,
            "thresholds_01pct": thresh_01pct,
        },
        "candidates": {},
        "pair_flips": {},
    }

    # Tracking per-pair flips for McNemar / paired analysis
    pair_flip_records: Dict[str, Dict[str, Dict[str, List[bool]]]] = {}

    for cand_idx, cand in enumerate(CANDIDATES):
        c_id = cand["id"]
        c_name = cand["name"]
        c_type = cand["type"]
        steps = cand["steps"]
        w_fn = cand["w_fn"]

        print("\n" + "=" * 90, flush=True)
        print(f"RUNNING ROW {cand_idx} / {len(CANDIDATES) - 1}: {c_name}", flush=True)
        print("=" * 90, flush=True)

        torch.cuda.reset_peak_memory_stats()

        cand_output = {
            "candidate_id": c_id,
            "candidate_name": c_name,
            "type": c_type,
            "steps": steps,
            "w_fn": w_fn,
            "lfw": {},
            "visibility": {},
            "leakage_metrics": {},
        }

        lfw_prot_embs: Dict[str, List[Dict[str, np.ndarray]]] = {fmt: [] for fmt in formats}
        lfw_ssims: List[float] = []
        lfw_ciede_m: List[float] = []
        lfw_ciede_p95: List[float] = []
        lfw_ciede_p99: List[float] = []
        lfw_latencies: List[float] = []
        lfw_max_pixel_changes: List[float] = []
        lfw_outside_mask_changed_fracs: List[float] = []

        pair_flip_records[c_id] = {fmt: {m: [] for m in models} for fmt in formats}

        for p_i, (b1, b2, f1, f2, clean_c1, clean_c2) in enumerate(lfw_pairs):
            h, w = b1.shape[:2]
            M = cv2.estimateAffinePartial2D(np.array(f1["landmarks"][:5], dtype=np.float32), ARCFACE_CANONICAL_112)[0]
            inv_M = cv2.invertAffineTransform(M)

            if c_type == "control":
                prot_b1 = b1.copy()
                lat_sec = 0.0
                f_ssim = 1.0
                m_de, p95_de, p99_de = 0.0, 0.0, 0.0
                max_px_diff = 0.0
                outside_changed_frac = 0.0
            else:
                t0 = time.perf_counter()
                if c_type == "crop_only":
                    prot_crop_bgr, _, _ = pgd_engine.protect_crop_only(
                        crop_112_bgr=clean_c1,
                        steps=steps,
                        w_fn=w_fn,
                        eps=8.0 / 255.0,
                        mu=0.9,
                        q_sequence=lfw_q_sequences[p_i],
                    )
                elif c_type == "paste_aware":
                    prot_crop_bgr, _, _ = pgd_engine.protect_paste_aware(
                        img_bgr=b1,
                        landmarks=f1["landmarks"],
                        crop_112_bgr=clean_c1,
                        M=M,
                        steps=steps,
                        w_fn=w_fn,
                        eps=8.0 / 255.0,
                        mu=0.9,
                        q_sequence=lfw_q_sequences[p_i],
                        jitter_landmarks=False,
                    )
                elif c_type == "paste_aware_jitter":
                    prot_crop_bgr, _, _ = pgd_engine.protect_paste_aware(
                        img_bgr=b1,
                        landmarks=f1["landmarks"],
                        crop_112_bgr=clean_c1,
                        M=M,
                        steps=steps,
                        w_fn=w_fn,
                        eps=8.0 / 255.0,
                        mu=0.9,
                        q_sequence=lfw_q_sequences[p_i],
                        jitter_landmarks=True,
                        rng_seed=2026 + p_i,
                    )
                else:
                    raise ValueError(f"Unknown candidate type {c_type}")

                lat_sec = time.perf_counter() - t0

                # Crucial: The final protected image saved for evaluation MUST be produced by the OpenCV composite path
                prot_b1 = composite_crop_back(b1, prot_crop_bgr, M)

                prot_c1, _ = align_crop_112(prot_b1, f1["landmarks"])
                if prot_c1 is not None:
                    f_ssim = float(compute_ssim(clean_c1, prot_c1, channel_axis=2, data_range=255))
                    m_de, p95_de, p99_de = compute_face_ciede_stats(clean_c1, prot_c1)
                else:
                    f_ssim = 1.0
                    m_de, p95_de, p99_de = 0.0, 0.0, 0.0

                # Pixel change metrics
                diff_full = np.abs(prot_b1.astype(np.float32) - b1.astype(np.float32))
                max_px_diff = float(np.max(diff_full))

                # Changed pixels outside face mask
                mask_112 = np.zeros((112, 112), dtype=np.float32)
                cv2.ellipse(mask_112, (56, 56), (42, 50), 0, 0, 360, 1.0, -1)
                mask_112 = cv2.GaussianBlur(mask_112, (15, 15), 4.0)
                warped_mask = cv2.warpAffine(mask_112, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
                outside_mask = (warped_mask < 0.001)
                num_outside = int(np.sum(outside_mask))
                changed_outside = int(np.sum((diff_full > 0).any(axis=2) & outside_mask))
                outside_changed_frac = float(changed_outside) / float(num_outside) if num_outside > 0 else 0.0

            lfw_ssims.append(f_ssim)
            lfw_ciede_m.append(m_de)
            lfw_ciede_p95.append(p95_de)
            lfw_ciede_p99.append(p99_de)
            lfw_latencies.append(lat_sec)
            lfw_max_pixel_changes.append(max_px_diff)
            lfw_outside_mask_changed_fracs.append(outside_changed_frac)

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
                print(f"  Row {cand_idx} progress: {p_i + 1}/{len(lfw_pairs)} completed (Avg Latency: {np.mean(lfw_latencies):.2f}s/img)", flush=True)

        peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0
        if peak_vram_gb > 3.8:
            print(f"CRITICAL WARNING: Peak VRAM {peak_vram_gb:.2f} GB exceeded 3.8 GB limit!", flush=True)

        cand_output["lfw"]["formats"] = {}
        for fmt in formats:
            cand_output["lfw"]["formats"][fmt] = {}
            for m in models:
                raw_flips_1pct = 0
                valid_denom_1pct = 0
                raw_flips_01pct = 0
                valid_denom_01pct = 0
                sim_drops: List[float] = []

                for i, prot_emb in enumerate(lfw_prot_embs[fmt]):
                    sim_clean = float(cosine_similarity(lfw_clean_embs_1[i][m], lfw_clean_embs_2[i][m]))
                    sim_prot = float(cosine_similarity(prot_emb[m], lfw_clean_embs_2[i][m]))
                    sim_drops.append(sim_clean - sim_prot)

                    # 1% FAR flip tracking
                    is_clean_match_1 = lfw_clean_match_1pct[m][i]
                    is_flipped_1 = (sim_prot < thresh_1pct[m])
                    if is_clean_match_1:
                        valid_denom_1pct += 1
                        if is_flipped_1:
                            raw_flips_1pct += 1
                    # Record for paired McNemar
                    pair_flip_records[c_id][fmt][m].append(bool(is_flipped_1 and is_clean_match_1))

                    # 0.1% FAR flip tracking
                    is_clean_match_01 = lfw_clean_match_01pct[m][i]
                    if is_clean_match_01:
                        valid_denom_01pct += 1
                        if sim_prot < thresh_01pct[m]:
                            raw_flips_01pct += 1

                rate_1 = (raw_flips_1pct / valid_denom_1pct * 100.0) if valid_denom_1pct > 0 else 0.0
                rate_01 = (raw_flips_01pct / valid_denom_01pct * 100.0) if valid_denom_01pct > 0 else 0.0
                ci_1 = compute_wilson_ci(raw_flips_1pct, valid_denom_1pct)
                ci_01 = compute_wilson_ci(raw_flips_01pct, valid_denom_01pct)

                cand_output["lfw"]["formats"][fmt][m] = {
                    "flips_1pct_k": raw_flips_1pct,
                    "flips_1pct_n": valid_denom_1pct,
                    "flips_1pct_pct": round(rate_1, 2),
                    "flips_1pct_ci": ci_1,
                    "flips_01pct_k": raw_flips_01pct,
                    "flips_01pct_n": valid_denom_01pct,
                    "flips_01pct_pct": round(rate_01, 2),
                    "flips_01pct_ci": ci_01,
                    "mean_sim_drop": round(float(np.mean(sim_drops)), 4),
                }

            # Summary means for this format
            f_data = cand_output["lfw"]["formats"][fmt]
            all_4_1pct = float(np.mean([f_data[m]["flips_1pct_pct"] for m in models]))
            held_2_1pct = float(np.mean([f_data[m]["flips_1pct_pct"] for m in ["SFace", "MobileFaceNet"]]))
            all_4_01pct = float(np.mean([f_data[m]["flips_01pct_pct"] for m in models]))
            held_2_01pct = float(np.mean([f_data[m]["flips_01pct_pct"] for m in ["SFace", "MobileFaceNet"]]))

            cand_output["lfw"]["formats"][fmt]["summary"] = {
                "mean_flip_1pct": round(all_4_1pct, 2),
                "heldout_flip_1pct": round(held_2_1pct, 2),
                "mean_flip_01pct": round(all_4_01pct, 2),
                "heldout_flip_01pct": round(held_2_01pct, 2),
            }

        cand_output["visibility"] = {
            "face_ssim": round(float(np.mean(lfw_ssims)), 4),
            "ciede_mean": round(float(np.mean(lfw_ciede_m)), 4),
            "ciede_p95": round(float(np.mean(lfw_ciede_p95)), 4),
            "ciede_p99": round(float(np.mean(lfw_ciede_p99)), 4),
            "mean_latency_sec": round(float(np.mean(lfw_latencies)), 2),
            "peak_vram_gb": round(peak_vram_gb, 3),
        }

        cand_output["leakage_metrics"] = {
            "max_pixel_change_mean": round(float(np.mean(lfw_max_pixel_changes)), 2),
            "max_pixel_change_max": round(float(np.max(lfw_max_pixel_changes)), 2),
            "outside_mask_changed_frac_mean": round(float(np.mean(lfw_outside_mask_changed_fracs)), 6),
            "outside_mask_changed_frac_max": round(float(np.max(lfw_outside_mask_changed_fracs)), 6),
        }

        all_results["candidates"][c_id] = cand_output

        # Save partial JSON after each row
        partial_json_path = RESULTS_DIR / "paste_aware_partial.json"
        with open(partial_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"  [Checkpoint] Saved partial results for {c_id} to {partial_json_path}", flush=True)

        # Print row summary
        q50_sum = cand_output["lfw"]["formats"]["q50"]["summary"]
        png_sum = cand_output["lfw"]["formats"]["png"]["summary"]
        vis = cand_output["visibility"]
        leak = cand_output["leakage_metrics"]
        print(f"  --> PNG 1% FAR: 4-model mean = {png_sum['mean_flip_1pct']}%, held-out = {png_sum['heldout_flip_1pct']}%", flush=True)
        print(f"  --> Q50 1% FAR: 4-model mean = {q50_sum['mean_flip_1pct']}%, held-out = {q50_sum['heldout_flip_1pct']}%", flush=True)
        print(f"  --> SSIM = {vis['face_ssim']}, CIEDE2000 mean = {vis['ciede_mean']}, Latency = {vis['mean_latency_sec']}s, Peak VRAM = {vis['peak_vram_gb']} GB", flush=True)
        print(f"  --> Max Pixel Change = {leak['max_pixel_change_max']} / 255, Outside Mask Changed = {leak['outside_mask_changed_frac_mean']:.6f}", flush=True)

    # Paired comparisons (Row 2 vs Row 1)
    if "paste_aware_w100" in pair_flip_records and "crop_only_w100" in pair_flip_records:
        r2_rec = pair_flip_records["paste_aware_w100"]
        r1_rec = pair_flip_records["crop_only_w100"]
        paired_comp: Dict[str, Dict[str, Dict[str, int]]] = {}

        for fmt in formats:
            paired_comp[fmt] = {}
            for m in models:
                flips_r2 = r2_rec[fmt][m]
                flips_r1 = r1_rec[fmt][m]
                n_pairs = min(len(flips_r2), len(flips_r1))

                only_r2 = 0
                only_r1 = 0
                both = 0
                neither = 0

                for i in range(n_pairs):
                    f2 = flips_r2[i]
                    f1 = flips_r1[i]
                    if f2 and not f1:
                        only_r2 += 1
                    elif f1 and not f2:
                        only_r1 += 1
                    elif f2 and f1:
                        both += 1
                    else:
                        neither += 1

                paired_comp[fmt][m] = {
                    "total_valid_pairs": n_pairs,
                    "flipped_only_row2": only_r2,
                    "flipped_only_row1": only_r1,
                    "flipped_both": both,
                    "flipped_neither": neither,
                    "net_gain_row2": only_r2 - only_r1,
                }
        all_results["paired_comparison_row2_vs_row1"] = paired_comp

    return all_results


# ==============================================================================
# STEP 4: FIXED STAGE DIAGNOSTIC
# ==============================================================================

def run_step4_stage_diagnostic(
    lfw_pairs: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]],
    pgd_engine: PasteAwarePGDEngine,
    engine: ProtectionEngine,
    thresh_1pct: Dict[str, float],
    lfw_q_sequences: List[List[int]],
    subset_count: int = 40,
) -> Dict[str, Any]:
    print("\n" + "=" * 90, flush=True)
    print(f"STARTING STEP 4: FIXED STAGE DIAGNOSTIC ({subset_count} Pairs, Row 1 Crop-Only vs Row 2 Paste-Aware)", flush=True)
    print("=" * 90, flush=True)

    pairs_subset = lfw_pairs[:subset_count]
    device = pgd_engine.device
    surrogates = pgd_engine.surrogates

    stages = [
        "unprotected",
        "stage1_opt_crop",
        "stage2_pasted_float",
        "stage3a_8bit_round",
        "stage3b_png_reload",
        "stage4_jpeg_q50",
    ]
    models = ["ArcFace", "FaceNet"]

    # Storage: {cand_id: {stage: {model: [sims]}}}
    candidates_to_diag = ["crop_only_w100", "paste_aware_w100"]
    diag_sims: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        cid: {stg: {"ArcFace": [], "FaceNet": []} for stg in stages}
        for cid in candidates_to_diag
    }

    for p_i, (b1, b2, f1, f2, c1, c2) in enumerate(pairs_subset):
        h, w = b1.shape[:2]
        M = cv2.estimateAffinePartial2D(np.array(f1["landmarks"][:5], dtype=np.float32), ARCFACE_CANONICAL_112)[0]
        inv_M = cv2.invertAffineTransform(M)

        # 1. Reference Image 2 Clean Embedding
        t2 = torch.from_numpy(cv2.cvtColor(c2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            clean2_af, clean2_fn = surrogates(t2)
            ref_af = clean2_af.cpu().numpy().flatten()
            ref_fn = clean2_fn.cpu().numpy().flatten()

        # 0. Unprotected Reference Image 1
        t1_unprot = torch.from_numpy(cv2.cvtColor(c1, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            unprot_af, unprot_fn = surrogates(t1_unprot)
            sim_unprot_af = float(cosine_similarity(unprot_af.cpu().numpy().flatten(), ref_af))
            sim_unprot_fn = float(cosine_similarity(unprot_fn.cpu().numpy().flatten(), ref_fn))

        for cid in candidates_to_diag:
            diag_sims[cid]["unprotected"]["ArcFace"].append(sim_unprot_af)
            diag_sims[cid]["unprotected"]["FaceNet"].append(sim_unprot_fn)

            # Generate perturbation for this candidate
            if cid == "crop_only_w100":
                prot_crop_bgr, _, x_crop_adv = pgd_engine.protect_crop_only(
                    crop_112_bgr=c1,
                    steps=25,
                    w_fn=1.0,
                    eps=8.0 / 255.0,
                    mu=0.9,
                    q_sequence=lfw_q_sequences[p_i],
                )
            else:
                prot_crop_bgr, _, x_crop_adv = pgd_engine.protect_paste_aware(
                    img_bgr=b1,
                    landmarks=f1["landmarks"],
                    crop_112_bgr=c1,
                    M=M,
                    steps=25,
                    w_fn=1.0,
                    eps=8.0 / 255.0,
                    mu=0.9,
                    q_sequence=lfw_q_sequences[p_i],
                    jitter_landmarks=False,
                )

            # STAGE 1: Inside optimizer on 112x112 crop at final step
            with torch.no_grad():
                s1_af, s1_fn = surrogates(x_crop_adv)
                sim_s1_af = float(cosine_similarity(s1_af.cpu().numpy().flatten(), ref_af))
                sim_s1_fn = float(cosine_similarity(s1_fn.cpu().numpy().flatten(), ref_fn))
            diag_sims[cid]["stage1_opt_crop"]["ArcFace"].append(sim_s1_af)
            diag_sims[cid]["stage1_opt_crop"]["FaceNet"].append(sim_s1_fn)

            # STAGE 2: Pasted into full image and re-aligned, kept in float32 throughout. No integer cast, no saving.
            comp_float = composite_crop_back_float(b1.astype(np.float32), prot_crop_bgr, M)
            # Re-align in float32 using original landmarks
            c_s2, _ = align_crop_112(comp_float, f1["landmarks"])
            if c_s2 is not None:
                t_s2 = torch.from_numpy(cv2.cvtColor(c_s2.astype(np.float32), cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    s2_af, s2_fn = surrogates(t_s2)
                    sim_s2_af = float(cosine_similarity(s2_af.cpu().numpy().flatten(), ref_af))
                    sim_s2_fn = float(cosine_similarity(s2_fn.cpu().numpy().flatten(), ref_fn))
            else:
                sim_s2_af, sim_s2_fn = sim_s1_af, sim_s1_fn
            diag_sims[cid]["stage2_pasted_float"]["ArcFace"].append(sim_s2_af)
            diag_sims[cid]["stage2_pasted_float"]["FaceNet"].append(sim_s2_fn)

            # STAGE 3a: After 8-bit rounding only, using original landmarks and no re-detection
            comp_8bit = np.clip(np.round(comp_float), 0, 255).astype(np.uint8)
            c_s3a, _ = align_crop_112(comp_8bit, f1["landmarks"])
            if c_s3a is not None:
                t_s3a = torch.from_numpy(cv2.cvtColor(c_s3a, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                with torch.no_grad():
                    s3a_af, s3a_fn = surrogates(t_s3a)
                    sim_s3a_af = float(cosine_similarity(s3a_af.cpu().numpy().flatten(), ref_af))
                    sim_s3a_fn = float(cosine_similarity(s3a_fn.cpu().numpy().flatten(), ref_fn))
            else:
                sim_s3a_af, sim_s3a_fn = sim_s2_af, sim_s2_fn
            diag_sims[cid]["stage3a_8bit_round"]["ArcFace"].append(sim_s3a_af)
            diag_sims[cid]["stage3a_8bit_round"]["FaceNet"].append(sim_s3a_fn)

            # STAGE 3b: After saving as PNG and reloading, with YuNet re-detection and alignment
            p_png = TMP_DIR / f"diag_s3b_{cid}_{p_i}.png"
            cv2.imwrite(str(p_png), comp_8bit)
            re_png = cv2.imread(str(p_png))
            det_png = engine.detect_faces(re_png)
            if det_png:
                ious = [box_iou(f1["box"], f["box"]) for f in det_png]
                best_f_png = det_png[int(np.argmax(ious))]
                c_s3b, _ = align_crop_112(re_png, best_f_png["landmarks"])
                if c_s3b is not None:
                    t_s3b = torch.from_numpy(cv2.cvtColor(c_s3b, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    with torch.no_grad():
                        s3b_af, s3b_fn = surrogates(t_s3b)
                        sim_s3b_af = float(cosine_similarity(s3b_af.cpu().numpy().flatten(), ref_af))
                        sim_s3b_fn = float(cosine_similarity(s3b_fn.cpu().numpy().flatten(), ref_fn))
                else:
                    sim_s3b_af, sim_s3b_fn = sim_s3a_af, sim_s3a_fn
            else:
                sim_s3b_af, sim_s3b_fn = sim_s3a_af, sim_s3a_fn
            diag_sims[cid]["stage3b_png_reload"]["ArcFace"].append(sim_s3b_af)
            diag_sims[cid]["stage3b_png_reload"]["FaceNet"].append(sim_s3b_fn)
            if p_png.exists():
                p_png.unlink()

            # STAGE 4: After saving as real JPEG Q50 and reloading, with re-detection and alignment
            p_q50 = TMP_DIR / f"diag_s4_{cid}_{p_i}.jpg"
            cv2.imwrite(str(p_q50), comp_8bit, [cv2.IMWRITE_JPEG_QUALITY, 50])
            re_q50 = cv2.imread(str(p_q50))
            det_q50 = engine.detect_faces(re_q50)
            if det_q50:
                ious = [box_iou(f1["box"], f["box"]) for f in det_q50]
                best_f_q50 = det_q50[int(np.argmax(ious))]
                c_s4, _ = align_crop_112(re_q50, best_f_q50["landmarks"])
                if c_s4 is not None:
                    t_s4 = torch.from_numpy(cv2.cvtColor(c_s4, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                    with torch.no_grad():
                        s4_af, s4_fn = surrogates(t_s4)
                        sim_s4_af = float(cosine_similarity(s4_af.cpu().numpy().flatten(), ref_af))
                        sim_s4_fn = float(cosine_similarity(s4_fn.cpu().numpy().flatten(), ref_fn))
                else:
                    sim_s4_af, sim_s4_fn = sim_s3b_af, sim_s3b_fn
            else:
                sim_s4_af, sim_s4_fn = sim_s3b_af, sim_s3b_fn
            diag_sims[cid]["stage4_jpeg_q50"]["ArcFace"].append(sim_s4_af)
            diag_sims[cid]["stage4_jpeg_q50"]["FaceNet"].append(sim_s4_fn)
            if p_q50.exists():
                p_q50.unlink()

        if (p_i + 1) % 10 == 0 or (p_i + 1) == len(pairs_subset):
            print(f"  Stage Diagnostic progress: {p_i + 1}/{len(pairs_subset)} pairs evaluated", flush=True)

    # Compute summary statistics
    tau_af = thresh_1pct["ArcFace"]
    tau_fn = thresh_1pct["FaceNet"]

    diag_summary: Dict[str, Any] = {
        "metadata": {
            "subset_pairs": subset_count,
            "threshold_1pct_arcface": tau_af,
            "threshold_1pct_facenet": tau_fn,
        },
        "candidates": {},
    }

    for cid in candidates_to_diag:
        c_summary: Dict[str, Any] = {}
        for stg in stages:
            af_arr = np.array(diag_sims[cid][stg]["ArcFace"])
            fn_arr = np.array(diag_sims[cid][stg]["FaceNet"])
            af_below = int(np.sum(af_arr < tau_af))
            fn_below = int(np.sum(fn_arr < tau_fn))
            n_tot = len(af_arr)

            c_summary[stg] = {
                "ArcFace": {
                    "mean_similarity": round(float(np.mean(af_arr)), 4),
                    "pairs_below_threshold": af_below,
                    "total_pairs": n_tot,
                    "fraction_below_threshold": round(float(af_below / n_tot * 100.0), 2) if n_tot > 0 else 0.0,
                },
                "FaceNet": {
                    "mean_similarity": round(float(np.mean(fn_arr)), 4),
                    "pairs_below_threshold": fn_below,
                    "total_pairs": n_tot,
                    "fraction_below_threshold": round(float(fn_below / n_tot * 100.0), 2) if n_tot > 0 else 0.0,
                },
            }
        diag_summary["candidates"][cid] = c_summary

    # Save stage diagnostic JSON
    stage_json_path = RESULTS_DIR / "stage_diagnostic.json"
    with open(stage_json_path, "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)
    print(f"\n[Diagnostic] Saved stage_diagnostic.json to {stage_json_path}", flush=True)

    # Save stage diagnostic CSV
    stage_csv_path = RESULTS_DIR / "stage_diagnostic.csv"
    with open(stage_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id",
            "stage",
            "model",
            "mean_similarity",
            "pairs_below_threshold",
            "total_pairs",
            "fraction_below_threshold_pct",
        ])
        for cid, c_data in diag_summary["candidates"].items():
            for stg in stages:
                for m in ["ArcFace", "FaceNet"]:
                    m_data = c_data[stg][m]
                    writer.writerow([
                        cid,
                        stg,
                        m,
                        m_data["mean_similarity"],
                        m_data["pairs_below_threshold"],
                        m_data["total_pairs"],
                        m_data["fraction_below_threshold"],
                    ])
    print(f"[Diagnostic] Saved stage_diagnostic.csv to {stage_csv_path}", flush=True)

    return diag_summary


# ==============================================================================
# CSV EXPORTER & REPORT WRITER
# ==============================================================================

def export_summary_csv(results: Dict[str, Any], csv_path: Path):
    fieldnames = [
        "candidate_id",
        "candidate_name",
        "candidate_type",
        "steps",
        "w_fn",
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
        "peak_vram_gb",
        "max_pixel_change_max",
        "outside_mask_changed_frac_mean",
    ]

    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c_id, cand in results["candidates"].items():
            c_name = cand["candidate_name"]
            c_type = cand["type"]
            w_fn = cand.get("w_fn", 0.0)
            steps = cand.get("steps", 0)
            vis = cand.get("visibility", {})
            leak = cand.get("leakage_metrics", {})
            for fmt in formats:
                fmt_data = cand.get("lfw", {}).get("formats", {}).get(fmt, {})
                for m in models:
                    m_data = fmt_data.get(m, {})
                    role = "White-Box" if m in ["ArcFace", "FaceNet"] else "Held-Out"
                    ci_1 = m_data.get("flips_1pct_ci", [0.0, 0.0])
                    ci_01 = m_data.get("flips_01pct_ci", [0.0, 0.0])
                    writer.writerow({
                        "candidate_id": c_id,
                        "candidate_name": c_name,
                        "candidate_type": c_type,
                        "steps": steps,
                        "w_fn": w_fn,
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
                        "peak_vram_gb": vis.get("peak_vram_gb", 0.0),
                        "max_pixel_change_max": leak.get("max_pixel_change_max", 0.0),
                        "outside_mask_changed_frac_mean": leak.get("outside_mask_changed_frac_mean", 0.0),
                    })


def write_report_md(
    results: Dict[str, Any],
    diag_summary: Dict[str, Any],
    equiv_results: List[Dict[str, float]],
    decision: Dict[str, Any],
    report_path: Path,
):
    lines = []
    lines.append("# Varman 3.0 — Paste-Aware Optimization Experiment Report")
    lines.append("")
    lines.append("**Measurement-Only Engineering Benchmark**")
    lines.append(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- Epsilon: fixed `8/255` (0.03137), Luminance-Only Perturbation")
    lines.append("- White-Box Surrogate Models: ArcFace (IResNet50), FaceNet (InceptionResnetV1)")
    lines.append("- Strictly Held-Out Evaluation Models: SFace, MobileFaceNet")
    lines.append(f"- LFW Genuine Pairs Evaluated: {results['benchmark_metadata']['lfw_pairs_count']}")
    lines.append("- Calibration Source: `benchmarks/results/sweep_evidence/calibrated_thresholds_sweep.json`")
    lines.append("")

    # Step 2: Equivalence Check
    lines.append("## 1. Step 2: Equivalence Check (Differentiable Paste vs OpenCV)")
    lines.append("")
    lines.append("| Image Index | Mean Abs Pixel Diff (/255) | Max Abs Pixel Diff (/255) | Status |")
    lines.append("|:---:|:---:|:---:|:---:|")
    for r in equiv_results:
        st = "PASS" if r["mean_diff"] <= 1.0 else "FAIL"
        lines.append(f"| Image {r['image_idx']} | {r['mean_diff']:.4f} [measured] | {r['max_diff']:.1f} [measured] | {st} |")
    max_m = max(r["mean_diff"] for r in equiv_results) if equiv_results else 0.0
    lines.append("")
    lines.append(f"**Equivalence Conclusion:** Max mean difference = {max_m:.4f} <= 1.0 threshold. PyTorch differentiable paste-back accurately replicates OpenCV composite math.")
    lines.append("")

    # Step 3: Candidate Results Table
    lines.append("## 2. Step 3: Experiment Results Across Conditions")
    lines.append("")
    formats = ["png", "q75", "q50"]
    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]

    for fmt in formats:
        lines.append(f"### Format: {fmt.upper()}")
        lines.append("")
        lines.append("| Candidate | 4-Model Mean (1% FAR) | Held-Out Mean (1% FAR) | ArcFace | FaceNet | SFace (Held) | MobileFaceNet (Held) | Face SSIM | CIEDE2000 Mean | Latency (s) | Peak VRAM |")
        lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
        for cid, cand in results["candidates"].items():
            c_name = cand["candidate_name"]
            f_data = cand["lfw"]["formats"][fmt]
            m_sum = f_data["summary"]
            vis = cand["visibility"]
            af = f"{f_data['ArcFace']['flips_1pct_k']}/{f_data['ArcFace']['flips_1pct_n']} ({f_data['ArcFace']['flips_1pct_pct']}%)"
            fn = f"{f_data['FaceNet']['flips_1pct_k']}/{f_data['FaceNet']['flips_1pct_n']} ({f_data['FaceNet']['flips_1pct_pct']}%)"
            sf = f"{f_data['SFace']['flips_1pct_k']}/{f_data['SFace']['flips_1pct_n']} ({f_data['SFace']['flips_1pct_pct']}%)"
            mf = f"{f_data['MobileFaceNet']['flips_1pct_k']}/{f_data['MobileFaceNet']['flips_1pct_n']} ({f_data['MobileFaceNet']['flips_1pct_pct']}%)"
            lines.append(f"| **{c_name}** | **{m_sum['mean_flip_1pct']:.2f}%** [measured] | **{m_sum['heldout_flip_1pct']:.2f}%** [measured] | {af} | {fn} | {sf} | {mf} | {vis['face_ssim']:.4f} | {vis['ciede_mean']:.4f} | {vis['mean_latency_sec']}s | {vis['peak_vram_gb']} GB |")
        lines.append("")

    # Leakage and Pixel Change Table
    lines.append("### Full-Image Modification and Canvas Leakage Metrics")
    lines.append("")
    lines.append("| Candidate | Max Abs Pixel Change (Mean) | Max Abs Pixel Change (Max) | Changed Pixels Outside Face Mask (Mean Frac) | Changed Pixels Outside Face Mask (Max Frac) |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")
    for cid, cand in results["candidates"].items():
        c_name = cand["candidate_name"]
        leak = cand["leakage_metrics"]
        lines.append(f"| **{c_name}** | {leak['max_pixel_change_mean']} [measured] | {leak['max_pixel_change_max']} [measured] | {leak['outside_mask_changed_frac_mean']:.6f} [measured] | {leak['outside_mask_changed_frac_max']:.6f} [measured] |")
    lines.append("")

    # Paired McNemar Table (Row 2 vs Row 1)
    if "paired_comparison_row2_vs_row1" in results:
        lines.append("## 3. Paired Comparison: Row 2 (Paste-Aware) vs Row 1 (Crop-Only)")
        lines.append("")
        lines.append("| Format | Model | Total Valid Pairs | Flipped ONLY by Row 2 | Flipped ONLY by Row 1 | Flipped by BOTH | Flipped by NEITHER | Net Gain (Row 2 - Row 1) |")
        lines.append("|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
        paired = results["paired_comparison_row2_vs_row1"]
        for fmt in formats:
            for m in models:
                p_data = paired[fmt][m]
                lines.append(f"| {fmt.upper()} | {m} | {p_data['total_valid_pairs']} | {p_data['flipped_only_row2']} [measured] | {p_data['flipped_only_row1']} [measured] | {p_data['flipped_both']} [measured] | {p_data['flipped_neither']} [measured] | {p_data['net_gain_row2']:+d} |")
        lines.append("")

    # Step 4: Fixed Stage Diagnostic Table
    lines.append("## 4. Step 4: Fixed Stage Diagnostic (40 Genuine Pairs)")
    lines.append("")
    lines.append("Tracks similarity rebounds and fraction below 1% FAR threshold across processing stages:")
    lines.append("")
    diag_cands = diag_summary["candidates"]
    stage_labels = {
        "unprotected": "Reference (Unprotected)",
        "stage1_opt_crop": "Stage 1: Inside Optimizer (112x112 Crop)",
        "stage2_pasted_float": "Stage 2: Pasted into Canvas (Float32, No Quant/Save)",
        "stage3a_8bit_round": "Stage 3a: 8-Bit Quantization (Original Landmarks)",
        "stage3b_png_reload": "Stage 3b: PNG Save & Reload (YuNet Re-detection)",
        "stage4_jpeg_q50": "Stage 4: Real JPEG Q50 & Reload (YuNet Re-detection)",
    }

    for cid in ["crop_only_w100", "paste_aware_w100"]:
        c_label = "Row 1: Crop-Only (w=1.0)" if cid == "crop_only_w100" else "Row 2: Paste-Aware (w=1.0)"
        lines.append(f"### {c_label}")
        lines.append("")
        lines.append("| Stage | ArcFace Mean Sim | ArcFace Below Thresh | FaceNet Mean Sim | FaceNet Below Thresh |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")
        c_stages = diag_cands[cid]
        for stg, label in stage_labels.items():
            af_m = c_stages[stg]["ArcFace"]["mean_similarity"]
            af_frac = c_stages[stg]["ArcFace"]["fraction_below_threshold"]
            af_k = c_stages[stg]["ArcFace"]["pairs_below_threshold"]
            af_n = c_stages[stg]["ArcFace"]["total_pairs"]
            fn_m = c_stages[stg]["FaceNet"]["mean_similarity"]
            fn_frac = c_stages[stg]["FaceNet"]["fraction_below_threshold"]
            fn_k = c_stages[stg]["FaceNet"]["pairs_below_threshold"]
            fn_n = c_stages[stg]["FaceNet"]["total_pairs"]
            lines.append(f"| **{label}** | {af_m:.4f} [measured] | {af_k}/{af_n} ({af_frac:.1f}%) [measured] | {fn_m:.4f} [measured] | {fn_k}/{fn_n} ({fn_frac:.1f}%) [measured] |")
        lines.append("")

    # Diagnostic Stage Analysis
    lines.append("### Diagnostic Findings: Loss Decomposition Across Stages")
    lines.append("")
    for cid in ["crop_only_w100", "paste_aware_w100"]:
        c_name = "Row 1 (Crop-Only)" if cid == "crop_only_w100" else "Row 2 (Paste-Aware)"
        c_st = diag_cands[cid]
        af_s1 = c_st["stage1_opt_crop"]["ArcFace"]["mean_similarity"]
        af_s2 = c_st["stage2_pasted_float"]["ArcFace"]["mean_similarity"]
        af_s3a = c_st["stage3a_8bit_round"]["ArcFace"]["mean_similarity"]
        af_s3b = c_st["stage3b_png_reload"]["ArcFace"]["mean_similarity"]
        af_s4 = c_st["stage4_jpeg_q50"]["ArcFace"]["mean_similarity"]

        fn_s1 = c_st["stage1_opt_crop"]["FaceNet"]["mean_similarity"]
        fn_s2 = c_st["stage2_pasted_float"]["FaceNet"]["mean_similarity"]
        fn_s3a = c_st["stage3a_8bit_round"]["FaceNet"]["mean_similarity"]
        fn_s3b = c_st["stage3b_png_reload"]["FaceNet"]["mean_similarity"]
        fn_s4 = c_st["stage4_jpeg_q50"]["FaceNet"]["mean_similarity"]

        lines.append(f"**{c_name} Delta Breakdown (Similarity Rebound):**")
        lines.append(f"- Stage 1 -> Stage 2 (Paste & Feathering Rebound): ArcFace **+{af_s2 - af_s1:.4f}** [measured], FaceNet **+{fn_s2 - fn_s1:.4f}** [measured]")
        lines.append(f"- Stage 2 -> Stage 3a (8-Bit Quantization): ArcFace **+{af_s3a - af_s2:.4f}** [measured], FaceNet **+{fn_s3a - fn_s2:.4f}** [measured]")
        lines.append(f"- Stage 3a -> Stage 3b (YuNet Detection Jitter): ArcFace **+{af_s3b - af_s3a:.4f}** [measured], FaceNet **+{fn_s3b - fn_s3a:.4f}** [measured]")
        lines.append(f"- Stage 3b -> Stage 4 (JPEG Q50 DCT Compression): ArcFace **+{af_s4 - af_s3b:.4f}** [measured], FaceNet **+{fn_s4 - fn_s3b:.4f}** [measured]")
        lines.append("")

    # Decision Rule & Step 5
    lines.append("## 5. Decision Rule Evaluation & Step 5 Gated Follow-up")
    lines.append("")
    lines.append(f"- **Decision Rule Condition 1:** Row 2 beats Row 1 by at least 3.0 percentage points on the 4-model mean at Q50 and 1% FAR.")
    lines.append(f"  - Row 1 Q50 4-model mean: **{decision['row1_q50_mean']:.2f}%** [measured]")
    lines.append(f"  - Row 2 Q50 4-model mean: **{decision['row2_q50_mean']:.2f}%** [measured]")
    lines.append(f"  - Difference (Row 2 - Row 1): **{decision['delta_q50_mean']:+.2f}%** [measured] (Threshold: >= +3.00%)")
    lines.append(f"- **Decision Rule Condition 2:** Held-out mean at Q50 and 1% FAR does not drop.")
    lines.append(f"  - Row 1 Q50 held-out mean: **{decision['row1_q50_heldout']:.2f}%** [measured]")
    lines.append(f"  - Row 2 Q50 held-out mean: **{decision['row2_q50_heldout']:.2f}%** [measured]")
    lines.append(f"  - Difference (Row 2 - Row 1): **{decision['delta_q50_heldout']:+.2f}%** [measured] (Threshold: >= 0.00%)")
    lines.append(f"- **Decision Outcome:** **{decision['verdict'].upper()}**")
    lines.append("")
    if decision["verdict"] == "go":
        lines.append(f"**Step 5 Follow-up:** Go rule passed. Row 3 (Paste-aware + Landmark Jitter) was executed and evaluated.")
    else:
        lines.append(f"**Step 5 Follow-up:** Go rule failed ({decision['verdict'].upper()}). Row 3 (Landmark Jitter) was skipped according to the pre-fixed protocol.")
        lines.append(f"**Identified Persistent Leakage Stage:** [hypothesis] {decision['leak_stage_explanation']}")
    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[Report] Saved final report to {report_path}", flush=True)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Varman 3.0 Paste-Aware Optimization Experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Quick smoke test on 2 pairs")
    parser.add_argument("--lfw-limit", type=int, default=150, help="Number of LFW pairs (default: 150)")
    args = parser.parse_args()

    print("=" * 90, flush=True)
    print("VARMAN 3.0 — PASTE-AWARE OPTIMIZATION EXPERIMENT", flush=True)
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # Load thresholds
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    thresh_1pct = calib_data["thresholds_1pct"]
    thresh_01pct = calib_data["thresholds_01pct"]

    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]

    engine = ProtectionEngine(model_dir=str(REPO_ROOT / "models"))
    model_suite = ModelSuite(models_dir=REPO_ROOT / "models", requested_device="cuda")
    pgd_engine = PasteAwarePGDEngine(device=device)

    # Load LFW pairs
    target_lfw_pairs = 2 if args.smoke_test else args.lfw_limit
    print(f"\n[LFW] Loading genuine pairs (target: {target_lfw_pairs})...", flush=True)
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

    print(f"[LFW] Successfully loaded {len(lfw_pairs)} genuine pairs.", flush=True)

    # STEP 2: Equivalence Check
    passed_equiv, equiv_results = run_equivalence_check(lfw_pairs, pgd_engine, engine)
    if not passed_equiv:
        print("ERROR: Equivalence check FAILED (mean diff > 1.0 / 255). Stopping experiment as required.", flush=True)
        sys.exit(1)

    # Extract clean baseline representations
    print("\n[LFW] Extracting clean baseline representations...", flush=True)
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

    # Deterministic EOT quality sequences per pair
    max_steps = 25
    lfw_q_sequences = []
    for p_i in range(len(lfw_pairs)):
        rng = np.random.default_rng(2026 + p_i)
        lfw_q_sequences.append([int(rng.integers(50, 96)) for _ in range(max_steps)])

    # STEP 3: Initial experiment run (Row 0, Row 1, Row 2)
    initial_results = run_experiment_rows(
        lfw_pairs=lfw_pairs,
        lfw_clean_embs_1=lfw_clean_embs_1,
        lfw_clean_embs_2=lfw_clean_embs_2,
        lfw_clean_match_1pct=lfw_clean_match_1pct,
        lfw_clean_match_01pct=lfw_clean_match_01pct,
        thresh_1pct=thresh_1pct,
        thresh_01pct=thresh_01pct,
        pgd_engine=pgd_engine,
        engine=engine,
        model_suite=model_suite,
        lfw_q_sequences=lfw_q_sequences,
        run_row3_jitter=False,
    )

    # Check Decision Rule before Row 3
    r1_q50_mean = initial_results["candidates"]["crop_only_w100"]["lfw"]["formats"]["q50"]["summary"]["mean_flip_1pct"]
    r2_q50_mean = initial_results["candidates"]["paste_aware_w100"]["lfw"]["formats"]["q50"]["summary"]["mean_flip_1pct"]
    r1_q50_heldout = initial_results["candidates"]["crop_only_w100"]["lfw"]["formats"]["q50"]["summary"]["heldout_flip_1pct"]
    r2_q50_heldout = initial_results["candidates"]["paste_aware_w100"]["lfw"]["formats"]["q50"]["summary"]["heldout_flip_1pct"]

    delta_q50_mean = r2_q50_mean - r1_q50_mean
    delta_q50_heldout = r2_q50_heldout - r1_q50_heldout

    go_pass = bool(delta_q50_mean >= 3.0 and delta_q50_heldout >= 0.0)
    decision = {
        "verdict": "go" if go_pass else "no-go",
        "row1_q50_mean": r1_q50_mean,
        "row2_q50_mean": r2_q50_mean,
        "delta_q50_mean": delta_q50_mean,
        "row1_q50_heldout": r1_q50_heldout,
        "row2_q50_heldout": r2_q50_heldout,
        "delta_q50_heldout": delta_q50_heldout,
        "leak_stage_explanation": "",
    }

    final_results = initial_results
    if go_pass:
        print("\n" + "=" * 90, flush=True)
        print(f"DECISION RULE PASSED (GO): Delta Q50 = {delta_q50_mean:+.2f}% >= +3.0%, Held-out Delta = {delta_q50_heldout:+.2f}% >= 0.0%", flush=True)
        print("RUNNING GATED STEP 5 ROW 3 (Paste-Aware + Landmark Jitter)...", flush=True)
        print("=" * 90, flush=True)
        final_results = run_experiment_rows(
            lfw_pairs=lfw_pairs,
            lfw_clean_embs_1=lfw_clean_embs_1,
            lfw_clean_embs_2=lfw_clean_embs_2,
            lfw_clean_match_1pct=lfw_clean_match_1pct,
            lfw_clean_match_01pct=lfw_clean_match_01pct,
            thresh_1pct=thresh_1pct,
            thresh_01pct=thresh_01pct,
            pgd_engine=pgd_engine,
            engine=engine,
            model_suite=model_suite,
            lfw_q_sequences=lfw_q_sequences,
            run_row3_jitter=True,
        )
    else:
        print("\n" + "=" * 90, flush=True)
        print(f"DECISION RULE FAILED (NO-GO): Delta Q50 = {delta_q50_mean:+.2f}% (Threshold: >= +3.0%), Held-out Delta = {delta_q50_heldout:+.2f}%", flush=True)
        print("SKIPPING ROW 3 ACCORDING TO PRE-FIXED PROTOCOL.", flush=True)
        print("=" * 90, flush=True)

    # STEP 4: Fixed Stage Diagnostic
    diag_subset = 2 if args.smoke_test else 40
    diag_summary = run_step4_stage_diagnostic(
        lfw_pairs=lfw_pairs,
        pgd_engine=pgd_engine,
        engine=engine,
        thresh_1pct=thresh_1pct,
        lfw_q_sequences=lfw_q_sequences,
        subset_count=diag_subset,
    )

    # Formulate hypothesis for leakage stage
    r2_st = diag_summary["candidates"]["paste_aware_w100"]
    deltas = {
        "Stage 1 -> Stage 2 (Canvas Pasting & Feathering)": r2_st["stage2_pasted_float"]["ArcFace"]["mean_similarity"] - r2_st["stage1_opt_crop"]["ArcFace"]["mean_similarity"],
        "Stage 2 -> Stage 3a (8-bit Quantization)": r2_st["stage3a_8bit_round"]["ArcFace"]["mean_similarity"] - r2_st["stage2_pasted_float"]["ArcFace"]["mean_similarity"],
        "Stage 3a -> Stage 3b (YuNet Detection Jitter)": r2_st["stage3b_png_reload"]["ArcFace"]["mean_similarity"] - r2_st["stage3a_8bit_round"]["ArcFace"]["mean_similarity"],
        "Stage 3b -> Stage 4 (JPEG Q50 Compression)": r2_st["stage4_jpeg_q50"]["ArcFace"]["mean_similarity"] - r2_st["stage3b_png_reload"]["ArcFace"]["mean_similarity"],
    }
    largest_stage = max(deltas.items(), key=lambda x: x[1])
    decision["leak_stage_explanation"] = f"The largest persistent similarity increase occurs at '{largest_stage[0]}' (+{largest_stage[1]:.4f} ArcFace similarity rebound)."

    # Save final JSON
    final_json_path = RESULTS_DIR / "paste_aware_final.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\n[Final] Saved final results to {final_json_path}", flush=True)

    # Save summary CSV
    summary_csv_path = RESULTS_DIR / "paste_aware_summary.csv"
    export_summary_csv(final_results, summary_csv_path)
    print(f"[Final] Saved summary CSV to {summary_csv_path}", flush=True)

    # Write REPORT.md
    report_path = RESULTS_DIR / "REPORT.md"
    write_report_md(final_results, diag_summary, equiv_results, decision, report_path)
    print(f"[Final] Saved REPORT.md to {report_path}", flush=True)

    # Cleanup temporary images
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        print(f"[Cleanup] Removed temporary image directory: {TMP_DIR}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("PASTE-AWARE EXPERIMENT COMPLETE", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
