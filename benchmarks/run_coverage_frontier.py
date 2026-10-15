"""
Varman Coverage Frontier & Metric Reconciliation Experiment
Evaluates the coverage frontier across mask scale (1.0, 1.15, 1.3) and epsilon (8, 7, 6, 5 / 255)
on 150 LFW genuine pairs, evaluated under real saved-and-reloaded JPEG Q50 at 1% and 0.1% FAR.
Computes visibility metrics under Definition B (standard: fixed canonical face mask on composite),
Definition A (full 112 crop on composite), and Definition C (raw PGD crop).
Includes Step 3: Graded-grain background extension (amplitudes 0.25 and 0.50 with radial falloff).
"""

from __future__ import annotations

import argparse
import csv
import json
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
from benchmarks.evaluate_face_models import ModelSuite, cosine_similarity
from benchmarks.run_paste_aware_experiment import (
    compute_wilson_ci,
)
from benchmarks.run_region_ablation import (
    build_scaled_mask,
    composite_crop_back_scaled,
    RegionAblationEngine,
    run_mcnemar_exact,
    holm_bonferroni,
)

CALIB_PATH = REPO_ROOT / "benchmarks" / "results" / "sweep_evidence" / "calibrated_thresholds_sweep.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "coverage_frontier"
IMAGE_STORE_DIR = Path(r"D:\JavaProjects\Varman-research\frontier2_2026-10-15")

CANONICAL_FACE_MASK_BASE = (get_canonical_face_mask_112() >= 0.10)


def compute_visibility_all_defs(
    clean_full_bgr: np.ndarray,
    prot_full_bgr: np.ndarray,
    clean_crop_bgr: np.ndarray,
    raw_prot_crop_bgr: np.ndarray,
    reextracted_prot_crop_bgr: np.ndarray,
    M: np.ndarray,
    fixed_face_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes visibility metrics under:
      - Definition B (Standard): Fixed canonical face mask (>=0.10) on re-extracted composite crop
      - Definition A: Full 112x112 re-extracted composite crop
      - Definition C: Raw unfeathered PGD crop
      - Whole image: Canvas-wide SSIM and CIEDE2000
    """
    h, w = clean_full_bgr.shape[:2]

    # Whole image
    w_ssim = float(compute_ssim(clean_full_bgr, prot_full_bgr, channel_axis=2, data_range=255))
    lab_clean_full = rgb2lab(cv2.cvtColor(clean_full_bgr, cv2.COLOR_BGR2RGB) / 255.0)
    lab_prot_full = rgb2lab(cv2.cvtColor(prot_full_bgr, cv2.COLOR_BGR2RGB) / 255.0)
    de_full = deltaE_ciede2000(lab_clean_full, lab_prot_full)
    w_ciede_m = float(np.mean(de_full))
    w_ciede_p95 = float(np.percentile(de_full, 95.0))

    # Delta differences for crop re-extracted from composite
    lab_clean_c = rgb2lab(cv2.cvtColor(clean_crop_bgr, cv2.COLOR_BGR2RGB) / 255.0)
    lab_prot_c = rgb2lab(cv2.cvtColor(reextracted_prot_crop_bgr, cv2.COLOR_BGR2RGB) / 255.0)
    de_crop_composite = deltaE_ciede2000(lab_clean_c, lab_prot_c) # (112, 112)

    # Definition A: full 112x112 re-extracted composite crop
    ssim_A = float(compute_ssim(clean_crop_bgr, reextracted_prot_crop_bgr, channel_axis=2, data_range=255))
    ciede_m_A = float(np.mean(de_crop_composite))
    ciede_p95_A = float(np.percentile(de_crop_composite, 95.0))

    # Definition B (Standard): fixed canonical face mask on composite
    _, ssim_map = compute_ssim(clean_crop_bgr, reextracted_prot_crop_bgr, channel_axis=2, data_range=255, full=True)
    ssim_map_2d = np.mean(ssim_map, axis=2)
    ssim_B = float(np.mean(ssim_map_2d[fixed_face_mask]))
    de_B = de_crop_composite[fixed_face_mask]
    ciede_m_B = float(np.mean(de_B))
    ciede_p95_B = float(np.percentile(de_B, 95.0))
    ciede_p99_B = float(np.percentile(de_B, 99.0))

    # Definition C: raw unfeathered PGD crop
    ssim_C = float(compute_ssim(clean_crop_bgr, raw_prot_crop_bgr, channel_axis=2, data_range=255))
    lab_raw_c = rgb2lab(cv2.cvtColor(raw_prot_crop_bgr, cv2.COLOR_BGR2RGB) / 255.0)
    de_raw = deltaE_ciede2000(lab_clean_c, lab_raw_c)
    ciede_m_C = float(np.mean(de_raw[fixed_face_mask]))
    ciede_p95_C = float(np.percentile(de_raw[fixed_face_mask], 95.0))

    # Pixel change metrics on full canvas
    diff_full = np.abs(prot_full_bgr.astype(np.float32) - clean_full_bgr.astype(np.float32))
    max_px_diff = float(np.max(diff_full))
    mean_px_diff = float(np.mean(diff_full))

    # Footprint change fractions
    inv_M = cv2.invertAffineTransform(M)
    crop_footprint_112 = np.ones((112, 112), dtype=np.float32)
    warped_footprint = cv2.warpAffine(crop_footprint_112, inv_M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

    is_changed = (diff_full > 0).any(axis=2)
    in_fp = (warped_footprint > 0.5)
    out_fp = ~in_fp
    ch_in = float(np.sum(is_changed & in_fp)) / float(np.sum(in_fp)) if np.sum(in_fp) > 0 else 0.0
    ch_out = float(np.sum(is_changed & out_fp)) / float(np.sum(out_fp)) if np.sum(out_fp) > 0 else 0.0

    return {
        "def_B_standard": {
            "face_ssim": ssim_B,
            "ciede_mean": ciede_m_B,
            "ciede_p95": ciede_p95_B,
            "ciede_p99": ciede_p99_B,
        },
        "def_A_full_112": {
            "face_ssim": ssim_A,
            "ciede_mean": ciede_m_A,
            "ciede_p95": ciede_p95_A,
        },
        "def_C_raw_pgd": {
            "face_ssim": ssim_C,
            "ciede_mean": ciede_m_C,
            "ciede_p95": ciede_p95_C,
        },
        "whole_image": {
            "full_ssim": w_ssim,
            "full_ciede_mean": w_ciede_m,
            "full_ciede_p95": w_ciede_p95,
        },
        "deltas": {
            "max_pixel_change": max_px_diff,
            "mean_pixel_change": mean_px_diff,
            "changed_inside_footprint": ch_in,
            "changed_outside_footprint": ch_out,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Varman Coverage Frontier & Metric Reconciliation")
    parser.add_argument("--smoke-test", action="store_true", help="Quick run on 2 pairs")
    parser.add_argument("--lfw-limit", type=int, default=150, help="Number of LFW pairs (default: 150)")
    args = parser.parse_args()

    print("=" * 90, flush=True)
    print("VARMAN — COVERAGE FRONTIER EXPERIMENT (150 LFW Genuine Pairs)", flush=True)
    print(f"Start Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 90, flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # Check free disk space
    c_free_gb = shutil.disk_usage("C:\\").free / (1024 ** 3)
    d_free_gb = shutil.disk_usage("D:\\").free / (1024 ** 3)
    print(f"[Preflight] Free Disk Space: C: {c_free_gb:.2f} GB, D: {d_free_gb:.2f} GB", flush=True)
    if c_free_gb < 10.0:
        raise RuntimeError(f"STOP CONDITION: Drive C: free space ({c_free_gb:.2f} GB) is below 10 GB limit.")

    IMAGE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load thresholds
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    thresh_1pct = calib_data["thresholds_1pct"]
    thresh_01pct = calib_data["thresholds_01pct"]

    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    formats = ["png", "q75", "q50"]

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
    paired_q_sequences: List[List[int]] = []
    paired_jitter_params: List[List[Tuple[float, float, float]]] = []
    for p_i in range(n_pairs):
        rng = np.random.default_rng(2026 + p_i)
        q_seq = [int(rng.integers(50, 96)) for _ in range(25)]
        jit = [(float(rng.uniform(0.97, 1.03)), float(rng.uniform(-2.0, 2.0)), float(rng.uniform(-2.0, 2.0))) for _ in range(25)]
        paired_q_sequences.append(q_seq)
        paired_jitter_params.append(jit)

    # Rows ordered as requested:
    # F0: eps 8, scale 1.0 (same-run baseline)
    # F1: eps 8, scale 1.3
    # F2: eps 7, scale 1.3
    # F3: eps 6, scale 1.3
    # F4: eps 5, scale 1.3
    # F5: eps 8, scale 1.15 (dose-response check; run last)
    ROWS = [
        {"id": "F0_eps8_scale10", "name": "F0: eps 8/255, scale 1.0 (Baseline)", "eps": 8.0 / 255.0, "mask_scale": 1.0},
        {"id": "F1_eps8_scale13", "name": "F1: eps 8/255, scale 1.3", "eps": 8.0 / 255.0, "mask_scale": 1.3},
        {"id": "F2_eps7_scale13", "name": "F2: eps 7/255, scale 1.3", "eps": 7.0 / 255.0, "mask_scale": 1.3},
        {"id": "F3_eps6_scale13", "name": "F3: eps 6/255, scale 1.3", "eps": 6.0 / 255.0, "mask_scale": 1.3},
        {"id": "F4_eps5_scale13", "name": "F4: eps 5/255, scale 1.3", "eps": 5.0 / 255.0, "mask_scale": 1.3},
        {"id": "F5_eps8_scale115", "name": "F5: eps 8/255, scale 1.15 (Dose-Response)", "eps": 8.0 / 255.0, "mask_scale": 1.15},
    ]

    all_results: Dict[str, Any] = {
        "benchmark_metadata": {
            "name": "Varman Coverage Frontier Experiment",
            "date": time.strftime("%Y-%m-%d"),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "n_pairs": n_pairs,
            "thresholds_1pct": thresh_1pct,
            "thresholds_01pct": thresh_01pct,
            "clean_denoms_1pct": clean_denoms_1pct,
            "clean_denoms_01pct": clean_denoms_01pct,
            "standard_definition": "Definition B: SSIM and CIEDE2000 computed on re-extracted composite crop within fixed canonical face mask (>=0.10).",
        },
        "candidates": {},
        "pair_flips": {},
        "pair_sims": {},
    }

    recent_latencies: List[float] = []

    # Execute Frontier Rows
    for row_idx, row_cfg in enumerate(ROWS):
        r_id = row_cfg["id"]
        r_name = row_cfg["name"]
        eps = row_cfg["eps"]
        m_scale = row_cfg["mask_scale"]

        row_dir = IMAGE_STORE_DIR / r_id
        row_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90, flush=True)
        print(f"RUNNING {r_name} (Row {row_idx + 1} of {len(ROWS)})", flush=True)
        print(f"  eps={eps:.5f} ({int(round(eps*255))}/255), mask_scale={m_scale}", flush=True)
        print("=" * 90, flush=True)

        cand_output: Dict[str, Any] = {
            "config": row_cfg,
            "results_by_format": {},
        }

        cand_pair_flips: Dict[str, Dict[str, List[bool]]] = {fmt: {m: [] for m in models} for fmt in formats}
        cand_pair_sims: Dict[str, Dict[str, List[float]]] = {fmt: {m: [] for m in models} for fmt in formats}

        vis_records_defB = []
        vis_records_defA = []
        vis_records_defC = []
        vis_records_whole = []
        vis_records_deltas = []
        row_latencies = []

        row_t0 = time.perf_counter()

        for p_i, (b1, b2, f1, f2, c1, c2) in enumerate(lfw_pairs):
            t_pair0 = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device) if torch.cuda.is_available() else None

            # Seed for pair reproducibility across rows
            torch.manual_seed(2026 + p_i)

            M1 = cv2.estimateAffinePartial2D(np.array(f1["landmarks"][:5], dtype=np.float32), ARCFACE_CANONICAL_112)[0]
            raw_prot_crop_bgr, _, _ = pgd_engine.protect(
                img_bgr=b1,
                landmarks=f1["landmarks"],
                crop_112_bgr=c1,
                M=M1,
                mask_scale=m_scale,
                steps=25,
                w_fn=1.0,
                eps=eps,
                mu=0.9,
                q_sequence=paired_q_sequences[p_i],
                jitter_params=paired_jitter_params[p_i],
            )

            prot_full_bgr, _ = composite_crop_back_scaled(
                original_bgr=b1,
                protected_crop_112=raw_prot_crop_bgr,
                affine_matrix_M=M1,
                mask_scale=m_scale,
                feather_sigma=4.0,
            )

            t_pair1 = time.perf_counter()
            lat_sec = t_pair1 - t_pair0
            row_latencies.append(lat_sec)
            recent_latencies.append(lat_sec)
            if len(recent_latencies) > 5:
                recent_latencies.pop(0)

            # Safety assertions
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

            # Reload from disk (real saved-and-reloaded files)
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

            # Re-extract crop from composite PNG for Definition A & B
            reextracted_crop, _ = align_crop_112(prot_full_bgr, f1["landmarks"])
            if reextracted_crop is None:
                reextracted_crop = raw_prot_crop_bgr

            # Multi-definition visibility computation
            vis_m = compute_visibility_all_defs(
                clean_full_bgr=b1,
                prot_full_bgr=prot_full_bgr,
                clean_crop_bgr=c1,
                raw_prot_crop_bgr=raw_prot_crop_bgr,
                reextracted_prot_crop_bgr=reextracted_crop,
                M=M1,
                fixed_face_mask=CANONICAL_FACE_MASK_BASE,
            )
            vis_records_defB.append(vis_m["def_B_standard"])
            vis_records_defA.append(vis_m["def_A_full_112"])
            vis_records_defC.append(vis_m["def_C_raw_pgd"])
            vis_records_whole.append(vis_m["whole_image"])
            vis_records_deltas.append(vis_m["deltas"])

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

        # Compile multi-definition visibility
        cand_output["visibility"] = {
            "standard_def_B": {
                "face_ssim": round(float(np.mean([x["face_ssim"] for x in vis_records_defB])), 4),
                "ciede_mean": round(float(np.mean([x["ciede_mean"] for x in vis_records_defB])), 4),
                "ciede_p95": round(float(np.mean([x["ciede_p95"] for x in vis_records_defB])), 4),
                "ciede_p99": round(float(np.mean([x["ciede_p99"] for x in vis_records_defB])), 4),
            },
            "def_A_full_112_crop": {
                "face_ssim": round(float(np.mean([x["face_ssim"] for x in vis_records_defA])), 4),
                "ciede_mean": round(float(np.mean([x["ciede_mean"] for x in vis_records_defA])), 4),
                "ciede_p95": round(float(np.mean([x["ciede_p95"] for x in vis_records_defA])), 4),
            },
            "def_C_raw_pgd_crop": {
                "face_ssim": round(float(np.mean([x["face_ssim"] for x in vis_records_defC])), 4),
                "ciede_mean": round(float(np.mean([x["ciede_mean"] for x in vis_records_defC])), 4),
                "ciede_p95": round(float(np.mean([x["ciede_p95"] for x in vis_records_defC])), 4),
            },
            "whole_image": {
                "full_ssim": round(float(np.mean([x["full_ssim"] for x in vis_records_whole])), 4),
                "full_ciede_mean": round(float(np.mean([x["full_ciede_mean"] for x in vis_records_whole])), 4),
                "full_ciede_p95": round(float(np.mean([x["full_ciede_p95"] for x in vis_records_whole])), 4),
            },
            "deltas": {
                "mean_pixel_change": round(float(np.mean([x["mean_pixel_change"] for x in vis_records_deltas])), 4),
                "max_pixel_change": round(float(np.max([x["max_pixel_change"] for x in vis_records_deltas])), 2),
                "changed_inside_footprint": round(float(np.mean([x["changed_inside_footprint"] for x in vis_records_deltas])), 4),
                "changed_outside_footprint": round(float(np.mean([x["changed_outside_footprint"] for x in vis_records_deltas])), 6),
            },
            "mean_latency_sec": round(float(np.mean(row_latencies)), 2),
            "total_row_time_sec": round(row_time, 2),
            "peak_vram_gb": round(peak_vram, 3),
        }

        all_results["candidates"][r_id] = cand_output
        all_results["pair_flips"][r_id] = cand_pair_flips
        all_results["pair_sims"][r_id] = cand_pair_sims

        print(f"  --> Completed {r_id} in {row_time:.1f}s")
        print(f"  --> Q50 1% FAR: 4-model mean = {cand_output['results_by_format']['q50']['summary']['mean_flip_1pct']}%, held-out = {cand_output['results_by_format']['q50']['summary']['heldout_flip_1pct']}%")
        print(f"  --> Visibility (Standard Def B): Face SSIM={cand_output['visibility']['standard_def_B']['face_ssim']}, CIEDE Mean={cand_output['visibility']['standard_def_B']['ciede_mean']}, p95={cand_output['visibility']['standard_def_B']['ciede_p95']}", flush=True)

        # Write partial results after every row
        partial_json_path = RESULTS_DIR / "coverage_frontier_partial.json"
        with open(partial_json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

    # ==========================================================================
    # PAIRED STATISTICAL TESTS (Two Families with Holm-Bonferroni correction)
    # ==========================================================================
    print("\n" + "=" * 90, flush=True)
    print("COMPUTING PAIRED EXACT MCNEMAR TESTS (Q50, 1% FAR)", flush=True)
    print("=" * 90, flush=True)

    f0_flips = all_results["pair_flips"]["F0_eps8_scale10"]["q50"]
    f1_flips = all_results["pair_flips"]["F1_eps8_scale13"]["q50"]

    mcnemar_family_1: Dict[str, Any] = {} # F1-F5 vs F0
    mcnemar_family_2: Dict[str, Any] = {} # F2-F4 vs F1

    # Family 1: vs F0
    for cand_id in ["F1_eps8_scale13", "F2_eps7_scale13", "F3_eps6_scale13", "F4_eps5_scale13", "F5_eps8_scale115"]:
        cand_flips = all_results["pair_flips"][cand_id]["q50"]
        c_tests = {}
        raw_p = []
        for m in models:
            valid_idx = [i for i in range(n_pairs) if clean_match_1pct[m][i]]
            f_base = [f0_flips[m][i] for i in valid_idx]
            f_cand = [cand_flips[m][i] for i in valid_idx]
            t = run_mcnemar_exact(f_base, f_cand)
            c_tests[m] = t
            raw_p.append(t["exact_p"])
        adj_p = holm_bonferroni(raw_p)
        for idx, m in enumerate(models):
            c_tests[m]["holm_adj_p"] = round(adj_p[idx], 5)
            c_tests[m]["significant_05"] = bool(adj_p[idx] < 0.05)
        mcnemar_family_1[cand_id] = c_tests

    # Family 2: vs F1
    for cand_id in ["F2_eps7_scale13", "F3_eps6_scale13", "F4_eps5_scale13"]:
        cand_flips = all_results["pair_flips"][cand_id]["q50"]
        c_tests = {}
        raw_p = []
        for m in models:
            valid_idx = [i for i in range(n_pairs) if clean_match_1pct[m][i]]
            f_base = [f1_flips[m][i] for i in valid_idx]
            f_cand = [cand_flips[m][i] for i in valid_idx]
            t = run_mcnemar_exact(f_base, f_cand)
            c_tests[m] = t
            raw_p.append(t["exact_p"])
        adj_p = holm_bonferroni(raw_p)
        for idx, m in enumerate(models):
            c_tests[m]["holm_adj_p"] = round(adj_p[idx], 5)
            c_tests[m]["significant_05"] = bool(adj_p[idx] < 0.05)
        mcnemar_family_2[cand_id] = c_tests

    all_results["mcnemar_tests"] = {
        "family_1_vs_F0": mcnemar_family_1,
        "family_2_vs_F1": mcnemar_family_2,
    }

    # ==========================================================================
    # STEP 3: GRADED GRAIN MEASUREMENT (N=50 pairs)
    # ==========================================================================
    print("\n" + "=" * 90, flush=True)
    print("STEP 3: GRADED GRAIN MEASUREMENT (No PGD, First 50 Pairs of F1)", flush=True)
    print("=" * 90, flush=True)

    grain_dir = IMAGE_STORE_DIR / "graded_grain"
    grain_dir.mkdir(parents=True, exist_ok=True)
    f1_dir = IMAGE_STORE_DIR / "F1_eps8_scale13"

    grain_pairs_count = min(50, n_pairs)
    kernel_ring = np.ones((31, 31), np.uint8)

    graded_results = {
        "pairs_evaluated": grain_pairs_count,
        "amplitudes": [0.25, 0.50],
        "eval_by_amp": {},
    }

    for amp_scale in [0.25, 0.50]:
        amp_key = f"amp_{int(amp_scale*100):02d}"
        g_flips_png = {m: [] for m in models}
        g_flips_q50 = {m: [] for m in models}
        f1_50_flips_q50 = {m: [] for m in models}
        f1_50_flips_png = {m: [] for m in models}

        disp_list = []
        det_count = 0
        w_ssims = []
        w_ciedes = []
        ch_fracs = []
        r1_rms_in = []
        r1_rms_rg = []
        g_rms_in = []
        g_rms_rg = []

        for p_i in range(grain_pairs_count):
            b1, b2, f1, f2, c1, c2 = lfw_pairs[p_i]
            f1_png_path = f1_dir / f"pair_{p_i:04d}_png.png"
            f1_img = cv2.imread(str(f1_png_path))
            if f1_img is None:
                continue

            h, w = b1.shape[:2]
            delta_f1 = f1_img.astype(np.float32) - b1.astype(np.float32)

            M1 = cv2.estimateAffinePartial2D(np.array(f1["landmarks"][:5], dtype=np.float32), ARCFACE_CANONICAL_112)[0]
            mask_112 = build_scaled_mask(1.3, 4.0)
            inv_M = cv2.invertAffineTransform(M1)
            warped_mask = cv2.warpAffine(mask_112, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

            # Smooth radial falloff reaching zero at border
            # Center of face
            pts = np.array(f1["landmarks"][:5], dtype=np.float32)
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
            ys, xs = np.indices((h, w), dtype=np.float32)
            dists = np.sqrt((xs - cx)**2 + (ys - cy)**2)
            max_dist = max(np.sqrt(cx**2 + cy**2), np.sqrt((w - cx)**2 + cy**2),
                           np.sqrt(cx**2 + (h - cy)**2), np.sqrt((w - cx)**2 + (h - cy)**2))
            norm_dist = np.clip(dists / (max_dist + 1e-6), 0.0, 1.0)
            radial_falloff = np.cos(norm_dist * (np.pi / 2.0))

            # Phase-randomized FFT
            rng_grain = np.random.default_rng(2026 + p_i)
            grain = np.zeros_like(delta_f1)
            for ch in range(3):
                d_ch = delta_f1[:, :, ch]
                F_ch = np.fft.rfft2(d_ch)
                amp = np.abs(F_ch)
                phase = rng_grain.uniform(-np.pi, np.pi, size=F_ch.shape)
                phase[0, 0] = 0.0
                F_rand = amp * np.exp(1j * phase)
                syn = np.fft.irfft2(F_rand, s=(h, w))

                active_std = np.std(d_ch[warped_mask > 0.5]) if np.sum(warped_mask > 0.5) > 0 else 1.0
                syn_std = np.std(syn)
                if syn_std > 1e-6:
                    syn = syn * (active_std / syn_std)
                grain[:, :, ch] = syn * amp_scale * radial_falloff

            w_mask_3ch = warped_mask[:, :, None]
            graded_comp = w_mask_3ch * f1_img.astype(np.float32) + (1.0 - w_mask_3ch) * (b1.astype(np.float32) + grain)
            graded_img = np.clip(graded_comp, 0.0, 255.0).astype(np.uint8)

            # Save real files
            g_png = grain_dir / f"graded_{amp_key}_p{p_i:04d}_png.png"
            g_q50 = grain_dir / f"graded_{amp_key}_p{p_i:04d}_q50.jpg"
            cv2.imwrite(str(g_png), graded_img)
            cv2.imwrite(str(g_q50), graded_img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            g_png_rel = cv2.imread(str(g_png))
            g_q50_rel = cv2.imread(str(g_q50))

            faces_g = engine.detect_faces(g_png_rel)
            if faces_g:
                det_count += 1
                fg = faces_g[0]
                lm_b1 = np.array(f1["landmarks"][:5], dtype=np.float32)
                lm_g = np.array(fg["landmarks"][:5], dtype=np.float32)
                disp_list.append(float(np.mean(np.linalg.norm(lm_g - lm_b1, axis=1))))
            else:
                fg = f1

            embs_g_png, _ = model_suite.extract_model_features(g_png_rel, fg)
            embs_g_q50, _ = model_suite.extract_model_features(g_q50_rel, fg)
            e2 = clean_embs_2[p_i]

            for m in models:
                s_png = float(cosine_similarity(embs_g_png[m], e2[m])) if (embs_g_png[m] is not None and e2[m] is not None) else 1.0
                s_q50 = float(cosine_similarity(embs_g_q50[m], e2[m])) if (embs_g_q50[m] is not None and e2[m] is not None) else 1.0
                g_flips_png[m].append(bool(s_png < thresh_1pct[m]))
                g_flips_q50[m].append(bool(s_q50 < thresh_1pct[m]))

                f1_flips_png_val = all_results["pair_flips"]["F1_eps8_scale13"]["png"][m][p_i]
                f1_flips_q50_val = all_results["pair_flips"]["F1_eps8_scale13"]["q50"][m][p_i]
                f1_50_flips_png[m].append(f1_flips_png_val)
                f1_50_flips_q50[m].append(f1_flips_q50_val)

            # Whole-image visibility
            w_ssims.append(float(compute_ssim(b1, graded_img, channel_axis=2, data_range=255)))
            lab_b1 = rgb2lab(cv2.cvtColor(b1, cv2.COLOR_BGR2RGB) / 255.0)
            lab_g = rgb2lab(cv2.cvtColor(graded_img, cv2.COLOR_BGR2RGB) / 255.0)
            w_ciedes.append(float(np.mean(deltaE_ciede2000(lab_b1, lab_g))))

            diff_g = np.abs(graded_img.astype(np.float32) - b1.astype(np.float32))
            ch_fracs.append(float(np.mean((diff_g > 0).any(axis=2))))

            # Conspicuity proxy
            def eval_conspicuity(delta_arr, mask_2d):
                blur = cv2.GaussianBlur(delta_arr, (9, 9), 2.0)
                hp = delta_arr - blur
                hp_lum = np.mean(hp, axis=2)
                core_mask = (mask_2d >= 0.5)
                ring_outer = cv2.dilate((mask_2d >= 0.05).astype(np.uint8), kernel_ring, iterations=1)
                ring_mask = (ring_outer > 0) & (mask_2d < 0.05)
                rms_in = float(np.sqrt(np.mean(hp_lum[core_mask] ** 2))) if np.sum(core_mask) > 0 else 0.0
                rms_rg = float(np.sqrt(np.mean(hp_lum[ring_mask] ** 2))) if np.sum(ring_mask) > 0 else 0.0
                return rms_in, rms_rg

            r_in_f1, r_rg_f1 = eval_conspicuity(delta_f1, warped_mask)
            r_in_gr, r_rg_gr = eval_conspicuity(graded_comp - b1.astype(np.float32), warped_mask)
            r1_rms_in.append(r_in_f1)
            r1_rms_rg.append(r_rg_f1)
            g_rms_in.append(r_in_gr)
            g_rms_rg.append(r_rg_gr)

        # Summarize amplitude
        amp_res = {
            "amplitude": amp_scale,
            "yunet_detection_rate": round(float(det_count / grain_pairs_count * 100.0), 2),
            "mean_landmark_displacement_px": round(float(np.mean(disp_list)) if disp_list else 0.0, 3),
            "whole_image_ssim": round(float(np.mean(w_ssims)), 4),
            "whole_image_ciede": round(float(np.mean(w_ciedes)), 4),
            "changed_pixel_fraction": round(float(np.mean(ch_fracs)), 4),
            "conspicuity_proxy": {
                "f1_conspicuity_ratio": round(float(np.mean(r1_rms_in)) / (float(np.mean(r1_rms_rg)) + 1e-6), 2),
                "graded_conspicuity_ratio": round(float(np.mean(g_rms_in)) / (float(np.mean(g_rms_rg)) + 1e-6), 2),
                "f1_rms_inside": round(float(np.mean(r1_rms_in)), 4),
                "f1_rms_ring": round(float(np.mean(r1_rms_rg)), 4),
                "graded_rms_inside": round(float(np.mean(g_rms_in)), 4),
                "graded_rms_ring": round(float(np.mean(g_rms_rg)), 4),
            },
            "eval_by_format": {},
        }
        for fmt_name, g_fl, f_fl in [("png", g_flips_png, f1_50_flips_png), ("q50", g_flips_q50, f1_50_flips_q50)]:
            f_d = {}
            for m in models:
                v_idx = [i for i in range(grain_pairs_count) if clean_match_1pct[m][i]]
                k_g = sum([g_fl[m][i] for i in v_idx])
                k_f = sum([f_fl[m][i] for i in v_idx])
                n_v = len(v_idx)
                f_d[m] = {
                    "graded_flips": f"{k_g}/{n_v} ({(k_g/n_v*100.0) if n_v > 0 else 0.0:.1f}%)",
                    "f1_flips": f"{k_f}/{n_v} ({(k_f/n_v*100.0) if n_v > 0 else 0.0:.1f}%)",
                }
            f_d["mean_4m_graded"] = round(float(np.mean([(sum([g_fl[m][i] for i in [x for x in range(grain_pairs_count) if clean_match_1pct[m][x]]]) / len([x for x in range(grain_pairs_count) if clean_match_1pct[m][x]]) * 100.0) for m in models])), 2)
            f_d["mean_4m_f1"] = round(float(np.mean([(sum([f_fl[m][i] for i in [x for x in range(grain_pairs_count) if clean_match_1pct[m][x]]]) / len([x for x in range(grain_pairs_count) if clean_match_1pct[m][x]]) * 100.0) for m in models])), 2)
            amp_res["eval_by_format"][fmt_name] = f_d
        graded_results["eval_by_amp"][amp_key] = amp_res

    all_results["step3_graded_grain_results"] = graded_results

    # Save final JSON and pair flips JSON
    final_json_path = RESULTS_DIR / "coverage_frontier_final.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Final JSON results: {final_json_path}", flush=True)

    pair_flips_path = RESULTS_DIR / "coverage_frontier_pair_flips.json"
    with open(pair_flips_path, "w", encoding="utf-8") as f:
        json.dump({"flips": all_results["pair_flips"], "sims": all_results["pair_sims"]}, f)
    print(f"[Saved] Per-pair flip vectors: {pair_flips_path}", flush=True)

    # Export Summary CSV
    csv_path = RESULTS_DIR / "coverage_frontier_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_id", "row_name", "format",
            "mean_flip_1pct", "heldout_flip_1pct",
            "FaceNet_1pct", "ArcFace_1pct", "SFace_1pct", "MobileFaceNet_1pct",
            "defB_face_ssim", "defB_ciede_mean", "defB_ciede_p95",
            "defA_face_ssim", "defA_ciede_mean", "defC_raw_ssim",
            "full_ssim", "full_ciede_mean", "latency_sec", "peak_vram_gb"
        ])
        for r_id, cand in all_results["candidates"].items():
            vis = cand["visibility"]
            vB = vis["standard_def_B"]
            vA = vis["def_A_full_112_crop"]
            vC = vis["def_C_raw_pgd_crop"]
            vW = vis["whole_image"]
            for fmt in formats:
                f_data = cand["results_by_format"][fmt]
                writer.writerow([
                    r_id, cand["config"]["name"], fmt,
                    f_data["summary"]["mean_flip_1pct"], f_data["summary"]["heldout_flip_1pct"],
                    f_data["FaceNet"]["flips_1pct_pct"], f_data["ArcFace"]["flips_1pct_pct"],
                    f_data["SFace"]["flips_1pct_pct"], f_data["MobileFaceNet"]["flips_1pct_pct"],
                    vB["face_ssim"], vB["ciede_mean"], vB["ciede_p95"],
                    vA["face_ssim"], vA["ciede_mean"], vC["face_ssim"],
                    vW["full_ssim"], vW["full_ciede_mean"],
                    vis["mean_latency_sec"], vis["peak_vram_gb"]
                ])
    print(f"[Saved] Summary CSV: {csv_path}", flush=True)

    # Compact REPORT.md
    report_md_path = RESULTS_DIR / "REPORT.md"
    frontier_lines = [
        "# Varman Coverage Frontier Benchmark Results",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d')}  ",
        f"**Pairs:** {n_pairs} LFW genuine pairs  ",
        f"**Hardware:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}  ",
        f"**Standard Visibility Definition:** Definition B (fixed canonical face mask on re-extracted composite crop)  ",
        "",
        "## Coverage Frontier Table (Real JPEG Q50, 1% FAR)",
        "",
        "| Row ID | Epsilon | Mask Scale | Face SSIM (Def B) | CIEDE Mean | CIEDE p95 | Whole SSIM | Q50 1% FAR 4-Model Mean | Held-Out Mean |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r_id, cand in all_results["candidates"].items():
        vB = cand["visibility"]["standard_def_B"]
        vW = cand["visibility"]["whole_image"]
        q50 = cand["results_by_format"]["q50"]["summary"]
        eps_int = int(round(cand["config"]["eps"] * 255.0))
        m_sc = cand["config"]["mask_scale"]
        frontier_lines.append(
            f"| **{r_id}** | {eps_int}/255 | {m_sc:.2f} | {vB['face_ssim']:.4f} | {vB['ciede_mean']:.4f} | {vB['ciede_p95']:.4f} | {vW['full_ssim']:.4f} | **{q50['mean_flip_1pct']:.2f}%** | **{q50['heldout_flip_1pct']:.2f}%** |"
        )
    frontier_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(frontier_lines))
    print(f"[Saved] Compact REPORT.md: {report_md_path}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("ALL COVERAGE FRONTIER RUNS COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
