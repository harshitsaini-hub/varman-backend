"""
Varman Controlled Pilot: YuNet Landmark Displacement on 30 LFW Pairs (Measurement-Only)

Objective:
  Directly measure YuNet face detector landmark drift and affine alignment parameter shifts
  between clean originals, clean JPEG Q50 controls, protected PNG, and protected JPEG Q50
  on the first 30 LFW genuine pairs.

Operating Constraints & Standards:
  - Row 3 configuration: eps=8/255 (luminance-only), 25 steps, alpha=eps/7, momentum=0.9
  - White-box surrogates: ArcFace (1.0) and FaceNet (1.0)
  - Held-out evaluation models: SFace and MobileFaceNet
  - RNG seeds: 2026 + pair_index for EOT JPEG sequences, landmark jitter, and PyTorch seeds.
  - Image files saved to D:\\JavaProjects\\Varman-research\\pilot_landmark_2026-10-15\\ (outside repo)
  - Result metrics saved to benchmarks/results/landmark_drift_pilot/
  - Stop conditions: C: free space < 10 GB, peak VRAM > 3.8 GB, or per-image latency > 6s over 5 pairs.
"""

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

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_lfw_pairs

# Import verified components from existing benchmark harness
from benchmarks.run_paste_aware_experiment import (
    PasteAwarePGDEngine,
    ARCFACE_CANONICAL_112,
    composite_crop_back,
    align_crop_112,
    box_iou,
    compute_wilson_ci,
)
from benchmarks.evaluate_face_models import ModelSuite, cosine_similarity
from app.protection.protection_engine import ProtectionEngine

# Setup paths
REPO_ROOT = Path(__file__).resolve().parent.parent
CALIB_PATH = REPO_ROOT / "benchmarks" / "results" / "sweep_evidence" / "calibrated_thresholds_sweep.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results" / "landmark_drift_pilot"
IMAGE_STORE_DIR = Path(r"D:\JavaProjects\Varman-research\pilot_landmark_2026-10-15")

LANDMARK_NAMES = [
    "Right Eye",
    "Left Eye",
    "Nose Tip",
    "Right Mouth Corner",
    "Left Mouth Corner",
]

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LandmarkDriftPilot")


def decompose_similarity_transform(M: np.ndarray) -> Dict[str, float]:
    """
    Decomposes a 2x3 partial affine similarity transform matrix:
      M = [[a,  b, tx],
           [-b, a, ty]]
    into scale s, rotation angle theta (degrees), and translation (tx, ty).
    """
    a = float(M[0, 0])
    b = float(M[0, 1])
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    scale = float(np.sqrt(a**2 + b**2))
    theta_deg = float(np.degrees(np.arctan2(b, a)))
    return {
        "scale": scale,
        "theta_deg": theta_deg,
        "tx": tx,
        "ty": ty,
        "a": a,
        "b": b,
    }


def compute_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 2026) -> List[float]:
    """Computes a two-sided bootstrap confidence interval for the mean of diffs."""
    if len(diffs) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boot_means = []
    n = len(diffs)
    for _ in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return [round(low, 4), round(high, 4)]


def check_stop_conditions(latencies: List[float], peak_vram_gb: float) -> None:
    """Verifies all pre-declared safety stop conditions."""
    free_c_gb = shutil.disk_usage("C:\\").free / (1024**3)
    if free_c_gb < 10.0:
        raise RuntimeError(f"STOP CONDITION TRIGGERED: C: drive free space fell below 10.0 GB ({free_c_gb:.2f} GB [Measured]). Aborting.")
    
    if peak_vram_gb > 3.8:
        raise RuntimeError(f"STOP CONDITION TRIGGERED: Peak VRAM exceeded safety limit of 3.8 GB ({peak_vram_gb:.3f} GB [Measured]). Aborting.")
    
    if len(latencies) >= 5:
        recent_latency = float(np.mean(latencies[-5:]))
        if recent_latency > 6.0:
            raise RuntimeError(f"STOP CONDITION TRIGGERED: Sustained per-image latency exceeded 6.0s over 5 pairs ({recent_latency:.2f}s [Measured]). Aborting.")


def run_pilot(target_pairs: int = 30) -> None:
    print("=" * 90, flush=True)
    print("VARMAN CONTROLLED PILOT: YUNET LANDMARK DISPLACEMENT MEASUREMENT", flush=True)
    print(f"Target: {target_pairs} LFW Genuine Pairs | Settings: Row 3 (Paste-Aware + Jitter)", flush=True)
    print("=" * 90, flush=True)

    # Initial safety check
    free_c = shutil.disk_usage("C:\\").free / (1024**3)
    free_d = shutil.disk_usage("D:\\").free / (1024**3)
    print(f"[Preflight] Initial Storage: C: {free_c:.2f} GB free | D: {free_d:.2f} GB free", flush=True)
    if free_c < 10.0:
        raise RuntimeError(f"Initial C: free space is below 10 GB ({free_c:.2f} GB). Aborting.")

    # Create directories
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Directories] Image Store: {IMAGE_STORE_DIR}", flush=True)
    print(f"[Directories] Results: {RESULTS_DIR}", flush=True)

    # Load thresholds
    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"Calibrated thresholds not found at {CALIB_PATH}")
    with open(CALIB_PATH, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    thresh_1pct = calib_data["thresholds_1pct"]
    thresh_01pct = calib_data["thresholds_01pct"]

    models = ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Hardware] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # Initialize engines
    engine = ProtectionEngine(model_dir=str(REPO_ROOT / "models"))
    model_suite = ModelSuite(models_dir=REPO_ROOT / "models", requested_device="cuda")
    pgd_engine = PasteAwarePGDEngine(device=device)

    # Load LFW pairs (same loading sequence as Row 3)
    print(f"\n[LFW] Loading genuine pairs (target: {target_pairs})...", flush=True)
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
        if len(lfw_pairs) >= target_pairs:
            break

    print(f"[LFW] Loaded {len(lfw_pairs)} genuine pairs successfully.", flush=True)

    # Per-landmark displacement records for CSV export
    csv_rows: List[Dict[str, Any]] = []

    # Per-pair detailed results for final JSON
    pair_results: List[Dict[str, Any]] = []

    latencies: List[float] = []

    # Deterministic sequences per pair
    max_steps = 25
    q_sequences = []
    for p_i in range(len(lfw_pairs)):
        rng = np.random.default_rng(2026 + p_i)
        q_sequences.append([int(rng.integers(50, 96)) for _ in range(max_steps)])

    print("\n" + "=" * 90, flush=True)
    print("STARTING PER-PAIR MEASUREMENTS (Row 3 Settings: eps=8/255, 25 steps, w_fn=1.0, jitter=True)", flush=True)
    print("=" * 90, flush=True)

    for p_i, (b1, b2, f1, f2, clean_c1, clean_c2) in enumerate(lfw_pairs):
        # Set seeds per pair (RNG seed = 2026 + p_i)
        torch.manual_seed(2026 + p_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2026 + p_i)

        h, w = b1.shape[:2]
        orig_landmarks = np.array(f1["landmarks"][:5], dtype=np.float32)
        target_box = f1["box"]
        target_box_area = float(target_box[2] * target_box[3])

        # Clean similarity transform matrix M and decomposition
        M_clean, _ = cv2.estimateAffinePartial2D(orig_landmarks, ARCFACE_CANONICAL_112)
        clean_decomp = decompose_similarity_transform(M_clean)
        scale_clean = clean_decomp["scale"]

        # ----------------------------------------------------------------------
        # 1. CLEAN CONTROL: Save Clean PNG and Clean JPEG Q50
        # ----------------------------------------------------------------------
        p_clean_png = IMAGE_STORE_DIR / f"clean_{p_i:02d}.png"
        p_clean_q50 = IMAGE_STORE_DIR / f"clean_q50_{p_i:02d}.jpg"

        cv2.imwrite(str(p_clean_png), b1, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cv2.imwrite(str(p_clean_q50), b1, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

        reloaded_clean_q50 = cv2.imread(str(p_clean_q50))

        # ----------------------------------------------------------------------
        # 2. RUN ROW 3 PGD OPTIMIZATION (Protecting Image 1: b1)
        # ----------------------------------------------------------------------
        t0 = time.perf_counter()
        prot_crop_bgr, _, _ = pgd_engine.protect_paste_aware(
            img_bgr=b1,
            landmarks=f1["landmarks"],
            crop_112_bgr=clean_c1,
            M=M_clean,
            steps=25,
            w_fn=1.00,
            eps=8.0 / 255.0,
            mu=0.9,
            q_sequence=q_sequences[p_i],
            jitter_landmarks=True,
            rng_seed=2026 + p_i,
        )
        lat_sec = time.perf_counter() - t0
        latencies.append(lat_sec)

        # Composite protected crop back into full image via OpenCV composite path
        prot_b1 = composite_crop_back(b1, prot_crop_bgr, M_clean)

        # ----------------------------------------------------------------------
        # 3. SAVE PROTECTED COMPOSITES: PNG and JPEG Q50
        # ----------------------------------------------------------------------
        p_prot_png = IMAGE_STORE_DIR / f"prot_{p_i:02d}.png"
        p_prot_q50 = IMAGE_STORE_DIR / f"prot_q50_{p_i:02d}.jpg"

        cv2.imwrite(str(p_prot_png), prot_b1, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        cv2.imwrite(str(p_prot_q50), prot_b1, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

        reloaded_prot_png = cv2.imread(str(p_prot_png))
        reloaded_prot_q50 = cv2.imread(str(p_prot_q50))

        # ----------------------------------------------------------------------
        # 4. YUNET DETECTIONS & MATCHING ON ALL 4 IMAGES
        # ----------------------------------------------------------------------
        images_to_evaluate = {
            "clean_control": b1,
            "clean_q50": reloaded_clean_q50,
            "prot_png": reloaded_prot_png,
            "prot_q50": reloaded_prot_q50,
        }

        detections: Dict[str, Dict[str, Any]] = {}
        for cond_name, img_eval in images_to_evaluate.items():
            raw_dets = engine.detect_faces(img_eval)
            det_count = len(raw_dets) if raw_dets is not None else 0

            if det_count == 0:
                detections[cond_name] = {
                    "detected": False,
                    "det_count": 0,
                    "matched_face": None,
                    "match_rule": "none (0 detections)",
                    "match_iou": 0.0,
                    "box": None,
                    "landmarks": None,
                    "M": None,
                    "decomp": None,
                }
            else:
                # Match to original clean face box via maximum IoU rule
                ious = [box_iou(target_box, f["box"]) for f in raw_dets]
                best_idx = int(np.argmax(ious))
                best_face = raw_dets[best_idx]
                best_iou = float(ious[best_idx])
                rule_desc = f"max_iou (IoU={best_iou:.3f}, rank {best_idx+1}/{det_count})"

                det_lm = np.array(best_face["landmarks"][:5], dtype=np.float32)
                M_det, _ = cv2.estimateAffinePartial2D(det_lm, ARCFACE_CANONICAL_112)
                decomp_det = decompose_similarity_transform(M_det) if M_det is not None else None

                detections[cond_name] = {
                    "detected": True,
                    "det_count": det_count,
                    "matched_face": best_face,
                    "match_rule": rule_desc,
                    "match_iou": best_iou,
                    "box": best_face["box"],
                    "landmarks": det_lm,
                    "M": M_det,
                    "decomp": decomp_det,
                }

        # ----------------------------------------------------------------------
        # 5. DISPLACEMENT CALCULATIONS (Clean vs Clean-Q50, Prot-PNG, Prot-Q50)
        # ----------------------------------------------------------------------
        comp_records: Dict[str, Any] = {}

        for cond_name in ["clean_q50", "prot_png", "prot_q50"]:
            det_info = detections[cond_name]
            if not det_info["detected"] or det_info["landmarks"] is None:
                comp_records[cond_name] = {
                    "detected": False,
                    "failure": "detection_failed",
                    "landmarks": [],
                    "mean_full_euc": None,
                    "mean_crop_euc": None,
                }
                continue

            det_lm = det_info["landmarks"]
            decomp_det = det_info["decomp"]
            det_box = det_info["box"]
            det_box_area = float(det_box[2] * det_box[3])
            box_area_ratio_diff = abs(det_box_area - target_box_area) / target_box_area

            # Transform parameters comparison
            delta_scale = decomp_det["scale"] - clean_decomp["scale"]
            scale_change_pct = abs(delta_scale) / clean_decomp["scale"] * 100.0
            delta_theta_deg = decomp_det["theta_deg"] - clean_decomp["theta_deg"]
            delta_tx = decomp_det["tx"] - clean_decomp["tx"]
            delta_ty = decomp_det["ty"] - clean_decomp["ty"]

            lm_eucs_full: List[float] = []
            lm_eucs_crop: List[float] = []
            lm_records: List[Dict[str, Any]] = []

            for k in range(5):
                # Full image space
                x_clean, y_clean = float(orig_landmarks[k, 0]), float(orig_landmarks[k, 1])
                x_det, y_det = float(det_lm[k, 0]), float(det_lm[k, 1])
                dx_full = x_det - x_clean
                dy_full = y_det - y_clean
                euc_full = float(np.sqrt(dx_full**2 + dy_full**2))
                lm_eucs_full.append(euc_full)

                # Aligned 112x112 crop space: apply clean similarity transform M_clean
                a_cl, b_cl = clean_decomp["a"], clean_decomp["b"]
                dx_crop = a_cl * dx_full + b_cl * dy_full
                dy_crop = -b_cl * dx_full + a_cl * dy_full
                euc_crop = float(np.sqrt(dx_crop**2 + dy_crop**2))
                lm_eucs_crop.append(euc_crop)

                lm_rec = {
                    "landmark_index": k,
                    "landmark_name": LANDMARK_NAMES[k],
                    "clean_x": x_clean,
                    "clean_y": y_clean,
                    "det_x": x_det,
                    "det_y": y_det,
                    "dx_full": round(dx_full, 4),
                    "dy_full": round(dy_full, 4),
                    "euc_full": round(euc_full, 4),
                    "dx_crop": round(dx_crop, 4),
                    "dy_crop": round(dy_crop, 4),
                    "euc_crop": round(euc_crop, 4),
                }
                lm_records.append(lm_rec)

                # Append to flat CSV rows
                csv_rows.append({
                    "pair_index": p_i,
                    "condition": cond_name,
                    "landmark_index": k,
                    "landmark_name": LANDMARK_NAMES[k],
                    "clean_x": round(x_clean, 2),
                    "clean_y": round(y_clean, 2),
                    "detected_x": round(x_det, 2),
                    "detected_y": round(y_det, 2),
                    "dx_full": round(dx_full, 4),
                    "dy_full": round(dy_full, 4),
                    "dist_full": round(euc_full, 4),
                    "dx_crop": round(dx_crop, 4),
                    "dy_crop": round(dy_crop, 4),
                    "dist_crop": round(euc_crop, 4),
                    "clean_scale": round(clean_decomp["scale"], 4),
                    "detected_scale": round(decomp_det["scale"], 4),
                    "scale_change_pct": round(scale_change_pct, 4),
                    "clean_rot_deg": round(clean_decomp["theta_deg"], 4),
                    "detected_rot_deg": round(decomp_det["theta_deg"], 4),
                    "rot_change_deg": round(delta_theta_deg, 4),
                    "clean_tx": round(clean_decomp["tx"], 2),
                    "detected_tx": round(decomp_det["tx"], 2),
                    "clean_ty": round(clean_decomp["ty"], 2),
                    "detected_ty": round(decomp_det["ty"], 2),
                    "det_count": det_info["det_count"],
                    "match_iou": round(det_info["match_iou"], 4),
                    "orig_box_area": round(target_box_area, 1),
                    "det_box_area": round(det_box_area, 1),
                    "box_area_change_pct": round(box_area_ratio_diff * 100.0, 2),
                })

            comp_records[cond_name] = {
                "detected": True,
                "det_count": det_info["det_count"],
                "match_iou": round(det_info["match_iou"], 4),
                "box_area_change_pct": round(box_area_ratio_diff * 100.0, 2),
                "box_change_gt_10pct": bool(box_area_ratio_diff > 0.10),
                "scale_clean": round(clean_decomp["scale"], 4),
                "scale_det": round(decomp_det["scale"], 4),
                "scale_change_pct": round(scale_change_pct, 4),
                "scale_change_gt_3pct": bool(scale_change_pct > 3.0),
                "rot_change_deg": round(delta_theta_deg, 4),
                "tx_change": round(delta_tx, 4),
                "ty_change": round(delta_ty, 4),
                "mean_full_euc": round(float(np.mean(lm_eucs_full)), 4),
                "mean_crop_euc": round(float(np.mean(lm_eucs_crop)), 4),
                "landmarks": lm_records,
            }

        # ----------------------------------------------------------------------
        # 6. FEATURE EXTRACTION & THRESHOLD EVALUATION (Against Clean Reference b2)
        # ----------------------------------------------------------------------
        clean_e1, _ = model_suite.extract_model_features(b1, f1)
        e2, _ = model_suite.extract_model_features(b2, f2)
        clean_pair_sims = {m: float(cosine_similarity(clean_e1[m], e2[m])) for m in models}

        model_eval: Dict[str, Dict[str, Any]] = {}

        for cond_name, img_eval in images_to_evaluate.items():
            det_info = detections[cond_name]
            if not det_info["detected"] or det_info["matched_face"] is None:
                model_eval[cond_name] = {m: {"sim": 0.0, "flip_1pct": False, "flip_01pct": False, "valid_1pct": False, "valid_01pct": False} for m in models}
                continue

            e_cand, _ = model_suite.extract_model_features(img_eval, det_info["matched_face"])
            m_res: Dict[str, Any] = {}
            for m in models:
                sim = float(cosine_similarity(e_cand[m], e2[m]))
                sim_clean_pair = clean_pair_sims[m]

                valid_1 = bool(sim_clean_pair >= thresh_1pct[m])
                valid_01 = bool(sim_clean_pair >= thresh_01pct[m])

                flip_1 = bool(valid_1 and (sim < thresh_1pct[m]))
                flip_01 = bool(valid_01 and (sim < thresh_01pct[m]))

                m_res[m] = {
                    "sim": round(sim, 4),
                    "clean_sim": round(sim_clean_pair, 4),
                    "valid_1pct": valid_1,
                    "valid_01pct": valid_01,
                    "flip_1pct": flip_1,
                    "flip_01pct": flip_01,
                }
            model_eval[cond_name] = m_res

        # Record pair summary
        pair_data = {
            "pair_index": p_i,
            "latency_sec": round(lat_sec, 2),
            "target_box": target_box,
            "clean_decomp": {
                "scale": round(clean_decomp["scale"], 4),
                "theta_deg": round(clean_decomp["theta_deg"], 4),
                "tx": round(clean_decomp["tx"], 2),
                "ty": round(clean_decomp["ty"], 2),
            },
            "comparisons": comp_records,
            "model_eval": model_eval,
        }
        pair_results.append(pair_data)

        # Check safety stop conditions
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        check_stop_conditions(latencies, peak_vram)

        # Log progress per pair
        clean_q50_drift = comp_records["clean_q50"]["mean_crop_euc"] if comp_records["clean_q50"]["detected"] else 0.0
        prot_q50_drift = comp_records["prot_q50"]["mean_crop_euc"] if comp_records["prot_q50"]["detected"] else 0.0
        match_iou = comp_records["prot_q50"].get("match_iou", 0.0)
        curr_c_free = shutil.disk_usage("C:\\").free / (1024**3)

        print(f"[Pair {p_i+1:02d}/30] Latency: {lat_sec:4.2f}s | CleanQ50: {clean_q50_drift:4.2f}px | ProtQ50: {prot_q50_drift:4.2f}px (crop) | IoU: {match_iou:.3f} | C: {curr_c_free:.1f}GB", flush=True)

        # Save partial results every 5 pairs
        if (p_i + 1) % 5 == 0 or (p_i + 1) == len(lfw_pairs):
            partial_path = RESULTS_DIR / "landmark_drift_pilot_partial.json"
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump({
                    "progress": f"{p_i+1}/{len(lfw_pairs)}",
                    "latencies_mean": round(float(np.mean(latencies)), 2),
                    "pair_results": pair_results,
                }, f, indent=2)
            print(f"  [Checkpoint] Saved partial results to {partial_path.name}", flush=True)

    # --------------------------------------------------------------------------
    # 7. AGGREGATE SUMMARY & STATISTICAL ANALYSIS
    # --------------------------------------------------------------------------
    print("\n" + "=" * 90, flush=True)
    print("COMPUTING SUMMARY STATISTICS & STATISTICAL INFERENCE", flush=True)
    print("=" * 90, flush=True)

    summary_displacements: Dict[str, Any] = {}

    for cond_name in ["clean_q50", "prot_png", "prot_q50"]:
        full_dists = [row["dist_full"] for row in csv_rows if row["condition"] == cond_name]
        crop_dists = [row["dist_crop"] for row in csv_rows if row["condition"] == cond_name]
        dx_crops = [row["dx_crop"] for row in csv_rows if row["condition"] == cond_name]
        dy_crops = [row["dy_crop"] for row in csv_rows if row["condition"] == cond_name]

        # Per-landmark crop distances
        per_lm_stats = {}
        for k in range(5):
            k_dists = [row["dist_crop"] for row in csv_rows if row["condition"] == cond_name and row["landmark_index"] == k]
            per_lm_stats[LANDMARK_NAMES[k]] = {
                "mean": round(float(np.mean(k_dists)), 4),
                "median": round(float(np.median(k_dists)), 4),
                "p90": round(float(np.percentile(k_dists, 90)), 4),
                "p95": round(float(np.percentile(k_dists, 95)), 4),
                "max": round(float(np.max(k_dists)), 4),
            }

        # Covariance matrix of dx, dy in crop space
        if len(dx_crops) > 1:
            cov_matrix = np.cov(dx_crops, dy_crops)
            cov_formatted = [
                [round(float(cov_matrix[0, 0]), 4), round(float(cov_matrix[0, 1]), 4)],
                [round(float(cov_matrix[1, 0]), 4), round(float(cov_matrix[1, 1]), 4)],
            ]
        else:
            cov_formatted = [[0.0, 0.0], [0.0, 0.0]]

        # Failure rates across pairs
        cond_pairs = [p["comparisons"][cond_name] for p in pair_results]
        no_dets = sum(1 for cp in cond_pairs if not cp["detected"])
        mult_dets = sum(1 for cp in cond_pairs if cp.get("det_count", 0) > 1)
        box_gt_10 = sum(1 for cp in cond_pairs if cp.get("box_change_gt_10pct", False))
        scale_gt_3 = sum(1 for cp in cond_pairs if cp.get("scale_change_gt_3pct", False))

        summary_displacements[cond_name] = {
            "pooled_full_px": {
                "mean": round(float(np.mean(full_dists)), 4),
                "median": round(float(np.median(full_dists)), 4),
                "p90": round(float(np.percentile(full_dists, 90)), 4),
                "p95": round(float(np.percentile(full_dists, 95)), 4),
                "max": round(float(np.max(full_dists)), 4),
            },
            "pooled_crop_px": {
                "mean": round(float(np.mean(crop_dists)), 4),
                "median": round(float(np.median(crop_dists)), 4),
                "p90": round(float(np.percentile(crop_dists, 90)), 4),
                "p95": round(float(np.percentile(crop_dists, 95)), 4),
                "max": round(float(np.max(crop_dists)), 4),
            },
            "per_landmark_crop_px": per_lm_stats,
            "frac_instances_gt_1px_crop": round(float(np.mean([d > 1.0 for d in crop_dists])) * 100.0, 2),
            "frac_instances_gt_2px_crop": round(float(np.mean([d > 2.0 for d in crop_dists])) * 100.0, 2),
            "frac_pairs_scale_gt_3pct": round(float(scale_gt_3) / float(len(cond_pairs)) * 100.0, 2),
            "covariance_matrix_dx_dy_crop": cov_formatted,
            "failures": {
                "no_detections": no_dets,
                "multiple_detections": mult_dets,
                "box_change_gt_10pct": box_gt_10,
                "scale_change_gt_3pct": scale_gt_3,
            },
        }

    # --------------------------------------------------------------------------
    # 8. PAIRED COMPARISON: Protected Q50 vs Clean Q50 Control
    # --------------------------------------------------------------------------
    paired_diffs_crop = []
    paired_diffs_full = []
    for p in pair_results:
        c_q50 = p["comparisons"]["clean_q50"]
        p_q50 = p["comparisons"]["prot_q50"]
        if c_q50["detected"] and p_q50["detected"]:
            d_crop = p_q50["mean_crop_euc"] - c_q50["mean_crop_euc"]
            d_full = p_q50["mean_full_euc"] - c_q50["mean_full_euc"]
            paired_diffs_crop.append(d_crop)
            paired_diffs_full.append(d_full)

    diffs_crop_arr = np.array(paired_diffs_crop, dtype=np.float32)
    diffs_full_arr = np.array(paired_diffs_full, dtype=np.float32)

    ci_paired_crop = compute_bootstrap_ci(diffs_crop_arr, n_boot=10000, alpha=0.05, seed=2026)
    ci_paired_full = compute_bootstrap_ci(diffs_full_arr, n_boot=10000, alpha=0.05, seed=2026)

    mean_paired_crop = float(np.mean(diffs_crop_arr)) if len(diffs_crop_arr) > 0 else 0.0
    mean_paired_full = float(np.mean(diffs_full_arr)) if len(diffs_full_arr) > 0 else 0.0

    # Determine Pre-registered Reading
    # - If protected-Q50 displacement is not larger than clean-Q50 control (paired CI includes zero): detector noise.
    # - If protected displacement is larger than control but mostly within 2 px: current jitter range plausibly adequate.
    # - If protected displacement is larger than control and a meaningful fraction exceeds 2 px or 3% scale: jitter may be too narrow.
    prot_q50_crop_summary = summary_displacements["prot_q50"]
    frac_gt_2px = prot_q50_crop_summary["frac_instances_gt_2px_crop"]
    frac_scale_gt_3 = prot_q50_crop_summary["frac_pairs_scale_gt_3pct"]

    ci_excludes_zero = bool(ci_paired_crop[0] > 0.0 or ci_paired_crop[1] < 0.0)

    if not ci_excludes_zero:
        prereg_outcome = "Reading 1: Protected-Q50 displacement is not statistically distinguishable from the clean-Q50 control (bootstrap 95% CI includes zero). Drift is mostly detector noise/compression quantization, not protection-induced [Hypothesis]."
    else:
        if frac_gt_2px <= 15.0 and frac_scale_gt_3 <= 15.0:
            prereg_outcome = f"Reading 2: Protected displacement is statistically larger than clean control (CI excludes zero: {ci_paired_crop}), but landmark displacements are mostly within 2 px aligned-crop space ({frac_gt_2px}% > 2px, {frac_scale_gt_3}% scale > 3%). The current Row 3 jitter range (+/-2px, +/-3%) is plausibly adequate [Measured]."
        else:
            prereg_outcome = f"Reading 3: Protected displacement is statistically larger than clean control (CI excludes zero: {ci_paired_crop}) and a meaningful fraction exceeds 2 px aligned-crop space ({frac_gt_2px}% > 2px, {frac_scale_gt_3}% scale > 3%). The current jitter range (+/-2px, +/-3%) may be too narrow, justifying a future wider-jitter experiment [Hypothesis]."

    paired_analysis = {
        "n_valid_pairs": len(diffs_crop_arr),
        "mean_prot_q50_crop_px": summary_displacements["prot_q50"]["pooled_crop_px"]["mean"],
        "mean_clean_q50_crop_px": summary_displacements["clean_q50"]["pooled_crop_px"]["mean"],
        "mean_paired_diff_crop_px": round(mean_paired_crop, 4),
        "bootstrap_95_ci_diff_crop_px": ci_paired_crop,
        "mean_prot_q50_full_px": summary_displacements["prot_q50"]["pooled_full_px"]["mean"],
        "mean_clean_q50_full_px": summary_displacements["clean_q50"]["pooled_full_px"]["mean"],
        "mean_paired_diff_full_px": round(mean_paired_full, 4),
        "bootstrap_95_ci_diff_full_px": ci_paired_full,
        "preregistered_reading": prereg_outcome,
    }

    # --------------------------------------------------------------------------
    # 9. MODEL FLIPS SUMMARY
    # --------------------------------------------------------------------------
    model_flips_summary: Dict[str, Any] = {}
    for cond_name in ["clean_control", "clean_q50", "prot_png", "prot_q50"]:
        cond_m_data: Dict[str, Any] = {}
        for m in models:
            valid_1_count = 0
            flips_1_count = 0
            valid_01_count = 0
            flips_01_count = 0
            sims = []

            for p in pair_results:
                m_info = p["model_eval"][cond_name][m]
                sims.append(m_info["sim"])
                if m_info["valid_1pct"]:
                    valid_1_count += 1
                    if m_info["flip_1pct"]:
                        flips_1_count += 1
                if m_info["valid_01pct"]:
                    valid_01_count += 1
                    if m_info["flip_01pct"]:
                        flips_01_count += 1

            rate_1 = (float(flips_1_count) / float(valid_1_count) * 100.0) if valid_1_count > 0 else 0.0
            rate_01 = (float(flips_01_count) / float(valid_01_count) * 100.0) if valid_01_count > 0 else 0.0

            cond_m_data[m] = {
                "mean_sim": round(float(np.mean(sims)), 4),
                "flips_1pct": flips_1_count,
                "valid_1pct_n": valid_1_count,
                "flip_rate_1pct": round(rate_1, 2),
                "wilson_ci_1pct": compute_wilson_ci(flips_1_count, valid_1_count),
                "flips_01pct": flips_01_count,
                "valid_01pct_n": valid_01_count,
                "flip_rate_01pct": round(rate_01, 2),
                "wilson_ci_01pct": compute_wilson_ci(flips_01_count, valid_01_count),
            }
        model_flips_summary[cond_name] = cond_m_data

    # Final payload
    final_payload = {
        "benchmark_metadata": {
            "name": "Varman YuNet Landmark Displacement Controlled Pilot",
            "date": "2026-10-15",
            "branch": "adaptive-strength",
            "head_commit": "7268342290485666d5e68e826ad84f15a83e7268",
            "pairs_evaluated": len(lfw_pairs),
            "configuration": "Row 3 (Paste-Aware PGD + Online Landmark Jitter)",
            "epsilon": "8/255 (0.03137)",
            "steps": 25,
            "loss_weights": {"ArcFace": 1.0, "FaceNet": 1.0},
            "thresholds_1pct": thresh_1pct,
            "thresholds_01pct": thresh_01pct,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "peak_vram_gb": round(float(torch.cuda.max_memory_allocated() / (1024**3)), 3) if torch.cuda.is_available() else 0.0,
            "mean_latency_sec": round(float(np.mean(latencies)), 2),
            "image_store_dir": str(IMAGE_STORE_DIR),
        },
        "summary_displacements": summary_displacements,
        "paired_displacement_comparison": paired_analysis,
        "model_flips_summary": model_flips_summary,
        "pair_results": pair_results,
    }

    # --------------------------------------------------------------------------
    # 10. WRITE RESULT FILES (ONLY TO benchmarks/results/landmark_drift_pilot/)
    # --------------------------------------------------------------------------
    final_json_path = RESULTS_DIR / "landmark_drift_pilot_final.json"
    with open(final_json_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
    print(f"\n[Saved] Final JSON: {final_json_path}", flush=True)

    csv_path = RESULTS_DIR / "landmark_displacements.csv"
    if csv_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[Saved] CSV Displacements: {csv_path} ({len(csv_rows)} rows)", flush=True)

    flips_json_path = RESULTS_DIR / "per_pair_flips.json"
    per_pair_flips_payload = [
        {
            "pair_index": p["pair_index"],
            "model_eval": p["model_eval"],
        }
        for p in pair_results
    ]
    with open(flips_json_path, "w", encoding="utf-8") as f:
        json.dump(per_pair_flips_payload, f, indent=2)
    print(f"[Saved] Per-pair Flips: {flips_json_path}", flush=True)

    # Write REPORT.md
    report_md_path = RESULTS_DIR / "REPORT.md"
    write_report_markdown(final_payload, report_md_path)
    print(f"[Saved] REPORT.md: {report_md_path}", flush=True)

    # Final check: Verify NO image files inside RESULTS_DIR
    img_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    repo_leak_images = [f for f in RESULTS_DIR.rglob("*") if f.suffix.lower() in img_exts]
    if repo_leak_images:
        raise RuntimeError(f"LEAK ERROR: Found image files in repo results dir: {[f.name for f in repo_leak_images]}")
    print(f"[Verification PASS] Zero image files inside repository directory {RESULTS_DIR}", flush=True)

    print("\n" + "=" * 90, flush=True)
    print("PILOT MEASUREMENT COMPLETE SUCCESSFULLY", flush=True)
    print("=" * 90, flush=True)


def write_report_markdown(payload: Dict[str, Any], path: Path) -> None:
    meta = payload["benchmark_metadata"]
    displacements = payload["summary_displacements"]
    paired = payload["paired_displacement_comparison"]
    flips = payload["model_flips_summary"]

    lines = []
    lines.append("# Varman: YuNet Landmark Displacement Controlled Pilot Report (30 LFW Pairs)")
    lines.append("")
    lines.append("**Measurement-Only Engineering Benchmark**")
    lines.append("")
    lines.append(f"- **Date:** {meta['date']}")
    lines.append(f"- **Target Pairs:** {meta['pairs_evaluated']} LFW Genuine Pairs (indices 0 to 29) [Measured]")
    lines.append(f"- **Configuration:** {meta['configuration']}")
    lines.append(f"- **Epsilon:** {meta['epsilon']} (luminance-only)")
    lines.append(f"- **Hardware:** {meta['device_name']} | Peak VRAM: {meta['peak_vram_gb']} GB [Measured] | Avg Latency: {meta['mean_latency_sec']}s/img [Measured]")
    lines.append(f"- **Protected Image Location:** `{meta['image_store_dir']}` (Outside repo, not committed)")
    lines.append("")
    lines.append("> [!NOTE]")
    lines.append("> **Seeding Notice:** `torch.manual_seed(2026 + pair_index)` and `torch.cuda.manual_seed_all(2026 + pair_index)` were invoked at the start of each pair along with deterministic NumPy EOT quality and landmark jitter RNGs. This ensures input-level determinism, though atomic operations in PyTorch CUDA backward kernels do not guarantee bit-exact floating-point results across differing GPU architectures.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary & Pre-Registered Decision")
    lines.append("")
    lines.append(f"**Pre-Registered Reading Verdict:**")
    lines.append(f"> {paired['preregistered_reading']}")
    lines.append("")
    lines.append("| Metric | Clean JPEG Q50 (Control) | Protected Lossless PNG | Protected JPEG Q50 | Paired Difference (Prot Q50 - Clean Q50) | Bootstrap 95% CI |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
    lines.append(f"| **Mean Euclidean Drift (Crop 112x112 px)** | {displacements['clean_q50']['pooled_crop_px']['mean']} px [Measured] | {displacements['prot_png']['pooled_crop_px']['mean']} px [Measured] | {displacements['prot_q50']['pooled_crop_px']['mean']} px [Measured] | **{paired['mean_paired_diff_crop_px']:+.4f} px** [Measured] | **[{paired['bootstrap_95_ci_diff_crop_px'][0]}, {paired['bootstrap_95_ci_diff_crop_px'][1]}]** [Recomputed from artifacts] |")
    lines.append(f"| **Median Euclidean Drift (Crop px)** | {displacements['clean_q50']['pooled_crop_px']['median']} px [Measured] | {displacements['prot_png']['pooled_crop_px']['median']} px [Measured] | {displacements['prot_q50']['pooled_crop_px']['median']} px [Measured] | — | — |")
    lines.append(f"| **P90 Euclidean Drift (Crop px)** | {displacements['clean_q50']['pooled_crop_px']['p90']} px [Measured] | {displacements['prot_png']['pooled_crop_px']['p90']} px [Measured] | {displacements['prot_q50']['pooled_crop_px']['p90']} px [Measured] | — | — |")
    lines.append(f"| **P95 Euclidean Drift (Crop px)** | {displacements['clean_q50']['pooled_crop_px']['p95']} px [Measured] | {displacements['prot_png']['pooled_crop_px']['p95']} px [Measured] | {displacements['prot_q50']['pooled_crop_px']['p95']} px [Measured] | — | — |")
    lines.append(f"| **Max Euclidean Drift (Crop px)** | {displacements['clean_q50']['pooled_crop_px']['max']} px [Measured] | {displacements['prot_png']['pooled_crop_px']['max']} px [Measured] | {displacements['prot_q50']['pooled_crop_px']['max']} px [Measured] | — | — |")
    lines.append(f"| **Mean Euclidean Drift (Full Image px)** | {displacements['clean_q50']['pooled_full_px']['mean']} px [Measured] | {displacements['prot_png']['pooled_full_px']['mean']} px [Measured] | {displacements['prot_q50']['pooled_full_px']['mean']} px [Measured] | **{paired['mean_paired_diff_full_px']:+.4f} px** [Measured] | **[{paired['bootstrap_95_ci_diff_full_px'][0]}, {paired['bootstrap_95_ci_diff_full_px'][1]}]** [Recomputed from artifacts] |")
    lines.append(f"| **Instances with Drift > 1 px (Crop)** | {displacements['clean_q50']['frac_instances_gt_1px_crop']}% [Measured] | {displacements['prot_png']['frac_instances_gt_1px_crop']}% [Measured] | {displacements['prot_q50']['frac_instances_gt_1px_crop']}% [Measured] | — | — |")
    lines.append(f"| **Instances with Drift > 2 px (Crop)** | {displacements['clean_q50']['frac_instances_gt_2px_crop']}% [Measured] | {displacements['prot_png']['frac_instances_gt_2px_crop']}% [Measured] | {displacements['prot_q50']['frac_instances_gt_2px_crop']}% [Measured] | — | — |")
    lines.append(f"| **Pairs with Scale Shift > 3%** | {displacements['clean_q50']['frac_pairs_scale_gt_3pct']}% [Measured] | {displacements['prot_png']['frac_pairs_scale_gt_3pct']}% [Measured] | {displacements['prot_q50']['frac_pairs_scale_gt_3pct']}% [Measured] | — | — |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Per-Landmark Crop Displacement Breakdown (Aligned 112x112 px)")
    lines.append("")
    lines.append("| Landmark | Clean Q50 Mean [Med, P95] | Prot PNG Mean [Med, P95] | Prot Q50 Mean [Med, P95] |")
    lines.append("|:---|:---:|:---:|:---:|")
    for lm in LANDMARK_NAMES:
        c_lm = displacements["clean_q50"]["per_landmark_crop_px"][lm]
        p_png = displacements["prot_png"]["per_landmark_crop_px"][lm]
        p_q50 = displacements["prot_q50"]["per_landmark_crop_px"][lm]
        lines.append(f"| **{lm}** | {c_lm['mean']} px [{c_lm['median']}, {c_lm['p95']}] [Measured] | {p_png['mean']} px [{p_png['median']}, {p_png['p95']}] [Measured] | {p_q50['mean']} px [{p_q50['median']}, {p_q50['p95']}] [Measured] |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Displacement Covariance & Detection Stability")
    lines.append("")
    lines.append("### Displacement Covariance Matrix (Aligned Crop Space)")
    lines.append("- **Protected JPEG Q50 $\\boldsymbol{\\Sigma}_{\\Delta x, \\Delta y}$:**")
    cov_00 = displacements['prot_q50']['covariance_matrix_dx_dy_crop'][0][0]
    cov_01 = displacements['prot_q50']['covariance_matrix_dx_dy_crop'][0][1]
    cov_10 = displacements['prot_q50']['covariance_matrix_dx_dy_crop'][1][0]
    cov_11 = displacements['prot_q50']['covariance_matrix_dx_dy_crop'][1][1]
    lines.append("  ```")
    lines.append(f"  [[{cov_00:.4f}, {cov_01:.4f}],")
    lines.append(f"   [{cov_10:.4f}, {cov_11:.4f}]]")
    lines.append("  ```")
    lines.append("")
    lines.append("### Detection Failures & Anomalies Across 30 Pairs")
    lines.append("| Condition | No Detection Failures | Multiple Face Detections | Bounding Box Shift > 10% | Alignment Scale Change > 3% |")
    lines.append("|:---|:---:|:---:|:---:|:---:|")
    for cond_name in ["clean_q50", "prot_png", "prot_q50"]:
        f_info = displacements[cond_name]["failures"]
        lines.append(f"| **{cond_name}** | {f_info['no_detections']} / 30 [Measured] | {f_info['multiple_detections']} / 30 [Measured] | {f_info['box_change_gt_10pct']} / 30 [Measured] | {f_info['scale_change_gt_3pct']} / 30 [Measured] |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Verification Flip Rates on n=30 Pilot (1% and 0.1% FAR)")
    lines.append("")
    lines.append("> [!WARNING]")
    lines.append("> **Statistical Caveat:** Flip rates below reflect the $N=30$ pilot subset only and must NOT be directly compared to the $N=150$ benchmark figure (36.57%) without a paired confidence interval.")
    lines.append("")
    lines.append("| Condition | Model | Role | Mean Sim | 1% FAR Flips (k/n) | Flip Rate (1% FAR) | Wilson 95% CI | 0.1% FAR Flips (k/n) | Flip Rate (0.1% FAR) | Wilson 95% CI |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for cond_name in ["clean_control", "clean_q50", "prot_png", "prot_q50"]:
        for m in ["FaceNet", "ArcFace", "SFace", "MobileFaceNet"]:
            md = flips[cond_name][m]
            role = "White-Box" if m in ["FaceNet", "ArcFace"] else "Held-Out"
            lines.append(f"| {cond_name} | {m} | {role} | {md['mean_sim']} [Measured] | {md['flips_1pct']}/{md['valid_1pct_n']} [Measured] | {md['flip_rate_1pct']}% [Measured] | [{md['wilson_ci_1pct'][0]}, {md['wilson_ci_1pct'][1]}] | {md['flips_01pct']}/{md['valid_01pct_n']} [Measured] | {md['flip_rate_01pct']}% [Measured] | [{md['wilson_ci_01pct'][0]}, {md['wilson_ci_01pct'][1]}] |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Artifact Preservation Manifest")
    lines.append("")
    lines.append(f"- **Protected & Control Image Store:** `{meta['image_store_dir']}` (Preserved outside repo)")
    lines.append(f"- **Displacement CSV:** `{RESULTS_DIR / 'landmark_displacements.csv'}`")
    lines.append(f"- **Per-Pair Flip Records:** `{RESULTS_DIR / 'per_pair_flips.json'}`")
    lines.append(f"- **Final JSON:** `{RESULTS_DIR / 'landmark_drift_pilot_final.json'}`")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varman YuNet Landmark Displacement Controlled Pilot")
    parser.add_argument("--pairs", type=int, default=30, help="Number of LFW pairs (default: 30)")
    args = parser.parse_args()

    run_pilot(target_pairs=args.pairs)
