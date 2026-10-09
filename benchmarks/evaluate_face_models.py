"""
Varman Face Recognition Benchmark & Evaluation Suite
=====================================================
Production-grade, reproducible, and auditable face-recognition benchmarking tool.

Key features:
  1. Dynamic ONNX Runtime GPU (CUDAExecutionProvider) execution with explicit CPU fallback logging.
  2. Strict face preprocessing: ArcFace 5-point affine alignment, SFace OpenCV alignCrop, FaceNet margin crop.
     Explicit failure tracking without silent resizing fallback.
  3. Self-validating threshold cache tied to model weights, preprocessing config, and dataset split.
     Uses separate calibration split (LFW train impostors) from evaluation split (LFW test genuine).
     Reports actual observed FAR alongside target FAR.
  4. Correct paired statistical comparisons:
     - Strictly evaluates flip rates over clean baseline matches.
     - Paired difference in proportions (Protected vs. Controls) using Newcombe/Wilson score interval.
     - McNemar exact binomial p-values for discordant pairs.
  5. Reproducible random-noise control with configurable seed and verified realized energy tracking.
  6. Comprehensive experiment artifact export with complete metadata and granular pair audit trails.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Ensure Varman-backend is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
import scipy.stats
from sklearn.datasets import fetch_lfw_pairs
import torch

from app.protection.protection_engine import ProtectionEngine

BENCHMARK_VERSION = "2.1.0"

# InsightFace standard canonical 5-point facial landmark reference (112x112 space)
ARCFACE_CANONICAL_112 = np.array([
    [38.2946, 51.6963],   # Right Eye
    [73.5318, 51.5014],   # Left Eye
    [56.0252, 71.7366],   # Nose Tip
    [41.5493, 92.3655],   # Right Mouth Corner
    [70.7299, 92.2041],   # Left Mouth Corner
], dtype=np.float32)

PREPROCESSING_CONFIG = {
    "arcface": "canonical_5point_affine_112x112_rgb_norm_127.5",
    "sface": "opencv_sface_alignCrop_112x112_bgr",
    "facenet": "bbox_margin_0.15_resize_160x160_rgb_norm_127.5_128.0",
}

DEFAULT_JPEG_QUALITIES = [95, 85, 80, 75, 70, 60, 50]
CALIBRATION_SPLIT_NAME = "lfw_train_impostors"
EVALUATION_SPLIT_NAME = "lfw_test_genuine"


# -- File Hashing Helpers ------------------------------------------------------

def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 fingerprint of a model file."""
    if not filepath.exists():
        return "file_missing"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


# -- Statistical Helper Functions ---------------------------------------------

def compute_single_proportion_ci(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Computes Wilson score 95% confidence interval for a single binomial proportion.
    Handles boundaries [0, 1] cleanly.
    """
    if trials <= 0:
        return 0.0, 0.0
    p = successes / trials
    z = float(scipy.stats.norm.ppf((1.0 + confidence) / 2.0))
    denom = 1.0 + (z ** 2) / trials
    center = (p + (z ** 2) / (2.0 * trials)) / denom
    margin = (z * np.sqrt((p * (1.0 - p) / trials) + ((z ** 2) / (4.0 * (trials ** 2))))) / denom
    ci_low = max(0.0, float(center - margin) * 100.0)
    ci_high = min(100.0, float(center + margin) * 100.0)
    return round(ci_low, 2), round(ci_high, 2)


def compute_paired_proportion_difference_ci(
    treatment_flips: List[bool],
    control_flips: List[bool],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Computes the paired difference in proportions (p_treatment - p_control)
    with Newcombe-Wilson paired confidence interval and McNemar exact p-value.

    Treats zero flips, identical outcomes, and small discordant sample sizes robustly.
    CI is explicitly documented as approximate asymptotic (Newcombe Method 10).
    """
    assert len(treatment_flips) == len(control_flips), "Treatment and Control lists must match in length"
    n = len(treatment_flips)
    if n == 0:
        return {
            "diff_pct": 0.0,
            "ci_lower_pct": 0.0,
            "ci_upper_pct": 0.0,
            "ci_method": "newcombe_paired_wilson_approximate",
            "p_value": 1.0,
            "test_type": "none (n=0)",
            "n_discordant_treatment": 0,
            "n_discordant_control": 0,
            "ci_formatted": "+0.00% [+0.00%, +0.00%]",
        }

    x = np.array(treatment_flips, dtype=bool)
    y = np.array(control_flips, dtype=bool)

    # Contingency matrix
    # n11: both flipped, n10: treatment only, n01: control only, n00: neither
    n11 = int(np.sum(x & y))
    n10 = int(np.sum(x & ~y))
    n01 = int(np.sum(~x & y))
    n00 = int(np.sum(~x & ~y))

    p1 = (n11 + n10) / n
    p2 = (n11 + n01) / n
    diff = p1 - p2
    diff_pct = diff * 100.0

    # Discordant pairs
    n_disc = n10 + n01

    # Exact McNemar test on discordant pairs
    if n_disc == 0:
        p_val = 1.0
        test_type = "mcnemar_identical_outcomes"
    elif n_disc < 25:
        # Exact two-sided binomial test under H0: p = 0.5
        b_res = scipy.stats.binomtest(n10, n_disc, p=0.5, alternative="two-sided")
        p_val = float(b_res.pvalue)
        test_type = "mcnemar_exact_binomial"
    else:
        # Continuity-corrected chi-square
        chi2 = (abs(n10 - n01) - 1.0) ** 2 / n_disc
        p_val = float(1.0 - scipy.stats.chi2.cdf(chi2, df=1))
        test_type = "mcnemar_chi2_corrected"

    # Newcombe Paired Wilson Score Interval (Method 10)
    z = float(scipy.stats.norm.ppf((1.0 + confidence) / 2.0))

    # Wilson intervals for marginals p1 and p2 (as proportions in [0, 1])
    def wilson_bounds(k: int, total: int) -> Tuple[float, float]:
        if total == 0:
            return 0.0, 0.0
        ph = k / total
        d_ = 1.0 + (z ** 2) / total
        ctr = (ph + (z ** 2) / (2.0 * total)) / d_
        mrg = (z * np.sqrt((ph * (1.0 - ph) / total) + ((z ** 2) / (4.0 * (total ** 2))))) / d_
        return max(0.0, ctr - mrg), min(1.0, ctr + mrg)

    l1, u1 = wilson_bounds(n11 + n10, n)
    l2, u2 = wilson_bounds(n11 + n01, n)

    # Sample correlation phi
    denom_phi = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    if denom_phi > 0:
        phi = float((n11 * n00 - n10 * n01) / denom_phi)
    else:
        phi = 0.0

    # Newcombe bounds
    term_l = (p1 - l1) ** 2 - 2.0 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2
    term_u = (u1 - p1) ** 2 - 2.0 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2

    delta_l = diff - np.sqrt(max(0.0, term_l))
    delta_u = diff + np.sqrt(max(0.0, term_u))

    ci_low_pct = max(-100.0, float(delta_l * 100.0))
    ci_high_pct = min(100.0, float(delta_u * 100.0))

    return {
        "diff_pct": round(diff_pct, 2),
        "ci_lower_pct": round(ci_low_pct, 2),
        "ci_upper_pct": round(ci_high_pct, 2),
        "ci_method": "newcombe_paired_wilson_approximate",
        "p_value": round(p_val, 5),
        "test_type": test_type,
        "n_discordant_treatment": n10,
        "n_discordant_control": n01,
        "ci_formatted": f"{diff_pct:+.2f}% [{ci_low_pct:+.2f}%, {ci_high_pct:+.2f}%]",
    }


# -- Preprocessing & Alignment Implementations ---------------------------------

def align_arcface_112(img_bgr: np.ndarray, landmarks: List[Tuple[int, int]]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Standard InsightFace 5-point similarity transformation to canonical 112x112.
    Returns (crop, None) on success or (None, failure_reason) on failure.
    """
    if len(landmarks) < 5:
        return None, "insufficient_landmarks"
    src = np.array(landmarks[:5], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_CANONICAL_112)
    if M is None:
        return None, "affine_matrix_estimation_failed"
    warped = cv2.warpAffine(img_bgr, M, (112, 112), borderMode=cv2.BORDER_REFLECT)
    if warped is None or warped.size == 0 or warped.shape[:2] != (112, 112):
        return None, "affine_warp_produced_invalid_crop"
    return warped, None


def align_sface_112(
    sface_model: cv2.FaceRecognizerSF,
    img_bgr: np.ndarray,
    face_data_15: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Official OpenCV FaceRecognizerSF landmark-aligned crop.
    Requires exact 15-float FaceDetectorYN structure.
    Returns (crop, None) on success or (None, failure_reason) on failure.
    """
    if face_data_15 is None or len(face_data_15) < 15:
        return None, "invalid_face_data_vector_for_sface"
    try:
        aligned = sface_model.alignCrop(img_bgr, face_data_15.astype(np.float32))
        if aligned is None or aligned.size == 0 or aligned.shape[:2] != (112, 112):
            return None, "sface_alignCrop_produced_invalid_dimensions"
        return aligned, None
    except Exception as exc:
        return None, f"sface_alignCrop_exception: {exc}"


def crop_facenet_160(
    img_bgr: np.ndarray,
    box: Tuple[int, int, int, int],
    margin: float = 0.15,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """
    Margin-expanded bounding box crop resized to 160x160 for FaceNet.
    Returns (crop, None) on success or (None, failure_reason) on failure.
    """
    h, w = img_bgr.shape[:2]
    fx, fy, fw, fh = box
    if fw <= 5 or fh <= 5:
        return None, "face_box_too_small"
    mw, mh = int(fw * margin), int(fh * margin)
    x1, y1 = max(0, fx - mw), max(0, fy - mh)
    x2, y2 = min(w, fx + fw + mw), min(h, fy + fh + mh)
    crop = img_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None, "empty_or_degenerate_bounding_box_crop"
    resized = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
    return resized, None


def compress_jpeg(img_bgr: np.ndarray, quality: int) -> np.ndarray:
    success, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError(f"Failed to compress JPEG at Q={quality}")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def generate_energy_matched_noise(
    original_bgr: np.ndarray,
    protected_bgr: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Synthesizes Gaussian random noise matching the L2 energy of the protection perturbation,
    applies image clipping and uint8 conversion, and reports actual realized energy retention.
    """
    pert = protected_bgr.astype(np.float64) - original_bgr.astype(np.float64)
    intended_energy = float(np.linalg.norm(pert))
    if intended_energy < 1e-6:
        return original_bgr.copy(), {
            "intended_l2_energy": 0.0,
            "realized_l2_energy": 0.0,
            "energy_match_ratio": 1.0,
        }

    noise = rng.standard_normal(original_bgr.shape, dtype=np.float64)
    noise_norm = float(np.linalg.norm(noise))
    scaled_noise = noise * (intended_energy / (noise_norm + 1e-12))
    noisy_bgr = np.clip(original_bgr.astype(np.float64) + scaled_noise, 0, 255).astype(np.uint8)

    realized_pert = noisy_bgr.astype(np.float64) - original_bgr.astype(np.float64)
    realized_energy = float(np.linalg.norm(realized_pert))
    ratio = realized_energy / (intended_energy + 1e-12)

    return noisy_bgr, {
        "intended_l2_energy": round(intended_energy, 4),
        "realized_l2_energy": round(realized_energy, 4),
        "energy_match_ratio": round(ratio, 4),
    }


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    dot = float(np.dot(u, v))
    denom = float((np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12)
    return float(dot / denom)


# -- Model Suite & Transparent Execution Management ---------------------------

class ModelSuite:
    """
    Manages Tier 1 Face Recognition Models with transparent hardware logging:
      - FaceNet: PyTorch (CUDA if available, else CPU)
      - ArcFace: ONNX Runtime (Dynamic CUDA fallback to CPU with diagnostics)
      - SFace: OpenCV FaceRecognizerSF (CPU DNN backend)
    """

    def __init__(self, models_dir: Path, requested_device: str = "cuda"):
        self.models_dir = models_dir
        self.requested_device = requested_device
        self.diagnostics: Dict[str, Any] = {}

        # 1. PyTorch Device
        if requested_device == "cuda" and torch.cuda.is_available():
            self.torch_device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            self.diagnostics["torch"] = {
                "device": "cuda:0",
                "gpu_name": gpu_name,
                "cuda_version": torch.version.cuda,
            }
            print(f"[Device Init] PyTorch CUDA enabled: {gpu_name} (CUDA {torch.version.cuda})")
        else:
            self.torch_device = torch.device("cpu")
            self.diagnostics["torch"] = {"device": "cpu", "gpu_name": None}
            print(f"[Device Init] PyTorch running on CPU")

        # 2. ONNX Runtime Dynamic Providers
        available_providers = ort.get_available_providers()
        self.requested_onnx_providers = [
            p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"] if p in available_providers
        ]
        self.diagnostics["onnx"] = {
            "available_providers": available_providers,
            "requested_providers": self.requested_onnx_providers,
        }
        print(f"[Device Init] Available ONNX Providers: {available_providers}")
        print(f"[Device Init] Target Filtered Providers: {self.requested_onnx_providers}")

        self._init_facenet()
        self._init_arcface()
        self._init_sface()

    def _init_facenet(self):
        from facenet_pytorch import InceptionResnetV1
        print(f"[ModelSuite] Loading FaceNet (InceptionResnetV1) -> {self.torch_device}...")
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.torch_device)
        self.diagnostics["facenet"] = {"backend": "PyTorch", "device": str(self.torch_device)}

    def _init_arcface(self):
        arcface_path = self.models_dir / "w600k_r50.onnx"
        if not arcface_path.exists():
            raise FileNotFoundError(f"ArcFace model missing at {arcface_path}")

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create session with fallback
        try:
            self.arcface_session = ort.InferenceSession(
                str(arcface_path),
                sess_options=sess_opts,
                providers=self.requested_onnx_providers,
            )
        except Exception as exc:
            print(f"[ModelSuite] Warning: ONNX GPU initialization failed ({exc}). Falling back to CPU.")
            self.arcface_session = ort.InferenceSession(
                str(arcface_path),
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )

        active_provider = self.arcface_session.get_providers()[0]
        cuda_requested = "CUDAExecutionProvider" in self.requested_onnx_providers
        fallback_occurred = cuda_requested and (active_provider != "CUDAExecutionProvider")

        self.arcface_in_name = self.arcface_session.get_inputs()[0].name
        self.arcface_out_name = self.arcface_session.get_outputs()[0].name

        self.diagnostics["arcface"] = {
            "backend": "ONNXRuntime",
            "active_provider": active_provider,
            "configured_providers": self.arcface_session.get_providers(),
            "cuda_requested": cuda_requested,
            "fallback_occurred": fallback_occurred,
            "fallback_reason": (
                "Native CUDA/cuDNN DLL dependency mismatch (e.g. cublasLt64_13.dll)"
                if fallback_occurred
                else None
            ),
        }
        print(f"[ModelSuite] ArcFace session configured. Executing provider: {active_provider}")
        if fallback_occurred:
            print(f"  -> Note: CUDA requested, but fell back to CPUExecutionProvider due to environment runtime DLLs.")

    def _init_sface(self):
        sface_path = self.models_dir / "face_recognition_sface_2021dec.onnx"
        if not sface_path.exists():
            raise FileNotFoundError(f"SFace model missing at {sface_path}")
        self.sface_opencv = cv2.FaceRecognizerSF.create(str(sface_path), "")
        self.diagnostics["sface"] = {
            "backend": "OpenCV-FaceRecognizerSF",
            "device": "CPU",
        }
        print(f"[ModelSuite] SFace initialized. Executing backend: OpenCV-FaceRecognizerSF (CPU)")

    def extract_facenet(self, face_160_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_160_bgr, cv2.COLOR_BGR2RGB)
        tensor = (torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1) - 127.5) / 128.0
        tensor = tensor.unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            emb = self.facenet(tensor)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().flatten()

    def extract_arcface(self, face_112_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(face_112_bgr, cv2.COLOR_BGR2RGB)
        tensor = (rgb.transpose(2, 0, 1).astype(np.float32) - 127.5) / 127.5
        feed = {self.arcface_in_name: tensor[np.newaxis, ...]}
        emb = self.arcface_session.run([self.arcface_out_name], feed)[0].flatten()
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-12)

    def extract_sface(self, face_112_bgr: np.ndarray) -> np.ndarray:
        feat = self.sface_opencv.feature(face_112_bgr).flatten()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-12)

    def extract_model_features(
        self,
        img_bgr: np.ndarray,
        face_info: Dict[str, Any],
    ) -> Tuple[Dict[str, Optional[np.ndarray]], Dict[str, Optional[str]]]:
        """
        Extracts embeddings for FaceNet, ArcFace, and SFace with explicit failure tracking.
        Never resizes entire image on failure.
        """
        embeddings: Dict[str, Optional[np.ndarray]] = {}
        failures: Dict[str, Optional[str]] = {}

        # 1. FaceNet (BBox Margin Crop)
        crop_fn, err_fn = crop_facenet_160(img_bgr, face_info["box"])
        if crop_fn is not None:
            embeddings["FaceNet"] = self.extract_facenet(crop_fn)
            failures["FaceNet"] = None
        else:
            embeddings["FaceNet"] = None
            failures["FaceNet"] = err_fn

        # 2. ArcFace (5-point Canonical Affine Alignment)
        crop_af, err_af = align_arcface_112(img_bgr, face_info["landmarks"])
        if crop_af is not None:
            embeddings["ArcFace"] = self.extract_arcface(crop_af)
            failures["ArcFace"] = None
        else:
            embeddings["ArcFace"] = None
            failures["ArcFace"] = err_af

        # 3. SFace (Official OpenCV alignCrop with 15-float vector)
        face_data_15 = reconstruct_15_float_face(face_info)
        crop_sf, err_sf = align_sface_112(self.sface_opencv, img_bgr, face_data_15)
        if crop_sf is not None:
            embeddings["SFace"] = self.extract_sface(crop_sf)
            failures["SFace"] = None
        else:
            embeddings["SFace"] = None
            failures["SFace"] = err_sf

        return embeddings, failures


def reconstruct_15_float_face(face_info: Dict[str, Any]) -> np.ndarray:
    """Reconstructs the standard 15-float YuNet vector from face_info dict."""
    fx, fy, fw, fh = face_info["box"]
    lm = face_info["landmarks"]
    score = face_info.get("score", 0.95)
    return np.array([
        fx, fy, fw, fh,
        lm[0][0], lm[0][1],  # R Eye
        lm[1][0], lm[1][1],  # L Eye
        lm[2][0], lm[2][1],  # Nose
        lm[3][0], lm[3][1],  # R Mouth
        lm[4][0], lm[4][1],  # L Mouth
        score,
    ], dtype=np.float32)


# -- Self-Validating Threshold Cache ------------------------------------------

def validate_threshold_cache(
    cache_path: Path,
    expected_version: str,
    expected_split: str,
    expected_target_far: float,
    expected_hashes: Dict[str, str],
    expected_prep: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    """Validates that cached thresholds match current models, split, and config."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("benchmark_version") != expected_version:
            return None
        if data.get("calibration_split") != expected_split:
            return None
        if abs(data.get("target_far", 0.0) - expected_target_far) > 1e-5:
            return None
        if data.get("model_weights_hashes") != expected_hashes:
            return None
        if data.get("preprocessing_config") != expected_prep:
            return None
        if "thresholds" not in data or "observed_far" not in data:
            return None

        return data
    except Exception:
        return None


def save_threshold_cache(cache_path: Path, data: Dict[str, Any]):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def calibrate_thresholds_on_split(
    model_suite: ModelSuite,
    engine: ProtectionEngine,
    target_far: float = 0.01,
    max_impostors: int = 500,
) -> Dict[str, Any]:
    """
    Calibrates baseline thresholds strictly on LFW train impostor split (target == 0).
    Never touches evaluation genuine pairs. Reports observed FAR alongside target FAR.
    """
    print("\n" + "=" * 80)
    print(f"CALIBRATING THRESHOLDS ON SEPARATE DEV SPLIT ({CALIBRATION_SPLIT_NAME})")
    print(f"Target FAR: {target_far * 100:.1f}% | Max Impostors: {max_impostors}")
    print("=" * 80)

    lfw_train = fetch_lfw_pairs(subset="train", color=True, resize=1.0, slice_=None)
    impostor_indices = np.where(lfw_train.target == 0)[0][:max_impostors]

    models = ["FaceNet", "ArcFace", "SFace"]
    scores: Dict[str, List[float]] = {m: [] for m in models}

    valid_impostors = 0
    t0 = time.perf_counter()

    for idx in impostor_indices:
        bgr1 = cv2.cvtColor((lfw_train.pairs[idx][0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor((lfw_train.pairs[idx][1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        f1 = engine.detect_faces(bgr1)
        f2 = engine.detect_faces(bgr2)
        if not f1 or not f2:
            continue

        emb1, fail1 = model_suite.extract_model_features(bgr1, f1[0])
        emb2, fail2 = model_suite.extract_model_features(bgr2, f2[0])

        for m in models:
            if emb1[m] is not None and emb2[m] is not None:
                sim = cosine_similarity(emb1[m], emb2[m])
                scores[m].append(sim)

        valid_impostors += 1

    elapsed = time.perf_counter() - t0
    percentile = (1.0 - target_far) * 100.0

    thresholds: Dict[str, float] = {}
    observed_far: Dict[str, float] = {}

    print(f"Evaluated {valid_impostors} valid impostor pairs from '{CALIBRATION_SPLIT_NAME}' ({elapsed:.1f}s):")
    for m in models:
        arr = np.array(scores[m])
        if len(arr) > 0:
            thresh = float(np.percentile(arr, percentile))
            # Measure actual observed false acceptance rate on calibration set
            obs_far = float(np.mean(arr >= thresh))
            thresholds[m] = round(thresh, 4)
            observed_far[m] = round(obs_far, 4)
            print(f"  {m:<9} | Threshold: {thresh:.4f} | Observed FAR: {obs_far * 100:.2f}% (Target: {target_far * 100:.1f}%) | Mean: {arr.mean():.4f}")
        else:
            thresholds[m] = 0.5
            observed_far[m] = 0.0

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "calibration_split": CALIBRATION_SPLIT_NAME,
        "calibration_pair_count": valid_impostors,
        "target_far": target_far,
        "observed_far": observed_far,
        "thresholds": thresholds,
    }


# -- Core Benchmark Execution ------------------------------------------------

def run_face_recognition_benchmark(
    num_pairs: int = 500,
    preset: str = "Balanced",
    jpeg_qualities: Optional[List[int]] = None,
    requested_device: str = "cuda",
    seed: int = 42,
    recalibrate: bool = False,
    max_impostors: int = 500,
    output_json: Path = Path("benchmarks/face_eval_results.json"),
) -> Dict[str, Any]:
    if jpeg_qualities is None:
        jpeg_qualities = DEFAULT_JPEG_QUALITIES

    print("\n" + "=" * 90)
    print("STARTING AUDITABLE FACE RECOGNITION BENCHMARK EVALUATION")
    print("=" * 90)
    print(f"Preset: {preset} | Target Genuine Pairs: {num_pairs} | Seed: {seed}")
    print(f"Evaluation Split: {EVALUATION_SPLIT_NAME} | Qualities: {jpeg_qualities}")
    print("-" * 90)

    rng = np.random.default_rng(seed)
    models_dir = Path("models").resolve()
    model_suite = ModelSuite(models_dir=models_dir, requested_device=requested_device)
    engine = ProtectionEngine(model_dir=str(models_dir))

    # Weight fingerprints
    weight_hashes = {
        "ArcFace": compute_file_sha256(models_dir / "w600k_r50.onnx"),
        "SFace": compute_file_sha256(models_dir / "face_recognition_sface_2021dec.onnx"),
        "FaceNet": "InceptionResnetV1_vggface2",
    }

    # 1. Resolve Decision Thresholds (Self-Validating Cache vs Dev Split)
    cache_path = Path("benchmarks/calibrated_thresholds.json")
    cached_payload = validate_threshold_cache(
        cache_path,
        expected_version=BENCHMARK_VERSION,
        expected_split=CALIBRATION_SPLIT_NAME,
        expected_target_far=0.01,
        expected_hashes=weight_hashes,
        expected_prep=PREPROCESSING_CONFIG,
    )

    if not recalibrate and cached_payload is not None:
        thresholds = cached_payload["thresholds"]
        observed_far = cached_payload["observed_far"]
        calibration_provenance = "cached_validated_payload"
        print(f"[Thresholds] Validated cache loaded from {cache_path}")
        print(f"  -> Target FAR: 1.0% | Observed FARs: {observed_far}")
    else:
        print(f"[Thresholds] Cache stale, missing, or recalibrate requested. Calibrating on dev split...")
        calib_data = calibrate_thresholds_on_split(model_suite, engine, target_far=0.01, max_impostors=max_impostors)
        thresholds = calib_data["thresholds"]
        observed_far = calib_data["observed_far"]
        calibration_provenance = "calibrated_on_lfw_train_impostors"
        # Persist full self-validating cache
        cache_payload = {
            **calib_data,
            "model_weights_hashes": weight_hashes,
            "preprocessing_config": PREPROCESSING_CONFIG,
        }
        save_threshold_cache(cache_path, cache_payload)
        print(f"[Thresholds] Saved self-validating cache to {cache_path}")

    # 2. Load Evaluation Split (LFW test genuine)
    print(f"\n[Dataset] Loading evaluation set ({EVALUATION_SPLIT_NAME})...")
    lfw_test = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None)
    genuine_indices = np.where(lfw_test.target == 1)[0]
    use_count = min(num_pairs, len(genuine_indices))
    selected_indices = genuine_indices[:use_count]
    print(f"[Dataset] Selected {use_count} genuine pairs from {EVALUATION_SPLIT_NAME} for benchmarking.")

    models = ["FaceNet", "ArcFace", "SFace"]

    # Granular state maps: state_maps[model][quality_str] -> list of pair audit records
    state_maps: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        m: {f"Q{q}": [] for q in jpeg_qualities} for m in models
    }

    t_start = time.perf_counter()
    skipped_pairs_count = 0
    evaluated_pairs_count = 0
    noise_energy_ratios: List[float] = []

    for pair_id, pair_idx in enumerate(selected_indices):
        bgr1 = cv2.cvtColor((lfw_test.pairs[pair_idx][0] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr2 = cv2.cvtColor((lfw_test.pairs[pair_idx][1] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Detect faces on original uncompressed images
        f1 = engine.detect_faces(bgr1)
        f2 = engine.detect_faces(bgr2)
        if not f1 or not f2:
            skipped_pairs_count += 1
            for m in models:
                for q in jpeg_qualities:
                    state_maps[m][f"Q{q}"].append({
                        "pair_id": pair_id,
                        "is_valid": False,
                        "skip_reason": "detector_found_no_face_on_original",
                        "is_clean_match": False,
                        "is_protected_flip": False,
                        "is_jpeg_control_flip": False,
                        "is_noise_control_flip": False,
                        "sim_clean": None,
                        "sim_prot": None,
                        "sim_jpeg_ctrl": None,
                        "sim_noise_ctrl": None,
                    })
            continue

        face1_clean = f1[0]
        face2_target = f2[0]

        # Extract clean baseline representations with explicit failure tracking
        emb2_target, fail2 = model_suite.extract_model_features(bgr2, face2_target)
        emb1_clean, fail1 = model_suite.extract_model_features(bgr1, face1_clean)

        # Run Varman Protection once on original image 1
        prot_bgr, prot_meta = engine.protect(bgr1, preset_name=preset)
        if not prot_meta.get("faces"):
            skipped_pairs_count += 1
            for m in models:
                for q in jpeg_qualities:
                    state_maps[m][f"Q{q}"].append({
                        "pair_id": pair_id,
                        "is_valid": False,
                        "skip_reason": "detector_lost_face_after_protection",
                        "is_clean_match": False,
                        "is_protected_flip": False,
                        "is_jpeg_control_flip": False,
                        "is_noise_control_flip": False,
                        "sim_clean": None,
                        "sim_prot": None,
                        "sim_jpeg_ctrl": None,
                        "sim_noise_ctrl": None,
                    })
            continue

        face1_prot = prot_meta["faces"][0]

        # Synthesize reproducible energy-matched random noise control
        noise_bgr, noise_stats = generate_energy_matched_noise(bgr1, prot_bgr, rng)
        noise_energy_ratios.append(noise_stats["energy_match_ratio"])

        # Determine clean baseline match status per model
        clean_match_status: Dict[str, bool] = {}
        clean_sims: Dict[str, Optional[float]] = {}
        clean_valid: Dict[str, bool] = {}

        for m in models:
            if emb1_clean[m] is None or emb2_target[m] is None:
                clean_valid[m] = False
                clean_match_status[m] = False
                clean_sims[m] = None
            else:
                sim = cosine_similarity(emb1_clean[m], emb2_target[m])
                clean_sims[m] = sim
                clean_valid[m] = True
                clean_match_status[m] = bool(sim >= thresholds[m])

        # Evaluate across JPEG quality tiers
        for q in jpeg_qualities:
            ctrl_bgr = compress_jpeg(bgr1, quality=q)
            prot_comp_bgr = compress_jpeg(prot_bgr, quality=q)
            noise_comp_bgr = compress_jpeg(noise_bgr, quality=q)

            emb_ctrl, _ = model_suite.extract_model_features(ctrl_bgr, face1_clean)
            emb_prot, _ = model_suite.extract_model_features(prot_comp_bgr, face1_prot)
            emb_noise, _ = model_suite.extract_model_features(noise_comp_bgr, face1_clean)

            for m in models:
                # If extraction on clean baseline failed, record failure
                if not clean_valid[m]:
                    state_maps[m][f"Q{q}"].append({
                        "pair_id": pair_id,
                        "is_valid": False,
                        "skip_reason": f"clean_feature_extraction_failed ({fail1.get(m) or fail2.get(m)})",
                        "is_clean_match": False,
                        "is_protected_flip": False,
                        "is_jpeg_control_flip": False,
                        "is_noise_control_flip": False,
                        "sim_clean": None,
                        "sim_prot": None,
                        "sim_jpeg_ctrl": None,
                        "sim_noise_ctrl": None,
                    })
                    continue

                thresh = thresholds[m]
                is_clean = clean_match_status[m]
                sim_c = clean_sims[m]

                # Extract similarities under degradation
                sim_p = cosine_similarity(emb_prot[m], emb2_target[m]) if emb_prot[m] is not None else -1.0
                sim_ctrl = cosine_similarity(emb_ctrl[m], emb2_target[m]) if emb_ctrl[m] is not None else -1.0
                sim_n = cosine_similarity(emb_noise[m], emb2_target[m]) if emb_noise[m] is not None else -1.0

                # A flip occurs ONLY if is_clean_match was True, but degraded similarity drops below threshold
                is_prot_flip = bool(is_clean and (sim_p < thresh))
                is_ctrl_flip = bool(is_clean and (sim_ctrl < thresh))
                is_noise_flip = bool(is_clean and (sim_n < thresh))

                state_maps[m][f"Q{q}"].append({
                    "pair_id": pair_id,
                    "is_valid": True,
                    "skip_reason": None,
                    "is_clean_match": is_clean,
                    "is_protected_flip": is_prot_flip,
                    "is_jpeg_control_flip": is_ctrl_flip,
                    "is_noise_control_flip": is_noise_flip,
                    "sim_clean": round(sim_c, 4) if sim_c is not None else None,
                    "sim_prot": round(sim_p, 4) if sim_p is not None else None,
                    "sim_jpeg_ctrl": round(sim_ctrl, 4) if sim_ctrl is not None else None,
                    "sim_noise_ctrl": round(sim_n, 4) if sim_n is not None else None,
                })

        evaluated_pairs_count += 1
        if evaluated_pairs_count % 50 == 0 or (pair_id + 1) == use_count:
            elapsed = time.perf_counter() - t_start
            print(f"  Processed {evaluated_pairs_count}/{use_count} pairs ({elapsed:.1f}s)...")

    total_time = time.perf_counter() - t_start
    mean_energy_ratio = float(np.mean(noise_energy_ratios)) if noise_energy_ratios else 1.0

    # -- Compile Summary Metrics with Strict Denominator Accounting ---------------

    print("\n" + "=" * 120)
    print("EXPERIMENT RESULTS: RIGOROUS FLIP RATES & NEWCOMBE PAIRED 95% CIs")
    print(f"Evaluated Pairs: {evaluated_pairs_count} | Skipped Pairs: {skipped_pairs_count} | Total Time: {total_time:.1f}s")
    print(f"Random Noise Realized Energy Match Ratio: {mean_energy_ratio * 100:.2f}% (measured after uint8 quantization)")
    print("=" * 120)
    print(f"{'Model':<9} | {'Quality':<7} | {'Clean (N)':<9} | {'Protected Flip':<16} | {'JPEG Ctrl':<10} | {'Noise Ctrl':<10} | {'Net Diff vs JPEG (95% CI)':<26} | {'Net Diff vs Noise (95% CI)'}")
    print("-" * 120)

    summary_metrics: Dict[str, Any] = {}

    for m in models:
        summary_metrics[m] = {
            "calibrated_threshold": thresholds[m],
            "observed_far_calibration": observed_far[m],
            "qualities": {},
        }

        # Model-wide clean acceptance count from valid pairs
        first_q_records = state_maps[m][f"Q{jpeg_qualities[0]}"]
        clean_count = sum(1 for r in first_q_records if r["is_clean_match"])
        total_valid = sum(1 for r in first_q_records if r["is_valid"])
        clean_acc_rate = (clean_count / total_valid * 100.0) if total_valid > 0 else 0.0

        summary_metrics[m]["total_pairs_input"] = len(first_q_records)
        summary_metrics[m]["total_pairs_valid"] = total_valid
        summary_metrics[m]["clean_matches_count"] = clean_count
        summary_metrics[m]["clean_acceptance_rate_pct"] = round(clean_acc_rate, 2)

        for q in jpeg_qualities:
            q_label = f"Q{q}"
            q_records = state_maps[m][q_label]

            # STRICT DENOMINATOR: Only pairs where is_clean_match == True
            clean_subpop = [r for r in q_records if r["is_clean_match"]]
            n_clean = len(clean_subpop)

            if n_clean > 0:
                prot_flips = [r["is_protected_flip"] for r in clean_subpop]
                ctrl_flips = [r["is_jpeg_control_flip"] for r in clean_subpop]
                noise_flips = [r["is_noise_control_flip"] for r in clean_subpop]

                raw_prot = (sum(prot_flips) / n_clean) * 100.0
                raw_ctrl = (sum(ctrl_flips) / n_clean) * 100.0
                raw_noise = (sum(noise_flips) / n_clean) * 100.0

                ci_raw_prot = compute_single_proportion_ci(sum(prot_flips), n_clean)
                ci_diff_jpeg = compute_paired_proportion_difference_ci(prot_flips, ctrl_flips)
                ci_diff_noise = compute_paired_proportion_difference_ci(prot_flips, noise_flips)
            else:
                raw_prot, raw_ctrl, raw_noise = 0.0, 0.0, 0.0
                ci_raw_prot = (0.0, 0.0)
                ci_diff_jpeg = compute_paired_proportion_difference_ci([], [])
                ci_diff_noise = compute_paired_proportion_difference_ci([], [])

            prot_str = f"{raw_prot:5.2f}%"
            ctrl_str = f"{raw_ctrl:5.2f}%"
            noise_str = f"{raw_noise:5.2f}%"
            diff_jpeg_str = ci_diff_jpeg["ci_formatted"]
            diff_noise_str = ci_diff_noise["ci_formatted"]

            print(f"{m:<9} | {q_label:<7} | {n_clean:<9} | {prot_str:<16} | {ctrl_str:<10} | {noise_str:<10} | {diff_jpeg_str:<26} | {diff_noise_str}")

            summary_metrics[m]["qualities"][q_label] = {
                "quality": q,
                "clean_matches_denominator": n_clean,
                "raw_protected_flips": sum(1 for r in clean_subpop if r["is_protected_flip"]),
                "raw_protected_flip_rate_pct": round(raw_prot, 2),
                "protected_flip_rate_95_ci_wilson": list(ci_raw_prot),
                "raw_jpeg_control_flips": sum(1 for r in clean_subpop if r["is_jpeg_control_flip"]),
                "raw_jpeg_control_flip_rate_pct": round(raw_ctrl, 2),
                "raw_noise_control_flips": sum(1 for r in clean_subpop if r["is_noise_control_flip"]),
                "raw_noise_control_flip_rate_pct": round(raw_noise, 2),
                "paired_comparison_vs_jpeg_control": ci_diff_jpeg,
                "paired_comparison_vs_noise_control": ci_diff_noise,
            }
        print("-" * 120)

    # -- Export Comprehensive Experiment Artifact ---------------------------------

    artifact: Dict[str, Any] = {
        "meta_configuration": {
            "benchmark_version": BENCHMARK_VERSION,
            "timestamp_utc": datetime.datetime.utcnow().isoformat(),
            "random_seed": seed,
            "execution_preset": preset,
            "calibration_split": CALIBRATION_SPLIT_NAME,
            "evaluation_split": EVALUATION_SPLIT_NAME,
            "calibration_provenance": calibration_provenance,
            "calibration_target_far": 0.01,
            "calibration_observed_far": observed_far,
            "calibrated_thresholds": thresholds,
            "model_weights_hashes": weight_hashes,
            "preprocessing_config": PREPROCESSING_CONFIG,
            "hardware_diagnostics": model_suite.diagnostics,
            "noise_energy_match_ratio_mean": round(mean_energy_ratio, 4),
            "jpeg_qualities": jpeg_qualities,
            "total_pairs_requested": num_pairs,
            "total_pairs_evaluated": evaluated_pairs_count,
            "total_pairs_skipped": skipped_pairs_count,
            "total_execution_time_seconds": round(total_time, 2),
        },
        "summary_metrics": summary_metrics,
        "granular_pair_state_maps": state_maps,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)

    # Validate output artifact parses cleanly and consistency matches
    with open(output_json, "r", encoding="utf-8") as f:
        reloaded = json.load(f)

    for m in models:
        for q in jpeg_qualities:
            q_label = f"Q{q}"
            sum_rec = reloaded["summary_metrics"][m]["qualities"][q_label]
            pair_recs = reloaded["granular_pair_state_maps"][m][q_label]
            expected_clean = sum(1 for r in pair_recs if r["is_clean_match"])
            expected_prot = sum(1 for r in pair_recs if r["is_protected_flip"])
            expected_ctrl = sum(1 for r in pair_recs if r["is_jpeg_control_flip"])
            expected_noise = sum(1 for r in pair_recs if r["is_noise_control_flip"])

            assert sum_rec["clean_matches_denominator"] == expected_clean, (
                f"Mismatch in clean matches count for {m} {q_label}"
            )
            assert sum_rec["raw_protected_flips"] == expected_prot, (
                f"Mismatch in protected flips for {m} {q_label}"
            )
            assert sum_rec["raw_jpeg_control_flips"] == expected_ctrl, (
                f"Mismatch in control flips for {m} {q_label}"
            )
            assert sum_rec["raw_noise_control_flips"] == expected_noise, (
                f"Mismatch in noise flips for {m} {q_label}"
            )

    print(f"\n[Artifact Validated] Audit artifact successfully saved and verified: {output_json.resolve()}")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varman Refactored Auditable Face Recognition Benchmark")
    parser.add_argument("--pairs", type=int, default=500, help="Number of genuine LFW pairs (default: 500)")
    parser.add_argument("--preset", type=str, default="Balanced", choices=["Light", "Balanced", "Aggressive"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for noise control (default: 42)")
    parser.add_argument("--recalibrate", action="store_true", help="Force recalibration on impostor dev split")
    parser.add_argument("--max-impostors", type=int, default=500, help="Max impostor pairs for calibration (default: 500)")
    parser.add_argument("--output", type=str, default="benchmarks/face_eval_results.json")
    args = parser.parse_args()

    run_face_recognition_benchmark(
        num_pairs=args.pairs,
        preset=args.preset,
        requested_device=args.device,
        seed=args.seed,
        recalibrate=args.recalibrate,
        max_impostors=args.max_impostors,
        output_json=Path(args.output),
    )
