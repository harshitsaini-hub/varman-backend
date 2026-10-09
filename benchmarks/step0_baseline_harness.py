"""
Varman Step 0 Baseline Test Harness
===================================
Establishes the empirical baseline flip rate of the current Varman engine
against Tier 1 face recognition models across a JPEG compression sweep (Q95 down to Q50).

Models evaluated:
  - ArcFace (InsightFace ResNet50 w600k)
  - FaceNet (InceptionResnetV1 VGGFace2)
  - SFace (OpenCV FaceRecognizerSF)

Dataset:
  - LFW (Labeled Faces in the Wild) genuine pairs

Sweep:
  - JPEG Quality: Q95, Q85, Q80, Q75, Q70, Q60, Q50
  - Control: Unprotected image compressed at each Q
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure Varman-backend is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.datasets import fetch_lfw_pairs
import torch

from app.protection.protection_engine import ProtectionEngine

# Standard LFW verification cosine similarity thresholds
# Pairs with clean similarity >= THRESHOLD are considered genuine matches.
# If protected similarity < THRESHOLD, the verification is flipped.
DEFAULT_THRESHOLDS = {
    "FaceNet": 0.60,    # standard cosine similarity threshold
    "ArcFace": 0.50,    # standard threshold for 112x112 ResNet50
    "SFace": 0.363,     # OpenCV SFace official threshold
}

JPEG_QUALITIES = [95, 85, 80, 75, 70, 60, 50]


class ModelSuite:
    """Encapsulates Tier 1 face recognition models."""

    def __init__(self, models_dir: Path, device: str = "cpu"):
        self.models_dir = models_dir
        self.device = device
        self._init_facenet()
        self._init_arcface()
        self._init_sface()

    def _init_facenet(self):
        from facenet_pytorch import InceptionResnetV1
        print(f"[ModelSuite] Loading FaceNet (InceptionResnetV1, device={self.device})...")
        self.facenet = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def _init_arcface(self):
        arcface_path = self.models_dir / "w600k_r50.onnx"
        if not arcface_path.exists():
            raise FileNotFoundError(f"ArcFace ONNX not found: {arcface_path}")
        print(f"[ModelSuite] Loading ArcFace (w600k_r50.onnx, ONNX Runtime)...")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" else ["CPUExecutionProvider"]
        try:
            self.arcface_session = ort.InferenceSession(str(arcface_path), providers=providers)
        except Exception:
            self.arcface_session = ort.InferenceSession(str(arcface_path), providers=["CPUExecutionProvider"])
        self.arcface_input_name = self.arcface_session.get_inputs()[0].name
        self.arcface_output_name = self.arcface_session.get_outputs()[0].name

    def _init_sface(self):
        sface_path = self.models_dir / "face_recognition_sface_2021dec.onnx"
        if not sface_path.exists():
            raise FileNotFoundError(f"SFace ONNX not found: {sface_path}")
        print(f"[ModelSuite] Loading SFace (face_recognition_sface_2021dec.onnx, OpenCV)...")
        self.sface = cv2.FaceRecognizerSF.create(str(sface_path), "")

    def get_facenet_embedding(self, img_rgb: np.ndarray) -> np.ndarray:
        crop = cv2.resize(img_rgb, (160, 160))
        t = (torch.tensor(crop, dtype=torch.float32).permute(2, 0, 1) - 127.5) / 128.0
        t = t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.facenet(t)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            return emb.cpu().numpy().flatten()

    def get_arcface_embedding(self, img_rgb: np.ndarray) -> np.ndarray:
        crop = cv2.resize(img_rgb, (112, 112)).transpose(2, 0, 1).astype(np.float32)
        crop = (crop - 127.5) / 127.5
        feed = {self.arcface_input_name: crop[np.newaxis, ...]}
        emb = self.arcface_session.run([self.arcface_output_name], feed)[0].flatten()
        norm = np.linalg.norm(emb)
        return emb / (norm + 1e-12)

    def get_sface_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        # SFace expects 112x112 input
        crop = cv2.resize(img_bgr, (112, 112))
        feat = self.sface.feature(crop).flatten()
        norm = np.linalg.norm(feat)
        return feat / (norm + 1e-12)

    def extract_all(self, img_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return {
            "FaceNet": self.get_facenet_embedding(img_rgb),
            "ArcFace": self.get_arcface_embedding(img_rgb),
            "SFace": self.get_sface_embedding(img_bgr),
        }


def compress_jpeg(img_bgr: np.ndarray, quality: int) -> np.ndarray:
    """Simulate lossy JPEG encoding and decoding in memory."""
    success, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise RuntimeError(f"Failed to JPEG compress at quality {quality}")
    decompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decompressed


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    dot = np.dot(u, v)
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12
    return float(dot / denom)


def compute_ssim_psnr(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> Tuple[float, float]:
    g1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    # PSNR
    mse = np.mean((g1 - g2) ** 2)
    psnr = 100.0 if mse == 0 else float(20 * np.log10(255.0 / np.sqrt(mse)))
    # SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean()), psnr


def run_step0_evaluation(
    num_pairs: int = 100,
    preset: str = "Balanced",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_json: Path = Path("benchmarks/step0_baseline_results.json"),
) -> Dict[str, Any]:
    print("=" * 78)
    print("VARMAN STEP 0: BASELINE TEST HARNESS")
    print("=" * 78)
    print(f"Device: {device.upper()}")
    print(f"Protection Preset: {preset}")
    print(f"Requested LFW Pairs: {num_pairs}")
    print(f"JPEG Sweep Qualities: {JPEG_QUALITIES}")
    print("-" * 78)

    models_dir = Path("models").resolve()
    model_suite = ModelSuite(models_dir=models_dir, device=device)
    engine = ProtectionEngine(model_dir=str(models_dir))

    print("\n[Data] Fetching LFW test pairs (resize=1.0, slice_=None)...")
    lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None)

    # Filter to genuine pairs (target == 1)
    genuine_indices = np.where(lfw.target == 1)[0]
    total_available = len(genuine_indices)
    use_count = min(num_pairs, total_available)
    selected_indices = genuine_indices[:use_count]
    print(f"[Data] Selected {use_count} genuine positive pairs out of {total_available} available.")

    # Storage for per-model statistics
    models = ["FaceNet", "ArcFace", "SFace"]
    thresholds = DEFAULT_THRESHOLDS

    # Structure:
    # results[model][q] = {
    #   "total_clean_matches": int,
    #   "protected_flips": int,
    #   "control_flips": int,
    #   "clean_sims": [],
    #   "protected_sims": [],
    #   "control_sims": [],
    # }
    stats: Dict[str, Dict[int, Dict[str, Any]]] = {
        m: {
            q: {
                "total_clean_matches": 0,
                "protected_flips": 0,
                "control_flips": 0,
                "clean_sims": [],
                "protected_sims": [],
                "control_sims": [],
            }
            for q in JPEG_QUALITIES
        }
        for m in models
    }

    quality_metrics = {"ssim": [], "psnr": []}
    skipped_no_face = 0
    t_start = time.perf_counter()

    for idx_num, pair_idx in enumerate(selected_indices):
        # Convert float32 [0, 1] RGB to uint8 BGR
        pair_rgb1 = (lfw.pairs[pair_idx][0] * 255).astype(np.uint8)
        pair_rgb2 = (lfw.pairs[pair_idx][1] * 255).astype(np.uint8)
        img1_bgr = cv2.cvtColor(pair_rgb1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(pair_rgb2, cv2.COLOR_RGB2BGR)

        # 1. Clean Reference Embeddings
        emb2 = model_suite.extract_all(img2_bgr)
        emb1_clean = model_suite.extract_all(img1_bgr)

        # 2. Apply Current Varman Protection to Image 1
        prot_bgr, meta = engine.protect(img1_bgr, preset_name=preset)

        ssim_val, psnr_val = compute_ssim_psnr(img1_bgr, prot_bgr)
        quality_metrics["ssim"].append(ssim_val)
        quality_metrics["psnr"].append(psnr_val)

        # 3. For each JPEG Quality, test Protected vs Control
        for q in JPEG_QUALITIES:
            # Control: Image 1 compressed without protection
            ctrl_bgr = compress_jpeg(img1_bgr, quality=q)
            emb1_ctrl = model_suite.extract_all(ctrl_bgr)

            # Protected: Image 1 protected then compressed
            prot_comp_bgr = compress_jpeg(prot_bgr, quality=q)
            emb1_prot = model_suite.extract_all(prot_comp_bgr)

            for m in models:
                sim_clean = cosine_similarity(emb1_clean[m], emb2[m])
                sim_ctrl = cosine_similarity(emb1_ctrl[m], emb2[m])
                sim_prot = cosine_similarity(emb1_prot[m], emb2[m])

                thresh = thresholds[m]

                # Only evaluate pairs that were initially recognized as the same person
                if sim_clean >= thresh:
                    stats[m][q]["total_clean_matches"] += 1
                    stats[m][q]["clean_sims"].append(sim_clean)
                    stats[m][q]["control_sims"].append(sim_ctrl)
                    stats[m][q]["protected_sims"].append(sim_prot)

                    if sim_prot < thresh:
                        stats[m][q]["protected_flips"] += 1
                    if sim_ctrl < thresh:
                        stats[m][q]["control_flips"] += 1

        if (idx_num + 1) % 10 == 0 or (idx_num + 1) == use_count:
            elapsed = time.perf_counter() - t_start
            print(f"  Processed {idx_num + 1}/{use_count} pairs... ({elapsed:.1f}s)")

    total_time = time.perf_counter() - t_start
    mean_ssim = float(np.mean(quality_metrics["ssim"]))
    mean_psnr = float(np.mean(quality_metrics["psnr"]))

    # Compile Final Table
    summary_table = []
    print("\n" + "=" * 88)
    print(f"STEP 0 BASELINE RESULTS: FLIP RATE PER MODEL PER QUALITY (Current Engine: {preset})")
    print(f"Average Quality: SSIM = {mean_ssim:.4f} | PSNR = {mean_psnr:.2f} dB | Total Time: {total_time:.1f}s")
    print("=" * 88)
    print(f"{'Model':<10} | {'JPEG Quality':<12} | {'Clean Matches':<14} | {'Protected Flip Rate':<20} | {'Control Flip (JPEG)':<20} | {'Net Flip'}")
    print("-" * 88)

    compiled_results: Dict[str, Any] = {
        "engine_preset": preset,
        "evaluated_pairs": use_count,
        "mean_ssim": round(mean_ssim, 4),
        "mean_psnr": round(mean_psnr, 2),
        "total_time_seconds": round(total_time, 2),
        "models": {},
    }

    for m in models:
        compiled_results["models"][m] = {"threshold": thresholds[m], "qualities": {}}
        for q in JPEG_QUALITIES:
            data_q = stats[m][q]
            total = data_q["total_clean_matches"]
            if total > 0:
                prot_flip_pct = (data_q["protected_flips"] / total) * 100.0
                ctrl_flip_pct = (data_q["control_flips"] / total) * 100.0
                net_flip_pct = prot_flip_pct - ctrl_flip_pct
                mean_sim_clean = float(np.mean(data_q["clean_sims"]))
                mean_sim_prot = float(np.mean(data_q["protected_sims"]))
                mean_sim_ctrl = float(np.mean(data_q["control_sims"]))
            else:
                prot_flip_pct = 0.0
                ctrl_flip_pct = 0.0
                net_flip_pct = 0.0
                mean_sim_clean = 0.0
                mean_sim_prot = 0.0
                mean_sim_ctrl = 0.0

            q_label = f"Q{q}"
            print(f"{m:<10} | {q_label:<12} | {total:<14} | {prot_flip_pct:>18.2f}% | {ctrl_flip_pct:>18.2f}% | {net_flip_pct:>7.2f}%")

            compiled_results["models"][m]["qualities"][q_label] = {
                "quality": q,
                "clean_matches": total,
                "protected_flips": data_q["protected_flips"],
                "protected_flip_rate_pct": round(prot_flip_pct, 2),
                "control_flips": data_q["control_flips"],
                "control_flip_rate_pct": round(ctrl_flip_pct, 2),
                "net_flip_rate_pct": round(net_flip_pct, 2),
                "mean_sim_clean": round(mean_sim_clean, 4),
                "mean_sim_prot": round(mean_sim_prot, 4),
                "mean_sim_ctrl": round(mean_sim_ctrl, 4),
            }
        print("-" * 88)

    # Save to disk
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(compiled_results, f, indent=2)
    print(f"\n[Saved] Detailed JSON baseline report written to: {output_json.resolve()}")
    return compiled_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Varman Step 0 Baseline Test Harness")
    parser.add_argument("--pairs", type=int, default=50, help="Number of genuine LFW pairs to test (default: 50)")
    parser.add_argument("--preset", type=str, default="Balanced", choices=["Light", "Balanced", "Aggressive"], help="Varman preset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--output", type=str, default="benchmarks/step0_baseline_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_step0_evaluation(
        num_pairs=args.pairs,
        preset=args.preset,
        device=args.device,
        output_json=Path(args.output),
    )
