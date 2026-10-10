"""Phase 2 validation and latency profiling for the locked Varman pipeline.

The script does not introduce new protection algorithms. It runs the existing
ProtectionEngine/EvaluationEngine across multiple real face images and prints
preset metric tables plus per-stage timing summaries.

Inputs are explicit so large/binary assets are not committed:

    python benchmarks/phase2_validation.py \
      --yunet-model /path/to/face_detection_yunet_2023mar.onnx \
      --images /path/to/face1.jpg /path/to/face2.jpg ...

If --images is omitted, the script tries to use skimage.data.lfw_subset() when
available because it provides multiple real face crops. That dependency is only
an optional validation convenience and is not part of the runtime pipeline.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure Varman-backend is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from PIL import Image

from app.protection.protection_engine import ProtectionEngine
from app.protection.evaluation_engine import EvaluationEngine

PRESET_ORDER = ("Light", "Balanced", "Aggressive")
NPR_ORDER = ("Q95", "Q85", "Q75", "Q60", "Resize075_Q75")


def _load_optional_lfw_images(output_dir: Path, limit: int) -> list[Path]:
    try:
        from skimage import data  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("No --images were supplied and optional skimage.data is unavailable") from exc

    faces = data.lfw_subset()
    paths: list[Path] = []
    for idx, face in enumerate(faces[:limit]):
        arr = np.asarray(face)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([arr, arr, arr], axis=2)
        path = output_dir / f"lfw_subset_{idx:02d}.png"
        Image.fromarray(rgb).save(path)
        paths.append(path)
    return paths


def _mean(values: list[float | None]) -> float | None:
    concrete = [v for v in values if v is not None]
    if not concrete:
        return None
    return float(statistics.mean(concrete))


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def run_validation(yunet_model: Path, images: list[Path], output_dir: Path) -> dict[str, Any]:
    # ProtectionEngine and EvaluationEngine each manage their own YuNet
    # instance internally. We pass model_dir so they find the ONNX file.
    model_dir = str(yunet_model.parent)
    protection_eng = ProtectionEngine(model_dir=model_dir)
    evaluation_eng = EvaluationEngine(model_dir=model_dir)

    if protection_eng.yunet_detector is None:
        raise RuntimeError(f"YuNet model unavailable or OpenCV FaceDetectorYN missing: {yunet_model}")

    rows: list[dict[str, Any]] = []
    detection_counts = {"input_images": len(images), "faces_detected": 0}

    for image_path in images:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Unable to read validation image: {image_path}")

        # Check if face is detected at all
        faces_check = protection_eng.detect_faces(image_bgr)
        if not faces_check:
            continue
        detection_counts["faces_detected"] += 1

        for preset in PRESET_ORDER:
            # ── Protection phase with per-stage timing ──
            t_total = time.perf_counter()

            t_detect = time.perf_counter()
            faces = protection_eng.detect_faces(image_bgr)
            t_detect_ms = int((time.perf_counter() - t_detect) * 1000)

            preset_cfg = protection_eng.PRESETS.get(preset, protection_eng.PRESETS["Balanced"])
            warp_scale = preset_cfg["warp_scale"]
            dct_strength = preset_cfg["dct_strength"]

            t_warp = time.perf_counter()
            warped = protection_eng.apply_relative_soft_masked_warp(image_bgr, faces, warp_scale)
            t_warp_ms = int((time.perf_counter() - t_warp) * 1000)

            t_dct = time.perf_counter()
            _ = protection_eng.apply_y_channel_dct_perturbation(warped, dct_strength)
            t_dct_ms = int((time.perf_counter() - t_dct) * 1000)

            t_grid = time.perf_counter()
            # Grid disabled by default per locked spec
            t_grid_ms = int((time.perf_counter() - t_grid) * 1000)

            # Run full protect() for guardrail validation (includes attenuation loop)
            t_guardrail = time.perf_counter()
            protected_bgr_final, prot_meta = protection_eng.protect(
                img_bgr=image_bgr,
                preset_name=preset,
                enable_grid=False,
            )
            t_guardrail_ms = int((time.perf_counter() - t_guardrail) * 1000)

            # ── Evaluation phase ──
            t_eval = time.perf_counter()
            eval_metrics = evaluation_eng.evaluate(
                orig_bgr=image_bgr,
                protected_bgr=protected_bgr_final,
                faces=prot_meta["faces"],
            )
            t_eval_ms = int((time.perf_counter() - t_eval) * 1000)

            t_total_ms = int((time.perf_counter() - t_total) * 1000)

            timings = {
                "face_detection_ms": t_detect_ms,
                "face_warp_ms": t_warp_ms,
                "y_dct_ms": t_dct_ms,
                "grid_ms": t_grid_ms,
                "quality_guardrail_ms": t_guardrail_ms,
                "evaluation_ms": t_eval_ms,
                "total_ms": t_total_ms,
            }

            rows.append(
                {
                    "image": image_path.name,
                    "preset": preset,
                    "guardrail_satisfied": prot_meta["guardrail_satisfied"],
                    "attenuation_factor": prot_meta["attenuation_factor"],
                    "global_ssim": eval_metrics["global_ssim"],
                    "face_ssim": eval_metrics["face_ssim_1_2x"],
                    "psnr": eval_metrics["psnr_db"],
                    "linf": eval_metrics["l_inf_max_shift"],
                    "y_npr": eval_metrics["y_npr_breakdown"],
                    "avg_y_npr": eval_metrics["avg_y_npr_percent"],
                    "alignment": {
                        "landmark_shift_px": eval_metrics["landmark_shift_delta_px"],
                        "alignment_error": eval_metrics["alignment_error_orig"],
                        "alignment_error_protected": eval_metrics["alignment_error_protected"],
                        "alignment_error_pct_change": eval_metrics["alignment_error_pct_change"],
                    },
                    "timings": timings,
                }
            )

    summary: dict[str, dict[str, Any]] = {}
    for preset in PRESET_ORDER:
        preset_rows = [row for row in rows if row["preset"] == preset]
        summary[preset] = {
            "global_ssim": _mean([row["global_ssim"] for row in preset_rows]),
            "face_ssim": _mean([row["face_ssim"] for row in preset_rows]),
            "psnr": _mean([row["psnr"] for row in preset_rows]),
            "linf": _mean([row["linf"] for row in preset_rows]),
            "avg_y_npr": _mean([row["avg_y_npr"] for row in preset_rows]),
            "landmark_shift_px": _mean([row["alignment"]["landmark_shift_px"] for row in preset_rows]),
            "alignment_error": _mean([row["alignment"]["alignment_error"] for row in preset_rows]),
            "alignment_error_pct_change": _mean([row["alignment"]["alignment_error_pct_change"] for row in preset_rows]),
            "guardrail_satisfied_count": sum(1 for row in preset_rows if row["guardrail_satisfied"]),
            "count": len(preset_rows),
        }
        for tier in NPR_ORDER:
            summary[preset][tier] = _mean([row["y_npr"][tier] for row in preset_rows])
        timing_keys = sorted({key for row in preset_rows for key in row["timings"]})
        for key in timing_keys:
            summary[preset][key] = _mean([row["timings"].get(key) for row in preset_rows])

    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "cpu": platform.processor() or "unknown",
        },
        "yunet_model": str(yunet_model),
        "detection_counts": detection_counts,
        "rows": rows,
        "summary": summary,
    }


def print_tables(result: dict[str, Any]) -> None:
    summary = result["summary"]
    metrics = [
        ("Global SSIM", "global_ssim"),
        ("Face SSIM", "face_ssim"),
        ("PSNR", "psnr"),
        ("L-inf", "linf"),
        ("Avg Y-NPR", "avg_y_npr"),
        ("Y-NPR Q95", "Q95"),
        ("Y-NPR Q85", "Q85"),
        ("Y-NPR Q75", "Q75"),
        ("Y-NPR Q60", "Q60"),
        ("Resize+Q75", "Resize075_Q75"),
        ("Landmark delta-px", "landmark_shift_px"),
        ("Alignment error", "alignment_error"),
        ("Alignment err %chg", "alignment_error_pct_change"),
    ]
    print("\nMetric                     Light       Balanced    Aggressive")
    print("------------------------------------------------------------")
    for label, key in metrics:
        print(f"{label:<26} {_fmt(summary['Light'][key]):>10}  {_fmt(summary['Balanced'][key]):>10}  {_fmt(summary['Aggressive'][key]):>10}")

    timing_keys = ["face_detection_ms", "face_warp_ms", "y_dct_ms", "grid_ms", "quality_guardrail_ms", "evaluation_ms", "total_ms"]
    print("\nLatency ms                 Light       Balanced    Aggressive")
    print("------------------------------------------------------------")
    for key in timing_keys:
        print(f"{key:<26} {_fmt(summary['Light'].get(key)):>10}  {_fmt(summary['Balanced'].get(key)):>10}  {_fmt(summary['Aggressive'].get(key)):>10}")

    print("\nGuardrail Results:")
    for preset in PRESET_ORDER:
        s = summary[preset]
        print(f"  {preset}: {s['guardrail_satisfied_count']}/{s['count']} satisfied")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yunet-model", required=True, type=Path)
    parser.add_argument("--images", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.mkdtemp(prefix="varman_phase2_")))
    parser.add_argument("--lfw-limit", type=int, default=6)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lfw").mkdir(parents=True, exist_ok=True)
    images = args.images or _load_optional_lfw_images(args.output_dir / "lfw", args.lfw_limit)
    result = run_validation(args.yunet_model, images, args.output_dir)
    print(json.dumps(result["environment"], indent=2))
    print(json.dumps(result["detection_counts"], indent=2))
    print_tables(result)
    if args.json_output:
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
