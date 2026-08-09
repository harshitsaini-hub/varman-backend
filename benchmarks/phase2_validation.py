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
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.protection.engine import EvaluationEngine, ProtectionEngine, YuNetFaceDetector

PRESET_ORDER = ("light", "balanced", "aggressive")
NPR_ORDER = ("jpeg_q95", "jpeg_q85", "jpeg_q75", "jpeg_q60", "resize_0_75_jpeg_q75")


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
    detector = YuNetFaceDetector(model_path=str(yunet_model), confidence_threshold=0.9)
    if not detector.available:
        raise RuntimeError(f"YuNet model unavailable or OpenCV FaceDetectorYN missing: {yunet_model}")

    rows: list[dict[str, Any]] = []
    detection_counts = {"input_images": len(images), "faces_detected": 0}
    for image_path in images:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Unable to read validation image: {image_path}")
        if detector.detect(image_bgr) is None:
            continue
        detection_counts["faces_detected"] += 1
        for preset in PRESET_ORDER:
            protected_path = output_dir / f"{image_path.stem}_{preset}.png"
            start = time.perf_counter()
            protection = ProtectionEngine(detector=detector).protect(str(image_path), str(protected_path), preset)
            evaluation = EvaluationEngine(detector=detector).evaluate(str(image_path), protection["output_path"])
            total_ms = int((time.perf_counter() - start) * 1000)
            rows.append(
                {
                    "image": image_path.name,
                    "preset": preset,
                    "guardrail_satisfied": protection["guardrail_satisfied"],
                    "attenuation_factor": protection["attenuation_factor"],
                    "global_ssim": evaluation["global_ssim"],
                    "face_ssim": evaluation["face_ssim"],
                    "psnr": evaluation["psnr"],
                    "linf": evaluation["linf"],
                    "y_npr": evaluation["y_npr"],
                    "alignment": evaluation["alignment"],
                    "timings": {**protection["timings"], "evaluation_ms": total_ms - protection["timings"].get("quality_guardrail_ms", 0), "total_ms": total_ms},
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
            "landmark_shift_px": _mean([row["alignment"]["landmark_shift_px"] for row in preset_rows]),
            "alignment_error": _mean([row["alignment"]["alignment_error"] for row in preset_rows]),
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
        ("L∞", "linf"),
        ("Y-NPR Q95", "jpeg_q95"),
        ("Y-NPR Q85", "jpeg_q85"),
        ("Y-NPR Q75", "jpeg_q75"),
        ("Y-NPR Q60", "jpeg_q60"),
        ("Resize+Q75", "resize_0_75_jpeg_q75"),
        ("Landmark Δpx", "landmark_shift_px"),
        ("Alignment error", "alignment_error"),
    ]
    print("\nMetric                     Light       Balanced    Aggressive")
    print("------------------------------------------------------------")
    for label, key in metrics:
        print(f"{label:<26} {_fmt(summary['light'][key]):>10}  {_fmt(summary['balanced'][key]):>10}  {_fmt(summary['aggressive'][key]):>10}")

    timing_keys = ["face_detection_ms", "face_warp_ms", "y_dct_ms", "grid_ms", "quality_guardrail_ms", "evaluation_ms", "total_ms"]
    print("\nLatency ms                 Light       Balanced    Aggressive")
    print("------------------------------------------------------------")
    for key in timing_keys:
        print(f"{key:<26} {_fmt(summary['light'].get(key)):>10}  {_fmt(summary['balanced'].get(key)):>10}  {_fmt(summary['aggressive'].get(key)):>10}")


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
