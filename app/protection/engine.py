import os
import cv2
import logging
from typing import Dict, Any, Tuple, Union
from app.protection.protection_engine import ProtectionEngine
from app.protection.evaluation_engine import EvaluationEngine

logger = logging.getLogger(__name__)

# Singletons for engines
_protection_engine = None
_evaluation_engine = None


def get_protection_engine() -> ProtectionEngine:
    global _protection_engine
    if _protection_engine is None:
        _protection_engine = ProtectionEngine()
    return _protection_engine


def get_evaluation_engine() -> EvaluationEngine:
    global _evaluation_engine
    if _evaluation_engine is None:
        _evaluation_engine = EvaluationEngine()
    return _evaluation_engine


def strength_to_preset(strength: float) -> str:
    """Map numeric strength (0.0 to 1.0) to preset name."""
    if strength < 0.34:
        return "Light"
    if strength > 0.67:
        return "Aggressive"
    return "Balanced"


def protect_image_pipeline(
    original_path: str,
    protected_path: str,
    preset_name: Union[str, float] = "Balanced",
    enable_grid: bool = False,
    strength: float | None = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Main Orchestrator Entrypoint for Varman Image Protection Engine.

    Pipeline Steps:
      1. Load original image using OpenCV (BGR).
      2. Execute ProtectionEngine (Face Detection -> Warp -> Y-DCT -> Grid -> Guardrail).
      3. Save protected output as lossless PNG.
      4. Execute EvaluationEngine (SSIM, Face SSIM, PSNR, Y-NPR breakdown, Landmark Drift).
      5. Return protected file path and scientific metrics dictionary.
    """
    if isinstance(preset_name, (int, float)):
        preset_str = strength_to_preset(float(preset_name))
    elif strength is not None:
        preset_str = strength_to_preset(float(strength))
    else:
        preset_str = str(preset_name).capitalize()

    logger.info(f"[Varman Orchestrator] Starting protection: {original_path} (Preset: {preset_str})")

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original image not found: {original_path}")

    # Load BGR image via OpenCV
    orig_bgr = cv2.imread(original_path)
    if orig_bgr is None:
        raise ValueError(f"Failed to decode image at {original_path}")

    protection_eng = get_protection_engine()
    evaluation_eng = get_evaluation_engine()

    # 1. Execute Protection Pipeline
    protected_bgr, prot_meta = protection_eng.protect(
        img_bgr=orig_bgr,
        preset_name=preset_str,
        enable_grid=enable_grid,
    )

    # 2. Save protected image as lossless PNG
    png_path = os.path.splitext(protected_path)[0] + ".png"
    dir_name = os.path.dirname(png_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    cv2.imwrite(png_path, protected_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    # 3. Execute Evaluation & Metrics Pipeline
    eval_metrics = evaluation_eng.evaluate(
        orig_bgr=orig_bgr,
        protected_bgr=protected_bgr,
        faces=prot_meta["faces"],
    )

    # Combine metadata and metrics
    combined_metrics = {
        "protection_meta": prot_meta,
        "evaluation_metrics": eval_metrics,
        "output_path": png_path,
        "ssim": eval_metrics["global_ssim"],
        "face_ssim": eval_metrics["face_ssim_1_2x"],
        "psnr": eval_metrics["psnr_db"],
        "linf": eval_metrics["l_inf_max_shift"],
        "guardrail_satisfied": prot_meta["guardrail_satisfied"],
        "attenuation_factor": prot_meta["attenuation_factor"],
        "status": "completed",
    }

    logger.info(
        f"[Varman Orchestrator] Protection complete: {png_path} | "
        f"Global SSIM: {eval_metrics['global_ssim']} | "
        f"Face SSIM: {eval_metrics['face_ssim_1_2x']} | "
        f"Avg Y-NPR: {eval_metrics['avg_y_npr_percent']}%"
    )

    return png_path, combined_metrics


def compute_quality_metrics(original_path: str, protected_path: str) -> Tuple[float, float]:
    """Compatibility helper for quality metrics."""
    orig_bgr = cv2.imread(original_path)
    prot_bgr = cv2.imread(protected_path)
    if orig_bgr is None or prot_bgr is None:
        return 0.0, 0.0
    eval_metrics = get_evaluation_engine().evaluate(orig_bgr, prot_bgr, faces=[])
    return float(eval_metrics["global_ssim"]), float(eval_metrics["psnr_db"])
