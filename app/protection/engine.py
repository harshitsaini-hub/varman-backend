import os
import cv2
import logging
import threading
from typing import Dict, Any, Tuple, Union, Optional
from app.protection.protection_engine import ProtectionEngine
from app.protection.evaluation_engine import EvaluationEngine

logger = logging.getLogger(__name__)

# Singletons for engines
_protection_engine = None
_evaluation_engine = None
_paste_aware_engine = None


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


def get_paste_aware_engine():
    global _paste_aware_engine
    if _paste_aware_engine is None:
        import torch
        from app.protection.paste_aware_engine import PasteAwareProtectionEngine
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this host for Hard Mode (Server GPU).")
        _paste_aware_engine = PasteAwareProtectionEngine()
    return _paste_aware_engine


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
    mode: str = "fast",
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Main Orchestrator Entrypoint for Varman Image Protection Engine.

    Modes:
      - 'fast': CPU YuNet micro-warping + Y-DCT frequency perturbation (~0.8s).
      - 'hard': Server GPU paste-aware EOT-PGD dual-surrogate perturbation (~2.0s).
    """
    mode_clean = str(mode).strip().lower()

    if isinstance(preset_name, (int, float)):
        preset_str = strength_to_preset(float(preset_name))
    elif strength is not None:
        preset_str = strength_to_preset(float(strength))
    else:
        preset_str = str(preset_name)

    logger.info(f"[Varman Orchestrator] Starting protection: {original_path} (Mode: {mode_clean}, Preset: {preset_str})")

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original image not found: {original_path}")

    # Load BGR image via OpenCV
    orig_bgr = cv2.imread(original_path)
    if orig_bgr is None:
        raise ValueError(f"Failed to decode image at {original_path}")

    protection_eng = get_protection_engine()
    evaluation_eng = get_evaluation_engine()

    if mode_clean == "hard":
        # ── Hard Mode (Server GPU Paste-Aware PGD) ──────────────────────
        paste_aware_eng = get_paste_aware_engine()
        faces = protection_eng.detect_faces(orig_bgr)

        protected_bgr, prot_meta = paste_aware_eng.protect_image(
            img_bgr=orig_bgr,
            faces=faces,
            steps=25,
            mask_scale=1.15,
            w_fn=1.0,
            feather_sigma=4.0,
            cancel_event=cancel_event,
        )
        prot_meta["faces"] = faces
        prot_meta["guardrail_satisfied"] = True
        prot_meta["attenuation_factor"] = 1.0
        # Dynamic parameters used by the engine (eot_steps_executed, eps_budget,
        # alpha_step, w_fn, mask_scale) are directly populated in prot_meta by the engine.
    else:
        # ── Fast Mode (CPU YuNet + DCT) ─────────────────────────────────
        preset_clean = preset_str.capitalize()
        protected_bgr, prot_meta = protection_eng.protect(
            img_bgr=orig_bgr,
            preset_name=preset_clean,
            enable_grid=enable_grid,
        )
        prot_meta["eot_steps_executed"] = None
        prot_meta["eps_budget"] = None
        prot_meta["alpha_step"] = None
        prot_meta["w_fn"] = None
        prot_meta["mask_scale"] = None

    # 2. Save protected image as lossless PNG
    png_path = os.path.splitext(protected_path)[0] + ".png"
    dir_name = os.path.dirname(png_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    cv2.imwrite(png_path, protected_bgr, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

    # 3. Execute Evaluation & Metrics Pipeline
    faces_to_eval = prot_meta.get("faces", [])
    eval_metrics = evaluation_eng.evaluate(
        orig_bgr=orig_bgr,
        protected_bgr=protected_bgr,
        faces=faces_to_eval,
    )

    combined_metrics = {
        "protection_meta": prot_meta,
        "evaluation_metrics": eval_metrics,
        "output_path": png_path,
        "ssim": eval_metrics["global_ssim"],
        "face_ssim": eval_metrics["face_ssim_1_2x"],
        "psnr": eval_metrics["psnr_db"],
        "linf": eval_metrics["l_inf_max_shift"],
        "guardrail_satisfied": prot_meta.get("guardrail_satisfied", True),
        "attenuation_factor": prot_meta.get("attenuation_factor", 1.0),
        "status": "completed",
        "mode": mode_clean,
    }

    logger.info(
        f"[Varman Orchestrator] Protection complete: {png_path} | Mode: {mode_clean} | "
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
