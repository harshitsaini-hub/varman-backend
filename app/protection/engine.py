"""Varman protection/evaluation orchestration.

This module intentionally keeps the locked architecture separated:
ProtectionEngine generates the protected image, EvaluationEngine measures it,
and protect_image_pipeline orchestrates both for the API.
"""

from __future__ import annotations

import io
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

PRESET_THRESHOLDS = {
    "light": 0.95,
    "balanced": 0.92,
    "aggressive": 0.90,
}
FACE_SSIM_THRESHOLD = 0.90
DEFAULT_PRESET = "balanced"


@dataclass(frozen=True)
class FaceDetection:
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray
    score: float


@dataclass(frozen=True)
class ProtectionPreset:
    name: str
    global_ssim_threshold: float
    warp_fraction: float
    dct_strength: float
    enable_grid_layer: bool = False


PRESETS = {
    "light": ProtectionPreset("light", PRESET_THRESHOLDS["light"], 0.005, 1.4),
    "balanced": ProtectionPreset("balanced", PRESET_THRESHOLDS["balanced"], 0.010, 2.2),
    "aggressive": ProtectionPreset("aggressive", PRESET_THRESHOLDS["aggressive"], 0.015, 3.0),
}


class YuNetFaceDetector:
    """YuNet ONNX face detector with conditional multi-scale YuNet fallback."""

    def __init__(self, model_path: str | None = None, confidence_threshold: float = 0.9):
        self.model_path = model_path or getattr(settings, "yunet_model_path", "weights/face_detection_yunet_2023mar.onnx")
        self.confidence_threshold = confidence_threshold
        self._available = bool(os.path.exists(self.model_path) and hasattr(cv2, "FaceDetectorYN_create"))

    @property
    def available(self) -> bool:
        return self._available

    def detect(self, image_bgr: np.ndarray) -> FaceDetection | None:
        if not self._available:
            logger.warning("YuNet model unavailable at %s; face warp/alignment metrics disabled", self.model_path)
            return None
        h, w = image_bgr.shape[:2]
        primary = self._detect_at_size(image_bgr, (w, h), 1.0)
        if primary is not None:
            return primary
        for scale in (1.25, 1.5, 2.0, 0.75):
            resized = cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
            found = self._detect_at_size(resized, (resized.shape[1], resized.shape[0]), scale)
            if found is not None:
                return found
        return None

    def _detect_at_size(self, image_bgr: np.ndarray, size: tuple[int, int], scale: float) -> FaceDetection | None:
        detector = cv2.FaceDetectorYN_create(self.model_path, "", size, self.confidence_threshold, 0.3, 5000)
        _, faces = detector.detect(image_bgr)
        if faces is None or len(faces) == 0:
            return None
        face = max(faces, key=lambda row: row[-1])
        x, y, bw, bh = (float(v) / scale for v in face[:4])
        landmarks = (face[4:14].reshape(5, 2) / scale).astype(np.float32)
        return FaceDetection((x, y, bw, bh), landmarks, float(face[-1]))


class ProtectionEngine:
    """Generate protected images using the locked Varman protection pipeline."""

    def __init__(self, detector: YuNetFaceDetector | None = None, min_attenuation: float = 0.125):
        self.detector = detector or YuNetFaceDetector()
        self.min_attenuation = min_attenuation

    def protect(self, original_path: str, protected_path: str, preset_name: str = DEFAULT_PRESET) -> dict[str, Any]:
        image_bgr = cv2.imread(original_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Unable to read image: {original_path}")
        preset = PRESETS.get(preset_name.lower(), PRESETS[DEFAULT_PRESET])
        detection_start = time.perf_counter()
        face = self.detector.detect(image_bgr)
        detection_ms = int((time.perf_counter() - detection_start) * 1000)
        timings: dict[str, int] = {}
        t0 = time.perf_counter()
        attenuation = 1.0
        best = image_bgr.copy()
        best_metrics: dict[str, float | None] = {}
        guardrail_satisfied = False
        while attenuation >= self.min_attenuation - 1e-9:
            candidate = image_bgr.copy()
            tw = time.perf_counter()
            if face is not None:
                candidate = self._apply_face_warp(
                    candidate,
                    face,
                    preset.warp_fraction,
                    attenuation,
                )
            timings["face_warp_ms"] = int((time.perf_counter() - tw) * 1000)
            td = time.perf_counter()
            candidate = self._apply_y_channel_dct(candidate, preset.dct_strength * attenuation)
            timings["y_dct_ms"] = int((time.perf_counter() - td) * 1000)
            tg = time.perf_counter()
            if preset.enable_grid_layer:
                candidate = self._apply_grid_layer(candidate, attenuation)
            timings["grid_ms"] = int((time.perf_counter() - tg) * 1000)
            metrics = EvaluationEngine(detector=self.detector).compute_quality(image_bgr, candidate, face)
            best, best_metrics = candidate, metrics
            global_ok = (metrics["global_ssim"] or 0.0) >= preset.global_ssim_threshold
            face_ok = metrics["face_ssim"] is None or metrics["face_ssim"] >= FACE_SSIM_THRESHOLD
            if global_ok and face_ok:
                guardrail_satisfied = True
                break
            attenuation *= 0.5
        timings["quality_guardrail_ms"] = int((time.perf_counter() - t0) * 1000)
        timings["face_detection_ms"] = detection_ms
        out_path = os.path.splitext(protected_path)[0] + ".png"
        Image.fromarray(cv2.cvtColor(best, cv2.COLOR_BGR2RGB)).save(out_path)
        return {
            "output_path": out_path,
            "face_detected": face is not None,
            "face_bbox": face.bbox if face else None,
            "guardrail_satisfied": guardrail_satisfied,
            "attenuation_factor": attenuation,
            "preset": preset.name,
            "enable_grid_layer": preset.enable_grid_layer,
            **best_metrics,
            "timings": timings,
        }

    def _apply_face_warp(
        self,
        image_bgr: np.ndarray,
        face: FaceDetection,
        displacement_fraction: float,
        attenuation: float = 1.0,
    ) -> np.ndarray:
        x, y, w, h = face.bbox
        displacement_px = w * displacement_fraction * attenuation
        yy, xx = np.mgrid[0:image_bgr.shape[0], 0:image_bgr.shape[1]].astype(np.float32)
        cx, cy = x + w / 2.0, y + h / 2.0
        sigma_x, sigma_y = max(w * 0.42, 1.0), max(h * 0.42, 1.0)
        mask = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x**2) + ((yy - cy) ** 2) / (2 * sigma_y**2))).astype(np.float32)
        dx = displacement_px * np.sin((yy - cy) / max(h, 1.0) * math.pi) * mask
        dy = displacement_px * np.cos((xx - cx) / max(w, 1.0) * math.pi) * mask
        map_x = (xx + dx).astype(np.float32)
        map_y = (yy + dy).astype(np.float32)
        return cv2.remap(image_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)

    def _apply_y_channel_dct(self, image_bgr: np.ndarray, strength: float) -> np.ndarray:
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        y = ycrcb[:, :, 0]
        h, w = y.shape
        out = y.copy()
        for r in range(0, h - h % 8, 8):
            for c in range(0, w - w % 8, 8):
                block = y[r:r+8, c:c+8] - 128.0
                coeff = cv2.dct(block)
                phase = 1.0 if ((r // 8) * 3 + (c // 8) * 5) % 2 == 0 else -1.0
                spatial = 0.65 + 0.35 * math.sin((r + 17) * 0.071 + (c + 11) * 0.043)
                for u, v in ((2, 3), (3, 2), (3, 4), (4, 3)):
                    coeff[u, v] += phase * strength * spatial * (1.0 + 0.1 * (u - v))
                out[r:r+8, c:c+8] = cv2.idct(coeff) + 128.0
        ycrcb[:, :, 0] = np.clip(out, 0, 255)
        return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)

    def _apply_grid_layer(self, image_bgr: np.ndarray, attenuation: float) -> np.ndarray:
        out = image_bgr.copy().astype(np.int16)
        out[::32, :, :] += int(2 * attenuation)
        out[:, ::32, :] -= int(2 * attenuation)
        return np.clip(out, 0, 255).astype(np.uint8)


class EvaluationEngine:
    """Independently evaluate protected images produced by ProtectionEngine."""

    def __init__(self, detector: YuNetFaceDetector | None = None):
        self.detector = detector or YuNetFaceDetector()

    def evaluate(self, original_path: str, protected_path: str) -> dict[str, Any]:
        original = cv2.imread(original_path, cv2.IMREAD_COLOR)
        protected = cv2.imread(protected_path, cv2.IMREAD_COLOR)
        if original is None or protected is None:
            raise ValueError("Unable to read original or protected image for evaluation")
        if original.shape != protected.shape:
            protected = cv2.resize(protected, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
        original_face = self.detector.detect(original)
        protected_face = self.detector.detect(protected)
        quality = self.compute_quality(original, protected, original_face)
        return {**quality, "y_npr": self._y_npr_suite(original, protected), "alignment": self._alignment_metrics(original_face, protected_face)}

    def compute_quality(self, original: np.ndarray, protected: np.ndarray, face: FaceDetection | None) -> dict[str, float | None]:
        return {
            "global_ssim": self._ssim(original, protected),
            "face_ssim": self._expanded_face_ssim(original, protected, face),
            "psnr": self._psnr(original, protected),
            "linf": float(np.max(np.abs(original.astype(np.int16) - protected.astype(np.int16)))),
        }

    def _expanded_face_ssim(self, original: np.ndarray, protected: np.ndarray, face: FaceDetection | None) -> float | None:
        if face is None:
            return None
        x, y, w, h = face.bbox
        cx, cy = x + w / 2, y + h / 2
        ew, eh = w * 1.2, h * 1.2
        x0, y0 = max(0, int(cx - ew / 2)), max(0, int(cy - eh / 2))
        x1, y1 = min(original.shape[1], int(cx + ew / 2)), min(original.shape[0], int(cy + eh / 2))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return self._ssim(original[y0:y1, x0:x1], protected[y0:y1, x0:x1])

    def _ssim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.ndim == 3:
            return float(np.mean([self._ssim(a[:, :, channel], b[:, :, channel]) for channel in range(a.shape[2])]))
        a = a.astype(np.float64); b = b.astype(np.float64)
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu_a = cv2.GaussianBlur(a, (7, 7), 1.5); mu_b = cv2.GaussianBlur(b, (7, 7), 1.5)
        sigma_a = cv2.GaussianBlur(a*a, (7, 7), 1.5) - mu_a*mu_a
        sigma_b = cv2.GaussianBlur(b*b, (7, 7), 1.5) - mu_b*mu_b
        sigma_ab = cv2.GaussianBlur(a*b, (7, 7), 1.5) - mu_a*mu_b
        score = ((2*mu_a*mu_b + c1) * (2*sigma_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2))
        return float(np.mean(score))

    def _psnr(self, a: np.ndarray, b: np.ndarray) -> float:
        mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
        return float("inf") if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse)))

    def _jpeg_roundtrip_y(self, image_bgr: np.ndarray, quality: int, resize: float | None = None) -> np.ndarray:
        source = image_bgr
        if resize is not None:
            h, w = image_bgr.shape[:2]
            small = cv2.resize(image_bgr, (max(1, int(w * resize)), max(1, int(h * resize))), interpolation=cv2.INTER_AREA)
            source = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        arr = np.array(Image.open(buf).convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)

    def _y_npr_suite(self, original: np.ndarray, protected: np.ndarray) -> dict[str, float]:
        y_o = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        y_p = cv2.cvtColor(protected, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        denom = float(np.linalg.norm(y_p - y_o))
        tiers = {"jpeg_q95": (95, None), "jpeg_q85": (85, None), "jpeg_q75": (75, None), "jpeg_q60": (60, None), "resize_0_75_jpeg_q75": (75, 0.75)}
        if denom == 0:
            return {name: 0.0 for name in tiers}
        results = {}
        for name, (quality, resize) in tiers.items():
            c0 = self._jpeg_roundtrip_y(original, quality, resize)
            c = self._jpeg_roundtrip_y(protected, quality, resize)
            results[name] = float(np.linalg.norm(c - c0) / denom * 100.0)
        return results

    def _alignment_metrics(self, original_face: FaceDetection | None, protected_face: FaceDetection | None) -> dict[str, float | None]:
        if original_face is None or protected_face is None:
            return {"landmark_shift_px": None, "alignment_error": None, "alignment_error_pct_change": None}
        orig = original_face.landmarks.astype(np.float32)
        prot = protected_face.landmarks.astype(np.float32)
        shifts = np.linalg.norm(prot - orig, axis=1)
        aligned_prot = self._similarity_align(prot, orig)
        alignment_error = float(np.sqrt(np.mean(np.sum((aligned_prot - orig) ** 2, axis=1))))
        eye_distance = float(np.linalg.norm(orig[1] - orig[0]))
        pct = None if eye_distance == 0 else float(alignment_error / eye_distance * 100.0)
        return {
            "landmark_shift_px": float(np.mean(shifts)),
            "alignment_error": alignment_error,
            "alignment_error_pct_change": pct,
        }

    def _similarity_align(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_center = source.mean(axis=0, keepdims=True)
        target_center = target.mean(axis=0, keepdims=True)
        source_zero = source - source_center
        target_zero = target - target_center
        source_norm = float(np.linalg.norm(source_zero))
        target_norm = float(np.linalg.norm(target_zero))
        if source_norm == 0 or target_norm == 0:
            return np.repeat(target_center, len(source), axis=0)
        source_unit = source_zero / source_norm
        target_unit = target_zero / target_norm
        u, _, vt = np.linalg.svd(source_unit.T @ target_unit)
        rotation = u @ vt
        scale = target_norm / source_norm
        return (source_zero @ rotation) * scale + target_center


def strength_to_preset(strength: float) -> str:
    if strength < 0.34:
        return "light"
    if strength > 0.67:
        return "aggressive"
    return "balanced"


def protect_image_pipeline(original_path: str, protected_path: str, strength: float = 0.5) -> dict[str, Any]:
    """Orchestrate protection then independent evaluation."""
    t0 = time.perf_counter()
    preset = strength_to_preset(strength)
    detector = YuNetFaceDetector()
    protection = ProtectionEngine(detector=detector).protect(original_path, protected_path, preset)
    timings = protection.pop("timings", {})
    evaluation_start = time.perf_counter()
    evaluation = EvaluationEngine(detector=detector).evaluate(original_path, protection["output_path"])
    timings["evaluation_ms"] = int((time.perf_counter() - evaluation_start) * 1000)
    timings["total_ms"] = int((time.perf_counter() - t0) * 1000)
    result = {**protection, **evaluation, "ssim": evaluation["global_ssim"], "status": "completed", "timings": timings}
    return result


def compute_quality_metrics(original_path: str, protected_path: str) -> tuple[float, float]:
    metrics = EvaluationEngine().evaluate(original_path, protected_path)
    return float(metrics["global_ssim"]), float(metrics["psnr"])
