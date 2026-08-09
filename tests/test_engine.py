import os

import numpy as np
from PIL import Image

from app.protection.engine import (
    EvaluationEngine,
    FaceDetection,
    PRESETS,
    ProtectionEngine,
    YuNetFaceDetector,
    protect_image_pipeline,
)


class NoFaceDetector(YuNetFaceDetector):
    def __init__(self):
        pass

    @property
    def available(self):
        return False

    def detect(self, image_bgr):
        return None


class FixedFaceDetector(YuNetFaceDetector):
    def __init__(self, original_landmarks=None, protected_landmarks=None):
        self.calls = 0
        self.original_landmarks = original_landmarks
        self.protected_landmarks = protected_landmarks or original_landmarks

    @property
    def available(self):
        return True

    def detect(self, image_bgr):
        self.calls += 1
        landmarks = self.original_landmarks if self.calls == 1 else self.protected_landmarks
        if landmarks is None:
            landmarks = np.array(
                [[48, 50], [80, 50], [64, 66], [52, 84], [76, 84]],
                dtype=np.float32,
            )
        return FaceDetection((32.0, 32.0, 64.0, 64.0), landmarks.astype(np.float32), 0.99)


def _make_test_image(path):
    yy, xx = np.mgrid[0:128, 0:128]
    pixels = np.stack([
        (xx * 2) % 256,
        (yy * 2) % 256,
        ((xx + yy) * 3) % 256,
    ], axis=2).astype(np.uint8)
    Image.fromarray(pixels).save(path)


def test_no_face_still_applies_y_channel_dct_and_guardrail(tmp_path):
    original = tmp_path / "input.png"
    protected = tmp_path / "output.png"
    _make_test_image(original)

    result = ProtectionEngine(detector=NoFaceDetector()).protect(str(original), str(protected), "balanced")

    assert result["face_detected"] is False
    assert result["output_path"].endswith(".png")
    assert os.path.exists(result["output_path"])
    assert result["enable_grid_layer"] is False
    assert "guardrail_satisfied" in result
    assert "attenuation_factor" in result
    assert result["global_ssim"] >= PRESETS["balanced"].global_ssim_threshold
    assert result["linf"] > 0


def test_evaluation_exposes_required_y_npr_and_quality_metrics(tmp_path):
    original = tmp_path / "input.png"
    protected = tmp_path / "output.png"
    _make_test_image(original)
    result = ProtectionEngine(detector=NoFaceDetector()).protect(str(original), str(protected), "light")

    metrics = EvaluationEngine(detector=NoFaceDetector()).evaluate(str(original), result["output_path"])

    assert set(metrics["y_npr"]) == {"jpeg_q95", "jpeg_q85", "jpeg_q75", "jpeg_q60", "resize_0_75_jpeg_q75"}
    assert metrics["global_ssim"] >= PRESETS["light"].global_ssim_threshold
    assert metrics["face_ssim"] is None
    assert metrics["psnr"] > 0
    assert metrics["linf"] > 0
    assert metrics["alignment"] == {
        "landmark_shift_px": None,
        "alignment_error": None,
        "alignment_error_pct_change": None,
    }


def test_pipeline_orchestrates_protection_and_evaluation(tmp_path):
    original = tmp_path / "input.png"
    protected = tmp_path / "output.png"
    _make_test_image(original)

    result = protect_image_pipeline(str(original), str(protected), strength=0.2)

    assert result["preset"] == "light"
    assert result["ssim"] == result["global_ssim"]
    assert "y_npr" in result
    assert "alignment" in result
    assert result["status"] == "completed"
    assert result["timings"]["total_ms"] >= 0


def test_quality_metrics_are_full_image_not_y_only():
    original = np.zeros((16, 16, 3), dtype=np.uint8)
    protected = original.copy()
    protected[:, :, 2] = 120

    metrics = EvaluationEngine(detector=NoFaceDetector()).compute_quality(original, protected, None)

    assert metrics["global_ssim"] < 1.0
    assert metrics["psnr"] < float("inf")
    assert metrics["linf"] == 120.0


def test_face_warp_displacement_is_scaled_by_face_width():
    engine = ProtectionEngine(detector=NoFaceDetector())
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(128, dtype=np.uint8)
    small_face = FaceDetection((32.0, 32.0, 40.0, 40.0), np.zeros((5, 2), dtype=np.float32), 0.99)
    large_face = FaceDetection((16.0, 16.0, 80.0, 80.0), np.zeros((5, 2), dtype=np.float32), 0.99)

    small = engine._apply_face_warp(image, small_face, 0.015)
    large = engine._apply_face_warp(image, large_face, 0.015)

    assert np.max(np.abs(large.astype(np.int16) - image.astype(np.int16))) > np.max(
        np.abs(small.astype(np.int16) - image.astype(np.int16))
    )


def test_expanded_face_ssim_uses_1_2_bbox_region():
    evaluator = EvaluationEngine(detector=NoFaceDetector())
    original = np.zeros((100, 100, 3), dtype=np.uint8)
    protected = original.copy()
    face = FaceDetection((30.0, 30.0, 40.0, 40.0), np.zeros((5, 2), dtype=np.float32), 0.99)
    protected[26:74, 26:74, :] = 255

    face_ssim = evaluator._expanded_face_ssim(original, protected, face)

    assert face_ssim is not None
    assert face_ssim < 0.1


def test_presets_and_guardrail_regenerate_with_attenuation(tmp_path):
    original = tmp_path / "input.png"
    _make_test_image(original)

    light = ProtectionEngine(detector=NoFaceDetector()).protect(str(original), str(tmp_path / "light.png"), "light")
    aggressive = ProtectionEngine(detector=NoFaceDetector()).protect(str(original), str(tmp_path / "aggressive.png"), "aggressive")
    forced = ProtectionEngine(detector=NoFaceDetector(), min_attenuation=0.25).protect(
        str(original),
        str(tmp_path / "forced.png"),
        "light",
    )

    light_linf = light["linf"]
    aggressive_linf = aggressive["linf"]
    assert aggressive_linf >= light_linf
    assert forced["guardrail_satisfied"] is True
    assert forced["attenuation_factor"] == 1.0


def test_guardrail_attenuates_and_regenerates_candidates(monkeypatch, tmp_path):
    original = tmp_path / "input.png"
    _make_test_image(original)
    strengths = []

    class RecordingEngine(ProtectionEngine):
        def _apply_y_channel_dct(self, image_bgr, strength):
            strengths.append(strength)
            return super()._apply_y_channel_dct(image_bgr, strength)

    calls = []

    def fake_quality(self, original_image, protected_image, face):
        calls.append(1)
        if len(calls) == 1:
            return {"global_ssim": 0.0, "face_ssim": None, "psnr": 1.0, "linf": 1.0}
        return {"global_ssim": 1.0, "face_ssim": None, "psnr": 99.0, "linf": 1.0}

    monkeypatch.setattr("app.protection.engine.EvaluationEngine.compute_quality", fake_quality)

    result = RecordingEngine(detector=NoFaceDetector()).protect(str(original), str(tmp_path / "attenuated.png"), "light")

    assert result["guardrail_satisfied"] is True
    assert result["attenuation_factor"] == 0.5
    assert strengths == [PRESETS["light"].dct_strength, PRESETS["light"].dct_strength * 0.5]


def test_y_npr_formula_uses_y_channel_baseline_corrected_norm(monkeypatch):
    evaluator = EvaluationEngine(detector=NoFaceDetector())
    original = np.zeros((1, 2, 3), dtype=np.uint8)
    protected = original.copy()
    protected[0, 0, :] = 10

    y_o = np.dstack([np.array([[0.0, 0.0]], dtype=np.float32)] * 3)
    y_p = np.dstack([np.array([[3.0, 4.0]], dtype=np.float32)] * 3)
    compressed_original = np.array([[1.0, 1.0]], dtype=np.float32)
    compressed_protected = np.array([[4.0, 5.0]], dtype=np.float32)

    monkeypatch.setattr(
        "app.protection.engine.cv2.cvtColor",
        lambda image, code: y_o if image is original else y_p,
    )
    monkeypatch.setattr(
        evaluator,
        "_jpeg_roundtrip_y",
        lambda image, quality, resize=None: compressed_original if image is original else compressed_protected,
    )

    results = evaluator._y_npr_suite(original, protected)

    assert all(value == 100.0 for value in results.values())


def test_alignment_metrics_compare_original_and_protected_landmarks():
    original_landmarks = np.array([[0, 0], [10, 0], [5, 5], [2, 10], [8, 10]], dtype=np.float32)
    protected_landmarks = original_landmarks.copy()
    protected_landmarks[2] += np.array([3, 0], dtype=np.float32)

    metrics = EvaluationEngine(detector=NoFaceDetector())._alignment_metrics(
        FaceDetection((0, 0, 10, 10), original_landmarks, 0.99),
        FaceDetection((0, 0, 10, 10), protected_landmarks, 0.99),
    )

    assert metrics["landmark_shift_px"] > 0
    assert metrics["alignment_error"] > 0
    assert metrics["alignment_error_pct_change"] > 0
