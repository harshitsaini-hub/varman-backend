import numpy as np
import cv2
from PIL import Image
from app.protection.protection_engine import ProtectionEngine
from app.protection.evaluation_engine import EvaluationEngine


class NoFaceDetector:
    def detect_faces(self, image_bgr):
        return []


def test_guardrail_reports_unsatisfied_at_minimum_attenuation(monkeypatch, tmp_path):
    image_path = str(tmp_path / "input.jpg")
    protected_path = str(tmp_path / "protected.png")
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(image_path, image)

    # Force fast SSIM in guardrail loop to fail
    engine = ProtectionEngine()
    monkeypatch.setattr(engine, "_compute_fast_ssim", lambda img1, img2: 0.0)

    protected_bgr, meta = engine.protect(image, preset_name="Light", enable_grid=False)

    assert meta["guardrail_satisfied"] is False
    assert meta["attenuation_factor"] <= 0.1


def test_no_face_evaluation_keeps_alignment_metrics_null():
    evaluator = EvaluationEngine()
    metrics = evaluator._compute_alignment_effectiveness(np.zeros((32, 32, 3), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8))
    assert metrics["landmark_shift_delta_px"] == 0.0
    assert metrics["alignment_error_orig"] == 0.0
    assert metrics["alignment_error_pct_change"] == 0.0


def test_alignment_metrics_are_zero_for_identical_real_landmark_geometry():
    evaluator = EvaluationEngine()
    landmarks = [(0, 0), (10, 0), (5, 5), (2, 10), (8, 10)]
    faces = [{"box": (0, 0, 10, 10), "landmarks": landmarks, "score": 0.99}]
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    metrics = evaluator.evaluate(img, img, faces)
    assert metrics["global_ssim"] == 1.0
    assert metrics["face_ssim_1_2x"] == 1.0
