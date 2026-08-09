import numpy as np

from app.protection.engine import EvaluationEngine, FaceDetection, ProtectionEngine, YuNetFaceDetector


class NoFaceDetector(YuNetFaceDetector):
    def __init__(self):
        pass

    def detect(self, image_bgr):
        return None


def test_guardrail_reports_unsatisfied_at_minimum_attenuation(monkeypatch, tmp_path):
    image_path = tmp_path / "input.png"
    protected_path = tmp_path / "protected.png"
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    from PIL import Image

    Image.fromarray(image).save(image_path)

    def always_fails(self, original, protected, face):
        return {"global_ssim": 0.0, "face_ssim": None, "psnr": 0.0, "linf": 0.0}

    monkeypatch.setattr("app.protection.engine.EvaluationEngine.compute_quality", always_fails)

    result = ProtectionEngine(detector=NoFaceDetector(), min_attenuation=0.5).protect(
        str(image_path),
        str(protected_path),
        "light",
    )

    assert result["guardrail_satisfied"] is False
    assert result["attenuation_factor"] < 0.5


def test_no_face_evaluation_keeps_alignment_metrics_null():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    metrics = EvaluationEngine(detector=NoFaceDetector())._alignment_metrics(None, None)
    assert metrics == {
        "landmark_shift_px": None,
        "alignment_error": None,
        "alignment_error_pct_change": None,
    }


def test_alignment_metrics_are_zero_for_identical_real_landmark_geometry():
    landmarks = np.array([[0, 0], [10, 0], [5, 5], [2, 10], [8, 10]], dtype=np.float32)
    face = FaceDetection((0, 0, 10, 10), landmarks, 0.99)
    metrics = EvaluationEngine(detector=NoFaceDetector())._alignment_metrics(face, face)
    assert metrics["landmark_shift_px"] == 0.0
    assert metrics["alignment_error"] == 0.0
    assert metrics["alignment_error_pct_change"] == 0.0
