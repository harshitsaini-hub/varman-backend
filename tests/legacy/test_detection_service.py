# type: ignore
# pyright: reportMissingImports=false
# pyright: reportUndefinedVariable=false
from io import BytesIO

import numpy as np
from PIL import Image

from services import detection_service
from services.db_service import PhashMatch
from utils.hashing import compute_phash_candidates


def _image_bytes() -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", (96, 96), color=(240, 240, 240))
    for x in range(24, 72):
        for y in range(24, 72):
            image.putpixel((x, y), (20, 80, 160))
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_compute_phash_candidates_includes_full_and_crops():
    image = np.array(Image.open(BytesIO(_image_bytes())).convert("RGB"))
    candidates = compute_phash_candidates(image)

    assert candidates[0].hash_kind == "full"
    assert len(candidates) > 1
    assert len({candidate.phash for candidate in candidates}) == len(candidates)


def test_detection_uses_region_phash_after_whole_image_miss(monkeypatch):
    def miss_whole_phash(db, phash, threshold):
        return None

    def hit_region_phash(db, phash_candidates, threshold):
        assert phash_candidates
        return PhashMatch(
            user_id="u1",
            phash="abc",
            watermark_id="wm",
            confidence_score=0.9,
            distance=4,
        )

    def fail_if_face_called(image_array):
        raise AssertionError("face fallback should not run after region pHash match")

    monkeypatch.setattr(detection_service.db_service, "lookup_phash_global", miss_whole_phash)
    monkeypatch.setattr(
        detection_service.db_service,
        "lookup_phash_candidates_global",
        hit_region_phash,
    )
    monkeypatch.setattr(detection_service, "extract_face_encoding", fail_if_face_called)

    match = detection_service.detect_suspect_image_bytes(object(), _image_bytes())

    assert match is not None
    assert match.user_id == "u1"
    assert match.detection_method == "region_phash"


def test_detection_uses_face_fallback_after_hash_misses(monkeypatch):
    face_vector = np.zeros(128)

    monkeypatch.setattr(
        detection_service.db_service,
        "lookup_phash_global",
        lambda db, phash, threshold: None,
    )
    monkeypatch.setattr(
        detection_service.db_service,
        "lookup_phash_candidates_global",
        lambda db, phash_candidates, threshold: None,
    )
    monkeypatch.setattr(detection_service, "extract_face_encoding", lambda image_array: face_vector)

    def hit_face(db, submitted_face_vector, threshold):
        assert submitted_face_vector is face_vector
        return PhashMatch(
            user_id="u2",
            phash="def",
            watermark_id="wm2",
            confidence_score=0.8,
            detection_method="face",
            distance=0.33,
        )

    monkeypatch.setattr(detection_service.db_service, "lookup_face_global", hit_face)

    match = detection_service.detect_suspect_image_bytes(object(), _image_bytes())

    assert match is not None
    assert match.user_id == "u2"
    assert match.detection_method == "face"
