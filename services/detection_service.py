import numpy as np

from config import (
    FACE_MATCH_DISTANCE_THRESHOLD,
    PHASH_MATCH_THRESHOLD,
    REGION_PHASH_MATCH_THRESHOLD,
)
from services import db_service
from utils.face import extract_face_encoding
from utils.hashing import (
    compute_phash_candidates,
    image_array_from_bytes,
    read_image_bytes_from_url,
)


def detect_suspect_image_url(db, image_url: str) -> db_service.PhashMatch | None:
    image_bytes = read_image_bytes_from_url(image_url)
    if image_bytes is None:
        return None
    return detect_suspect_image_bytes(db, image_bytes)


def detect_suspect_image_bytes(db, image_bytes: bytes) -> db_service.PhashMatch | None:
    image_array = image_array_from_bytes(image_bytes)
    if image_array is None:
        return None
    return detect_suspect_image_array(db, image_array)


def detect_suspect_image_array(db, image_array: np.ndarray) -> db_service.PhashMatch | None:
    """Detect a protected image after reposts, screenshots, crops, and face-preserving edits."""
    candidates = compute_phash_candidates(image_array)
    full_phash = next(
        (candidate.phash for candidate in candidates if candidate.hash_kind == "full"),
        None,
    )

    if full_phash is not None:
        match = db_service.lookup_phash_global(db, full_phash, threshold=PHASH_MATCH_THRESHOLD)
        if match:
            match.detection_method = "phash"
            return match

    region_candidates = [
        (candidate.hash_kind, candidate.phash)
        for candidate in candidates
        if candidate.hash_kind != "full"
    ]
    match = db_service.lookup_phash_candidates_global(
        db,
        region_candidates,
        threshold=REGION_PHASH_MATCH_THRESHOLD,
    )
    if match:
        match.detection_method = "region_phash"
        return match

    face_vector = extract_face_encoding(image_array)
    if face_vector is None:
        return None

    match = db_service.lookup_face_global(
        db,
        face_vector,
        threshold=FACE_MATCH_DISTANCE_THRESHOLD,
    )
    if match:
        match.detection_method = "face"
    return match


def describe_match(match: db_service.PhashMatch) -> str:
    if match.detection_method == "face":
        if match.distance is None:
            return "face match"
        return f"face match, distance={match.distance:.3f}"
    if match.detection_method == "region_phash":
        return (
            "region pHash match"
            f", suspect={match.suspect_hash_kind}"
            f", stored={match.matched_hash_kind}"
            f", distance={match.distance:.0f}"
        )
    if match.distance is None:
        return "pHash match"
    return f"pHash match, distance={match.distance:.0f}"
