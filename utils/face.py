import importlib
import importlib.util
import logging

import numpy as np

logger = logging.getLogger(__name__)


def extract_face_encoding(image_array: np.ndarray) -> np.ndarray | None:
    if importlib.util.find_spec("face_recognition") is None:
        logger.warning("face_recognition is not installed; skipping face encoding extraction.")
        return None

    face_recognition = importlib.import_module("face_recognition")
    encodings = face_recognition.face_encodings(image_array)
    if not encodings:
        return None
    return encodings[0]
