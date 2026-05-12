import io
import logging
import urllib.request

import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


def compute_phash(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array.astype("uint8"), mode="RGB")
    return str(imagehash.phash(image))


def compute_phash_from_bytes(image_bytes: bytes) -> str | None:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, UnidentifiedImageError):
        logger.warning("Unable to compute pHash: bytes are not a supported image.")
        return None
    return str(imagehash.phash(image))


def compute_phash_from_url(url: str, timeout: int = 10) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return compute_phash_from_bytes(response.read())
    except OSError:
        logger.warning("Unable to compute pHash from URL: %s", url)
        return None
