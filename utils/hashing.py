import io
import urllib.request

import imagehash
import numpy as np
from PIL import Image


def compute_phash(image_array: np.ndarray) -> str:
    image = Image.fromarray(image_array.astype("uint8"), mode="RGB")
    return str(imagehash.phash(image))


def compute_phash_from_bytes(image_bytes: bytes) -> str | None:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return str(imagehash.phash(image))


def compute_phash_from_url(url: str, timeout: int = 10) -> str | None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return compute_phash_from_bytes(response.read())
