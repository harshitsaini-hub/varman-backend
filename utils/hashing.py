import io
import logging
from dataclasses import dataclass

import httpx
import imagehash
import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhashCandidate:
    hash_kind: str
    phash: str


def compute_phash(image_array: np.ndarray) -> str:
    return compute_phash_from_image(Image.fromarray(image_array.astype("uint8"), mode="RGB"))


def compute_phash_from_image(image: Image.Image) -> str:
    return str(imagehash.phash(image.convert("RGB")))


def image_array_from_bytes(image_bytes: bytes) -> np.ndarray | None:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (OSError, UnidentifiedImageError):
        logger.warning("Unable to read image: bytes are not a supported image.")
        return None
    return np.array(image)


def compute_phash_from_bytes(image_bytes: bytes) -> str | None:
    image_array = image_array_from_bytes(image_bytes)
    if image_array is None:
        return None
    return compute_phash(image_array)


def compute_phash_from_url(url: str, timeout: int = 10) -> str | None:
    image_bytes = read_image_bytes_from_url(url, timeout=timeout)
    if image_bytes is None:
        return None
    return compute_phash_from_bytes(image_bytes)


def read_image_bytes_from_url(url: str, timeout: int = 10) -> bytes | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            return response.content
    except httpx.RequestError as exc:
        logger.warning(f"Unable to fetch image URL: {url} - {exc}")
        return None


def compute_phash_candidates(image_array: np.ndarray) -> list[PhashCandidate]:
    """Return whole-image and center-crop hashes for screenshot/crop-tolerant matching."""
    image = Image.fromarray(image_array.astype("uint8"), mode="RGB").convert("RGB")
    crop_specs = [
        ("full", 1.0, 1.0),
        ("center_92", 0.92, 0.92),
        ("center_84", 0.84, 0.84),
        ("center_76", 0.76, 0.76),
        ("center_68", 0.68, 0.68),
        ("center_wide_90x70", 0.90, 0.70),
        ("center_tall_70x90", 0.70, 0.90),
    ]

    candidates: list[PhashCandidate] = []
    seen: set[str] = set()
    for hash_kind, width_ratio, height_ratio in crop_specs:
        crop = _center_crop(image, width_ratio, height_ratio)
        phash = compute_phash_from_image(crop)
        if phash not in seen:
            candidates.append(PhashCandidate(hash_kind=hash_kind, phash=phash))
            seen.add(phash)
    return candidates


def _center_crop(image: Image.Image, width_ratio: float, height_ratio: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(width * width_ratio))
    crop_height = max(1, int(height * height_ratio))
    left = max(0, (width - crop_width) // 2)
    upper = max(0, (height - crop_height) // 2)
    return image.crop((left, upper, left + crop_width, upper + crop_height))
