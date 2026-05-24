import io

import numpy as np
from PIL import Image

from services.amor_service import extract_watermark_bits, _uuid_to_bits, _bits_match


def simulate_platform_compression(image_array: np.ndarray, quality: int = 75) -> np.ndarray:
    """Simulate Instagram/Telegram-level JPEG compression."""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))


def validate_armor(original_armored: np.ndarray, watermark_id: str, wm_length: int = 32) -> dict:
    compressed = simulate_platform_compression(original_armored, quality=75)
    expected_bits = _uuid_to_bits(watermark_id)
    recovered_bits = extract_watermark_bits(compressed)

    survived = _bits_match(expected_bits, recovered_bits)
    matched = sum(1 for e, r in zip(expected_bits, recovered_bits) if int(e) == int(r)) if recovered_bits else 0

    return {
        "watermark_survived_compression": survived,
        "compression_quality_tested": 75,
        "bits_matched": f"{matched}/32",
        "warning": None if survived else f"Only {matched}/32 bits recovered. Watermark may not survive platform compression.",
    }