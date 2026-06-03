import io

import numpy as np
from PIL import Image

from services.amor_service import extract_watermark


def simulate_platform_compression(image_array: np.ndarray, quality: int = 75) -> np.ndarray:
    """Simulate Instagram/Telegram-level JPEG compression."""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))


def validate_armor(original_armored: np.ndarray, watermark_id: str, wm_length: int = 32) -> dict:
    compressed = simulate_platform_compression(original_armored, quality=75)
    expected_watermark = watermark_id.replace("-", "")[:4]
    recovered_watermark = extract_watermark(compressed, len(expected_watermark))

    survived = expected_watermark in recovered_watermark
    matched = (
        sum(
            1
            for expected_char, recovered_char in zip(
                expected_watermark,
                recovered_watermark,
                strict=False,
            )
            if expected_char == recovered_char
        )
        if recovered_watermark
        else 0
    )
    expected_length = len(expected_watermark)

    return {
        "watermark_survived_compression": survived,
        "compression_quality_tested": 75,
        "bits_matched": f"{matched}/{expected_length}",
        "warning": (
            None
            if survived
            else (
                f"Only {matched}/{expected_length} watermark characters recovered. "
                "Watermark may not survive platform compression."
            )
        ),
    }
