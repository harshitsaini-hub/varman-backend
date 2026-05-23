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


def validate_armor(original_armored: np.ndarray, watermark_id: str, wm_length: int) -> dict:
    compressed = simulate_platform_compression(original_armored, quality=75)
    recovered_bytes = extract_watermark(compressed, wm_length)

    if isinstance(recovered_bytes, bytes):
        recovered_str = recovered_bytes.decode('utf-8', errors='ignore')
    else:
        recovered_str = str(recovered_bytes)
        
    watermark_survived = watermark_id[:8] in recovered_str

    return {
        "watermark_survived_compression": watermark_survived,
        "compression_quality_tested": 75,
        "warning": None if watermark_survived else "Watermark may not survive platform compression",
    }