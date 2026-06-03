import io

import numpy as np
from blind_watermark import WaterMark
from PIL import Image

from config import ARMOR_VALIDATION_MIN_QUALITY


def _pad_for_dwt(image_array: np.ndarray) -> tuple[np.ndarray, int, int]:
    h, w = image_array.shape[:2]
    pad_h = 1 if h % 2 != 0 else 0
    pad_w = 1 if w % 2 != 0 else 0
    
    if pad_h > 0 or pad_w > 0:
        # Pad bottom and/or right edges by duplicating the last pixel line
        padded = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        return padded, h, w
    return image_array, h, w

def inject_watermark(image_array: np.ndarray, watermark_id: str) -> np.ndarray:
    # 1. Universally handle odd dimensions
    padded_array, orig_h, orig_w = _pad_for_dwt(image_array)

    # 2. Embed
    bwm = WaterMark(password_wm=1, password_img=1)
    bwm.read_img(img=padded_array)
    bwm.read_wm(watermark_id, mode='str')
    embedded_float_array = bwm.embed()
    
    # 3. Clean the math overflow
    clean_array = np.clip(embedded_float_array, 0, 255).astype(np.uint8)
    
    # 4. Slice off the temporary padding before giving it back to the user
    return clean_array[:orig_h, :orig_w]

def extract_watermark(image_array: np.ndarray, wm_byte_length: int) -> str:
    # Re-apply the exact same padding rule for extraction
    padded_array, _, _ = _pad_for_dwt(image_array)

    bwm = WaterMark(password_wm=1, password_img=1)
    try:
        extracted_str = bwm.extract(padded_array, wm_shape=wm_byte_length, mode='str')
        return str(extracted_str)
    except Exception:
        return ""

def simulate_platform_compression(image_array: np.ndarray, quality: int = 75) -> np.ndarray:
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def validate_armor(armored_array: np.ndarray, watermark_id: str) -> dict:
    compressed = simulate_platform_compression(armored_array, quality=ARMOR_VALIDATION_MIN_QUALITY)
    recovered = extract_watermark(compressed, len(watermark_id))
    
    prefix = watermark_id[:8]
    survived = prefix in recovered

    return {
        "passed": survived,
        "watermark_id": watermark_id,
        "recovered_prefix": recovered[:8] if recovered else "",
        "compression_quality_tested": ARMOR_VALIDATION_MIN_QUALITY,
        "warning": (
            None
            if survived
            else "Watermark did not survive compression simulation. Do not deliver."
        ),
    }

def armor_image(image_array: np.ndarray, watermark_id: str) -> tuple[np.ndarray, dict]:
    short_id = watermark_id.replace("-", "")[:4] 
    
    watermarked = inject_watermark(image_array, short_id)
    validation = validate_armor(watermarked, short_id)

    if not validation["passed"]:
        print(f"[ARMOR WARNING] Watermark validation failed for ID {watermark_id}")

    # The DB still tracks the full UUID, we just hide a tiny chunk of it
    validation["watermark_id"] = watermark_id
    return watermarked, validation
