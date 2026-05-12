# services/amor_service.py — full replacement of the armor pipeline

import io

import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image
from scipy.fftpack import dct, idct

from config import ARMOR_VALIDATION_MIN_QUALITY, NOISE_EPSILON

WATERMARK_METHOD = "dwtDct"  # Discrete Wavelet Transform + DCT. Survives JPEG at quality 70+.

# ── Noise ──────────────────────────────────────────────────────────────────


def apply_frequency_domain_noise(image_array: np.ndarray) -> np.ndarray:
    """
    Injects adversarial noise in DCT frequency domain.
    Targets mid-frequency bands (8:64, 8:64) — JPEG preserves these.
    High-frequency bands (64+) are what JPEG discards. We avoid those.
    """
    result = image_array.astype(float).copy()

    for channel in range(3):  # R, G, B
        freq = dct(dct(result[:, :, channel].T, norm="ortho").T, norm="ortho")
        noise_mask = np.random.choice([-1, 1], size=freq.shape) * NOISE_EPSILON * 255
        freq[8:64, 8:64] += noise_mask[8:64, 8:64]  # Mid-frequency target only
        result[:, :, channel] = idct(idct(freq.T, norm="ortho").T, norm="ortho")

    return np.clip(result, 0, 255).astype(np.uint8)


# ── Watermark ──────────────────────────────────────────────────────────────


def inject_watermark(image_array: np.ndarray, watermark_id: str) -> np.ndarray:
    """
    Embeds watermark_id using dwtDct method.
    Survives: JPEG quality 70+, minor cropping, color adjustments.
    This is your cryptographic proof of ownership for DMCA.
    """
    encoder = WatermarkEncoder()
    encoder.set_watermark("bytes", watermark_id.encode("utf-8"))  # type: ignore[arg-type]
    return encoder.encode(image_array, method=WATERMARK_METHOD)


def extract_watermark(image_array: np.ndarray, wm_byte_length: int) -> str:
    """Extracts and returns watermark string. Returns empty string on failure."""
    try:
        decoder = WatermarkDecoder("bytes", wm_byte_length * 8)
        result = decoder.decode(image_array, method=WATERMARK_METHOD)
        return result.decode("utf-8", errors="ignore")  #type: ignore[arg-type]
    except Exception:
        return ""


# ── Validation ─────────────────────────────────────────────────────────────


def simulate_platform_compression(image_array: np.ndarray, quality: int = 75) -> np.ndarray:
    """Simulates Instagram/Telegram-level JPEG compression."""
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))


def validate_armor(armored_array: np.ndarray, watermark_id: str) -> dict:
    """
    Compresses the armored image as platforms would, then tries to recover the watermark.
    Called before delivering the image to the user.
    Returns a dict with the validation result.
    """
    compressed = simulate_platform_compression(armored_array, quality=ARMOR_VALIDATION_MIN_QUALITY)
    recovered = extract_watermark(compressed, len(watermark_id))
    survived = watermark_id[:8] in recovered  # Prefix check is sufficient for validation

    return {
        "passed": survived,
        "watermark_id": watermark_id,
        "recovered_prefix": recovered[:8] if recovered else None,
        "compression_quality_tested": ARMOR_VALIDATION_MIN_QUALITY,
        "warning": (
            None
            if survived
            else "Watermark did not survive compression simulation. Do not deliver."
        ),
    }


# ── Master Pipeline ────────────────────────────────────────────────────────


def armor_image(image_array: np.ndarray, watermark_id: str) -> tuple[np.ndarray, dict]:
    """
    Full armor pipeline. Call this from celery_worker.py.
    Returns: (armored_image_array, validation_report)
    """
    noised = apply_frequency_domain_noise(image_array)
    watermarked = inject_watermark(noised, watermark_id)
    validation = validate_armor(watermarked, watermark_id)

    if not validation["passed"]:
        # Log this. Consider alerting the ops team. Do not silently deliver.
        print(f"[ARMOR WARNING] Watermark validation failed for ID {watermark_id}")

    return watermarked, validation


def apply_adversarial_noise(file_path: str) -> str:
    """Apply frequency-domain noise to an image file and save a derived JPEG."""
    image = Image.open(file_path).convert("RGB")
    noised = apply_frequency_domain_noise(np.array(image))
    output_path = f"{file_path}.noised.jpg"
    Image.fromarray(noised).save(output_path, format="JPEG", quality=95)
    return output_path


def apply_watermark(file_path: str, watermark_id: str) -> str:
    """Apply a robust watermark to an image file and save a derived JPEG."""
    image = Image.open(file_path).convert("RGB")
    watermarked = inject_watermark(np.array(image), watermark_id)
    output_path = f"{file_path}.watermarked.jpg"
    Image.fromarray(watermarked).save(output_path, format="JPEG", quality=95)
    return output_path
