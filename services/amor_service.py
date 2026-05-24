# services/amor_service.py

import hashlib
import io

import numpy as np
from imwatermark import WatermarkDecoder, WatermarkEncoder
from PIL import Image
from scipy.fftpack import dct, idct

from config import ARMOR_VALIDATION_MIN_QUALITY, NOISE_EPSILON

WATERMARK_METHOD = "dwtDctSvd"
WM_BITS = 32          # 32 bits survives JPEG well; 1-in-4B collision chance
MIN_WM_SIZE = 256
BIT_MATCH_THRESHOLD = 0.875  # 28/32 bits must match (tolerates minor JPEG noise)


# ── Bits conversion ────────────────────────────────────────────────────────


def _uuid_to_bits(watermark_id: str) -> list[int]:
    """Convert a UUID to a stable 32-bit sequence via SHA-256."""
    digest = hashlib.sha256(watermark_id.encode()).digest()
    bits = []
    for byte in digest[: WM_BITS // 8]:  # first 4 bytes = 32 bits
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bits_match(embedded: list[int], recovered) -> bool:
    """Compare bits; tolerates up to 12.5% bit-flip error from JPEG."""
    if len(embedded) != len(recovered):
        return False
    matches = sum(1 for e, r in zip(embedded, recovered) if int(e) == int(r))
    return matches >= int(WM_BITS * BIT_MATCH_THRESHOLD)


# ── Noise ──────────────────────────────────────────────────────────────────


def apply_frequency_domain_noise(image_array: np.ndarray) -> np.ndarray:
    """
    Injects adversarial noise in DCT frequency domain.
    Targets mid-frequency bands (8:64, 8:64) — JPEG preserves these.
    """
    result = image_array.astype(float).copy()
    for channel in range(3):
        freq = dct(dct(result[:, :, channel].T, norm="ortho").T, norm="ortho")
        noise_mask = np.random.choice([-1, 1], size=freq.shape) * NOISE_EPSILON * 255
        freq[8:64, 8:64] += noise_mask[8:64, 8:64]
        result[:, :, channel] = idct(idct(freq.T, norm="ortho").T, norm="ortho")
    return np.clip(result, 0, 255).astype(np.uint8)


# ── Watermark ──────────────────────────────────────────────────────────────


def inject_watermark(image_array: np.ndarray, watermark_id: str) -> np.ndarray:
    """
    Embeds watermark_id as 32 bits using dwtDctSvd.
    Handles images smaller than 256x256 via edge-padding (no upscale).
    """
    h, w = image_array.shape[:2]
    print(f"[ARMOR DEBUG] Image: {w}x{h}, watermark_id: {watermark_id}")

    needs_padding = h < MIN_WM_SIZE or w < MIN_WM_SIZE
    if needs_padding:
        pad_h = max(0, MIN_WM_SIZE - h)
        pad_w = max(0, MIN_WM_SIZE - w)
        padded = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    else:
        padded = image_array

    bits = _uuid_to_bits(watermark_id)
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", bits) # type: ignore[arg-type]
    watermarked_padded = encoder.encode(padded, method=WATERMARK_METHOD)

    return watermarked_padded[:h, :w]  # crop back to original size


def extract_watermark_bits(image_array: np.ndarray) -> list[int]:
    """Extracts WM_BITS bits from the image. Returns empty list on failure."""
    try:
        # Pad if needed for decoder
        h, w = image_array.shape[:2]
        needs_padding = h < MIN_WM_SIZE or w < MIN_WM_SIZE
        if needs_padding:
            pad_h = max(0, MIN_WM_SIZE - h)
            pad_w = max(0, MIN_WM_SIZE - w)
            image_array = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

        decoder = WatermarkDecoder("bits", WM_BITS)
        result = decoder.decode(image_array, method=WATERMARK_METHOD)
        return [int(b) for b in result]
    except Exception:
        return []


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
    Compresses the armored image, extracts bits, compares against expected.
    """
    compressed = simulate_platform_compression(armored_array, quality=ARMOR_VALIDATION_MIN_QUALITY)
    expected_bits = _uuid_to_bits(watermark_id)
    recovered_bits = extract_watermark_bits(compressed)
    survived = _bits_match(expected_bits, recovered_bits)

    matched = sum(1 for e, r in zip(expected_bits, recovered_bits) if int(e) == int(r)) if recovered_bits else 0

    return {
        "passed": survived,
        "watermark_id": watermark_id,
        "recovered_prefix": f"{matched}/{WM_BITS} bits matched",
        "compression_quality_tested": ARMOR_VALIDATION_MIN_QUALITY,
        "warning": (
            None
            if survived
            else f"Only {matched}/{WM_BITS} bits recovered. Do not deliver."
        ),
    }


# ── Master Pipeline ────────────────────────────────────────────────────────


def armor_image(image_array: np.ndarray, watermark_id: str) -> tuple[np.ndarray, dict]:
    """Full armor pipeline. Returns (armored_image_array, validation_report)."""
    noised = image_array  # noise bypass still in place

    watermarked = inject_watermark(noised, watermark_id)
    validation = validate_armor(watermarked, watermark_id)

    if not validation["passed"]:
        print(f"[ARMOR WARNING] Watermark validation failed for ID {watermark_id}")

    return watermarked, validation


# ── File-level helpers ─────────────────────────────────────────────────────


def apply_adversarial_noise(file_path: str) -> str:
    image = Image.open(file_path).convert("RGB")
    noised = apply_frequency_domain_noise(np.array(image))
    output_path = f"{file_path}.noised.jpg"
    Image.fromarray(noised).save(output_path, format="JPEG", quality=95)
    return output_path


def apply_watermark(file_path: str, watermark_id: str) -> str:
    image = Image.open(file_path).convert("RGB")
    watermarked = inject_watermark(np.array(image), watermark_id)
    output_path = f"{file_path}.watermarked.jpg"
    Image.fromarray(watermarked).save(output_path, format="JPEG", quality=95)
    return output_path