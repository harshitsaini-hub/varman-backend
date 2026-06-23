import os
import random
import logging

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from app.config import settings
from app.protection.diff_jpeg import DiffJPEGProxy
from app.protection.surrogate_models import FaceNetSurrogate
from app.protection.face_mask import create_face_mask
from app.protection.watermark import embed_watermark
from app.protection.quality import compute_quality_metrics

logger = logging.getLogger(__name__)

# ── The Nectar Rule: absolute maximum pixel shift ──────────────────────────────
# Reads from config so epsilon can be tuned via .env without touching code.
# Default: 0.035 (9/255) — invisible at 720×1280, sufficient for identity disruption.
EPSILON = settings.epsilon_max


def apply_spectral_filter(
    gradients: torch.Tensor,
    r_min: float = 0.1,
    r_max: float = 0.45,
) -> torch.Tensor:
    """
    Band-pass FFT filter applied to PGD gradients.

    Isolates mid-frequency components (r_min ≤ radius ≤ r_max):
      - Blocks low-frequencies  (radius < 0.1)  → prevents global colour shifts
      - Blocks high-frequencies (radius > 0.45) → prevents sharp visible edges
      - Preserves mid-band      (0.1 – 0.45)    → JPEG-robust, perceptually invisible

    Args:
        gradients: (B, C, H, W) gradient tensor on the correct device.
        r_min:     inner radius cutoff (normalised, 0–1).
        r_max:     outer radius cutoff (normalised, 0–1).

    Returns:
        Filtered gradient tensor of the same shape, real-valued.
    """
    _, _, H, W = gradients.shape
    fft_grads  = torch.fft.fftshift(torch.fft.fft2(gradients))

    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=gradients.device),
        torch.linspace(-1, 1, W, device=gradients.device),
        indexing="ij",
    )
    radius    = torch.sqrt(X ** 2 + Y ** 2)
    freq_mask = ((radius >= r_min) & (radius <= r_max)).float()
    freq_mask = freq_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    filtered = fft_grads * freq_mask
    return torch.fft.ifft2(torch.fft.ifftshift(filtered)).real


def protect_image_pipeline(
    original_path: str,
    protected_path: str,
    watermark_id: str = "",
    watermark_enabled: bool = True,
    strength: float = 0.5,
):
    """
    Varman Protection Engine — Spectral FaceNet Attack.
    "Poison to AI, Nectar to eyes."

    Pipeline:
      1. Letterbox image to 512×512 canvas
      2. Embed invisible watermark (optional)
      3. Detect face → spatial mask + bounding box
      4. FaceNet surrogate: extract original identity embedding (face-cropped)
      5. PGD loop (50 iters):
           a. Apply delta × face_mask
           b. DiffJPEG augmentation (Q 65–90) — EoT for compression robustness
           c. FaceNet forward on face crop
           d. Cosine similarity loss (minimise → maximise cosine distance)
           e. Backprop → FFT Spectral Filter (0.1–0.45 band)
           f. PGD step + L∞ clamp (The Nectar Rule)
      6. Restore original resolution
      7. Quality gate: SSIM ≥ 0.92
    """
    device_str = settings.device
    device     = torch.device(device_str)
    logger.info(f"[Varman] Starting protection: {original_path} on {device}")

    # ── 1. Load & letterbox to processing_resolution ──────────────────────────
    img     = Image.open(original_path).convert("RGB")
    orig_w, orig_h = img.size
    max_dim = max(orig_w, orig_h)
    scale   = settings.processing_resolution / max_dim
    new_w   = int(orig_w * scale)
    new_h   = int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new(
        "RGB",
        (settings.processing_resolution, settings.processing_resolution),
        (0, 0, 0) # type:ignore
    )
    offset_x = (settings.processing_resolution - new_w) // 2
    offset_y = (settings.processing_resolution - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    temp_canvas_path = f"{protected_path}.temp.jpg"
    canvas.save(temp_canvas_path, quality=100)

    # ── 2. Watermark (optional) ───────────────────────────────────────────────
    if watermark_enabled and watermark_id:
        embed_watermark(temp_canvas_path, temp_canvas_path, watermark_id)

    canvas_tensor = TF.to_tensor(Image.open(temp_canvas_path)).unsqueeze(0).to(device)

    # ── 3. Face detection → spatial mask + bbox ───────────────────────────────
    face_mask, num_faces, face_bboxes = create_face_mask(
        temp_canvas_path,
        (settings.processing_resolution, settings.processing_resolution),
        device_str,
    )
    face_bbox = face_bboxes[0] if face_bboxes else None

    if num_faces > 0:
        logger.info(f"[Varman] {num_faces} face(s) detected. Primary bbox: {face_bbox}")
    else:
        logger.info("[Varman] No faces detected — uniform perturbation applied.")

    # ── 4. Load FaceNet surrogate ─────────────────────────────────────────────
    logger.info("[Varman] Loading FaceNetSurrogate (VGGFace2).")
    surrogate     = FaceNetSurrogate(device=device_str)
    extract_kwargs = {"face_bbox": face_bbox} if face_bbox is not None else {}

    with torch.no_grad():
        original_embedding = surrogate.extract_features(canvas_tensor, **extract_kwargs)

    # ── 5. PGD optimisation loop ──────────────────────────────────────────────
    delta      = torch.zeros_like(canvas_tensor, requires_grad=True, device=device)
    alpha      = 0.5 / 255.0
    iterations = settings.eot_iterations  # default 50

    logger.info(
        f"[Varman] PGD+EoT: {iterations} iters, "
        f"ε={EPSILON:.5f}, α={alpha:.5f}, JPEG Q=65–90"
    )

    for i in range(iterations):
        # a. Apply delta with spatial face mask
        x_adv = torch.clamp(canvas_tensor + delta * face_mask, 0.0, 1.0)

        # b. DiffJPEG augmentation — EoT: train through Instagram-style compression
        q           = random.randint(65, 90)
        diff_jpeg   = DiffJPEGProxy(quality=q).to(device)
        x_compressed = diff_jpeg(x_adv)

        # c. FaceNet forward on face-cropped region
        adv_embedding = surrogate.extract_features(x_compressed, **extract_kwargs)

        # d. Cosine similarity loss — minimising this maximises cosine distance
        loss = torch.nn.functional.cosine_similarity(
            adv_embedding, original_embedding
        ).mean()

        # e. Backprop
        loss.backward()

        with torch.no_grad():
            if delta.grad is not None:
                # f. FFT Spectral Filter — force gradients into mid-frequency band
                robust_grads = apply_spectral_filter(delta.grad)
                # g. PGD step
                delta.data -= alpha * robust_grads.sign()
                # The Nectar Rule — hard L∞ clamp
                delta.data.clamp_(-EPSILON, EPSILON)
                delta.grad.zero_()

    # ── 6. Apply final perturbation & restore original resolution ─────────────
    with torch.no_grad():
        final_adv = torch.clamp(canvas_tensor + delta * face_mask, 0.0, 1.0)

    final_adv_img  = TF.to_pil_image(final_adv.squeeze(0).cpu())
    final_cropped  = final_adv_img.crop(
        (offset_x, offset_y, offset_x + new_w, offset_y + new_h)
    )
    final_restored = final_cropped.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    final_restored.save(protected_path, quality=95)

    if os.path.exists(temp_canvas_path):
        os.remove(temp_canvas_path)

    # ── 7. Quality gate ───────────────────────────────────────────────────────
    ssim_score, psnr_score = compute_quality_metrics(original_path, protected_path)
    status = "completed" if ssim_score >= settings.ssim_min_threshold else "failed"

    logger.info(
        f"[Varman] {status.upper()} — SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f}dB, "
        f"faces={num_faces}, ε={EPSILON:.5f}"
    )

    return {
        "ssim":          ssim_score,
        "psnr":          psnr_score,
        "epsilon_used":  EPSILON,
        "faces_detected": num_faces,
        "status":        status,
    }
