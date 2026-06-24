import os
import torch
import random
import torchvision.transforms.functional as TF
from PIL import Image
import logging

from app.config import settings
from app.protection.diff_jpeg import DiffJPEGProxy
from app.protection.surrogate_models import FaceNetArcFaceEnsemble
from app.protection.face_mask import create_face_mask
from app.protection.watermark import embed_watermark
from app.protection.quality import compute_quality_metrics

logger = logging.getLogger(__name__)

# ── The Nectar Rule: absolute maximum pixel shift ──────────────────────────────
EPSILON = settings.epsilon_max  # Reads from config — tune via .env


def apply_spectral_filter(
    gradients: torch.Tensor,
    r_min: float = 0.1,
    r_max: float = 0.45,
) -> torch.Tensor:
    """
    Band-pass FFT filter on gradients.

    Isolates mid-frequency bands that survive JPEG compression while
    remaining invisible to the human eye.
      - Blocks low-frequencies  (r < 0.1): global colour/brightness shifts
      - Blocks high-frequencies (r > 0.45): sharp edges, aliasing artefacts
      - Passes mid-frequencies  (0.1–0.45): texture detail — JPEG-robust band

    Args:
        gradients: (B, C, H, W) gradient tensor on any device
        r_min: inner radius of band-pass (normalised [0, 1])
        r_max: outer radius of band-pass (normalised [0, 1])

    Returns:
        Filtered gradients in spatial domain, same shape and device.
    """
    _, _, H, W = gradients.shape
    fft_grads = torch.fft.fftshift(torch.fft.fft2(gradients))

    Y, X = torch.meshgrid(
        torch.linspace(-1, 1, H, device=gradients.device),
        torch.linspace(-1, 1, W, device=gradients.device),
        indexing="ij",
    )
    radius = torch.sqrt(X**2 + Y**2)
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
    Varman Protection Engine — Spectral Ensemble Attack.
    "Poison to AI, Nectar to eyes."

    Pipeline:
      1. Letterbox image to 512×512 canvas
      2. Embed watermark (optional)
      3. Detect face → spatial mask + bbox
      4. FaceNetArcFaceEnsemble: extract original embeddings from BOTH models
      5. EoT loop (50 iters):
           a. Apply delta × face_mask
           b. DiffJPEG augmentation (Q 65–90) — EoT compression robustness
           c. Ensemble forward pass → sum of both cosine similarity losses
           d. Backprop → FFT spectral filter → PGD step → L∞ clamp
      6. Restore original resolution
      7. Quality gate (SSIM ≥ 0.92)

    Key invariants:
      - EPSILON from settings.epsilon_max (tune via .env)
      - Dual surrogate: FaceNet (107MB) + ArcFace (168MB) = 275MB total VRAM
      - DiffJPEG kept: EoT forces perturbation to survive real JPEG compression
      - FFT band-pass (0.15–0.35): narrowed to reduce visible ripple on skin
      - Face spatial mask: noise concentrated on face, background untouched
    """
    device_str = settings.device
    device = torch.device(device_str)
    logger.info(f"[Varman] Starting protection: {original_path} on {device}")

    # ── 1. Load & letterbox ───────────────────────────────────────────────────
    img = Image.open(original_path).convert("RGB")
    orig_w, orig_h = img.size
    max_dim = max(orig_w, orig_h)
    scale = settings.processing_resolution / max_dim
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

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

    # ── 4. Load ensemble surrogate (FaceNet + ArcFace) ───────────────────────
    # Both models are frozen — no gradient storage through weights.
    # Combined VRAM: 275MB. Safe on 4GB RTX 2050.
    logger.info("[Varman] Loading FaceNetArcFaceEnsemble (VGGFace2 + IResNet50/MS1MV3).")
    surrogate = FaceNetArcFaceEnsemble(device=device_str)
    extract_kwargs = {"face_bbox": face_bbox} if face_bbox is not None else {}

    with torch.no_grad():
        # Returns (fn_emb, arc_emb) — both (1, 512), gradient-compatible
        fn_orig, arc_orig = surrogate.extract_features(canvas_tensor, **extract_kwargs)

    # ── 5. EoT optimisation loop ──────────────────────────────────────────────
    delta = torch.zeros_like(canvas_tensor, requires_grad=True, device=device)
    alpha = 0.5 / 255.0
    iterations = settings.eot_iterations  # default 50

    logger.info(
        f"[Varman] EoT loop: {iterations} iters, "
        f"ε={EPSILON:.5f}, α={alpha:.5f}, DiffJPEG Q=65–90"
    )

    for i in range(iterations):
        # Apply delta with spatial face mask
        x_adv = torch.clamp(canvas_tensor + delta * face_mask, 0.0, 1.0)

        # DiffJPEG augmentation — EoT: train perturbation to survive JPEG compression.
        # Randomly samples quality factor each iteration so the noise must be
        # robust across the Q65–90 range Instagram/social media uses.
        q = random.randint(65, 90)
        diff_jpeg = DiffJPEGProxy(quality=q).to(device)
        x_compressed = diff_jpeg(x_adv)

        # Ensemble forward pass — both models see the JPEG-augmented image
        fn_adv, arc_adv = surrogate.extract_features(x_compressed, **extract_kwargs)

        # Dual cosine similarity loss — minimise similarity to BOTH models.
        # Perturbation must fool FaceNet AND ArcFace simultaneously.
        # This forces noise into the intersection of both models blind spots,
        # producing more targeted (lower amplitude) perturbations → better face SSIM.
        cos_fn  = torch.nn.functional.cosine_similarity(fn_adv,  fn_orig)
        cos_arc = torch.nn.functional.cosine_similarity(arc_adv, arc_orig)
        loss = cos_fn.mean() + cos_arc.mean()

        loss.backward()

        with torch.no_grad():
            if delta.grad is not None:
                # FFT spectral filter — narrowed band (0.15-0.35) vs original (0.10-0.45).
                # Cuts the low-mid frequencies that produce visible ripple/wave patterns
                # on smooth skin while keeping the core JPEG-robust band intact.
                robust_grads = apply_spectral_filter(delta.grad, r_min=0.15, r_max=0.35)
                delta.data -= alpha * robust_grads.sign()
                # The Nectar Rule — hard L∞ clamp
                delta.data.clamp_(-EPSILON, EPSILON)
                delta.grad.zero_()

    # ── 6. Apply final perturbation & restore original resolution ─────────────
    with torch.no_grad():
        final_adv = torch.clamp(canvas_tensor + delta * face_mask, 0.0, 1.0)

    final_adv_img = TF.to_pil_image(final_adv.squeeze(0).cpu())
    final_cropped = final_adv_img.crop(
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
        f"[Varman] {status.upper()} — "
        f"SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f}dB, "
        f"faces={num_faces}, ε={EPSILON:.5f}"
    )

    return {
        "ssim": ssim_score,
        "psnr": psnr_score,
        "epsilon_used": EPSILON,
        "faces_detected": num_faces,
        "status": status,
    }
