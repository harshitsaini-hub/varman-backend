import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import random
import logging

from app.config import settings
from app.protection.diff_jpeg import DiffJPEGProxy
from app.protection.surrogate_models import SurrogateEnsemble, FaceNetSurrogate
from app.protection.face_mask import create_face_mask
from app.protection.watermark import embed_watermark
from app.protection.quality import compute_quality_metrics

logger = logging.getLogger(__name__)

# ── The Nectar Rule: absolute maximum pixel shift ──────────────────────
EPSILON = 12.0 / 255.0


def protect_image_pipeline(
    original_path: str,
    protected_path: str,
    watermark_id: str = "",
    watermark_enabled: bool = True,
    strength: float = 0.5,
):
    """
    Surgical Protection Engine: "Poison to AI, Nectar to eyes."

    Three pillars:
      1. Spatial Mask    – isolate noise to face; background stays identical.
      2. Real EoT Loop   – train delta against CLIP + ResNet50 surrogate
                           ensemble through DiffJPEG compression.
      3. L-inf Clamp     – no pixel shifts more than 8/255 (~3%).
    """
    device_str = settings.device
    device = torch.device(device_str)
    logger.info(f"Starting protection for {original_path} on {device}")

    # ── 1. Load & Letterbox to processing_resolution ──────────────────
    img = Image.open(original_path).convert("RGB")
    orig_w, orig_h = img.size
    max_dim = max(orig_w, orig_h)
    scale = settings.processing_resolution / max_dim
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new(
        "RGB",
        (settings.processing_resolution, settings.processing_resolution),
        (0, 0, 0)  # type: ignore[arg-type]
    )
    offset_x = (settings.processing_resolution - new_w) // 2
    offset_y = (settings.processing_resolution - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))

    # Save temp for watermark and face detection
    temp_canvas_path = f"{protected_path}.temp.jpg"
    canvas.save(temp_canvas_path, quality=100)

    # ── 2. Watermark (optional) ───────────────────────────────────────
    if watermark_enabled and watermark_id:
        embed_watermark(temp_canvas_path, temp_canvas_path, watermark_id)

    # Load to tensor
    canvas_tensor = TF.to_tensor(Image.open(temp_canvas_path)).unsqueeze(0).to(device)

    # ── 3. Spatial Mask (MediaPipe face detection) ────────────────────
    face_mask, num_faces, face_bboxes = create_face_mask(
        temp_canvas_path,
        (settings.processing_resolution, settings.processing_resolution),
        device_str,
    )

    # Use the first detected face bbox for FaceNet cropping
    face_bbox = face_bboxes[0] if face_bboxes else None

    if num_faces > 0:
        logger.info(
            f"Targeting {num_faces} detected face(s). Primary face bbox: {face_bbox}"
        )
    else:
        logger.info(
            "No faces detected. Applying uniform perturbation across image."
        )

    # ── 4. Load surrogate ensemble ────────────────────────────────────────
    if settings.surrogate_mode == "vae":
        logger.info("Using VAESurrogate for Generative AI Latent Attack.")
        from app.protection.surrogate_models import VAESurrogate
        ensemble = VAESurrogate(device=device_str)
    elif settings.surrogate_mode == "facenet":
        logger.info("Using FaceNetSurrogate for identity-aware gradients.")
        ensemble = FaceNetSurrogate(device=device_str)
    else:
        logger.info("Using legacy SurrogateEnsemble (CLIP+ResNet50).")
        ensemble = SurrogateEnsemble(device=device_str)

    # Extract the ORIGINAL feature embedding
    # For FaceNet: pass face_bbox so we embed only the face crop (matches MTCNN behavior)
    extract_kwargs = {}
    if settings.surrogate_mode == "facenet" and face_bbox is not None:
        extract_kwargs["face_bbox"] = face_bbox

    with torch.no_grad():
        target_features = ensemble.extract_features(canvas_tensor, **extract_kwargs)
        
        # Out-of-Distribution Clamping (only for VAE)
        if settings.surrogate_mode == "vae":
            # T = -E(x), but bounded to [-3, 3] to avoid the target AI clamping it
            target_features = torch.clamp(-target_features, min=-3.0, max=3.0)

    # ── 5. Initialise delta & optimizer ───────────────────────────────
    delta = torch.zeros_like(canvas_tensor, requires_grad=True, device=device)
    alpha = 0.5 / 255.0  # PGD step size

    iterations = settings.eot_iterations  # default 50
    logger.info(f"Running EoT loop for {iterations} iterations (eps={EPSILON:.6f})...")

    # ── 6. EoT Optimisation Loop ──────────────────────────────────────
    for i in range(iterations):
        # Apply delta with spatial mask
        x_adv = torch.clamp(canvas_tensor + delta * face_mask, 0.0, 1.0)

        # DiffJPEG augmentation — simulate Instagram's harsh compression
        q = random.randint(65, 90)
        diff_jpeg = DiffJPEGProxy(quality=q).to(device)
        x_compressed = diff_jpeg(x_adv)

        # Forward pass through surrogate ensemble (with face crop for FaceNet)
        adv_features = ensemble.extract_features(x_compressed, **extract_kwargs)

        # Loss Calculation
        if settings.surrogate_mode == "vae":
            # We want to MINIMIZE distance to the mathematical void
            loss = torch.nn.functional.mse_loss(adv_features, target_features)
        elif settings.surrogate_mode == "facenet":
            # Cosine similarity loss: directly targets the metric face-recognition uses.
            # Minimizing cosine similarity = maximizing cosine distance.
            cos_sim = torch.nn.functional.cosine_similarity(adv_features, target_features)
            loss = cos_sim.mean()  # minimize similarity → maximize distance
        else:
            # Legacy: maximize MSE distance from original embedding
            loss = -torch.nn.functional.mse_loss(adv_features, target_features)

        # Backward
        loss.backward()

        # FFT Spectral Gradient Filter & Update
        with torch.no_grad():
            if delta.grad is not None:
                # FFT Spectral Gradient Filter — applied to ALL surrogate modes.
                # Isolates mid-frequency gradients that survive JPEG compression
                # while remaining invisible to the human eye.
                grads = delta.grad
                _, channels, height, width = grads.shape
                fft_grads = torch.fft.fftshift(torch.fft.fft2(grads))
                
                Y, X = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing='ij')
                radius = torch.sqrt(X**2 + Y**2).to(device)
                
                # Isolate mid-frequencies: Inner radius 0.1, outer radius 0.45
                mask = (radius >= 0.1) & (radius <= 0.45)
                mask = mask.unsqueeze(0).unsqueeze(0).float()
                
                filtered_fft_grads = fft_grads * mask
                robust_grads = torch.fft.ifft2(torch.fft.ifftshift(filtered_fft_grads)).real
                
                # Apply step
                delta.data -= alpha * robust_grads.sign()
                # ── The Nectar Rule: strict L-inf clamp ───────────
                delta.data.clamp_(-EPSILON, EPSILON)
                delta.grad.zero_()

    # ── 7. Apply final perturbation & restore original resolution ─────
    with torch.no_grad():
        final_adv = torch.clamp(canvas_tensor + delta * face_mask, 0.0, 1.0)

    final_adv_img = TF.to_pil_image(final_adv.squeeze(0).cpu())

    # Crop letterbox padding
    final_cropped = final_adv_img.crop(
        (offset_x, offset_y, offset_x + new_w, offset_y + new_h)
    )

    # Resize back to original dimensions
    final_restored = final_cropped.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

    # Save final protected image
    final_restored.save(protected_path, quality=95)

    # Cleanup temp
    if os.path.exists(temp_canvas_path):
        os.remove(temp_canvas_path)

    # ── 8. Quality gate ───────────────────────────────────────────────
    ssim_score, psnr_score = compute_quality_metrics(original_path, protected_path)

    status = "completed" if ssim_score >= settings.ssim_min_threshold else "failed"

    logger.info(
        f"Protection {status}. SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f}dB, "
        f"faces={num_faces}, eps={EPSILON:.6f}"
    )

    return {
        "ssim": ssim_score,
        "psnr": psnr_score,
        "epsilon_used": EPSILON,
        "faces_detected": num_faces,
        "status": status,
    }
