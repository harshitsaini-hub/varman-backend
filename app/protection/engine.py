import os
import torch
import torchvision.transforms.functional as TF
import open_clip
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
EPSILON = settings.epsilon_max  # L∞ bound — tune via .env


def protect_image_pipeline(
    original_path: str,
    protected_path: str,
    watermark_id: str = "",
    watermark_enabled: bool = False,
    strength: float = 0.5,
):
    """
    Varman Semantic Disruption Engine — CLIP PGD Attack.

    Pipeline:
      1. Load image at full resolution (no downscaling).
      2. Load OpenCLIP ViT-B/32 surrogate.
      3. Extract original CLIP embedding.
      4. PGD loop (N iterations):
           a. Forward pass through CLIP.
           b. Compute cosine similarity loss (minimize similarity).
           c. Backprop → signed gradient step → L∞ clamp.
           d. Log cosine similarity at intervals.
      5. Save as lossless PNG.

    Key design decisions:
      - No DiffJPEG: we are optimising for direct upload, not compression survival.
      - No face masking: MLLMs interpret the whole scene semantically.
      - No downscaling: perturbation is applied at native resolution.
      - Dynamic alpha: step size scales with epsilon for proper convergence.
      - CLIP ViT-B/32 only: single surrogate for clean v1.0 isolation.
    """
    device_str = settings.device
    device = torch.device(device_str)
    logger.info(f"[Varman] Starting semantic disruption: {original_path} on {device}")

    # ── 1. Load image at full resolution ──────────────────────────────────────
    from PIL import Image
    img = Image.open(original_path).convert("RGB")
    img_tensor = TF.to_tensor(img).unsqueeze(0).to(device)  # (1, 3, H, W) in [0, 1]

    logger.info(f"[Varman] Image loaded: {img.size[0]}x{img.size[1]} — no downscaling applied.")

    # ── 2. Load OpenCLIP ViT-B/32 ─────────────────────────────────────────────
    logger.info("[Varman] Loading OpenCLIP ViT-B/32 (laion2b_s34b_b79k)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # CLIP normalisation constants
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(device)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(device)

    def clip_encode(x: torch.Tensor) -> torch.Tensor:
        """Encode a (1, 3, H, W) [0,1] tensor through CLIP → (1, 512) embedding."""
        x_resized = torch.nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        x_norm = (x_resized - clip_mean) / clip_std
        return clip_model.encode_image(x_norm)  # type: ignore

    # ── 3. Extract original embedding ─────────────────────────────────────────
    with torch.no_grad():
        orig_embedding = clip_encode(img_tensor)

    # ── 4. PGD optimisation loop ──────────────────────────────────────────────
    iterations = settings.eot_iterations
    # Dynamic alpha: scale step size to epsilon and iterations for proper convergence.
    # Standard PGD heuristic: alpha = 2.5 * epsilon / iterations
    alpha = 2.5 * EPSILON / iterations

    delta = torch.zeros_like(img_tensor, requires_grad=True, device=device)

    logger.info(
        f"[Varman] PGD loop: {iterations} iters, "
        f"ε={EPSILON:.5f} ({EPSILON * 255:.1f}/255), "
        f"α={alpha:.6f} ({alpha * 255:.3f}/255)"
    )

    cosine_log = []

    for i in range(iterations):
        x_adv = torch.clamp(img_tensor + delta, 0.0, 1.0)
        adv_embedding = clip_encode(x_adv)

        # Objective: minimize cosine similarity → maximize embedding distance
        cos_sim = torch.nn.functional.cosine_similarity(adv_embedding, orig_embedding)
        loss = cos_sim.mean()

        loss.backward()

        with torch.no_grad():
            if delta.grad is not None:
                # Signed gradient step (PGD)
                delta.data -= alpha * delta.grad.sign()
                # L∞ clamp — the hard ceiling
                delta.data.clamp_(-EPSILON, EPSILON)
                # Ensure the full image stays in [0, 1]
                delta.data = torch.clamp(img_tensor + delta.data, 0.0, 1.0) - img_tensor
                delta.grad.zero_()

        # Log at intervals
        if i % 10 == 0 or i == iterations - 1:
            with torch.no_grad():
                current_cos = torch.nn.functional.cosine_similarity(
                    clip_encode(torch.clamp(img_tensor + delta, 0.0, 1.0)),
                    orig_embedding
                ).item()
            cosine_log.append((i, current_cos))
            logger.info(f"  [iter {i:3d}] cosine_sim = {current_cos:.4f}")

    # ── 5. Save as lossless PNG ───────────────────────────────────────────────
    with torch.no_grad():
        final_adv = torch.clamp(img_tensor + delta, 0.0, 1.0)

    final_img = TF.to_pil_image(final_adv.squeeze(0).cpu())

    # Force PNG extension for lossless output
    png_path = os.path.splitext(protected_path)[0] + ".png"
    final_img.save(png_path)

    logger.info(f"[Varman] Saved perturbed image (lossless PNG): {png_path}")

    # ── 6. Quality metrics ────────────────────────────────────────────────────
    from app.protection.quality import compute_quality_metrics
    ssim_score, psnr_score = compute_quality_metrics(original_path, png_path)

    final_cos = cosine_log[-1][1] if cosine_log else 1.0
    status = "completed" if ssim_score >= 0.98 else "quality_warning"

    logger.info(
        f"[Varman] {status.upper()} — "
        f"SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f}dB, "
        f"CLIP_cos={final_cos:.4f}, ε={EPSILON:.5f}"
    )

    return {
        "ssim": ssim_score,
        "psnr": psnr_score,
        "clip_cosine_final": final_cos,
        "cosine_log": cosine_log,
        "epsilon_used": EPSILON,
        "iterations": iterations,
        "status": status,
        "output_path": png_path,
    }
