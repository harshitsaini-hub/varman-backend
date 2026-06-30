import os
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import open_clip
import torchvision.models as models
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
EPSILON = settings.epsilon_max  # L∞ bound — tune via .env


def protect_image_pipeline(
    original_path: str,
    protected_path: str,
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

    # ── 2b. Load ResNet50 (CNN Surrogate) ─────────────────────────────────────
    logger.info("[Varman] Loading ResNet50 (IMAGENET1K_V1)...")
    resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device)
    resnet_model.eval()
    for param in resnet_model.parameters():
        param.requires_grad = False

    resnet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    resnet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def resnet_encode(x: torch.Tensor) -> torch.Tensor:
        """Encode a (1, 3, H, W) [0,1] tensor through ResNet50 → (1, 1000) logits."""
        x_resized = torch.nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        x_norm = (x_resized - resnet_mean) / resnet_std
        return resnet_model(x_norm)

    # ── 3. Extract original/target embeddings ─────────────────────────────────────────
    with torch.no_grad():
        orig_clip_embedding = clip_encode(img_tensor)
        orig_resnet_embedding = resnet_encode(img_tensor)
        
        # TARGETED ATTACK: Define malicious target semantics
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_tokens = tokenizer(["a blurry photo of a completely empty white room"]).to(device)
        target_clip_embedding = clip_model.encode_text(text_tokens)

    # ── 4. PGD optimisation loop (Targeted + EoT) ──────────────────────────────────────────────
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
    
    # Define EoT random augmentations
    eot_transform = T.Compose([
        T.RandomResizedCrop(size=(img.size[1], img.size[0]), scale=(0.8, 1.0)),
        T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3)
    ])

    for i in range(iterations):
        x_adv = torch.clamp(img_tensor + delta, 0.0, 1.0)
        
        # Apply EoT
        x_adv_eot = eot_transform(x_adv)
        
        adv_clip = clip_encode(x_adv_eot)
        adv_resnet = resnet_encode(x_adv_eot)

        # Objective: 
        # 1. MAXIMIZE similarity to target text (Minimize 1 - cos_sim_target)
        # 2. MINIMIZE similarity to original image in CLIP (Minimize cos_sim_orig_clip)
        # 3. MINIMIZE similarity to original image in ResNet (Minimize cos_sim_resnet)
        cos_sim_target = torch.nn.functional.cosine_similarity(adv_clip, target_clip_embedding).mean()
        cos_sim_orig_clip = torch.nn.functional.cosine_similarity(adv_clip, orig_clip_embedding).mean()
        cos_sim_resnet = torch.nn.functional.cosine_similarity(adv_resnet, orig_resnet_embedding).mean()
        
        # Loss components (all should be minimized)
        loss_clip_target = 1.0 - cos_sim_target      # Push towards target (ideal 0)
        loss_clip_orig = cos_sim_orig_clip           # Push away from orig (ideal -1)
        loss_resnet = cos_sim_resnet                 # Push away from orig (ideal -1)
        
        # We combine the two CLIP objectives:
        loss_clip_combined = loss_clip_target + loss_clip_orig
        
        # Dynamic Weighting: The model that is further from its goal gets more weight.
        with torch.no_grad():
            # Distances from ideal states
            dist_clip = loss_clip_target.abs() + (1.0 + cos_sim_orig_clip).abs() 
            dist_resnet = (1.0 + cos_sim_resnet).abs() 
            total_dist = dist_clip + dist_resnet + 1e-6
            w_clip = dist_clip / total_dist
            w_resnet = dist_resnet / total_dist
            
        loss = w_clip * loss_clip_combined + w_resnet * loss_resnet

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
                x_eval = torch.clamp(img_tensor + delta, 0.0, 1.0)
                eval_clip = clip_encode(x_eval)
                eval_resnet = resnet_encode(x_eval)
                
                current_cos_orig_clip = torch.nn.functional.cosine_similarity(eval_clip, orig_clip_embedding).mean().item()
                current_cos_target = torch.nn.functional.cosine_similarity(eval_clip, target_clip_embedding).mean().item()
                current_cos_resnet = torch.nn.functional.cosine_similarity(eval_resnet, orig_resnet_embedding).mean().item()
                
            cosine_log.append((i, current_cos_orig_clip, current_cos_target, current_cos_resnet))
            logger.info(f"  [iter {i:3d}] Target_cos = {current_cos_target:.4f} | OrigCLIP_cos = {current_cos_orig_clip:.4f} | ResNet_cos = {current_cos_resnet:.4f}")

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

    final_orig_clip_cos = cosine_log[-1][1] if cosine_log else 1.0
    final_target_cos = cosine_log[-1][2] if cosine_log else 0.0
    final_resnet_cos = cosine_log[-1][3] if cosine_log else 1.0
    status = "completed" if ssim_score >= 0.95 else "quality_warning"

    logger.info(
        f"[Varman] {status.upper()} — "
        f"SSIM={ssim_score:.4f}, PSNR={psnr_score:.2f}dB, "
        f"Target_cos={final_target_cos:.4f}, OrigCLIP_cos={final_orig_clip_cos:.4f}, ResNet_cos={final_resnet_cos:.4f}, ε={EPSILON:.5f}"
    )

    return {
        "ssim": ssim_score,
        "psnr": psnr_score,
        "clip_cosine_final": final_orig_clip_cos,
        "target_cosine_final": final_target_cos,
        "resnet_cosine_final": final_resnet_cos,
        "cosine_log": cosine_log,
        "epsilon_used": EPSILON,
        "iterations": iterations,
        "status": status,
        "output_path": png_path,
    }
