import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import random
import logging

from app.config import settings
from app.protection.diff_jpeg import DiffJPEGProxy
from app.protection.surrogate_models import SurrogateEnsemble
from app.protection.face_mask import create_face_mask
from app.protection.watermark import embed_watermark
from app.protection.quality import compute_quality_metrics

logger = logging.getLogger(__name__)

def protect_image_pipeline(
    original_path: str, 
    protected_path: str, 
    watermark_id: str = "",
    watermark_enabled: bool = True,
    strength: float = 0.5
):
    """
    Full EoT + DiffJPEG Protection Pipeline.
    Runs completely synchronously on the executor thread.
    """
    device_str = settings.device
    device = torch.device(device_str)
    logger.info(f"Starting protection for {original_path} on {device}")
    
    # 1. Resize & Pad (Aspect-preserving to 512x512)
    img = Image.open(original_path).convert("RGB")
    w, h = img.size
    max_dim = max(w, h)
    scale = settings.processing_resolution / max_dim
    new_w, new_h = int(w * scale), int(h * scale)
    
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create letterbox canvas
    canvas = Image.new("RGB", (settings.processing_resolution, settings.processing_resolution), (0, 0, 0))
    offset_x = (settings.processing_resolution - new_w) // 2
    offset_y = (settings.processing_resolution - new_h) // 2
    canvas.paste(img_resized, (offset_x, offset_y))
    
    # Save temp for watermark and masking
    temp_canvas_path = f"{protected_path}.temp.jpg"
    canvas.save(temp_canvas_path, quality=100)
    
    # 2. Watermark embedding
    if watermark_enabled and watermark_id:
        embed_watermark(temp_canvas_path, temp_canvas_path, watermark_id)
        
    # Load back to tensor
    canvas_tensor = TF.to_tensor(Image.open(temp_canvas_path)).unsqueeze(0).to(device)
    
    # 3. Create Face Mask
    face_mask = create_face_mask(temp_canvas_path, (settings.processing_resolution, settings.processing_resolution), device_str)
    
    # 4. Initialize δ and Optimizer
    delta = torch.zeros_like(canvas_tensor, requires_grad=True, device=device)
    
    # Scale epsilon based on UI strength (0.0 to 1.0) -> (0.01 to settings.epsilon_max)
    epsilon = 0.01 + (settings.epsilon_max - 0.01) * strength
    alpha = epsilon / 10.0  # step size
    
    # Load ensemble
    ensemble = SurrogateEnsemble(device=device_str)
    
    # Target features (we want to maximize distance from these)
    with torch.no_grad():
        target_features = ensemble.extract_features(canvas_tensor)
        
    # 5. EoT Loop
    iterations = settings.eot_iterations
    
    logger.info(f"Running EoT loop for {iterations} iterations...")
    for i in range(iterations):
        # Apply delta with mask
        x_adv = torch.clamp(canvas_tensor + delta * face_mask, 0.0, 1.0)
        
        # DiffJPEG augmentation
        q = random.randint(65, 90)
        diff_jpeg = DiffJPEGProxy(quality=q).to(device)
        x_compressed = diff_jpeg(x_adv)
        
        # Forward pass
        adv_features = ensemble.extract_features(x_compressed)
        
        # Loss: maximize feature distance (disrupt AI)
        loss = -torch.nn.functional.mse_loss(adv_features, target_features)
        
        # Backward
        loss.backward()
        
        # FGSM-style step
        with torch.no_grad():
            if delta.grad is not None:
                delta.data -= alpha * delta.grad.sign()
                delta.data.clamp_(-epsilon, epsilon)
                delta.grad.zero_()
            
    # 6. Apply final perturbation and crop back to original aspect ratio
    with torch.no_grad():
        final_adv = torch.clamp(canvas_tensor + delta * face_mask, 0.0, 1.0)
        
    # Convert back to PIL
    final_adv_img = TF.to_pil_image(final_adv.squeeze(0).cpu())
    
    # Crop letterbox
    final_cropped = final_adv_img.crop((offset_x, offset_y, offset_x + new_w, offset_y + new_h))
    
    # Resize back to original
    final_restored = final_cropped.resize((w, h), Image.Resampling.LANCZOS)
    
    # Save final
    final_restored.save(protected_path, quality=95)
    
    # Cleanup temp
    if os.path.exists(temp_canvas_path):
        os.remove(temp_canvas_path)
        
    # 7. Quality Check
    ssim_score, psnr_score = compute_quality_metrics(original_path, protected_path)
    
    return {
        "ssim": ssim_score,
        "psnr": psnr_score,
        "epsilon_used": epsilon,
        "status": "completed" if ssim_score >= settings.ssim_min_threshold else "failed"
    }
