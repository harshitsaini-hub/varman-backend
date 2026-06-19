import os
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
import logging

# We mock these configurations per the strict constraints.
# In a real setup, these would come from your settings.
EPSILON = 8.0 / 255.0
EOT_ITERATIONS = 10
LR = 0.01

logger = logging.getLogger(__name__)

def protect_image_pipeline(original_path: str, protected_path: str):
    """
    Surgical Protection Engine: "Poison to AI, Nectar to eyes."
    Applies strict L-Infinity clamping and spatial masking.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting surgical protection for {original_path} on {device}")
    
    # 1. Load Image and convert to tensor
    img = Image.open(original_path).convert("RGB")
    tensor_img = TF.to_tensor(img).unsqueeze(0).to(device)  # Shape: (1, 3, H, W)
    _, _, h, w = tensor_img.shape
    
    # 2. The Spatial Mask (Background Protection)
    # Mocking a 1.0 bounding box mask for the face region (simulating MediaPipe)
    # We'll assume the face is in the center 50% of the image.
    face_mask = torch.zeros_like(tensor_img, device=device)
    top, bottom = int(h * 0.25), int(h * 0.75)
    left, right = int(w * 0.25), int(w * 0.75)
    face_mask[:, :, top:bottom, left:right] = 1.0
    
    # Initialize noise (delta) as a trainable parameter
    delta = torch.zeros_like(tensor_img, requires_grad=True, device=device)
    
    # 3. The Expectation over Transformation (EoT) Loop
    # Using Adam optimizer
    optimizer = optim.Adam([delta], lr=LR)
    
    logger.info(f"Running Surgical EoT loop for {EOT_ITERATIONS} iterations...")
    for i in range(EOT_ITERATIONS):
        optimizer.zero_grad()
        
        # Apply delta strictly to the spatial mask
        # We clamp early to ensure the forward pass uses the clamped values
        clamped_delta = torch.clamp(delta, -EPSILON, EPSILON)
        x_adv = tensor_img + clamped_delta * face_mask
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # Calculate a dummy AI activation (sum of the perturbed tensor * 0.5)
        # We apply a negative loss to maximize the AI's error
        dummy_ai_activation = torch.sum(x_adv * 0.5)
        loss = -dummy_ai_activation
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # 4. Strict L-Infinity Clamp (The Nectar Rule)
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -EPSILON, EPSILON)
            
    # Apply final clamped perturbation
    with torch.no_grad():
        final_adv = tensor_img + delta * face_mask
        final_adv = torch.clamp(final_adv, 0.0, 1.0)
        
    # Convert back to PIL Image
    final_adv_img = TF.to_pil_image(final_adv.squeeze(0).cpu())
    
    # Save protected image
    final_adv_img.save(protected_path, quality=95)
    
    logger.info("Surgical protection completed successfully.")
    
    return {
        "epsilon_used": EPSILON,
        "iterations": EOT_ITERATIONS,
        "status": "completed"
    }
