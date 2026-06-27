"""
Varman Semantic Disruption Benchmark
====================================
Evaluates the mathematical success of a semantic disruption attack.
Measures embedding distance (OpenCLIP), structural similarity (SSIM),
and perceptual quality (LPIPS).

Usage:
    python varman_benchmark.py <original_image> <perturbed_image>

Output:
    Prints CLIP Cosine Similarity, SSIM, LPIPS, and Max Pixel Shift.
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as ssim

# Allow running from the benchmarks/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import lpips
    import open_clip
except ImportError:
    print("Error: Required packages missing. Please install lpips and open_clip_torch.")
    sys.exit(1)


def compute_metrics(orig_path: str, pert_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load images
    print("Loading images...")
    orig_pil = Image.open(orig_path).convert('RGB')
    pert_pil = Image.open(pert_path).convert('RGB')
    
    if pert_pil.size != orig_pil.size:
        print(f"Warning: Sizes differ. Resizing perturbed image from {pert_pil.size} to {orig_pil.size}.")
        pert_pil = pert_pil.resize(orig_pil.size, Image.Resampling.LANCZOS)
    
    orig_arr = np.array(orig_pil)
    pert_arr = np.array(pert_pil)
    
    orig_t = TF.to_tensor(orig_pil).unsqueeze(0).to(device)
    pert_t = TF.to_tensor(pert_pil).unsqueeze(0).to(device)

    # 1. SSIM
    print("Computing SSIM...")
    ssim_val = ssim(orig_arr, pert_arr, channel_axis=2, data_range=255)

    # 2. Max Perturbation
    max_pert = np.max(np.abs(orig_arr.astype(float) - pert_arr.astype(float)))

    # 3. LPIPS
    print("Loading LPIPS (AlexNet)...")
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    # LPIPS expects [-1, 1] range
    orig_lpips = orig_t * 2.0 - 1.0
    pert_lpips = pert_t * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = lpips_fn(orig_lpips, pert_lpips).item()
    del lpips_fn
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 4. CLIP Cosine
    print("Loading OpenCLIP ViT-B/32...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
    )
    clip_model.eval()
    
    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(device)
    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(device)

    def encode(x):
        x_r = torch.nn.functional.interpolate(x, size=(224,224), mode='bilinear', align_corners=False)
        x_n = (x_r - clip_mean) / clip_std
        with torch.no_grad():
            
            return clip_model.encode_image(x_n)

    print("Computing CLIP embeddings...")
    orig_emb = encode(orig_t)
    pert_emb = encode(pert_t)
    cos_sim = torch.nn.functional.cosine_similarity(pert_emb, orig_emb).item()

    # Report
    print("\n" + "="*50)
    print("  VARMAN SEMANTIC DISRUPTION BENCHMARK")
    print("="*50)
    print(f"Original:  {orig_path}")
    print(f"Perturbed: {pert_path}\n")
    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 33)
    print(f"{'CLIP Cosine Sim':<20} | {cos_sim:>10.4f}")
    print(f"{'SSIM':<20} | {ssim_val:>10.4f}")
    print(f"{'LPIPS':<20} | {lpips_val:>10.4f}")
    print(f"{'Max Pixel Shift':<20} | {max_pert:>10.1f}")
    print("="*50 + "\n")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    orig_path = sys.argv[1]
    pert_path = sys.argv[2]
    
    if not os.path.exists(orig_path):
        print(f"Error: Original image not found at {orig_path}")
        sys.exit(1)
    if not os.path.exists(pert_path):
        print(f"Error: Perturbed image not found at {pert_path}")
        sys.exit(1)

    compute_metrics(orig_path, pert_path)


if __name__ == "__main__":
    main()
