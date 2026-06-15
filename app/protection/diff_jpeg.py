import torch
import torch.nn as nn

class DiffJPEGProxy(nn.Module):
    """
    A lightweight, differentiable proxy for JPEG compression.
    Simulates the quantization step in a differentiable manner 
    using the Straight-Through Estimator (STE).
    """
    def __init__(self, quality=80):
        super().__init__()
        self.quality = quality

    def forward(self, x):
        # x is assumed to be in [0, 1] range, shape (B, C, H, W)
        # Simulate quantization: lower quality -> coarser bins
        # A quality of 100 means ~256 bins. Quality 50 means ~32 bins.
        bins = max(2, int(256 * (self.quality / 100.0)))
        
        # Quantize
        x_scaled = x * bins
        x_quantized = torch.round(x_scaled) / bins
        
        # Straight-Through Estimator (STE)
        # Forward pass uses x_quantized, backward pass copies gradients to x
        x_diff = x_quantized.detach() - x.detach()
        return x + x_diff
