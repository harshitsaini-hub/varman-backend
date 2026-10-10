"""
Unit tests for Standalone DiffJPEG Module
========================================
Validates differentiable JPEG simulation against real Pillow libjpeg.
"""

import io
import numpy as np
import pytest
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

from app.protection.diff_jpeg import DiffJPEG


def _pillow_jpeg(img_rgb_uint8: np.ndarray, quality: int) -> np.ndarray:
    pil_img = Image.fromarray(img_rgb_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality, subsampling=0)
    buf.seek(0)
    return np.array(Image.open(buf), dtype=np.uint8)


def test_diff_jpeg_vs_pillow_agreement():
    """Verifies that DiffJPEG outputs agree with Pillow libjpeg at Q50, Q75, Q95."""
    rng = np.random.default_rng(42)
    # Synthetic face-like gradient patch
    x = np.linspace(50, 200, 112, dtype=np.float32)
    y = np.linspace(60, 210, 112, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    test_img = np.stack([xx, yy, (xx + yy) / 2.0], axis=-1).astype(np.uint8)

    diff_layer = DiffJPEG().eval()

    for q in [50, 75, 95]:
        pil_recon = _pillow_jpeg(test_img, quality=q)

        tensor_in = torch.from_numpy(test_img).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            diff_recon_tensor = diff_layer(tensor_in, quality=q)
        diff_recon = (
            diff_recon_tensor.squeeze(0)
            .permute(1, 2, 0)
            .numpy()
            .round()
            .clip(0, 255)
            .astype(np.uint8)
        )

        s_val = compute_ssim(pil_recon, diff_recon, channel_axis=2, data_range=255)
        p_val = compute_psnr(pil_recon, diff_recon, data_range=255)
        mae = np.mean(np.abs(pil_recon.astype(float) - diff_recon.astype(float)))

        assert s_val >= 0.990, f"SSIM mismatch at Q{q}: {s_val}"
        assert p_val >= 40.0, f"PSNR too low at Q{q}: {p_val}"
        assert mae <= 1.0, f"MAE too high at Q{q}: {mae}"


def test_diff_jpeg_autograd_flow():
    """Verifies that gradients pass cleanly through DiffJPEG to input pixels."""
    diff_layer = DiffJPEG(quality=75).eval()
    x = torch.rand((1, 3, 112, 112), dtype=torch.float32, requires_grad=True)
    out = diff_layer(x)
    loss = (out ** 2).sum()
    loss.backward()

    assert x.grad is not None, "Gradient must be computed"
    assert not torch.isnan(x.grad).any(), "Gradients must not contain NaN"
    assert x.grad.norm().item() > 1e-4, "Gradient must be non-zero"


def test_diff_jpeg_monotonic_quality():
    """Verifies that Q95 preserves image better than Q50."""
    rng = np.random.default_rng(123)
    img = rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

    diff_layer = DiffJPEG().eval()
    with torch.no_grad():
        out_50 = diff_layer(tensor, quality=50).squeeze(0).permute(1, 2, 0).numpy()
        out_95 = diff_layer(tensor, quality=95).squeeze(0).permute(1, 2, 0).numpy()

    mse_50 = np.mean((img.astype(float) - out_50) ** 2)
    mse_95 = np.mean((img.astype(float) - out_95) ** 2)

    assert mse_95 < mse_50, f"Q95 error ({mse_95}) should be less than Q50 error ({mse_50})"
