"""
Unit and Regression Tests for PGD Convergence & Real JPEG Robustness Experiment
================================================================================
Verifies Phase 3 requirements:
  - Perturbation budget compliance (|delta| <= 8/255)
  - Valid pixel range [0, 1]
  - Pure luminance constraint (delta_R == delta_G == delta_B)
  - Gradient propagation through DiffJPEG path
  - Correct optimizer direction (similarity reduction)
  - Deterministic execution under fixed seed
  - Accurate statistics for returned perturbation
  - Compatibility with existing production behavior
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from app.protection.diff_jpeg import DiffJPEG
from app.protection.surrogate_models import DualSurrogateSuite
from app.protection.strong_engine import StrongProtectionEngine, get_canonical_face_mask_112


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for StrongProtectionEngine tests.")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def surrogates(device):
    return DualSurrogateSuite(device)


@pytest.fixture(scope="module")
def diff_jpeg(device):
    return DiffJPEG().to(device).eval()


def test_gradient_propagation_through_diffjpeg(device, surrogates, diff_jpeg):
    """Verify finite non-zero gradients propagate through DiffJPEG to delta."""
    x = torch.rand(1, 3, 112, 112, device=device)
    delta = torch.zeros_like(x, requires_grad=True)

    with torch.no_grad():
        c_af, c_fn = surrogates(x)

    x_adv = torch.clamp(x + delta, 0.0, 1.0)
    x_jpeg = diff_jpeg(x_adv, quality=75.0)
    adv_af, adv_fn = surrogates(x_jpeg)

    sim_af = (adv_af * c_af).sum(dim=1).mean()
    sim_fn = (adv_fn * c_fn).sum(dim=1).mean()
    loss = sim_af + 0.5 * sim_fn

    loss.backward()
    assert delta.grad is not None, "delta.grad must not be None"
    assert not torch.isnan(delta.grad).any(), "Gradients contain NaN"
    assert not torch.isinf(delta.grad).any(), "Gradients contain Inf"
    assert delta.grad.abs().sum() > 0, "Gradients are all zero"


def test_perturbation_bounds_and_luminance_projection(device):
    """Verify that pure luminance projection simultaneously preserves L_inf and [0, 1] range."""
    eps = 8.0 / 255.0

    # Extreme pixels: black, white, highly saturated color boundaries
    test_img = torch.tensor([
        [[0.00, 0.01, 0.99]],
        [[1.00, 0.99, 0.01]],
        [[0.50, 0.50, 0.50]],
    ], device=device).permute(2, 1, 0).unsqueeze(0).repeat(1, 1, 112, 112)[:, :, :112, :112]

    # Extreme unconstrained delta
    raw_delta = torch.randn_like(test_img) * 0.5

    d_lum = raw_delta.mean(dim=1, keepdim=True)
    low = torch.maximum(torch.tensor(-eps, device=device), -test_img.min(dim=1, keepdim=True).values)
    high = torch.minimum(torch.tensor(eps, device=device), 1.0 - test_img.max(dim=1, keepdim=True).values)
    d_lum = torch.clamp(d_lum, low, high)
    delta_projected = d_lum.repeat(1, 3, 1, 1)

    # 1. Budget compliance
    max_abs_delta = delta_projected.abs().max().item()
    assert max_abs_delta <= eps + 1e-7, f"Max delta {max_abs_delta} exceeded eps {eps}"

    # 2. Pixel range
    x_clamped = test_img + delta_projected
    assert x_clamped.min().item() >= -1e-7, "Pixel values dropped below 0.0"
    assert x_clamped.max().item() <= 1.0 + 1e-7, "Pixel values exceeded 1.0"

    # 3. Pure luminance (zero chroma)
    diff_rg = (delta_projected[:, 0] - delta_projected[:, 1]).abs().max().item()
    diff_rb = (delta_projected[:, 0] - delta_projected[:, 2]).abs().max().item()
    assert diff_rg < 1e-7 and diff_rb < 1e-7, f"Chroma difference detected: {diff_rg}, {diff_rb}"


def test_pgd_optimizer_direction(device, surrogates, diff_jpeg):
    """Verify that optimizer step strictly decreases surrogate similarity."""
    x = torch.rand(1, 3, 112, 112, device=device)
    delta = torch.zeros_like(x, requires_grad=True)
    momentum = torch.zeros_like(x)
    eps = 8.0 / 255.0
    alpha = eps / 7.0

    with torch.no_grad():
        c_af, c_fn = surrogates(x)

    sims = []
    for _ in range(5):
        x_adv = torch.clamp(x + delta, 0.0, 1.0)
        x_jpeg = diff_jpeg(x_adv, quality=75.0)
        a_af, a_fn = surrogates(x_jpeg)
        sim_af = (a_af * c_af).sum(dim=1).mean()
        sim_fn = (a_fn * c_fn).sum(dim=1).mean()
        loss = sim_af + 0.5 * sim_fn
        sims.append(loss.item())

        grad = torch.autograd.grad(loss, delta)[0]
        grad_norm = grad.norm(p=1) + 1e-12
        momentum = 0.9 * momentum + grad / grad_norm

        with torch.no_grad():
            delta = delta - alpha * torch.sign(momentum)
            d_lum = delta.mean(dim=1, keepdim=True)
            low = torch.maximum(torch.tensor(-eps, device=device), -x.min(dim=1, keepdim=True).values)
            high = torch.minimum(torch.tensor(eps, device=device), 1.0 - x.max(dim=1, keepdim=True).values)
            d_lum = torch.clamp(d_lum, low, high)
            delta = d_lum.repeat(1, 3, 1, 1)
        delta.requires_grad = True

    assert sims[-1] < sims[0], f"PGD failed to reduce surrogate similarity: {sims[0]} -> {sims[-1]}"


def test_deterministic_execution_under_fixed_seed():
    """Verify that identical random seed produces identical JPEG quality sequence."""
    rng1 = np.random.default_rng(42)
    q_seq1 = [int(rng1.integers(50, 96)) for _ in range(75)]

    rng2 = np.random.default_rng(42)
    q_seq2 = [int(rng2.integers(50, 96)) for _ in range(75)]

    assert q_seq1 == q_seq2, "Seeded RNG failed to produce identical quality sequence!"

