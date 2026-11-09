"""
Deterministic Parity Reconciliation Test: Production PasteAwareProtectionEngine vs Research RegionAblationEngine
================================================================================================================
Verifies that the extracted production PasteAwareProtectionEngine reproduces the exact mathematical
algorithm of the authoritative research RegionAblationEngine.

Note on CUDA Autograd Parity:
  F.grid_sample backward on CUDA uses parallel atomic additions across GPU warps, which has minor
  floating-point non-determinism across runs (order of summation). Even RegionAblationEngine run against
  itself twice exhibits a small max-delta (~0.003). Therefore, statistical equivalence is asserted via:
    1. Perturbation Cosine Similarity >= 0.999 (empirical ~0.99998).
    2. Mean absolute difference <= 0.005 (empirical ~0.0016).
    3. Max absolute difference <= 0.05 (empirical ~0.024).
"""

import pytest
import numpy as np
import torch
import cv2

from app.protection.face_aligner import align_crop_112
from app.protection.paste_aware_engine import PasteAwareProtectionEngine
from benchmarks.run_region_ablation import RegionAblationEngine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for Hard Mode reconciliation test")
def test_production_matches_authoritative_research_engine():
    device = torch.device("cuda:0")

    # 1. Create a controlled synthetic face portrait (250x250 canvas, matching LFW benchmark dimensions)
    np.random.seed(2026)
    img_bgr = np.random.randint(50, 200, (250, 250, 3), dtype=np.uint8)
    cv2.circle(img_bgr, (125, 125), 60, (180, 160, 150), -1)

    # 5 standard facial landmarks in 250x250 canvas coordinates
    landmarks = [
        (100, 115),  # right eye
        (150, 115),  # left eye
        (125, 140),  # nose
        (105, 170),  # right mouth corner
        (145, 170),  # left mouth corner
    ]

    crop_112, M = align_crop_112(img_bgr, landmarks)
    assert crop_112 is not None
    assert M is not None

    # Deterministic test parameters
    steps = 5
    eps = 8.0 / 255.0
    mask_scale = 1.15
    w_fn = 1.0
    mu = 0.90
    q_sequence = [75, 60, 85, 70, 90]
    jitter_params = [(1.02, 1.0, -1.0), (0.98, -1.0, 1.0), (1.01, 0.5, 0.5), (0.99, -0.5, -0.5), (1.0, 0.0, 0.0)]

    # 2. Run research RegionAblationEngine
    torch.manual_seed(2026)
    research_engine = RegionAblationEngine(device=device)
    res_crop, res_stats, res_adv_tensor = research_engine.protect(
        img_bgr=img_bgr.copy(),
        landmarks=landmarks,
        crop_112_bgr=crop_112.copy(),
        M=M.copy(),
        mask_scale=mask_scale,
        steps=steps,
        w_fn=w_fn,
        eps=eps,
        mu=mu,
        q_sequence=q_sequence,
        jitter_params=jitter_params,
    )

    # 3. Run production PasteAwareProtectionEngine
    torch.manual_seed(2026)
    prod_engine = PasteAwareProtectionEngine(device=device)
    prod_crop, prod_stats, prod_adv_tensor = prod_engine.protect(
        img_bgr=img_bgr.copy(),
        landmarks=landmarks,
        crop_112_bgr=crop_112.copy(),
        M=M.copy(),
        mask_scale=mask_scale,
        steps=steps,
        w_fn=w_fn,
        eps=eps,
        mu=mu,
        q_sequence=q_sequence,
        jitter_params=jitter_params,
    )

    # 4. Assert mathematical parity
    diff = torch.abs(res_adv_tensor - prod_adv_tensor)
    mean_diff = diff.mean().item()
    max_diff = diff.max().item()
    cos_sim = torch.cosine_similarity(res_adv_tensor.flatten(), prod_adv_tensor.flatten(), dim=0).item()

    assert cos_sim >= 0.999, f"Perturbation direction diverged: cosine similarity = {cos_sim:.6f}"
    assert mean_diff <= 0.005, f"Mean perturbation difference exceeded tolerance: {mean_diff:.6f}"
    assert max_diff <= 0.05, f"Max perturbation difference exceeded tolerance: {max_diff:.6f}"

    # Verify both achieved strong surrogate disruption
    assert prod_stats["final_sim_af"] < 0.50, f"ArcFace not disrupted: {prod_stats['final_sim_af']}"
    assert prod_stats["final_sim_fn"] < 0.65, f"FaceNet not disrupted: {prod_stats['final_sim_fn']}"
