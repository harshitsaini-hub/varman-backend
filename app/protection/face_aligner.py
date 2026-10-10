"""
Face Alignment & Inverse Warping Composite Module
=================================================
Handles canonical 5-point landmark affine transformation (112x112 space)
and seamless soft-feathered inverse composite back to full-resolution images.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np

# Canonical InsightFace 5-point landmark reference in 112x112 space
ARCFACE_CANONICAL_112 = np.array([
    [38.2946, 51.6963],   # Right Eye
    [73.5318, 51.5014],   # Left Eye
    [56.0252, 71.7366],   # Nose Tip
    [41.5493, 92.3655],   # Right Mouth Corner
    [70.7299, 92.2041],   # Left Mouth Corner
], dtype=np.float32)


def align_crop_112(
    img_bgr: np.ndarray,
    landmarks: List[Tuple[int, int]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Computes standard 5-point affine transformation matrix and warps face to canonical 112x112.

    Returns:
        (aligned_112_bgr, affine_matrix_M) or (None, None) if alignment fails.
    """
    if len(landmarks) < 5:
        return None, None
    src = np.array(landmarks[:5], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, ARCFACE_CANONICAL_112)
    if M is None:
        return None, None
    warped = cv2.warpAffine(img_bgr, M, (112, 112), borderMode=cv2.BORDER_REFLECT)
    if warped is None or warped.shape[:2] != (112, 112):
        return None, None
    return warped, M


def composite_crop_back(
    original_bgr: np.ndarray,
    protected_crop_112: np.ndarray,
    affine_matrix_M: np.ndarray,
    feather_sigma: float = 4.0,
) -> np.ndarray:
    """
    Inverts affine transform M and blends the protected face crop back onto the original photo
    using an elliptical Gaussian-feathered mask to prevent boundary seams.
    """
    h, w = original_bgr.shape[:2]
    inv_M = cv2.invertAffineTransform(affine_matrix_M)

    # 1. Construct elliptical blending mask in 112x112 space
    mask_112 = np.zeros((112, 112), dtype=np.float32)
    # Ellipse covering facial oval (center=56,56; radii=42,50)
    cv2.ellipse(mask_112, (56, 56), (42, 50), 0, 0, 360, 1.0, -1)
    mask_112 = cv2.GaussianBlur(mask_112, (15, 15), feather_sigma)

    # 2. Warp protected crop and mask back to original resolution
    warped_prot = cv2.warpAffine(
        protected_crop_112, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )
    warped_mask = cv2.warpAffine(
        mask_112, inv_M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0
    )
    warped_mask_3ch = np.repeat(warped_mask[:, :, np.newaxis], 3, axis=2)

    # 3. Alpha blend onto original image
    composited = (
        warped_mask_3ch * warped_prot.astype(np.float32)
        + (1.0 - warped_mask_3ch) * original_bgr.astype(np.float32)
    )
    return np.clip(composited, 0, 255).astype(np.uint8)
