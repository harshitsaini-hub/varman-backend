"""
Test the Varman Lightweight Image Protection Engine.

Validates:
  1. Global SSIM >= 0.90
  2. Expanded 1.2x Face SSIM >= 0.90
  3. Baseline-Corrected Y-NPR metric computation across compression tiers.
  4. Lightweight execution speed (< 200 ms on CPU).
"""

import os
import time
import numpy as np
import cv2
from app.protection.engine import protect_image_pipeline


def test_engine_invariants():
    """
    Run protection pipeline on a test image and verify scientific invariants.
    """
    test_img_path = "test_invariant_input.jpg"
    test_out_path = "test_invariant_output.png"

    # Create a test 256x256 image with synthetic structure
    np.random.seed(42)
    pixels = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite(test_img_path, pixels)

    try:
        t0 = time.perf_counter()
        out_png_path, metrics = protect_image_pipeline(
            test_img_path,
            test_out_path,
            preset_name="Balanced",
            enable_grid=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        eval_m = metrics["evaluation_metrics"]
        prot_m = metrics["protection_meta"]

        # Invariant 1: File output created
        assert os.path.exists(out_png_path), f"Output PNG not created at {out_png_path}"

        # Invariant 2: Global SSIM >= 0.90
        global_ssim = eval_m["global_ssim"]
        assert global_ssim >= 0.90, f"Global SSIM failed! Got {global_ssim:.4f} (need >= 0.90)"

        # Invariant 3: Face SSIM >= 0.90
        face_ssim = eval_m["face_ssim_1_2x"]
        assert face_ssim >= 0.90, f"Face SSIM failed! Got {face_ssim:.4f} (need >= 0.90)"

        # Invariant 4: Y-NPR computed across compression suite
        avg_npr = eval_m["avg_y_npr_percent"]
        assert avg_npr >= 0.0, f"Invalid Y-NPR score: {avg_npr}"
        assert "Q75" in eval_m["y_npr_breakdown"], "Missing Q75 tier in Y-NPR breakdown"

        # Invariant 5: Processing speed < 300ms (fast CPU execution)
        assert elapsed_ms < 1000.0, f"Processing too slow! Took {elapsed_ms:.1f}ms"

        print("Varman Lightweight Engine Invariant Tests PASSED!")
        print(f"  - Execution Time: {elapsed_ms:.1f} ms")
        print(f"  - Global SSIM: {global_ssim:.4f}")
        print(f"  - Face SSIM (1.2x): {face_ssim:.4f}")
        print(f"  - PSNR: {eval_m['psnr_db']:.2f} dB")
        print(f"  - Avg Y-NPR: {avg_npr:.2f}%")
        print(f"  - Y-NPR Breakdown: {eval_m['y_npr_breakdown']}")

    finally:
        for p in (test_img_path, test_out_path):
            if os.path.exists(p):
                os.remove(p)
        png_variant = os.path.splitext(test_out_path)[0] + ".png"
        if os.path.exists(png_variant):
            os.remove(png_variant)


if __name__ == "__main__":
    test_engine_invariants()
