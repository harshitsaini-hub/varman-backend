import os
import torch
from PIL import Image
from app.protection.engine import protect_image_pipeline, EPSILON

def test_engine_invariants():
    """
    Test 1: Background Protection (delta outside mask is exactly 0)
    Test 2: Epsilon Bound (maximum delta is <= EPSILON)
    """
    # Setup test image
    test_img_path = "test_image.jpg"
    test_out_path = "test_output.jpg"
    
    # Create a dummy image
    img = Image.new("RGB", (256, 256), color="white") # type:ignore
    img.save(test_img_path)
    
    try:
        # Run protection pipeline
        result = protect_image_pipeline(test_img_path, test_out_path)
        
        delta = result["delta_tensor"]
        face_mask = result["face_mask_tensor"]
        
        # Test 1: Background Protection Test
        # Assert that where face_mask == 0.0, delta is exactly 0.0
        background_delta = delta[face_mask == 0.0]
        assert torch.all(background_delta == 0.0), "Background protection failed! Delta is non-zero outside the face mask."
        
        # Test 2: Epsilon Bound Test
        # Assert that max(abs(delta)) <= EPSILON + small_float_tolerance
        tolerance = 1e-6
        max_shift = torch.max(torch.abs(delta)).item()
        assert max_shift <= EPSILON + tolerance, f"Epsilon bound failed! Max shift {max_shift} exceeded {EPSILON}"
        
        print("[OK] Engine Invariants Passed:")
        print("   - Background pixels strictly protected.")
        print(f"   - Max pixel shift within bounds: {max_shift:.6f} <= {EPSILON:.6f}")
        
    finally:
        # Cleanup
        if os.path.exists(test_img_path):
            os.remove(test_img_path)
        if os.path.exists(test_out_path):
            os.remove(test_out_path)

if __name__ == "__main__":
    test_engine_invariants()
