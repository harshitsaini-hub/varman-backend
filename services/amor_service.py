import cv2
import numpy as np
import os
from imwatermark import WatermarkEncoder
from config import NOISE_EPSILON, STORAGE_DIR

def apply_adversarial_noise(image_path):
 
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    img = img.astype(np.float32) / 255.0

    noise = np.random.uniform(-1, 1, img.shape).astype(np.float32)

    amored_img = img + NOISE_EPSILON * np.sign(noise)

    amored_img = np.clip(amored_img, 0, 1)
    
    output_path = os.path.join(STORAGE_DIR, "noised_" + os.path.basename(image_path))
    cv2.imwrite(output_path, (amored_img * 255).astype(np.uint8))
    
    return output_path

def apply_watermark(image_path, watermark_text):

    bgr = cv2.imread(image_path)
    
    if bgr is None:
         raise ValueError(f"Could not read image at {image_path} for watermarking")
        
    encoder = WatermarkEncoder()
    encoder.set_watermark('bytes', watermark_text.encode('utf-8'))

    bgr_encoded = encoder.encode(bgr, 'dwtDct')
    
    final_path = os.path.join(STORAGE_DIR, "amored_" + os.path.basename(image_path))
    cv2.imwrite(final_path, bgr_encoded)
    
    return final_path