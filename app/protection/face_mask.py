import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch

# Build path to the downloaded model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "models", "blaze_face_short_range.tflite")

def create_face_mask(image_path: str, output_size=(512, 512), device="cpu"):
    """
    Detect faces using MediaPipe Tasks API and create a soft spatial mask.
    Face regions = 1.0 (max perturbation)
    Background = 0.3 (minimal perturbation)
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        return torch.ones((1, 3, output_size[0], output_size[1]), device=device), 0

    h, w, _ = image_cv.shape
    mask = np.ones((h, w), dtype=np.float32) * 0.3  # Background weight
    num_faces = 0
    
    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)
        
        mp_image = mp.Image.create_from_file(image_path)
        detection_result = detector.detect(mp_image)
        
        num_faces = len(detection_result.detections)
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            xmin = int(bbox.origin_x)
            ymin = int(bbox.origin_y)
            width = int(bbox.width)
            height = int(bbox.height)
            
            # Expand box slightly
            xmin = max(0, xmin - int(width * 0.2))
            ymin = max(0, ymin - int(height * 0.2))
            xmax = min(w, xmin + int(width * 1.4))
            ymax = min(h, ymin + int(height * 1.4))
            
            # Face weight
            mask[ymin:ymax, xmin:xmax] = 1.0
            
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"MediaPipe Face Detection failed: {e}")
        
    mask_resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_LINEAR)
    mask_blurred = cv2.GaussianBlur(mask_resized, (21, 21), 0)
    
    mask_tensor = torch.from_numpy(mask_blurred).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(device)
    return mask_tensor, num_faces
