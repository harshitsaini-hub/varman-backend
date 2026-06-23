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

    Face regions   = 1.0  (full perturbation budget)
    Background     = 0.0  (zero perturbation — background noise wastes budget
                           and hurts visual quality without helping protection)

    Returns:
        mask_tensor:  (1, 3, H, W) soft mask on `device`
        num_faces:    int
        face_bboxes:  list of (x1, y1, x2, y2) in output_size coordinates
    """
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        # No image — fall back to full-image uniform mask
        return torch.ones((1, 3, output_size[0], output_size[1]), device=device), 0, []

    h, w, _ = image_cv.shape

    # ── Background weight = 0.0 ───────────────────────────────────────────────
    # 0.3 was causing visible wave artefacts on sky/ground.
    # FaceNet reads identity from the face crop only — background noise is
    # pure visual degradation with zero benefit to protection strength.
    mask = np.zeros((h, w), dtype=np.float32)

    num_faces = 0
    face_bboxes_orig = []

    try:
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options)
        detector = vision.FaceDetector.create_from_options(options)

        mp_image = mp.Image.create_from_file(image_path)
        detection_result = detector.detect(mp_image)

        num_faces = len(detection_result.detections)

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            orig_xmin = int(bbox.origin_x)
            orig_ymin = int(bbox.origin_y)
            orig_w    = int(bbox.width)
            orig_h    = int(bbox.height)

            # ── Expand bounding box by 20% on each side ───────────────────────
            # BUG FIX: compute padding from the ORIGINAL bbox dimensions,
            # then apply symmetrically. Previous code reused the shifted xmin
            # when computing xmax, causing the expanded box to drift and
            # potentially extend far beyond the actual face region.
            pad_x = int(orig_w * 0.2)
            pad_y = int(orig_h * 0.2)

            xmin = max(0, orig_xmin - pad_x)
            ymin = max(0, orig_ymin - pad_y)
            xmax = min(w, orig_xmin + orig_w + pad_x)
            ymax = min(h, orig_ymin + orig_h + pad_y)

            mask[ymin:ymax, xmin:xmax] = 1.0
            face_bboxes_orig.append((xmin, ymin, xmax, ymax))

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[Varman] MediaPipe Face Detection failed: {e}")

    # Resize mask to processing canvas size
    mask_resized = cv2.resize(mask, output_size, interpolation=cv2.INTER_LINEAR)

    # Gaussian blur softens the hard bbox edge to avoid sharp mask boundaries
    # showing up as rectangular artefacts in the protected image
    mask_blurred = cv2.GaussianBlur(mask_resized, (21, 21), 0)

    # Scale bounding boxes to output_size coordinates for FaceNet crop
    scale_x = output_size[0] / w
    scale_y = output_size[1] / h
    face_bboxes = [
        (
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y),
        )
        for x1, y1, x2, y2 in face_bboxes_orig
    ]

    mask_tensor = (
        torch.from_numpy(mask_blurred)
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(1, 3, 1, 1)
        .to(device)
    )
    return mask_tensor, num_faces, face_bboxes
