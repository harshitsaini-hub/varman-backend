import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class EvaluationEngine:
    """
    Varman Dedicated Evaluation & Metrics Engine.
    
    Provides scientific evaluation of:
      1. Global SSIM & Expanded 1.2x Face SSIM
      2. PSNR & L_infinity Max Pixel Shift
      3. Y-Channel Baseline-Corrected Normalized Perturbation Retention (Y-NPR)
         evaluated across compression suite (Q95, Q85, Q75, Q60, 0.75x Resize + Q75)
      4. Downstream Landmark Effectiveness Metrics (Reproducible 5-point YuNet alignment):
         - Landmark shift delta (px)
         - Alignment error
         - Alignment error percentage change (%)
    """

    COMPRESSION_TIERS = {
        "Q95": {"q": 95, "resize": 1.0},
        "Q85": {"q": 85, "resize": 1.0},
        "Q75": {"q": 75, "resize": 1.0},
        "Q60": {"q": 60, "resize": 1.0},
        "Resize075_Q75": {"q": 75, "resize": 0.75},
    }

    # Standard reference 5-point facial landmark shape for canonical alignment (256x256 space)
    CANONICAL_5_LANDMARKS = np.array([
        [90.0, 110.0],  # Right Eye
        [166.0, 110.0], # Left Eye
        [128.0, 155.0], # Nose Tip
        [98.0, 195.0],  # Right Mouth Corner
        [158.0, 195.0], # Left Mouth Corner
    ], dtype=np.float32)

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.path.join(settings.base_dir, "models")
        self.yunet_path = os.path.join(self.model_dir, "face_detection_yunet_2023mar.onnx")
        self.reproducible_detector = None
        self._init_reproducible_detector()

    def _init_reproducible_detector(self):
        """Initialize fixed, deterministic YuNet detector for reproducible landmark alignment metrics."""
        if not os.path.exists(self.yunet_path):
            alt_path = os.path.join(settings.base_dir, "weights", "face_detection_yunet_2023mar.onnx")
            if os.path.exists(alt_path):
                self.yunet_path = alt_path
        if os.path.exists(self.yunet_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.reproducible_detector = cv2.FaceDetectorYN.create(
                    model=self.yunet_path,
                    config="",
                    input_size=(300, 300),
                    score_threshold=0.40,  # Fixed threshold for evaluation
                    nms_threshold=0.30,
                    top_k=5000,
                )
            except Exception as e:
                logger.warning(f"[EvaluationEngine] Failed to init evaluation YuNet detector: {e}")
                self.reproducible_detector = None

    def evaluate(
        self,
        orig_bgr: np.ndarray,
        protected_bgr: np.ndarray,
        faces: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Execute full scientific evaluation suite on original vs protected image.
        All reported metrics come exclusively from EvaluationEngine.
        """
        # 1. Quality Metrics
        global_ssim = self._compute_ssim(orig_bgr, protected_bgr)
        face_ssim = self._compute_face_ssim_expanded(orig_bgr, protected_bgr, faces)
        psnr = self._compute_psnr(orig_bgr, protected_bgr)
        l_inf = self._compute_l_inf(orig_bgr, protected_bgr)

        # 2. Y-Channel Baseline-Corrected NPR across compression suite
        npr_breakdown, avg_npr = self._compute_y_npr_suite(orig_bgr, protected_bgr)

        # 3. Downstream Effectiveness Metrics (Reproducible Landmark Shift & Alignment Error)
        eff_metrics = self._compute_alignment_effectiveness(orig_bgr, protected_bgr)

        return {
            "global_ssim": round(global_ssim, 4),
            "face_ssim_1_2x": round(face_ssim, 4),
            "psnr_db": round(psnr, 2),
            "l_inf_max_shift": int(l_inf),
            "avg_y_npr_percent": round(avg_npr, 2),
            "y_npr_breakdown": npr_breakdown,
            "landmark_shift_delta_px": round(eff_metrics["landmark_shift_delta_px"], 2),
            "alignment_error_orig": round(eff_metrics["alignment_error_orig"], 4),
            "alignment_error_protected": round(eff_metrics["alignment_error_protected"], 4),
            "alignment_error_pct_change": round(eff_metrics["alignment_error_pct_change"], 2),
        }

    def _compute_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Structural Similarity Index (SSIM) between two BGR images."""
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

        h, w = g1.shape[:2]
        if h < 11 or w < 11:
            g1 = cv2.resize(g1, (max(11, w), max(11, h)), interpolation=cv2.INTER_NEAREST)
            g2 = cv2.resize(g2, (max(11, w), max(11, h)), interpolation=cv2.INTER_NEAREST)

        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        mu1 = cv2.GaussianBlur(g1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(g2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(g1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(g2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(g1 * g2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return float(ssim_map.mean())

    def _compute_face_ssim_expanded(
        self, orig_bgr: np.ndarray, protected_bgr: np.ndarray, faces: List[Dict[str, Any]]
    ) -> float:
        """Compute Face SSIM over 1.2x expanded face bounding boxes."""
        if not faces:
            return self._compute_ssim(orig_bgr, protected_bgr)

        h, w = orig_bgr.shape[:2]
        ssim_scores = []

        for face in faces:
            fx, fy, fw, fh = face["box"]
            ex_w, ex_h = int(fw * 1.2), int(fh * 1.2)
            ex_x = max(0, fx - int(fw * 0.1))
            ex_y = max(0, fy - int(fh * 0.1))

            crop_o = orig_bgr[ex_y : min(h, ex_y + ex_h), ex_x : min(w, ex_x + ex_w)]
            crop_p = protected_bgr[ex_y : min(h, ex_y + ex_h), ex_x : min(w, ex_x + ex_w)]

            if crop_o.size > 0 and crop_p.size > 0:
                ssim_scores.append(self._compute_ssim(crop_o, crop_p))

        return float(np.mean(ssim_scores)) if ssim_scores else 1.0

    def _compute_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute Peak Signal-to-Noise Ratio (PSNR) in dB."""
        mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
        if mse == 0:
            return 100.0
        PIXEL_MAX = 255.0
        return float(20 * np.log10(PIXEL_MAX / np.sqrt(mse)))

    def _compute_l_inf(self, img1: np.ndarray, img2: np.ndarray) -> int:
        """Compute L_infinity (Max absolute pixel difference)."""
        diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))
        return int(np.max(diff))

    def _jpeg_compress_decompress_y(self, img_bgr: np.ndarray, quality: int, resize: float) -> np.ndarray:
        """Encode to JPEG and decode back, extracting Y (Luminance) channel."""
        h, w = img_bgr.shape[:2]
        if resize != 1.0:
            target_w, target_h = int(w * resize), int(h * resize)
            proc_img = cv2.resize(img_bgr, (target_w, target_h))
        else:
            proc_img = img_bgr

        _, buf = cv2.imencode(".jpg", proc_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        decompressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        if resize != 1.0:
            decompressed = cv2.resize(decompressed, (w, h))

        ycbcr = cv2.cvtColor(decompressed, cv2.COLOR_BGR2YCrCb)
        return ycbcr[:, :, 0].astype(np.float32)

    def _compute_y_npr_suite(
        self, orig_bgr: np.ndarray, protected_bgr: np.ndarray
    ) -> Tuple[Dict[str, float], float]:
        """
        Compute Y-Channel Baseline-Corrected Normalized Perturbation Retention (NPR):
        NPR = ||C - C0||2 / ||P - O||2 * 100%
        """
        O = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)
        P = cv2.cvtColor(protected_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float64)

        denom = np.linalg.norm(P - O)
        if denom < 1e-6:
            return {tier: 0.0 for tier in self.COMPRESSION_TIERS}, 0.0

        npr_scores = {}
        for tier_name, tier_cfg in self.COMPRESSION_TIERS.items():
            quality = tier_cfg["q"]
            resize_ratio = tier_cfg["resize"]

            C0 = self._jpeg_compress_decompress_y(orig_bgr, quality, resize_ratio).astype(np.float64)
            C = self._jpeg_compress_decompress_y(protected_bgr, quality, resize_ratio).astype(np.float64)

            num = np.linalg.norm(C - C0)
            npr = min(100.0, (num / denom) * 100.0)
            npr_scores[tier_name] = round(float(npr), 2)

        avg_npr = float(np.mean(list(npr_scores.values())))
        return npr_scores, avg_npr

    def _detect_landmarks_reproducible(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Detect 5 landmarks deterministically on image using evaluation detector."""
        if self.reproducible_detector is None:
            return None

        h, w = img_bgr.shape[:2]
        self.reproducible_detector.setInputSize((w, h))
        _, faces = self.reproducible_detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None

        # Return 5 landmarks of primary face as (5, 2) array
        face_data = faces[0]
        landmarks = np.array([
            [face_data[4], face_data[5]],   # R Eye
            [face_data[6], face_data[7]],   # L Eye
            [face_data[8], face_data[9]],   # Nose
            [face_data[10], face_data[11]], # R Mouth
            [face_data[12], face_data[13]], # L Mouth
        ], dtype=np.float32)
        return landmarks

    def _compute_alignment_error(self, lmks: np.ndarray) -> float:
        """Compute Affine Alignment Error against Canonical 5-point face shape."""
        # Estimate affine matrix transforming detected landmarks to canonical shape
        matrix, _ = cv2.estimateAffinePartial2D(lmks, self.CANONICAL_5_LANDMARKS)
        if matrix is None:
            return 0.0
        # Transform landmarks
        ones = np.ones((5, 1), dtype=np.float32)
        lmks_homo = np.hstack([lmks, ones])
        transformed = (matrix @ lmks_homo.T).T
        # Euclidean landmark point deviation (RMSE across 5 landmark coordinates)
        point_squared_errors = np.sum((transformed - self.CANONICAL_5_LANDMARKS) ** 2, axis=1)
        rmse = float(np.sqrt(np.mean(point_squared_errors)))
        return rmse

    def _compute_alignment_effectiveness(
        self, orig_bgr: np.ndarray, protected_bgr: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute landmark shift delta px, alignment error orig, alignment error protected,
        and alignment error percentage change.
        """
        lmks_orig = self._detect_landmarks_reproducible(orig_bgr)
        lmks_prot = self._detect_landmarks_reproducible(protected_bgr)

        if lmks_orig is None or lmks_prot is None:
            return {
                "landmark_shift_delta_px": 0.0,
                "alignment_error_orig": 0.0,
                "alignment_error_protected": 0.0,
                "alignment_error_pct_change": 0.0,
            }

        # 1. Landmark shift delta px
        shift_delta = float(np.mean(np.linalg.norm(lmks_prot - lmks_orig, axis=1)))

        # 2. Alignment error for original and protected faces
        err_orig = self._compute_alignment_error(lmks_orig)
        err_prot = self._compute_alignment_error(lmks_prot)

        # 3. Alignment error percentage change
        if err_orig > 1e-6:
            pct_change = ((err_prot - err_orig) / err_orig) * 100.0
        else:
            pct_change = 0.0

        return {
            "landmark_shift_delta_px": shift_delta,
            "alignment_error_orig": err_orig,
            "alignment_error_protected": err_prot,
            "alignment_error_pct_change": pct_change,
        }
