import os
import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.config import settings

logger = logging.getLogger(__name__)

# Initialize thread settings during module load to prevent CPU oversubscription
cv2.setNumThreads(settings.cv2_threads)


class ProtectionEngine:
    """
    Varman Dedicated Protection Engine (Generation Only).
    
    Order of Operations:
      1. YuNet Face & Landmark Detection (Primary -> Multi-Scale Fallback -> Skip if no face)
      2. Relative Soft-Masked Landmark Face Warping (Actual YuNet landmarks only)
      3. Y-Channel (Luminance) Mid-Frequency DCT Perturbation
      4. Optional Spatial Grid Layer
      5. Adaptive SSIM Quality Guardrail (enforces Global SSIM and 1.2x Face SSIM)
    """

    PRESETS = {
        "Light": {
            "warp_scale": 0.005,       # 0.5% of face width (Quality Mode)
            "dct_strength": 3.0,       # Subtle DCT shift
            "target_global_ssim": 0.95,
            "target_face_ssim": 0.95,
        },
        "Balanced": {
            "warp_scale": 0.015,       # Inherits previous Aggressive level (1.5% face width)
            "dct_strength": 10.0,      # Medium-high DCT shift
            "target_global_ssim": 0.90,
            "target_face_ssim": 0.90,
        },
        "Aggressive": {
            "warp_scale": 0.025,       # Higher disruption (2.5% face width)
            "dct_strength": 18.0,      # High-intensity DCT perturbation
            "target_global_ssim": 0.85, # SSIM Target Range: 0.85 <= SSIM <= 0.90
            "target_face_ssim": 0.85,
        },
    }

    STRONG_PRESETS: Dict[str, Dict[str, Any]] = {
        "F5": {
            "eps": 8.0 / 255.0,
            "mask_scale": 1.15,
            "steps": 25,
            "alpha": (8.0 / 255.0) / 7.0,
            "mu": 0.9,
            "w_fn": 1.0,
            "luminance_only": True,
            "chroma_scale": 0.0,
            "enable_guardrail": False,
            "feather_sigma": 4.0,
        },
    }

    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = model_dir or os.path.join(settings.base_dir, "models")
        self.yunet_path = os.path.join(self.model_dir, "face_detection_yunet_2023mar.onnx")
        self.yunet_detector = None
        self.strong_engine = None
        self._init_yunet()

    def _get_strong_engine(self):
        """Lazily initialize StrongProtectionEngine on CUDA device."""
        if self.strong_engine is None:
            from app.protection.strong_engine import StrongProtectionEngine
            self.strong_engine = StrongProtectionEngine()
        return self.strong_engine

    def _init_yunet(self):
        """Lazy load YuNet face detector if ONNX file is present."""
        if os.path.exists(self.yunet_path) and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet_detector = cv2.FaceDetectorYN.create(
                    model=self.yunet_path,
                    config="",
                    input_size=(300, 300),
                    score_threshold=0.6,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                logger.info(f"[ProtectionEngine] Loaded YuNet face detector from {self.yunet_path}")
            except Exception as e:
                logger.warning(f"[ProtectionEngine] Failed to load YuNet ONNX: {e}")
                self.yunet_detector = None
        else:
            logger.info("[ProtectionEngine] YuNet ONNX not found. Face warp will be skipped if model is absent.")

    def detect_faces(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces using YuNet with conditional multi-scale fallback.
        Returns actual 5-point YuNet landmarks only (R eye, L eye, nose, R mouth, L mouth).
        No Haar Cascade or synthetic landmarks used.
        """
        if self.yunet_detector is None:
            return []

        h, w = img_bgr.shape[:2]
        faces = []

        # 1. Primary YuNet pass (Score threshold = 0.60)
        self.yunet_detector.setInputSize((w, h))
        self.yunet_detector.setScoreThreshold(0.60)
        _, raw_faces = self.yunet_detector.detect(img_bgr)

        # 2. Conditional multi-scale fallback if no face detected or low confidence
        if raw_faces is None or len(raw_faces) == 0:
            self.yunet_detector.setScoreThreshold(0.40)
            for scale in (0.75, 1.25):
                sw, sh = int(w * scale), int(h * scale)
                resized = cv2.resize(img_bgr, (sw, sh))
                self.yunet_detector.setInputSize((sw, sh))
                _, multi_faces = self.yunet_detector.detect(resized)
                if multi_faces is not None and len(multi_faces) > 0:
                    multi_faces[:, :14] /= scale
                    raw_faces = multi_faces
                    break

        # 3. Parse actual 5-point landmarks
        if raw_faces is not None and len(raw_faces) > 0:
            for face_data in raw_faces:
                fx, fy, fw, fh = map(int, face_data[:4])
                score = float(face_data[-1])
                landmarks = [
                    (int(face_data[4]), int(face_data[5])),   # Right Eye
                    (int(face_data[6]), int(face_data[7])),   # Left Eye
                    (int(face_data[8]), int(face_data[9])),   # Nose Tip
                    (int(face_data[10]), int(face_data[11])), # Right Mouth Corner
                    (int(face_data[12]), int(face_data[13])), # Left Mouth Corner
                ]
                faces.append({
                    "box": (max(0, fx), max(0, fy), min(w - fx, fw), min(h - fy, fh)),
                    "landmarks": landmarks,
                    "score": score,
                    "method": "YuNet"
                })

        return faces

    def apply_relative_soft_masked_warp(
        self, img_bgr: np.ndarray, faces: List[Dict[str, Any]], warp_scale: float
    ) -> np.ndarray:
        """
        Apply spatial micro-warp relative to face width with Gaussian soft masking around YuNet landmarks.
        If faces list is empty, returns original image unchanged.
        """
        if not faces or warp_scale <= 0:
            return img_bgr.copy()

        h, w = img_bgr.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        dx = np.zeros((h, w), dtype=np.float32)
        dy = np.zeros((h, w), dtype=np.float32)

        for face in faces:
            fx, fy, fw, fh = face["box"]
            landmarks = face["landmarks"]
            if fw <= 0 or fh <= 0:
                continue

            max_disp = warp_scale * fw
            sigma = 0.25 * fw

            # Opposing offset multipliers for 5 YuNet landmarks
            offsets = [
                (max_disp, -max_disp * 0.5),   # Right Eye
                (-max_disp, max_disp * 0.5),   # Left Eye
                (0.0, max_disp),               # Nose Tip
                (-max_disp * 0.7, -max_disp),  # Right Mouth Corner
                (max_disp * 0.7, max_disp),    # Left Mouth Corner
            ]

            for (lx, ly), (off_x, off_y) in zip(landmarks, offsets):
                dist_sq = (map_x - lx) ** 2 + (map_y - ly) ** 2
                weight = np.exp(-dist_sq / (2.0 * sigma ** 2))
                dx += weight * off_x
                dy += weight * off_y

        remapped_x = np.clip(map_x + dx, 0, w - 1).astype(np.float32)
        remapped_y = np.clip(map_y + dy, 0, h - 1).astype(np.float32)
        warped = cv2.remap(img_bgr, remapped_x, remapped_y, interpolation=cv2.INTER_CUBIC)
        return warped

    def apply_y_channel_dct_perturbation(
        self, img_bgr: np.ndarray, dct_strength: float
    ) -> np.ndarray:
        """
        Apply mid-frequency DCT perturbation exclusively on the Y (Luminance) channel in YCbCr space.
        Uses a 2D mid-frequency coefficient mask with a spatially alternating sign pattern.
        """
        if dct_strength <= 0:
            return img_bgr.copy()

        ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        y_chan, cr, cb = cv2.split(ycbcr)
        y_float = y_chan.astype(np.float32)

        h, w = y_float.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            y_padded = np.pad(y_float, ((0, pad_h), (0, pad_w)), mode="reflect")
        else:
            y_padded = y_float

        ph, pw = y_padded.shape
        y_out = y_padded.copy()

        # Mid-frequency coefficient mask for 8x8 block (indices 2 to 5, excluding DC)
        mid_freq_mask = np.zeros((8, 8), dtype=np.float32)
        mid_freq_mask[2:6, 2:6] = 1.0
        mid_freq_mask[0, 0] = 0.0

        for r in range(0, ph, 8):
            for c in range(0, pw, 8):
                block = y_padded[r : r + 8, c : c + 8]
                dct_block = cv2.dct(block)
                
                sign_pattern = 1.0 if ((r // 8 + c // 8) % 2 == 0) else -1.0
                dct_block += mid_freq_mask * (dct_strength * sign_pattern)
                
                idct_block = cv2.idct(dct_block)
                y_out[r : r + 8, c : c + 8] = idct_block

        y_final = np.clip(y_out[:h, :w], 0, 255).astype(np.uint8)
        protected_ycbcr = cv2.merge([y_final, cr, cb])
        protected_bgr = cv2.cvtColor(protected_ycbcr, cv2.COLOR_YCrCb2BGR)
        return protected_bgr

    def apply_optional_grid_layer(self, img_bgr: np.ndarray, alpha: float = 0.02) -> np.ndarray:
        """Apply optional spatial grid layer."""
        if alpha <= 0:
            return img_bgr.copy()

        h, w = img_bgr.shape[:2]
        grid = np.zeros((h, w, 3), dtype=np.float32)
        grid[::8, :, :] = 1.0
        grid[:, ::8, :] = 1.0

        pattern = grid * 10.0
        out = img_bgr.astype(np.float32) + alpha * pattern
        return np.clip(out, 0, 255).astype(np.uint8)

    def protect(
        self,
        img_bgr: np.ndarray,
        preset_name: str = "Balanced",
        enable_grid: bool = False,
        mode: str = "fast",
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute image protection pipeline with explicit mode selection:
          - mode='fast' (default): Lightweight CPU-based YuNet + landmark warping + DCT perturbation.
          - mode='strong': GPU-based EOT-PGD directly optimizing aligned face crops.
                           If CUDA is unavailable, explicitly falls back to 'fast' with clear metadata flags.
        """
        requested_mode = mode.lower() if isinstance(mode, str) else "fast"

        if requested_mode == "strong":
            import torch
            if not torch.cuda.is_available():
                logger.warning(
                    "[ProtectionEngine] Strong mode requested, but CUDA is unavailable. "
                    "Falling back to fast CPU mode."
                )
                prot_bgr, meta = self._protect_fast(img_bgr, preset_name, enable_grid)
                meta["mode"] = "fast"
                meta["requested_mode"] = "strong"
                meta["fallback_to_fast"] = True
                meta["fallback_reason"] = "CUDA is not available on this host for strong mode execution"
                return prot_bgr, meta

            # CUDA is available - execute StrongProtectionEngine
            faces = self.detect_faces(img_bgr)
            logger.info(f"[ProtectionEngine] Strong mode: detected {len(faces)} face(s).")
            strong_engine = self._get_strong_engine()
            prot_bgr, strong_meta = strong_engine.protect_image(
                img_bgr, faces, **kwargs
            )
            strong_meta["preset"] = preset_name
            strong_meta["mode"] = "strong"
            strong_meta["requested_mode"] = "strong"
            strong_meta["fallback_to_fast"] = False
            strong_meta["face_count"] = len(faces)
            strong_meta["faces"] = faces
            return prot_bgr, strong_meta

        # Default: Fast CPU mode
        prot_bgr, meta = self._protect_fast(img_bgr, preset_name, enable_grid)
        meta["mode"] = "fast"
        meta["requested_mode"] = "fast"
        meta["fallback_to_fast"] = False
        return prot_bgr, meta

    def protect_image(
        self,
        img_bgr: np.ndarray,
        preset_name: str = "Balanced",
        enable_grid: bool = False,
        mode: str = "fast",
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Alias for protect()."""
        return self.protect(img_bgr, preset_name=preset_name, enable_grid=enable_grid, mode=mode, **kwargs)

    def _protect_fast(
        self,
        img_bgr: np.ndarray,
        preset_name: str = "Balanced",
        enable_grid: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Internal execution of the fast CPU-based YuNet + DCT engine."""
        preset = self.PRESETS.get(preset_name, self.PRESETS["Balanced"])
        warp_scale = preset["warp_scale"]
        dct_strength = preset["dct_strength"]
        target_global_ssim = preset["target_global_ssim"]
        target_face_ssim = preset["target_face_ssim"]

        faces = self.detect_faces(img_bgr)
        logger.info(f"[ProtectionEngine] Fast mode: detected {len(faces)} face(s) using preset '{preset_name}'.")

        attenuation_factor = 1.0
        min_attenuation = 0.1
        step_factor = 0.85
        guardrail_satisfied = False

        protected_bgr = img_bgr.copy()
        curr_img = img_bgr.copy()

        while attenuation_factor >= min_attenuation:
            curr_warp = warp_scale * attenuation_factor
            curr_dct = dct_strength * attenuation_factor

            step2_img = self.apply_relative_soft_masked_warp(img_bgr, faces, curr_warp)
            step3_img = self.apply_y_channel_dct_perturbation(step2_img, curr_dct)

            if enable_grid:
                curr_img = self.apply_optional_grid_layer(step3_img, alpha=0.02 * attenuation_factor)
            else:
                curr_img = step3_img

            # Internal fast SSIM check for guardrail loop
            g_ssim = self._compute_fast_ssim(img_bgr, curr_img)
            f_ssim = 1.0
            if faces:
                f_ssim_list = []
                h, w = img_bgr.shape[:2]
                for face in faces:
                    fx, fy, fw, fh = face["box"]
                    ex_w, ex_h = int(fw * 1.2), int(fh * 1.2)
                    ex_x = max(0, fx - int(fw * 0.1))
                    ex_y = max(0, fy - int(fh * 0.1))
                    crop_orig = img_bgr[ex_y : min(h, ex_y + ex_h), ex_x : min(w, ex_x + ex_w)]
                    crop_prot = curr_img[ex_y : min(h, ex_y + ex_h), ex_x : min(w, ex_x + ex_w)]
                    if crop_orig.size > 0 and crop_prot.size > 0:
                        f_ssim_list.append(self._compute_fast_ssim(crop_orig, crop_prot))
                if f_ssim_list:
                    f_ssim = min(f_ssim_list)

            if g_ssim >= target_global_ssim and f_ssim >= target_face_ssim:
                protected_bgr = curr_img
                guardrail_satisfied = True
                break

            attenuation_factor *= step_factor

        if not guardrail_satisfied:
            logger.warning(
                f"[ProtectionEngine] Guardrail could NOT satisfy target SSIM ({target_global_ssim}) "
                f"before min attenuation limit ({min_attenuation}). Setting guardrail_satisfied=False."
            )
            protected_bgr = curr_img

        meta = {
            "preset": preset_name,
            "face_count": len(faces),
            "faces": faces,
            "guardrail_satisfied": guardrail_satisfied,
            "attenuation_factor": round(attenuation_factor, 3),
            "applied_warp_scale": round(warp_scale * attenuation_factor, 4),
            "applied_dct_strength": round(dct_strength * attenuation_factor, 2),
            "grid_enabled": enable_grid,
        }

        return protected_bgr, meta

    def _compute_fast_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Internal fast SSIM metric for guardrail attenuation loop."""
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
