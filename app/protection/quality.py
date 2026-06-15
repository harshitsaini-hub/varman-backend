from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

def compute_quality_metrics(original_path: str, protected_path: str):
    """
    Compute SSIM and PSNR between original and protected images.
    """
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(protected_path)
    
    if img1 is None or img2 is None:
        return 0.0, 0.0
        
    # Ensure same size for metric calculation
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    ssim_val = ssim(img1_gray, img2_gray)
    psnr_val = psnr(img1_gray, img2_gray)
    
    # Type checker workaround: ssim() overload can return a tuple when full=True
    if isinstance(ssim_val, tuple):
        ssim_val = ssim_val[0]
    if isinstance(psnr_val, tuple):
        psnr_val = psnr_val[0]
        
    return float(ssim_val), float(psnr_val)
