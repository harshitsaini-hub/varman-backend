import cv2
import numpy as np

def compute_quality_metrics(original_path: str, protected_path: str):
    """
    Compute SSIM and PSNR between original and protected images using OpenCV/NumPy.
    """
    img1 = cv2.imread(original_path)
    img2 = cv2.imread(protected_path)
    
    if img1 is None or img2 is None:
        return 0.0, 0.0
        
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Compute PSNR
    mse = np.mean((g1 - g2) ** 2)
    if mse == 0:
        psnr_val = 100.0
    else:
        psnr_val = float(20 * np.log10(255.0 / np.sqrt(mse)))

    # Compute SSIM
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
    ssim_val = float(ssim_map.mean())
        
    return ssim_val, psnr_val
