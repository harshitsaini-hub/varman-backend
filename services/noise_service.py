import numpy as np
from scipy.fftpack import dct, idct


def apply_frequency_domain_noise(image_array: np.ndarray, epsilon: float = 0.04) -> np.ndarray:
    """
    Add adversarial noise in DCT frequency domain.
    Survives JPEG compression significantly better than pixel-space noise.
    """
    result = image_array.astype(float).copy()

    for channel in range(3):
        channel_data = result[:, :, channel]
        # Transform to frequency domain
        freq = dct(dct(channel_data.T, norm="ortho").T, norm="ortho")
        # Add perturbation to mid-frequency components (survives compression)
        noise_mask = np.random.choice([-1, 1], size=freq.shape) * epsilon * 255
        # Target mid-frequencies, not high-frequencies (which JPEG discards anyway)
        freq[8:64, 8:64] += noise_mask[8:64, 8:64]
        # Transform back
        result[:, :, channel] = idct(idct(freq.T, norm="ortho").T, norm="ortho")

    return np.clip(result, 0, 255).astype(np.uint8)
