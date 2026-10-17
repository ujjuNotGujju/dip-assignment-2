"""Image-quality metrics built from scratch (NumPy array arithmetic only)."""
import numpy as np


def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float) -> float:
    """Peak Signal-to-Noise Ratio between two same-shape arrays."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)
