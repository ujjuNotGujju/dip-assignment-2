"""Image-quality metrics built from scratch (NumPy array arithmetic only)."""
import numpy as np

from .utils import convolve2d_valid, gaussian_kernel2d


def psnr(img1: np.ndarray, img2: np.ndarray, data_range: float) -> float:
    """Peak Signal-to-Noise Ratio between two same-shape arrays."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((data_range ** 2) / mse)


def ssim(img1: np.ndarray, img2: np.ndarray, data_range: float,
         win_size: int = 11, sigma: float = 1.5) -> float:
    """Single-scale SSIM (Wang et al.) with an 11x11 Gaussian window, from scratch.

    Local means/variances/covariance computed via valid-mode Gaussian convolution
    (src/dip_a2/utils.py::convolve2d_valid), no scipy/skimage.
    """
    a = img1.astype(np.float64)
    b = img2.astype(np.float64)
    kernel = gaussian_kernel2d(win_size, sigma)

    mu1 = convolve2d_valid(a, kernel)
    mu2 = convolve2d_valid(b, kernel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = convolve2d_valid(a * a, kernel) - mu1_sq
    sigma2_sq = convolve2d_valid(b * b, kernel) - mu2_sq
    sigma12 = convolve2d_valid(a * b, kernel) - mu1_mu2

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())
