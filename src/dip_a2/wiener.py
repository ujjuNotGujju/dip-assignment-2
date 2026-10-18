"""Wiener denoising, from scratch (NumPy array arithmetic + numpy.fft)."""
import numpy as np

from .dft import fftshift2d, ifftshift2d
from .filters import radial_distance


def radial_average(array2d: np.ndarray) -> np.ndarray:
    """Isotropic smoothing of a centred 2-D array: bins by integer radius and
    replaces every pixel with its bin's mean (the PDF's "radially averaged
    periodogram"). `array2d` must already be in centred (fftshifted) layout.
    """
    D = radial_distance(array2d.shape)
    bins = np.round(D).astype(np.int64)
    flat_bins = bins.ravel()
    sums = np.bincount(flat_bins, weights=array2d.ravel())
    counts = np.bincount(flat_bins)
    profile = sums / counts
    return profile[bins]


def wiener_denoise(noisy: np.ndarray, sigma_noise: float, assumed_noise_scale: float = 1.0):
    """MMSE Wiener denoiser. Normalizes to [0,1] internally.

    Snn is set from the KNOWN injected noise variance (sigma_noise), scaled by
    `assumed_noise_scale` to deliberately mis-specify it (Problem 5.2). Sff is
    estimated purely from the noisy image's own radially-averaged periodogram minus
    Snn -- no access to the clean image, so this is NOT oracle-tuned.

    Returns (denoised_image_0_255, W_radial_profile) where W_radial_profile[r] is the
    Wiener gain at integer radius r (for the frequency-shrinkage discussion/plot).
    """
    M, N = noisy.shape
    g = noisy.astype(np.float64) / 255.0
    sigma_n01 = sigma_noise / 255.0

    G = fftshift2d(np.fft.fft2(g))
    periodogram = np.abs(G) ** 2

    Snn = (sigma_n01 ** 2) * M * N * assumed_noise_scale
    Sff_raw = radial_average(periodogram) - Snn
    eps = 1e-8 * M * N
    Sff = np.clip(Sff_raw, eps, None)

    W = Sff / (Sff + Snn)
    F_hat = W * G
    f_hat = np.real(np.fft.ifft2(ifftshift2d(F_hat)))

    denoised = np.clip(f_hat, 0.0, 1.0) * 255.0

    D = radial_distance((M, N))
    bins = np.round(D).astype(np.int64)
    flat_bins = bins.ravel()
    W_sum = np.bincount(flat_bins, weights=W.ravel())
    W_count = np.bincount(flat_bins)
    W_radial_profile = W_sum / W_count

    return denoised, W_radial_profile
