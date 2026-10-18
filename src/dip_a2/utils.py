"""Shared I/O and small helpers (image loading, zero-padding, etc.)."""
import numpy as np


def zero_pad_to(img: np.ndarray, shape: tuple) -> np.ndarray:
    """Zero-pad a 2-D array to ``shape`` (placed at the top-left corner)."""
    out = np.zeros(shape, dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img
    return out


def convolve2d_valid(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """'Valid'-mode 2-D convolution (no padding, output shrinks by kernel_size-1),
    via the shifted-offset-sum pattern: loop over the (small) kernel, not pixels.
    """
    kH, kW = kernel.shape
    H, W = img.shape
    out = np.zeros((H - kH + 1, W - kW + 1), dtype=np.float64)
    for i in range(kH):
        for j in range(kW):
            w = kernel[i, j]
            if w == 0:
                continue
            out += w * img[i:i + out.shape[0], j:j + out.shape[1]]
    return out


def gaussian_kernel2d(size: int, sigma: float) -> np.ndarray:
    """Normalized 2-D Gaussian kernel of shape (size, size), from scratch."""
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def histogram_equalize(image: np.ndarray) -> np.ndarray:
    """Global histogram equalization on an 8-bit-range image, from scratch.

    Builds the 256-bin histogram, its normalized CDF, and maps each pixel through it.
    """
    img_u8 = np.clip(image, 0, 255).astype(np.uint8)
    hist = np.bincount(img_u8.ravel(), minlength=256).astype(np.float64)
    cdf = np.cumsum(hist)
    cdf /= cdf[-1]
    lut = np.round(cdf * 255.0).astype(np.float64)
    return lut[img_u8]
