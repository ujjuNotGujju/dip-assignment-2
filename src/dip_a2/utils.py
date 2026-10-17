"""Shared I/O and small helpers (image loading, zero-padding, etc.)."""
import numpy as np


def zero_pad_to(img: np.ndarray, shape: tuple) -> np.ndarray:
    """Zero-pad a 2-D array to ``shape`` (placed at the top-left corner)."""
    out = np.zeros(shape, dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img
    return out


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
