"""Shared I/O and small helpers (image loading, zero-padding, etc.)."""
import numpy as np


def zero_pad_to(img: np.ndarray, shape: tuple) -> np.ndarray:
    """Zero-pad a 2-D array to ``shape`` (placed at the top-left corner)."""
    out = np.zeros(shape, dtype=img.dtype)
    out[: img.shape[0], : img.shape[1]] = img
    return out
