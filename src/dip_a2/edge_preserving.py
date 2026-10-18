"""Bilateral and Kuwahara edge-preserving filters, from scratch (NumPy array
arithmetic only). Both are vectorized by looping over the small window/sub-window
OFFSETS (not pixels) and accumulating shifted whole-image arrays -- the same pattern
used throughout this repo for spatial convolutions (see e.g. Problem 1's
circular_convolve_spatial).
"""
import numpy as np


def bilateral_filter(image: np.ndarray, sigma_s: float, sigma_r: float, radius: int = None) -> np.ndarray:
    """Joint spatial-Gaussian x intensity-range-Gaussian filter, normalized per pixel.

    w(i,j; di,dj) = exp(-(di^2+dj^2)/(2*sigma_s^2)) * exp(-(I(i,j)-I(i+di,j+dj))^2/(2*sigma_r^2))
    """
    if radius is None:
        radius = max(1, int(round(3 * sigma_s)))
    img = image.astype(np.float64)
    H, W = img.shape

    # reflect-pad so every pixel (including borders) has a full window available
    padded = np.pad(img, radius, mode="reflect")

    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)

    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            spatial_w = np.exp(-(di ** 2 + dj ** 2) / (2.0 * sigma_s ** 2))
            neighbor = padded[radius + di:radius + di + H, radius + dj:radius + dj + W]
            range_w = np.exp(-((neighbor - img) ** 2) / (2.0 * sigma_r ** 2))
            w = spatial_w * range_w
            num += w * neighbor
            den += w
    return num / den


def _subwindow_stats(padded: np.ndarray, H: int, W: int, q: int, quadrant: str):
    """Mean and variance of a (q+1)x(q+1) sub-window in `quadrant` relative to each
    centre pixel, over the whole HxW image, via the shifted-offset-sum pattern.
    `padded` must already be padded by `q` on every side (reflect)."""
    if quadrant == "tl":
        rows, cols = range(-q, 1), range(-q, 1)
    elif quadrant == "tr":
        rows, cols = range(-q, 1), range(0, q + 1)
    elif quadrant == "bl":
        rows, cols = range(0, q + 1), range(-q, 1)
    else:  # "br"
        rows, cols = range(0, q + 1), range(0, q + 1)

    s = np.zeros((H, W), dtype=np.float64)
    s2 = np.zeros((H, W), dtype=np.float64)
    n = 0
    for di in rows:
        for dj in cols:
            val = padded[q + di:q + di + H, q + dj:q + dj + W]
            s += val
            s2 += val * val
            n += 1
    mean = s / n
    var = s2 / n - mean ** 2
    return mean, var


def _kuwahara_subwindow_stats(image: np.ndarray, subwindow_size: int):
    """Returns (means, variances), each shape (4, H, W), for the 4 classical
    overlapping Kuwahara sub-windows (each subwindow_size x subwindow_size, sharing
    the centre pixel at a corner)."""
    q = subwindow_size - 1  # subwindow spans q+1 = subwindow_size pixels
    img = image.astype(np.float64)
    H, W = img.shape
    padded = np.pad(img, q, mode="reflect")

    means, variances = [], []
    for quadrant in ("tl", "tr", "bl", "br"):
        m, v = _subwindow_stats(padded, H, W, q, quadrant)
        means.append(m)
        variances.append(v)
    return np.stack(means, axis=0), np.stack(variances, axis=0)


def kuwahara_filter(image: np.ndarray, subwindow_size: int) -> np.ndarray:
    """Classical Kuwahara: output = mean of the minimum-variance sub-window (hard
    argmin per pixel). Total window span = 2*subwindow_size - 1."""
    means, variances = _kuwahara_subwindow_stats(image, subwindow_size)
    best = np.argmin(variances, axis=0)  # (H, W)
    H, W = image.shape
    out = np.take_along_axis(means, best[None, :, :], axis=0)[0]
    return out


def soft_kuwahara_filter(image: np.ndarray, subwindow_size: int, temperature: float) -> np.ndarray:
    """Soft (de-blocked) Kuwahara: output = softmax(-variance/temperature)-weighted
    combination of the 4 sub-window means, instead of a hard argmin."""
    means, variances = _kuwahara_subwindow_stats(image, subwindow_size)
    logits = -variances / temperature
    logits -= logits.max(axis=0, keepdims=True)  # numerical stability
    weights = np.exp(logits)
    weights /= weights.sum(axis=0, keepdims=True)
    return np.sum(weights * means, axis=0)
