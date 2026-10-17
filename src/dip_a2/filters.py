"""Frequency-domain filter transfer functions (ideal / Butterworth / Gaussian),
built from their formulas with plain NumPy array arithmetic, plus the mandated
zero-pad -> centre -> multiply -> invert -> crop filtering pipeline.

numpy.fft is used for the forward/inverse transforms themselves (explicitly allowed
for this problem by the assignment); only the transfer functions H(u,v) and the
pipeline plumbing are "from scratch".
"""
import numpy as np

from .dft import fftshift2d, ifftshift2d
from .utils import zero_pad_to


def radial_distance(shape) -> np.ndarray:
    """D(u,v): distance from the centre of a P x Q frequency rectangle."""
    P, Q = shape
    u = np.arange(P).reshape(-1, 1) - P // 2
    v = np.arange(Q).reshape(1, -1) - Q // 2
    return np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)


def make_filter(kind: str, shape, D0: float, n: int = None, highpass: bool = False) -> np.ndarray:
    """Return the P x Q transfer function H(u,v) for kind in {ideal, butterworth, gaussian}."""
    D = radial_distance(shape)
    if kind == "ideal":
        H = (D <= D0).astype(np.float64)
    elif kind == "butterworth":
        if n is None:
            raise ValueError("Butterworth filter requires an order n")
        with np.errstate(divide="ignore", invalid="ignore"):
            H = 1.0 / (1.0 + (D / D0) ** (2 * n))
        H = np.nan_to_num(H, nan=0.0, posinf=0.0)
        if D0 == 0:
            H[:] = 0.0
    else:
        if kind != "gaussian":
            raise ValueError(f"unknown filter kind: {kind}")
        H = np.exp(-(D ** 2) / (2.0 * D0 ** 2))
    if highpass:
        H = 1.0 - H
    return H


def filter_pipeline(image: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Zero-pad to H's shape, centre spectrum, multiply by H, invert, real part, crop.

    H must already be built on the padded P x Q grid (e.g. P=2M, Q=2N).
    """
    M, N = image.shape
    P, Q = H.shape
    padded = zero_pad_to(image.astype(np.float64), (P, Q))
    F = np.fft.fft2(padded)
    F_centered = fftshift2d(F)
    G_centered = F_centered * H
    G = ifftshift2d(G_centered)
    g_padded = np.real(np.fft.ifft2(G))
    return g_padded[:M, :N]
