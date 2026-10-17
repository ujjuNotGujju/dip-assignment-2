"""From-scratch Discrete Fourier Transform machinery (NumPy array arithmetic only).

Used across the assignment wherever a 2-D DFT/IDFT is needed. ``numpy.fft`` is only
used elsewhere for validation against this implementation, never to compute it.
"""
import numpy as np


def dft_matrix(N: int) -> np.ndarray:
    """Return the N x N DFT matrix W_N with W_N[k, n] = exp(-2j*pi*k*n/N)."""
    n = np.arange(N)
    k = n.reshape(-1, 1)
    return np.exp(-2j * np.pi * k * n / N)


def idft_matrix(N: int) -> np.ndarray:
    """Return the N x N inverse-DFT matrix, i.e. conj(W_N) / N."""
    return np.conj(dft_matrix(N)) / N


def dft2d(f: np.ndarray) -> np.ndarray:
    """2-D DFT via row-column separability: F = W_M f W_N^T for an M x N image."""
    M, N = f.shape
    W_M = dft_matrix(M)
    W_N = dft_matrix(N)
    return W_M @ f.astype(np.complex128) @ W_N.T


def idft2d(F: np.ndarray) -> np.ndarray:
    """2-D inverse DFT via row-column separability."""
    M, N = F.shape
    Wi_M = idft_matrix(M)
    Wi_N = idft_matrix(N)
    return Wi_M @ F @ Wi_N.T


def fftshift2d(F: np.ndarray) -> np.ndarray:
    """Shift zero-frequency component to the array centre (from scratch, no np.fft)."""
    M, N = F.shape
    return np.roll(np.roll(F, M // 2, axis=0), N // 2, axis=1)


def ifftshift2d(F: np.ndarray) -> np.ndarray:
    """Inverse of fftshift2d (correct for both even and odd dimensions)."""
    M, N = F.shape
    return np.roll(np.roll(F, -(M // 2), axis=0), -(N // 2), axis=1)
