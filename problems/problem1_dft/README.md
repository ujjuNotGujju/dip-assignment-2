# Problem 1 — DFT and the Machinery of Frequency (15 pts)

Status: **done**. See [REPORT.pdf](REPORT.pdf) for the full write-up.

Data: `data/provided/problem1/q1.png` (standard 512x512 grayscale test image), plus a
self-synthesized sinusoid-sum image (ground-truth parameters recorded before any DFT
is taken — no external image for this subtask).

## 1.1 Build and Validate the DFT (6 pts)
- [x] Custom 1-D DFT matrix `W_N`; 2-D DFT via `F = W_N f W_N^T`.
- [x] Validate vs `numpy.fft.fft2`: report `max |F_custom - F_numpy|`.
- [x] (a) Conjugate symmetry proof + redundant-region explanation.
- [x] (b) Empirical verification of 2-D Spatial Shift Theorem and 2-D Circular
      Convolution Theorem, with quantitative tables.
- [x] (c) Sinusoid parameter recovery from spectral peaks + Δf derivation.

## 1.2 Phase vs. Magnitude Isolation (5 pts)
- [x] Dated pre-implementation prediction written **before** running this part:
      `predictions/1.2_phase_magnitude_prediction.md`.
- [x] (a) Phase of test image + flat magnitude -> reconstruct, explain, reconcile.
- [x] (b) Magnitude of test image + zero phase -> reconstruct, explain via symmetry.

## 1.3 Padding, Wraparound, Computational Overhead (4 pts)
- [x] Frequency-domain convolution with a 5x5 Laplacian/Sobel kernel via custom DFT.
- [x] (a) Circular vs. zero-padded linear convolution; wraparound evidence; minimal
      padding-size derivation.
- [x] (b) Memory/runtime trade-off analysis, related to `scipy.signal.fftconvolve`
      (used here only as the explicitly-permitted comparison target).

## Deliverables
- Quantitative error table vs `numpy.fft.fft2`.
- Shift/convolution theorem verification tables.
- Dated prediction + reconstruction plots + reconciliation.
- Recovered sinusoid parameters + Δf derivation.
- Wraparound images + padding-size derivation.

Code: `problem1.py`. Results: `results/`.
