# Problem 6 — Motion Blur and Deblurring (20 pts)

Status: **mostly done — one sub-part pending real data**. See [REPORT.md](REPORT.md)
for the full write-up.

Data: `data/provided/problem5_6/reference_image.jpg` (shared with Problem 5) + a
genuinely motion-blurred self-captured photo (`data/own_captures/`) for §6.4 —
**not yet supplied, see below**.

## 6.1 Model and Synthesize (5 pts)
- [x] Derive continuous PSF for linear motion (length L, angle θ); discretize/normalize.
- [x] State boundary convention (circular, used consistently); blur + add noise at
      stated SNR (defined explicitly); not clipped before filtering.
- [x] Derive motion OTF `H = sinc(L(fx cosθ + fy sinθ))`; zero-line spacing 1/L,
      perpendicular to motion direction; numerically verified against the rasterized
      kernel's own DFT.

## 6.2 Naive Inverse Filtering (5 pts)
- [x] `F_hat = G/H` with a stated numerical floor; log-magnitude spectra with predicted
      zero locations marked; explained amplification there (20.5x at floored bins vs.
      other high-frequency bins).

## 6.3 Wiener Deconvolution (5 pts)
- [x] Derive/implement `F_hat = H* G / (|H|^2 + K)`; sweep K; report K* at 4 noise
      levels (10/20/30/40 dB) and how it shifts with noise (monotonically, as expected).

## 6.4 Estimate the Blur, Then Remove It (5 pts)
- [ ] **PENDING.** `data/own_captures/` is empty; this sub-part is explicitly
      incomplete (not faked with synthetic blur). `problem6.py` will process any
      motion-blurred photo dropped into `data/own_captures/` automatically on the next
      run — no code changes needed, though the blur-estimation algorithm itself has
      not yet been validated against real data.

## Deliverables
PSF derivation + zero-spacing formula; synthetic inverse-filter spectrum with predicted
zeros marked; Wiener K sweeps + optimum-vs-noise; real blur case study **(pending)**.
