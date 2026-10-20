# Problem 6 — Motion Blur and Deblurring (20 pts)

Status: **done**. See [REPORT.pdf](REPORT.pdf) for the full write-up.

Data: `data/provided/problem5_6/reference_image.jpg` (shared with Problem 5) + a
genuinely motion-blurred self-captured photo
(`data/own_captures/motion_blur.jpeg`, a hand swiped in front of the camera) for §6.4.

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
- [x] Estimated θ=99.0°, L=2.69px from the real photo's log-magnitude spectrum;
      Wiener-deconvolved (K=0.01). The estimate under-measures the true blur extent
      because the photo's blur is localized to a moving hand against an otherwise
      sharp background, violating the model's spatially-invariant-PSF assumption —
      documented as the required model-failure case study (see REPORT.pdf).

## Deliverables
PSF derivation + zero-spacing formula; synthetic inverse-filter spectrum with predicted
zeros marked; Wiener K sweeps + optimum-vs-noise; real blur case study with a
documented model-failure mode.
