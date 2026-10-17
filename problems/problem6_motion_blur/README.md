# Problem 6 — Motion Blur and Deblurring (20 pts)

Status: **not started**.

Data: `data/provided/problem5_6/reference_image.jpg` (shared with Problem 5) + a
genuinely motion-blurred self-captured photo (`data/own_captures/`) for §6.4.

## 6.1 Model and Synthesize (5 pts)
- [ ] Derive continuous PSF for linear motion (length L, angle θ); discretize/normalize.
- [ ] State boundary convention (circular recommended); blur + add noise at stated SNR
      (defined explicitly); don't clip before filtering.
- [ ] Derive motion OTF `H = sinc(L(fx cosθ + fy sinθ))`; derive zero-line spacing 1/L,
      perpendicular to motion direction.

## 6.2 Naive Inverse Filtering (5 pts)
- [ ] `F_hat = G/H` with a stated numerical floor; log-magnitude spectra with predicted
      zero locations marked; explain amplification there.

## 6.3 Wiener Deconvolution (5 pts)
- [ ] Derive/implement `F_hat = H* G / (|H|^2 + K)`; sweep K; report K* at >=3 noise
      levels and how it shifts with noise.

## 6.4 Estimate the Blur, Then Remove It (5 pts)
- [ ] From own motion-blurred photo: estimate θ, L from log-magnitude spectrum stripe
      spacing; build PSF; Wiener-deconvolve; qualitative evaluation + documented model
      failure (no ground truth available).

## Deliverables
PSF derivation + zero-spacing formula; synthetic inverse-filter spectrum with predicted
zeros marked; Wiener K sweeps + optimum-vs-noise; real blur case study with θ, L,
deblurred output, documented failure mode.
