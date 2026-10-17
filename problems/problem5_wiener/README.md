# Problem 5 — Denoising and the Wiener Filter (15 pts)

Status: **not started**.

Data: `data/provided/problem5_6/reference_image.jpg` (clean RGB, >=1024x1024, shared with
Problem 6) + a genuinely noisy self-captured photo (`data/own_captures/`) for §5.2.

## 5.1 The Wiener Filter as an MMSE Denoiser (7 pts)
- [ ] Derive `F_hat = W*G`, `W = Sff/(Sff+Snn)`.
- [ ] Implement; estimate Snn from known/flat-region noise, Sff from (noise-subtracted)
      radially-averaged periodogram or a stated 1/f^α model.
- [ ] Compare vs Problem 2's Gaussian/Butterworth LPF and Problem 4's bilateral filter at
      σ ∈ {5,10,20,40}; PSNR/SSIM table; label oracle-tuned results.
- [ ] Discuss frequency-dependent shrinkage vs fixed-cutoff LPF; note Wiener doesn't
      always win.

## 5.2 Test the Assumptions (8 pts)
- [ ] At σ=20, mis-specify noise power (0.5x, 2x); compare + explain via the formula.
- [ ] Explain why correctly-specified Wiener can still blur edges.
- [ ] Apply to a genuinely noisy self-captured photo; discuss departure from
      white/stationary/signal-independent noise assumptions (no PSNR/SSIM without a
      clean reference).

## Deliverables
Wiener/low-pass/bilateral PSNR+SSIM table across σ; parameter settings; noise-power
mis-estimation figures; real-noise case study.
