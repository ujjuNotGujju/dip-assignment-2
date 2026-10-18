# Problem 5 — Denoising and the Wiener Filter (15 pts)

Status: **mostly done — one sub-part pending real data**. See [REPORT.md](REPORT.md)
for the full write-up.

Data: `data/provided/problem5_6/reference_image.jpg` (clean RGB, >=1024x1024, shared
with Problem 6) + a genuinely noisy self-captured photo (`data/own_captures/`) for
§5.2 — **not yet supplied, see below**.

## 5.1 The Wiener Filter as an MMSE Denoiser (7 pts)
- [x] Derive `F_hat = W*G`, `W = Sff/(Sff+Snn)`.
- [x] Implement; estimate Snn from known injected noise variance, Sff from a
      radially-averaged noisy-image periodogram minus Snn. Estimation method and
      assumptions stated.
- [x] Compare vs Problem 2's Gaussian/Butterworth LPF and Problem 4's bilateral filter
      at σ ∈ {5,10,20,40}; PSNR/SSIM table; parameter settings; oracle-tuned labeled.
- [x] Discuss frequency-dependent shrinkage vs fixed-cutoff LPF; Wiener doesn't always
      win (bilateral edges it out slightly at every σ tested here).

## 5.2 Test the Assumptions (8 pts)
- [x] At σ=20, mis-specify noise power (0.5x, 2x); compare + explain via the formula.
- [x] Explain why correctly-specified Wiener can still blur edges.
- [ ] **Apply to a genuinely noisy self-captured photo — PENDING.** `data/own_captures/`
      is empty; this sub-part is explicitly incomplete (not faked with synthetic
      noise). `problem5.py` will process any image dropped into
      `data/own_captures/` automatically on the next run — no code changes needed.

## Deliverables
Wiener/low-pass/bilateral PSNR/SSIM table across σ; filter-parameter settings; figures
showing noise-power mis-estimation; real-noise case study **(pending)**.
