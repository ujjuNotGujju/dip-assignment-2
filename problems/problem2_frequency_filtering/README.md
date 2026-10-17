# Problem 2 — Frequency-Domain Filtering: Ideal, Butterworth, Gaussian (15 pts)

Status: **not started**.

Data: `data/provided/problem2/` (`characters_test_pattern`, `text_gaps`, `thumb_print`,
`cameraman.tif`), plus a self-synthesized 512x512 step-edge image (B=200 square, side
256, on A=50 background) — to be generated in `data/synthetic/`.

Pipeline (per the PDF): pad each NxN image to PxP with P=2N, center the spectrum,
multiply by a real zero-phase filter H(u,v), invert, take the real part, crop. Every
transfer function must be built from its formula — no library filter routines except
where a subtask explicitly asks for a comparison (2.2d vs
`scipy.ndimage.gaussian_filter`).

## 2.1 Filter Construction and Transfer-Function Analysis (4 pts)
- [ ] `make_filter(kind, P, D0, n=None, highpass=False)` for ILPF/BLPF/GLPF (+ HP via
      `1 - H_LP`).
- [ ] (a) Cutoff semantics + Gaussian/Butterworth half-amplitude matching derivation.
- [ ] (b) Energy-based cutoff table (α = 90/95/99/99.5%) on `characters_test_pattern`.
- [ ] (c) Inverse design problem (recover n, D0 from two measured H values); verify.

## 2.2 Ringing: Prediction, Measurement, Reconciliation (6 pts)
- [ ] **Dated pre-implementation prediction** (closed-form Si(·) edge response, overshoot
      %, first-peak distance) in `predictions/2.2_ringing_prediction.md`, written before
      running this section.
- [ ] (a) ILPF step-edge overshoot vs. D0 ∈ {10,20,40,80}; corner-profile discussion.
- [ ] (b) Butterworth order sweep at D0=40; kernel ratio |min h|/max h; non-negative-kernel proof.
- [ ] (c) Visual grid: ILPF/BLPF(n=2)/GLPF on characters_test_pattern at 2.1b radii.
- [ ] (d) Gaussian equivalence check vs `scipy.ndimage.gaussian_filter` (explicit,
      PDF-sanctioned comparison exception).

## 2.3 Applications: Denoising, Repair, Sharpening (5 pts)
- [ ] (a) Noise-blur trade-off: PSNR vs D0 sweep for all 3 families on noisy cameraman.
- [ ] (b) Character repair on text_gaps.tif.
- [ ] (c) High-pass + DC-term discussion, high-frequency emphasis, binary ridge map on
      thumb_print.tif.

## Deliverables
`make_filter` + radial profile plots; cutoff-matching derivation + power-retention table
+ inverse-design solution; dated ringing prediction + measured-vs-predicted tables +
kernel-ratio table + proof; σs derivation + Gaussian equivalence table; PSNR sweep +
repaired text image + high-pass/HFE results.
