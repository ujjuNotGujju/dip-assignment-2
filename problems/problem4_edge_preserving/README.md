# Problem 4 — Edge-Preserving Smoothing: Bilateral vs. Kuwahara (15 pts)

Status: **not started**.

Data: `data/provided/problem4/q4.png` (clean ground-truth image with flat regions and
sharp boundaries) + synthetic additive Gaussian noise (σ reported) + self-generated
step-edge images across a Noise-to-Contrast-Ratio gradient for §4.3.

## 4.1 Implementation and Matched-Strength Comparison (7 pts)
- [ ] Bilateral filter from scratch (joint spatial σs + range σr Gaussian, normalized).
- [ ] Classical Kuwahara filter from scratch (4 overlapping sub-windows, min-variance mean).
- [ ] (a) "Matched effective smoothing strength" definition + justification.
- [ ] (b) PSNR/SSIM vs. clean ground truth + cropped boundary comparisons.

## 4.2 Mitigating Kuwahara Blockiness and Complexity (5 pts)
- [ ] (a) Why hard arg-min variance causes blockiness (not blur/ringing).
- [ ] (b) One de-blocking modification (generalised/soft Kuwahara); before/after +
      blockiness metric (e.g. TV).
- [ ] (c) Per-pixel operation counts: Classical Kuwahara vs. Modified vs. Bilateral.

## 4.3 Quantitative Breakdown Analysis of Bilateral Filtering (3 pts)
- [ ] Synthetic step-edge NCR sweep; identify critical NCR*.
- [ ] Edge-preservation metric vs NCR plot with NCR* annotated.
- [ ] Mathematical explanation via the range kernel `w_r(ΔI)`.

## Deliverables
Matched-strength outputs + PSNR/SSIM tables + crops; Kuwahara blockiness explanation +
de-blocking code + before/after metrics + complexity breakdown; synthetic step-edge
generator + breakdown curve with NCR* identified.
