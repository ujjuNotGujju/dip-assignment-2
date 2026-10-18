# Problem 4 — Edge-Preserving Smoothing: Bilateral vs. Kuwahara (15 pts)

Status: **done**. See [REPORT.md](REPORT.md) for the full write-up.

Data & Input: `data/provided/problem4/q4.png` (clean ground-truth image with flat
regions and sharp boundaries) + synthetic `N(0, 20^2)` additive noise for 4.1/4.2.
Section 4.3 uses a programmatically-generated synthetic step-edge image (not a single
fixed noisy image) across a controlled Noise-to-Contrast Ratio gradient.

## 4.1 Implementation and Matched-Strength Comparison (7 pts)
- [x] Bilateral filter (spatial sigma_s + range sigma_r, normalized), from scratch.
- [x] Classical Kuwahara filter (4 overlapping sub-windows, min-variance mean), from
      scratch.
- [x] (a) Matched effective smoothing strength: equal variance-reduction factor on a
      flat patch, justified and numerically matched.
- [x] (b) PSNR/SSIM vs. clean ground truth (both metrics implemented from scratch);
      cropped zoomed boundary comparisons.

## 4.2 Mitigating Kuwahara Blockiness and Complexity Analysis (5 pts)
- [x] (a) Mechanism explanation: why hard argmin causes blockiness, not blur/ringing.
- [x] (b) Soft-Kuwahara (softmax variance weighting) de-blocking modification;
      before/after + Total Variation blockiness metric.
- [x] (c) Per-pixel operation-count complexity analysis for all three algorithms.

## 4.3 Quantitative Breakdown Analysis of Bilateral Filtering (3 pts)
- [x] Synthetic step-edge generator across a Noise-to-Contrast Ratio gradient.
- [x] Edge-preservation metric vs. NCR plot with critical NCR* annotated.
- [x] Mathematical explanation via the range kernel w_r(ΔI).

## Deliverables
Matched-strength filter outputs with PSNR/SSIM tables and cropped comparisons;
Kuwahara blockiness explanation + de-blocking code + before/after metrics + per-pixel
complexity breakdown; synthetic step-edge generator + breakdown curve with NCR*
identified, plotted, and mathematically explained.
