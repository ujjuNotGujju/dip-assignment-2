# Problem 4 — Edge-Preserving Smoothing: Bilateral vs. Kuwahara (15 pts)

Status: **done**.

Code: [`problem4.py`](problem4.py). Shared implementations:
[`src/dip_a2/edge_preserving.py`](../../src/dip_a2/edge_preserving.py) (bilateral,
classical Kuwahara, soft Kuwahara), [`src/dip_a2/metrics.py`](../../src/dip_a2/metrics.py)
(PSNR, SSIM — both from scratch). Figures/tables: [`results/`](results/). No dated
pre-implementation prediction is required for this problem — the PDF's Problem 4 text
contains no "Pre-implementation Task" instruction (unlike Problems 1 §1.2 and 2 §2.2).

Data: `data/provided/problem4/q4.png` — the provided clean ground-truth image (a flat
light-grey rectangle, a horizontal-gradient rectangle, and a near-black circle
overlapping it, on a mid-grey background), matching the PDF's Figure 2. Noisy input:
`clean + N(0, 20²)` (σ=20, fixed seed=11).

Both the bilateral filter and classical/soft Kuwahara are implemented by looping over
the small window/sub-window **offsets** (not pixels) and accumulating shifted
whole-image NumPy arrays — the same vectorization pattern used throughout this repo
since Problem 1 (e.g. `circular_convolve_spatial`), avoiding any disallowed library.

---

## 4.1 Implementation and Matched-Strength Comparison

### (a) Matched-strength definition

Both filters reduce noise by *locally averaging* pixels; both reduce the variance of a
flat, edge-free patch of the noisy image by some factor. Defined "matched effective
smoothing strength" as **equal variance-reduction factor on a flat patch of the noisy
image** (same noise realization) — a concrete, numeric criterion, not just a verbal
one. Procedure actually used:

1. Locate a flat region automatically (darkest-variance 30×30 block of the *clean*
   image, away from the border): `rows 30:60, cols 30:60`.
2. Fix Kuwahara's sub-window size at 5×5 (total window span 9×9); measure its
   variance-reduction factor on that block: **22.03×** (noisy variance 380.29 →
   17.26 after filtering).
3. Search over bilateral's `sigma_s` (with `sigma_r` **fixed at a normal, edge-aware
   value — 40 — not disabled**, used identically both during this search and in the
   final applied filter, so the match is self-consistent) for the value whose
   variance-reduction factor on the same block is closest to 22.03×. Found:
   `sigma_s = 4.50` → reduction **22.47×** (target 22.03×, within 2%).

   (An earlier attempt probed the search with `sigma_r=200`, effectively disabling
   range discrimination during calibration, then switched to a smaller `sigma_r` for
   the real run — that under-matched badly, since a flat-region calibration done under
   an inactive range kernel doesn't actually characterize the filter that gets applied.
   Using the *same* `sigma_r` throughout both steps fixed this.)

### (b) PSNR, SSIM, and boundary comparison

At matched strength (Kuwahara 5×5, bilateral `sigma_s=4.50, sigma_r=40`):

| filter | PSNR (dB) | SSIM |
|---|---|---|
| Kuwahara | **35.40** | **0.850** |
| Bilateral | 34.57 | 0.788 |

Kuwahara edges out bilateral on both metrics here, despite matched flat-region
denoising strength — consistent with the cropped boundary comparison
(`results/4_1_boundary_zoom.png`, around the circle/rectangle corner): Kuwahara's
edges stay visibly crisper (its sub-window selection actively seeks out the most
homogeneous side of a boundary, rather than blending across it like an
isotropic-footprint average), while bilateral's edge is visibly softer at this
matched strength — bilateral's spatial Gaussian still contributes *some* weight across
the boundary even where the range kernel down-weights it, whereas Kuwahara's
sub-window selection is a hard, all-or-nothing choice of an entirely same-side
neighbourhood. Full-image comparison: `results/4_1_comparison.png`.

---

## 4.2 Mitigating Kuwahara Blockiness and Complexity Analysis

### (a) Mechanism explanation

Classical Kuwahara's output at a pixel is **which of 4 candidate sub-window means get
selected**, chosen by a hard `argmin` over their variances. As the pixel position
slides across the image, the *identity* of the minimum-variance sub-window can flip
discontinuously from one pixel to the next (whenever two sub-windows' variances cross
over) — and each flip switches the output to a **different sub-window's mean**, which
is itself a different number. The output is therefore a **piecewise-constant mosaic of
whichever sub-window "won" in each local neighbourhood**, with sharp jumps exactly at
the boundaries between "win regions" — a discontinuity in the *output itself*, imposed
by the selection rule, independent of whether the underlying image content actually
has an edge there. This is structurally different from:
- **Blur**, which comes from *continuous* averaging — output changes smoothly as the
  window slides, with no discontinuity introduced by the operation itself;
- **Ringing**, which comes from a *sharp frequency-domain cutoff* (Problem 2) —
  oscillatory over/undershoot around a true edge, not a new false edge in a region that
  was originally flat.

Kuwahara's blockiness is thus a discontinuity artifact of the **selection** operation,
appearing as new, spurious piecewise-constant boundaries ("tiles") even inside
genuinely flat regions wherever two sub-windows' noise-driven variances happen to
cross as the window slides.

### (b) Soft-Kuwahara de-blocking

Implemented `soft_kuwahara_filter` (PDF-sanctioned "continuous Soft-Kuwahara using
softmax variance weighting"): instead of a hard `argmin`, the output is a
**softmax(-variance/T)-weighted combination of all 4 sub-window means**. As the
pixel slides and the relative variances change, the softmax weights shift
*continuously* rather than snapping discretely between one-hot selections — removing
the source of the blockiness identified in (a).

Quantified with **Total Variation (TV)**, measured specifically in a *near-edge,
still-flat* zone (pixels within the Kuwahara sub-window span of a true edge in the
clean image, but not themselves on the edge — found automatically via a gradient
threshold + dilation on the clean image) — this isolates spurious blockiness from the
TV contribution of genuine edges, which both filters must cross regardless:

| filter | TV in near-edge flat zone |
|---|---|
| Classical Kuwahara | 6.332 |
| Soft Kuwahara (T=15) | **5.779** (−8.7%) |

Soft Kuwahara measurably reduces spurious variation right next to true edges, in the
direction the PDF predicts, though at `T=15` the visual difference is subtle
(`results/4_2_blockiness_comparison.png`) — a larger temperature would blend more
aggressively (trading more blockiness reduction for softer edges, the same
edge-sharpness/artifact trade-off seen throughout this problem).

### (c) Per-pixel complexity analysis

Analytic operation counts (not timed/profiled), at the window sizes actually used in
this script (Kuwahara sub-window 5×5 → 100 total taps across 4 sub-windows;
bilateral window `w×w` with `w=49`, the span for a representative `sigma_s≈8`):

| algorithm | window taps | additions | multiplications | exponentiations | comparisons |
|---|---|---|---|---|---|
| Classical Kuwahara | 100 | 200 | 100 | **0** | 3 (argmin of 4) |
| Soft Kuwahara | 100 | 204 | 104 | **4** (softmax) | 0 |
| **Bilateral** | 2401 | 7203 | 4802 | **2401** | 0 |

**Bilateral is by far the most computationally expensive per pixel**, because it needs
**one `exp()` evaluation for every single window tap** (`w²` of them — 2401 for a
49×49 window) to build the data-dependent range kernel fresh at each offset, whereas
Kuwahara's sub-window statistics (sums and sums-of-squares) only need cheap
additions/multiplications, with the expensive `exp()` calls limited to a **fixed
small constant** (0 for the hard-argmin classical version, 4 for the softmax-weighted
soft version) **regardless of window size** — Kuwahara's exponentiation cost does not
scale with its window at all, while bilateral's scales quadratically with window
diameter.

---

## 4.3 Quantitative Breakdown Analysis of Bilateral Filtering

Synthetic step-edge generator: a 256×256 array split evenly in half (`I1=50` left,
`I2=200` right, `|ΔI|=150`) plus `N(0,σ²)` noise, swept across
`NCR = σ/|ΔI| ∈ [0.03, 2.0]`.

**Design choice (found during this problem, documented honestly):** the filter's
`sigma_s=6.0` is held fixed throughout the sweep, but `sigma_r` is set **adaptively as
`max(σ_noise, 5)`** — i.e. proportional to each test case's actual noise level, the
textbook recommendation (Tomasi & Manduchi) for choosing a bilateral filter's range
parameter when the noise level is known. A first attempt used a single literally-fixed
`sigma_r` across the whole sweep; at the higher end of the NCR range tested
(`σ_noise` up to 300), a small fixed `sigma_r` made the range kernel reject even
*same-side* neighbours (whose difference is dominated by noise far exceeding
`sigma_r`), so the filter did almost nothing at high NCR and never exhibited the
cross-edge-blending breakdown the PDF asks to demonstrate — the measured curve just
hovered noisily near 100% throughout. Scaling `sigma_r` with the noise level keeps the
filter actively denoising at every tested noise level, which is what actually lets the
*intended* failure mode (cross-edge blending once `σ_noise` becomes comparable to
`ΔI`) show up cleanly.

Edge-preservation metric: band-averaged (4-pixel) intensity difference across the
boundary, ensemble-averaged over 6 independent noise realizations per NCR value (to
keep the curve legible despite per-pixel measurement noise at high σ).

**Result** (`results/4_3_breakdown_curve.png`, `results/4_3_profiles.png`): a clean,
monotonically decreasing curve from ≈100% (NCR≤0.15) down to a high-NCR plateau around
≈65% (the curve does not decay to 0% — see explanation below). Defining `NCR*` as
where the curve crosses the **midpoint between its low- and high-NCR plateaus**
(82.3%, a self-consistent knee definition since the curve asymptotes well above zero
rather than at it):

```
low plateau  = 100.0%
high plateau = 64.6%
midpoint     = 82.3%
NCR*         = 0.430
```

The breakdown is sharp and visible between roughly NCR=0.2 (98.6% retained) and
NCR=0.8 (70.1% retained), with NCR*≈0.43 falling right in the middle of that
transition. `results/4_3_profiles.png` shows example filtered profiles at NCR=0.1
(edge sharply preserved) vs. NCR=1.2 (visibly smoothed/blended across the boundary).

### Mathematical explanation

The range kernel is `w_r(ΔI) = exp(-ΔI²/2σr²)`. For a same-side pixel pair, the
observed intensity difference is dominated by noise, with typical magnitude
`~σ_noise`; for a cross-side pair, the expected difference is
`~√(ΔI² + 2σ_noise²)` (the true step plus independent noise on each sample). At low
NCR (`σ_noise ≪ ΔI`), these two typical magnitudes are very different — same-side
differences stay small relative to `σr`, giving `w_r≈1` (full weight, genuine
denoising within one side), while cross-side differences stay large relative to `σr`,
giving `w_r≈0` (correctly rejected) — the filter discriminates cleanly. As
`σ_noise` grows and approaches `ΔI` (NCR→1), the cross-side typical difference
`√(ΔI²+2σ_noise²)` and the same-side typical difference `σ_noise` become comparable in
scale (both dominated by `σ_noise` once `σ_noise≫ΔI`), so `w_r(ΔI_same-side)` and
`w_r(ΔI_cross-side)` can no longer be reliably separated — the range kernel starts
admitting cross-side neighbours with weights similar to genuine same-side ones, and
the filter blends across the true edge instead of respecting it. This is exactly the
breakdown measured above, transitioning right around where `σ_noise` becomes a
non-negligible fraction of `ΔI` (NCR*≈0.43, i.e. `σ_noise` reaching roughly 43% of the
step height).

(Why the curve plateaus at ≈65% rather than 0%: once `σr` has grown large enough —
`σr=σ_noise` at high NCR — to no longer discriminate same- vs. cross-side pairs at
all, the filter degenerates toward a **pure spatial Gaussian average** with no range
restriction; measured 4 pixels off-centre from (not exactly at) the boundary, even a
pure spatial blur with `sigma_s=6` retains a non-zero fraction of the step contrast at
that offset, rather than averaging it away completely — hence the plateau sits well
above, not at, 0%.)

---

## Deliverables checklist

- [x] Matched-strength filter outputs + PSNR/SSIM tables + cropped comparison figures.
- [x] Kuwahara blockiness mechanism explanation; soft-Kuwahara implementation;
      before/after visual + TV metric; per-pixel complexity breakdown for all three.
- [x] Synthetic step-edge generator; quantitative breakdown curve with NCR* identified,
      plotted, and mathematically explained.
