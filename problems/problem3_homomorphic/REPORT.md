# Problem 3 — Homomorphic Filtering: Illumination and Reflectance (10 pts)

Status: **done**.

Code: [`problem3.py`](problem3.py). Figures/tables: [`results/`](results/). No dated
pre-implementation prediction is required for this problem — the PDF's Problem 3 text
contains no "Pre-implementation Task" instruction (unlike Problems 1 §1.2 and 2 §2.2).

Data: `data/provided/problem3/tunnel_image.jpg`, a 3000x3644 photo looking down a
tunnel/pipe — a bright, near-saturated circular light source surrounded by a smoothly
darkening wall, fading to deep, almost pure-black shadow filling most of the frame. The
source photo was downsampled (PIL, `LANCZOS`, to a 900px max dimension) purely as a
practical loading step before any filtering — not one of this problem's implemented
operations (homomorphic filtering, histogram equalization), so this is consistent with
the assignment's library restrictions.

## Environment

Same machine as Problems 1–2 — CPU Intel(R) Core(TM) i5-9300H @ 2.40GHz (8 logical
CPUs), 14.96 GB RAM, NumPy 2.5.4 (see `results/machine_info.json`).

---

## 3.1 The Multiplicative Model

**Model.** `I(x,y) = L(x,y)·R(x,y)`, where `L` (illumination) varies slowly/smoothly
over the image and `R` (reflectance) captures fine detail and texture. Taking the
natural log converts the product into a sum:

```
log I(x,y) = log L(x,y) + log R(x,y)
```

A linear, shift-invariant frequency-domain filter can only separate additive
components by how their *energy is distributed across frequency* — it has no notion of
a product at all. Since `L` varies slowly, `log L` is concentrated at low spatial
frequencies; since `R` carries edges/texture, `log R` contributes more to higher
frequencies. A filter that attenuates low frequencies and boosts high ones, applied to
`log I`, therefore suppresses the illumination term and emphasizes the reflectance term
— but this separation is only meaningful because the quantities being filtered
(`log L`, `log R`) **add**. There is no equivalent operation that could separate `L`
and `R` directly from `I = L·R` in the frequency domain, because multiplication in the
spatial domain corresponds to *convolution* in the frequency domain, not to a
frequency-selective split.

**Why the log must precede the FFT, not follow it.** The filter needs to act on
`log I(x,y)`, a real-valued *image* whose own Fourier transform `FFT(log I)` is a full
complex spectrum (magnitude **and** phase) — this is what makes the
filter → inverse-transform → exponentiate pipeline work at all, since inverting
`FFT(log I)` after multiplying by `H(u,v)` requires the phase to reconstruct a spatial
image.

If instead one took `FFT(I)` first and then tried to filter `log(|FFT(I)|)` — the
*log-magnitude spectrum*, i.e. the quantity normally only used for **display** — two
things go wrong:

1. **The phase is gone.** `log(|FFT(I)|)` is a real, non-negative array computed from
   the magnitude alone; it retains no phase information at all. There is no way to
   inverse-transform it back into a meaningful spatial-domain image afterward — any
   attempt to do so has nothing to invert *with*. (This is the same phase-vs-magnitude
   point established in Problem 1 §1.2: phase carries the structural/positional
   information; discarding it the way `log(|FFT(I)|)` does is irreversible.)
2. **It wouldn't correspond to filtering `log I` even in principle.** `log` and `FFT`
   do not commute: `log(|FFT(I)|) ≠ |FFT(log I)|` in general, because `FFT` is linear
   and `log` is not. A concrete, numerical check on the tunnel image
   (`results/3_1_log_order_demo.png`, `results/3_1_result.json`): computing both
   `log(|FFT(I)|)` and `log(|FFT(log I)|)` and correlating them (after normalizing each
   to zero mean/unit variance) gives a correlation of only **0.669** — related in
   coarse shape (both peak near DC and fall off radially, since any natural image's
   spectrum is low-frequency-dominated), but clearly **not** the same quantity. Acting
   on one is not a stand-in for acting on the other.

So the only correct order is: take the log of the *image* first (an operation on real
pixel intensities), *then* Fourier-transform the result (preserving phase), filter,
invert, and exponentiate.

---

## 3.2 Implement and Tune

**Filter.** `H(u,v) = (γH − γL)·[1 − exp(−c·D²/D0²)] + γL` (`γL<1<γH`), applied to
`log(1+I)` via the same zero-pad/centre/multiply/invert/crop machinery used in Problem
2 (`src/dip_a2/filters.py::make_homomorphic_filter`), then `expm1`'d back.
`log1p`/`expm1` (rather than plain `log`/`exp` with an ad hoc epsilon) are used
deliberately because this image has literal zero-intensity pixels in its deepest
shadow — `log1p(0) = 0` handles that exactly, with no epsilon hack needed.

### A padding bug, found and fixed during this problem

The first implementation reused Problem 2's `filter_pipeline` (zero-padding the image
into the top-left corner of a `2M x 2N` canvas). Running it produced a badly broken
result: the whole image collapsed to near-black except for a faint ring. Diagnosis
(see `git history` / this session's debugging): zero-padding necessarily creates a
sharp discontinuity right at the image's own boundary (real content jumping to 0), and
with homomorphic high-frequency emphasis using `γH` up to 2–3.5 (a much stronger,
more broadband high-frequency *gain* than anything used in Problem 2), that single
discontinuity gets amplified into a severe ringing spike concentrated at the image
corner — large enough (value ≈125, vs. a 99th-percentile of only ≈15 everywhere else)
to dominate the display min-max rescale and crush the entire rest of the image to near
black. (An initial attempt to fix this by *centring* the image in the padded canvas
instead of placing it at the corner made no difference at all — because circular
convolution is shift-invariant, moving where the image sits in the periodic canvas
cannot change anything about an artifact tied to the image's own content-to-zero edge.
That dead end is left in `git log` as part of the record, not hidden.) The actual fix:
pad with mirror-symmetric reflection instead of zeros
(`src/dip_a2/filters.py::filter_pipeline_reflect`, using `numpy.pad(mode="symmetric")`
— plain NumPy array manipulation, not a disallowed filtering routine). Reflection
removes the artificial discontinuity at the boundary altogether (the padded region
continues the image's own intensity smoothly), which dropped the spurious spike from
≈125 down to ≈11 (in line with the rest of the image's natural dynamic range) and
produced a correct result. This is the same "avoid an artificial sharp edge at the
image boundary" principle established in Problem 1 §1.3, just solved by reflecting
instead of by the minimal-zero-padding derivation used there — zero-padding is the
right tool when avoiding *aliasing* between unrelated content, but the wrong tool here
because the filter's own strong high-frequency gain turns the zero-boundary's
discontinuity into the dominant feature of the whole image.

### Parameter sweep

`γL ∈ {0.25, 0.5, 0.75} × γH ∈ {1.5, 2.0, 3.0}` at `D0=60`:
`results/3_2_gamma_sweep.png`. Clear trend: **lower `γL` (top row, 0.25) flattens
illumination most effectively** — the tunnel wall's ring texture becomes visible
almost uniformly across the whole frame, including what was deep shadow — while
**higher `γL` (bottom row, 0.75) preserves more of the original illumination
gradient** (walls still comparatively dark, the light source still dominant), since a
`γL` close to 1 barely attenuates the low-frequency illumination term at all. `γH`
mainly controls local contrast/sharpness of the recovered texture, visible as more
pronounced ring detail at `γH=3.0` vs. `γH=1.5` within each row.

**Halo search:** `D0 ∈ {10, 30, 80, 250}` at a deliberately aggressive
`γL=0.2, γH=3.5`: `results/3_2_D0_sweep_halo.png`. At **`D0=10`**, a clear halo is
visible: pronounced bright/dark ring-shaped banding around the light source and
scalloped fringes through the wall texture — the disc's edge sits right at the
cutoff radius, so the strong, broadband high-frequency gain rings around it exactly
like Problem 2's ideal-filter ringing, just combined here with an aggressive gain. At
**`D0=250`**, the result is comparably smooth with no visible halo, but also flattens
illumination noticeably less well (a larger `D0` classifies more of the genuinely
slowly-varying illumination as "high frequency" and leaves it comparatively
unboosted/unflattened) — illustrating the real tuning trade-off between flattening
strength and halo risk.

**Best-tuned setting used for the final comparison:** `D0=60, γL=0.4, γH=2.0, c=1.0`
— a middle ground that visibly flattens illumination with no visible halo artifact
(`results/3_2_comparison.png`, middle panel).

### Comparison against global histogram equalization

`src/dip_a2/utils.py::histogram_equalize` — classic global histogram equalization
(256-bin histogram, normalized CDF, pixel remapping), implemented from scratch with
NumPy (`np.bincount`/`np.cumsum`), since there is no Assignment 1 code in this repo to
reuse. Result: `results/3_2_comparison.png`, right panel.

**Local-contrast comparison**, using data-driven region selection (not hand-picked
coordinates): the original image was tiled into 50x50 blocks, and the darkest/brightest
block by mean intensity (excluding the outermost ring of blocks) was used as the
"shadow region" and "bright region" respectively (`results/3_2_result.json`):

| region | metric | original | homomorphic (best) | histogram equalization |
|---|---|---|---|---|
| shadow | mean | 10.41 | **17.68** | 12.82 |
| shadow | std (local contrast) | 0.86 | **5.38** | 14.72 |
| bright | mean | 255.00 | **185.76** | 254.999 |
| bright | max | 255.0 | **199.52** | 255.0 |
| bright | fraction ≥ 250 | 0.9996 | **0.0** | 1.0 |

**Homomorphic filtering recovers shadow detail without blowing out the bright
region, clearly better than histogram equalization here**: the shadow block's local
std (contrast) rises from 0.86 (original, essentially flat/noise-floor-only) to 5.38
under homomorphic filtering — real texture has been pulled out of what was a nearly
uniform black patch — while histogram equalization's shadow std (14.72) is larger in
raw numeric terms but for the wrong reason: global histogram equalization has no
notion of *where* in the image a pixel is, so it just globally redistributes the whole
image's intensity histogram, and because this image's histogram is extremely
imbalanced (dominated by near-black pixels), equalization maps a huge range of nearly
indistinguishable dark values across a wide output range — amplifying quantization/
compression noise in the shadow at least as much as genuine structure (visible as
grain in `results/3_2_comparison.png`'s right panel), rather than the smooth, texture-revealing recovery homomorphic filtering achieves via the actual illumination model.
More decisively, in the **bright region** histogram equalization still saturates
99.96%→100% of pixels at/above 250 (it cannot do otherwise — monotonic global
remapping preserves rank order, so the brightest pixels in the original stay the
brightest after equalization, pinned near 255), while the homomorphic filter's
explicit illumination/reflectance separation pulls the bright region's mean down to
185.76 with **zero** pixels left at-or-above 250 — recovering structure in the light
source too, not just the shadow, which histogram equalization cannot do by
construction.

---

## Deliverables checklist

- [x] Derivation of the log-domain argument (§3.1), including a concrete numerical
      demonstration of why log must precede the FFT.
- [x] Parameter sweep grid (`γL × γH`) with halo cases identified (`D0` sweep).
- [x] Local-contrast comparison table against histogram equalization.
