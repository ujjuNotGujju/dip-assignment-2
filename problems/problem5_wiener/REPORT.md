# Problem 5 — Denoising and the Wiener Filter (15 pts)

Status: **mostly done — one sub-part pending real data** (see §5.2's real-noisy-photo
case study). No dated pre-implementation prediction is required for this problem —
the PDF's Problem 5 text contains no "Pre-implementation Task" instruction.

Code: [`problem5.py`](problem5.py). Shared implementation:
[`src/dip_a2/wiener.py`](../../src/dip_a2/wiener.py). Figures/tables:
[`results/`](results/). Heavily reuses existing shared code: Problem 2's
`filters.make_filter`/`filter_pipeline` (Gaussian/Butterworth comparison baselines),
Problem 4's `edge_preserving.bilateral_filter`, and the shared from-scratch
`metrics.psnr`/`ssim`.

Data: `data/provided/problem5_6/reference_image.jpg` (3072×2048 RGB, well above the
required ≥1024×1024), converted to grayscale and downsampled to 768×512 (LANCZOS, a
practical step for tractable repeated FFT/bilateral sweeps across 4 noise levels × 4
methods — the same rationale as Problem 3's downsample).

## Environment

Same machine as Problems 1–4 — CPU Intel(R) Core(TM) i5-9300H @ 2.40GHz (8 logical
CPUs), 14.96 GB RAM, NumPy 2.5.4 (see `results/machine_info.json`).

---

## 5.1 The Wiener Filter as an MMSE Denoiser

### Derivation

Assume the clean image `f` and additive noise `n` are uncorrelated, zero-mean,
wide-sense-stationary random fields with power spectral densities `Sff(u,v)`,
`Snn(u,v)`, and `g = f+n`. We seek the **linear** estimator
`F_hat(u,v) = W(u,v)G(u,v)` minimizing the mean squared error
`E[|f(x,y) − f_hat(x,y)|²] = E[|F(u,v) − F_hat(u,v)|²]` (Parseval) at every frequency
independently (since the DFT diagonalizes circulant/stationary operators, the optimal
*linear* per-frequency filter can be found frequency-by-frequency). Writing the error
at frequency `(u,v)` as `F − WG = F − W(F+N) = (1−W)F − WN`, and using independence of
`F` and `N` (uncorrelated by assumption):

```
E[|(1−W)F − WN|²] = |1−W|² Sff + |W|² Snn
```

Differentiating w.r.t. `W` (treating `W` as real, since the optimal linear MMSE filter
here is real/zero-phase) and setting to zero:

```
d/dW [ (1−W)² Sff + W² Snn ] = −2(1−W)Sff + 2W·Snn = 0
⟹ W·Sff + W·Snn = Sff
⟹ W(u,v) = Sff(u,v) / (Sff(u,v) + Snn(u,v))
```

the stated result.

### Implementation and estimation method

`src/dip_a2/wiener.py::wiener_denoise`. Image intensities normalized to `[0,1]`.
- **`Snn`** is set from the **known injected noise variance**: for white noise of std
  `σ` (in `[0,1]` units, `σ/255`), the expected squared-magnitude of its (unnormalized)
  2-D DFT at every frequency bin is `Snn = (σ/255)² · M·N` (a flat/white periodogram
  level) — this is legitimate "known from the experiment setup," not oracle access to
  the clean image.
- **`Sff`** is estimated **purely from the noisy image's own periodogram**: compute
  `|G(u,v)|²`, radially average it (bin by integer radius via
  `filters.radial_distance`, average within each bin — `wiener.py::radial_average`,
  the PDF's suggested method), then subtract `Snn` (floored at a small positive
  epsilon to keep `Sff≥0`). This never touches the clean image either.
- `W = Sff/(Sff+Snn)`, applied to `G`, inverted, rescaled to `[0,255]`.

Noise: `σ ∈ {5,10,20,40}` (8-bit levels), one fixed noise realization per σ (seeded),
**not clipped before filtering** — clipping is applied only for display/metric
computation afterward, per the PDF's instruction.

### Comparison vs. Problem 2's LPFs and Problem 4's bilateral filter

| σ | method | PSNR (dB) | SSIM | parameters | oracle-tuned? |
|---|---|---|---|---|---|
| 5 | **Wiener** | **35.50** | **0.912** | (from noisy image + known σ) | **No** |
| 5 | Bilateral | 35.63 | 0.914 | σs=0.50, σr=12.5 | Yes |
| 5 | Gaussian LPF | 27.71 | 0.729 | D0=150 | Yes |
| 5 | Butterworth LPF (n=2) | 27.07 | 0.688 | D0=150 | Yes |
| 10 | **Wiener** | **31.26** | **0.815** | — | **No** |
| 10 | Bilateral | 31.69 | 0.839 | σs=1.57, σr=25.0 | Yes |
| 10 | Gaussian LPF | 27.58 | 0.717 | D0=150 | Yes |
| 10 | Butterworth LPF (n=2) | 26.98 | 0.681 | D0=150 | Yes |
| 20 | **Wiener** | **28.01** | **0.691** | — | **No** |
| 20 | Bilateral | 28.69 | 0.711 | σs=1.57, σr=50.0 | Yes |
| 20 | Gaussian LPF | 27.10 | 0.675 | D0=150 | Yes |
| 20 | Butterworth LPF (n=2) | 26.65 | 0.653 | D0=150 | Yes |
| 40 | **Wiener** | **25.72** | **0.576** | — | **No** |
| 40 | Bilateral | 26.01 | 0.546 | σs=2.64, σr=100.0 | Yes |
| 40 | Gaussian LPF | 25.66 | 0.574 | D0=134.4 | Yes |
| 40 | Butterworth LPF (n=2) | 25.57 | 0.575 | D0=150 | Yes |

(SSIM data range 255, 11×11 Gaussian window σ=1.5 — `src/dip_a2/metrics.py::ssim`.)
Full table: `results/5_1_results_table.json`. Visual comparison at σ=20:
`results/5_1_comparison_sigma20.png`; PSNR-vs-σ plot: `results/5_1_psnr_vs_sigma.png`.

**A fairness caveat, stated explicitly as the PDF requires:** the Gaussian/Butterworth
`D0` and the bilateral `(σs,σr)` are all **oracle-tuned** — selected by sweeping and
picking whichever value maximizes PSNR *against the clean reference*, which would not
be available in a real deployment. **Wiener's parameters are not oracle-tuned** — `Sff`
comes only from the noisy image's own periodogram and `Snn` from the known injected
variance, both legitimately available without the clean image. Despite this asymmetry
being stacked *against* Wiener, it still matches or nearly matches the oracle-tuned
bilateral filter at every noise level, and clearly beats both oracle-tuned fixed-cutoff
LPFs throughout — a meaningfully strong result given Wiener had no access to ground
truth at all.

(Also note, as a secondary honesty check: the Gaussian/Butterworth `D0` sweep range was
`[10,150]`; at σ=5,10,20 the oracle optimum sits right at the sweep's upper boundary,
`D0=150` — as in Problem 2, this means the *true* unconstrained optimum for these very
low noise levels may lie slightly beyond 150, i.e. these two methods might do marginally
better with a wider sweep; it would not change the qualitative conclusion above, since
at `D0=150` they are already a hard-cutoff/near-no-op and still clearly trail Wiener
and bilateral.)

**Wiener does not win every comparison here** — the bilateral filter (oracle-tuned)
edges it out at every σ tested, confirming the PDF's point that MMSE-linear optimality
does not guarantee beating a well-tuned nonlinear filter. The margin is small, though,
and Wiener achieves it *without* oracle tuning.

### Wiener shrinkage vs. frequency and noise power

`results/5_1_wiener_gain_profiles.png` plots `W`'s radial profile for each σ. Two
clear trends, both expected from the formula `W=Sff/(Sff+Snn)`:
- **Within one σ, `W` decreases smoothly with frequency** (radius): at low frequency,
  `Sff` (image energy, concentrated at low frequency in any natural image) vastly
  exceeds `Snn` (flat across frequency for white noise), so `W≈1` (pass through); at
  high frequency, `Sff` has decayed toward the noise floor while `Snn` stays constant,
  so `W` falls toward 0 (suppress).
- **Across σ, the whole curve drops faster and lower as σ grows**: larger `σ` raises
  `Snn` uniformly, so the frequency at which `Sff` and `Snn` become comparable (where
  `W` crosses ≈0.5) moves to progressively *lower* frequencies — more noise forces
  the filter to suppress a wider band, not just the very highest frequencies.

**Contrast with a fixed-cutoff LPF:** a Gaussian/Butterworth/ideal LPF's `D0` is a
single number chosen in advance (here, oracle-tuned per σ) and produces a *fixed*
transfer-function *shape*, regardless of how the image's own energy is actually
distributed across frequency — the same shape just gets re-scaled (re-centred) by
`D0`. Wiener's gain, by contrast, is **derived per-frequency from the data itself**
(the image's own radially-averaged periodogram): wherever the *specific* image has
unusually strong or weak energy at a given frequency (irrespective of simple radial
distance alone), `Sff` reflects that directly, and `W` adapts accordingly — a
content-adaptive, continuously-varying gain rather than one global cutoff radius. This
is the structural reason it can out-perform an oracle-tuned fixed-cutoff filter at every
noise level without itself needing the oracle.

---

## 5.2 Test the Assumptions

### Noise-power mis-specification (σ=20)

`results/5_2_noise_misspecification.png`, `results/5_2_result.json`:

| assumed `Snn` scale | PSNR (dB) | SSIM |
|---|---|---|
| 0.5× (under-estimated) | 25.87 | 0.551 |
| **1.0× (correct)** | **28.01** | **0.691** |
| 2.0× (over-estimated) | 26.48 | 0.643 |

Exactly the expected pattern from `W=Sff/(Sff+Snn)`: **under-estimating `Snn`** makes
`W` too close to 1 everywhere (the filter under-trusts how noisy the data really is),
leaving visible **residual noise** in the output (visible grain in the "under" panel);
**over-estimating `Snn`** makes `W` too close to 0 even at frequencies where real
signal energy still exceeds the *true* noise level, causing **over-smoothing** (visible
loss of fine detail/texture in the "over" panel, lower PSNR/SSIM than the correct
case). The correctly-specified filter sits at the peak of this trade-off, as expected.

### Why correctly-specified Wiener can still blur edges

The Wiener filter's optimality is derived entirely from **second-order statistics
assumed stationary over the whole image** (`Sff`, `Snn` are single functions of
frequency alone, with no spatial/positional dependence). A true edge is a highly
**local**, non-stationary feature — but the filter has no way to know *where* edges
are, or to treat the handful of frequencies that happen to carry a particular edge any
differently near that edge than far from it. The same global gain `W(u,v)` is applied
everywhere in the image, uniformly suppressing whichever frequencies are
noise-dominated *on average over the whole image*, even at specific spatial locations
(like a sharp edge) where those same high frequencies actually carry essential genuine
signal. **MMSE optimality under this model guarantees the smallest *average* squared
error over the whole image, given the stated (stationary, linear) assumptions — it
makes no guarantee at all about preserving any particular local feature such as an
edge.** Edge-region crop: `results/5_2_edge_region_blur.png` (the correctly-specified
Wiener output's edge region vs. clean/noisy), selected automatically as the
highest-gradient 40×40 block of the clean image — shows the Wiener output noticeably
softer right at the edge than the clean image, despite being "correctly specified" in
the global-statistics sense.

### Real noisy photo — **PENDING**

The PDF requires applying the white-noise Wiener filter to a **genuinely captured**
noisy photo (not synthetic noise) and comparing qualitatively against the synthetic
cases, discussing departures from the white/stationary/signal-independent noise
assumption (shot noise, demosaicing, sharpening, compression).

**This sub-deliverable is explicitly incomplete.** `data/own_captures/` is currently
empty — no real noisy photo has been supplied. Per the user's explicit choice when
asked about this during planning, the rest of Problem 5 was completed in full and this
one sub-part was left pending rather than faked with synthetic noise (which would not
satisfy the PDF's requirement, and would misrepresent what was actually tested).
`problem5.py::section_5_2` checks `data/own_captures/` for any image file and will
process it automatically (estimating noise power from a flat image corner, applying
`wiener_denoise`, and saving a before/after figure) the next time the script is run
once a real photo is added — no code changes needed, just drop a file in
`data/own_captures/` and re-run.

**To complete this**: capture a genuinely noisy photo (e.g. a high-ISO/low-light phone
shot), save it under `data/own_captures/`, and re-run `problem5.py`.

---

## Deliverables checklist

- [x] Wiener, low-pass, and bilateral PSNR/SSIM table across σ.
- [x] Filter-parameter settings for every method/σ, oracle-tuned status labeled.
- [x] Figures showing noise-power mis-specification (under/correct/over).
- [ ] **Real-noise case study — pending a genuinely captured noisy photo** (see above).
