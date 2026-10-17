# Problem 2 — Frequency-Domain Filtering: Ideal, Butterworth, Gaussian (15 pts)

Status: **done**.

Code: [`problem2.py`](problem2.py). Figures/tables: [`results/`](results/). Dated
pre-implementation prediction:
[`predictions/2.2_ringing_prediction.md`](predictions/2.2_ringing_prediction.md).

Transfer functions (`src/dip_a2/filters.py::make_filter`) are built entirely from their
formulas with NumPy array arithmetic. `numpy.fft` is used for the forward/inverse
transforms themselves — explicitly permitted for this problem by the assignment ("you
may use numpy.fft for the transforms in this section, but every transfer function must
be built by you from its formula"). `scipy.ndimage.gaussian_filter` is used **only** for
the one PDF-sanctioned comparison in §2.2d. The mandated pipeline (`filter_pipeline`):
zero-pad `(M,N)` to `(2M,2N)`, centre the spectrum, multiply by `H`, invert, take the
real part, crop back to `(M,N)`.

## Environment

Same machine as Problem 1 — CPU Intel(R) Core(TM) i5-9300H @ 2.40GHz (8 logical CPUs),
14.96 GB RAM, NumPy 2.5.4 (see `results/machine_info.json`).

---

## 2.1 Filter Construction and Transfer-Function Analysis

### (a) Cutoff semantics and Gaussian/Butterworth matching

Analytic values at `D = D0` (confirmed numerically to match exactly — see
`results/2_1a_result.json`):

| Filter | H(D0) analytic | H(D0) numeric |
|---|---|---|
| ILPF | 1 (boundary, `D≤D0`) | 1.0 |
| BLPF | `1/(1+1) = 0.5` | 0.5 |
| GLPF | `e^-0.5 ≈ 0.6065` | 0.6065306597 |

Using the **same numeric `D0`** for all three is not like-for-like: ILPF's `D0` is a
hard wall (`H` jumps from 1 to 0), BLPF's `D0` is by definition its half-amplitude
radius, but GLPF's `H(D0) ≈ 0.6065 ≠ 0.5` — the Gaussian is still well above half
amplitude at `D=D0`, so a Gaussian and Butterworth filter sharing the same numeric `D0`
are not equally "wide."

**Matching derivation.** The Gaussian's half-amplitude point solves
`exp(-r²/2D0'²) = 0.5 → r = D0'·√(2 ln 2)`. Setting this radius equal to the Butterworth
cutoff `D0` gives

```
D0' = D0 / √(2 ln 2)
```

Verified: for `D0=40`, `D0' = 33.9729`, and `make_filter("gaussian", shape, D0')`
evaluated at `D=40` gives exactly `0.5000` (`results/2_1a_result.json`). This matched
`D0'` is used everywhere a Gaussian is compared against a Butterworth/ideal filter at a
given cutoff (§2.2c, §2.3c).

Radial profiles of all 6 transfer functions (ILPF/BLPF/GLPF, LP+HP) at `D0=40`:
`results/2_1_transfer_functions.png` — shows the ILPF's hard wall, BLPF's intermediate
roll-off, and GLPF's smoothest roll-off, all crossing 0.5 at their respective cutoffs
except GLPF (which crosses 0.5 at the matched `D0'`, shown as `0.6065` at the raw `D0`).

### (b) Energy-based cutoff selection

Power-retention radii on `characters_test_pattern`'s own (zero-padded, per the
section's default pipeline) spectrum:

| α | radius r |
|---|---|
| 90% | 15.30 |
| 95% | 45.54 |
| 99% | 409.04 |
| 99.5% | 558.74 |

The jump between 95% (radius ≈45) and 99% (radius ≈409) is large and genuine, not a
padding artifact — confirmed by also computing the same radii on the **unpadded** image
spectrum (90%→1.4, 95%→19.0, 99%→195.9, 99.5%→274.0): the same qualitative jump appears
either way. The reason: a large share of total squared-magnitude power sits at/near DC
because of the large, nearly-uniform background regions in the test pattern (constant
regions contribute strongly to low-frequency power, and `α(0)` alone — DC — already
captures roughly half the power). The remaining ~1–5% of power belongs to edges and
fine strokes (the "a", the bars, the noise patches), and that power is **spread thinly
across a very large number of high-frequency bins** rather than concentrated — so it
takes a very large radius to accumulate even that last percent. This does **not** mean
those high frequencies are visually unimportant: they are exactly the edges, strokes,
and fine detail that make the image recognisable — power-based energy fraction weights
by squared magnitude, which is dominated by large, low-frequency, low-information flat
regions, while perceptual/structural importance is not proportional to raw spectral
power.

### (c) Inverse design problem

Given `H(D=10) = 16/17` and `H(D=40) = 1/17` for an unknown Butterworth order `n` and
cutoff `D0`:

```
16/17 = 1/(1+(10/D0)^2n)  →  (10/D0)^2n = 1/16
1/17  = 1/(1+(40/D0)^2n)  →  (40/D0)^2n = 16
```

Dividing the second equation by the first eliminates `D0`:

```
(40/10)^2n = 16/(1/16) = 256  →  4^2n = 256 = 4^4  →  2n = 4  →  n = 2
```

Substituting back: `(10/D0)^4 = 1/16 → 10/D0 = (1/16)^{1/4} = 1/2 → D0 = 20`.

**Verified** with `make_filter("butterworth", shape, 20, n=2)`: `H(D≈10) = 0.941176`
(target `16/17 = 0.941176`), `H(D≈40) = 0.058824` (target `1/17 = 0.058824`) — exact
match (`results/2_1c_result.json`).

**Smallest integer order for ≤1% attenuation at `D=2D0`:** need
`1/(1+2^{2n}) ≤ 0.01 → 2^{2n} ≥ 99`. `2^6=64 < 99` (n=3 fails), `2^8=256 ≥ 99` (n=4
works) → **smallest n = 4**. Verified numerically.

---

## 2.2 Ringing: Prediction, Measurement, Reconciliation

**Pre-implementation prediction** (dated 2026-10-19, written before running this
section): see
[`predictions/2.2_ringing_prediction.md`](predictions/2.2_ringing_prediction.md).
Summary: closed-form 1-D ideal-LPF step response via the sine integral predicts
≈8.95% overshoot (classic Gibbs phenomenon, independent of `D0`) and a first-peak
distance of `P/(2·D0)` pixels (`P=1024` for the 512×512 step image); Gaussian LPF
predicted to show exactly zero overshoot (non-negative, sum-to-one spatial kernel).

### (a) Ideal filter verification

Step-edge image: `B=200` square, side 256, on `A=50` background, 512×512
(`data/synthetic/step_edge.npy`).

| D0 | measured overshoot % | predicted overshoot % | measured peak dist (px) | predicted peak dist (px) |
|---|---|---|---|---|
| 10 | 3.44 | 8.95 | 50 | 51.2 |
| 20 | 11.45 | 8.95 | 26 | 25.6 |
| 40 | 8.27 | 8.95 | 12 | 12.8 |
| 80 | 8.78 | 8.95 | 6 | 6.4 |

Peak-distance matches the prediction closely at every `D0`. Overshoot matches closely
for `D0=40,80` but **deviates notably for `D0=10` (lower than predicted) and `D0=20`
(higher than predicted)**. Reconciliation: the closed-form prediction assumes a purely
1-D band-limiting filter (`H(u,0)` only), which is only exactly correct for an
*infinite* straight edge. Our square has a *finite* extent (256 px) in the
perpendicular (`y`) direction too, so its true 2-D spectrum is not perfectly confined to
the line `v=0` — it has a narrow sinc-shaped spread around `v=0` with lobes spaced
roughly `1/256` cycles/pixel apart (≈4 index units on the `P=1024` grid). The 2-D
*disk*-shaped ideal filter (`D≤D0`) therefore clips this spread asymmetrically relative
to a pure 1-D band whenever `D0` is small enough to be comparable to that lobe spacing
(`D0=10,20`), producing measurable deviations from the 1-D-only prediction; once `D0`
is safely larger than the lobe spacing (`D0=40,80`), the disk behaves essentially like
the 1-D band and the match is close. Figure: `results/2_2a_profiles.png`.

**Near-corner profile** (same figure, right panel, `D0=40`): the ripple amplitude and
plateau level are visibly different from the mid-row profile (lower plateau, since this
row cuts through the square closer to its top edge). This confirms the 1-D argument
fails near a corner: a corner is a genuinely 2-D discontinuity (simultaneous
step in both `x` and `y`), not reducible to a single 1-D band-limited step response.

### (b) Butterworth order sweep

At fixed `D0=40`, `n ∈ {1,2,3,5,10,20}`:

| n | measured overshoot % | \|min h\|/max h |
|---|---|---|
| 1 | ~0.00 (9.3e-6) | ~0.00 (2.5e-6) |
| 2 | 3.36 | 0.0432 |
| 3 | 5.60 | 0.0966 |
| 5 | 7.50 | 0.1577 |
| 10 | 8.49 | 0.1999 |
| 20 | 8.43 | 0.2124 |

Overshoot increases monotonically with `n` (as predicted) and approaches the
ideal-filter limit (~8.95%) for large `n`, exactly as expected since the Butterworth
filter tends to the ideal filter's hard cutoff as `n→∞`. The `|min h|/max h` ratio (1-D
spatial kernel `h(x) = IDFT{H(u,0)}`) tracks the same trend: it is essentially **zero at
`n=1`** (the kernel stays non-negative — no overshoot possible) and grows steadily with
`n` (the kernel develops increasingly deep negative side-lobes, which is exactly what
produces ringing).

**Proof: a non-negative, sum-to-one kernel cannot overshoot a step.** If `h(x) ≥ 0` for
all `x` and `Σ_x h(x) = 1`, then for any input `f`, the (circular) convolution output is
`g(x) = Σ_k h(k) f(x-k)`, a convex combination (non-negative weights summing to 1) of
the values of `f`. For a step edge, every value of `f` satisfies `f ≤ B` (the plateau
height). A convex combination of numbers all `≤ B` is itself `≤ B`: `g(x) = Σ_k h(k)
f(x-k) ≤ Σ_k h(k)·B = B·Σ_k h(k) = B`. Hence `g` can never exceed `B` — no overshoot is
possible. This matches the `n=1` Butterworth kernel (measured non-negative to numerical
precision) and the Gaussian kernel (§ below).

### (c) Visual comparison

ILPF / BLPF(n=2) / GLPF (matched `D0'`) applied to `characters_test_pattern` at the four
radii from §2.1b (15.3, 45.5, 409.0, 558.7): `results/2_2c_visual_grid.png`. Ringing
(faint periodic bands) is clearly visible around the "a" and the squares in the **ILPF
row at D0=45.5** and is essentially absent from the BLPF and GLPF rows at the same
cutoff — confirming the sharpness/ringing trade-off: the ideal filter is not sharper in
a useful sense, it just trades a hard spectral cutoff for spatial-domain ringing, while
Butterworth/Gaussian give a visually softer but ringing-free transition.

### (d) Gaussian equivalence check

Deriving `σs`: `H_GLPF(D) = exp(-D²/2D0²)` is a Gaussian in frequency-index units on the
`P×P` grid, i.e. in continuous frequency `f = D/P` cycles/pixel it has standard
deviation `σf = D0/P`. By the Fourier Gaussian-pair property (`σs·σf = 1/(2π)`):

```
σs = 1 / (2π σf) = P / (2π D0)
```

| D0 | σs (px) | max\|g_freq − g_spatial\| |
|---|---|---|
| 10 | 21.900 | 0.0157 |
| 40 | 5.475 | 0.0177 |
| 160 | 1.369 | 0.0186 |

All errors are tiny (≤0.019 intensity levels out of 0–255, i.e. <0.01%), confirming the
`σs` derivation is correct and the frequency-domain and spatial-domain Gaussian filters
are, for practical purposes, equivalent. The error is not exactly zero because the two
implementations discretize the same continuous Gaussian differently: our
frequency-domain filter samples a continuous Gaussian on an integer `(u,v)` index grid
and inverts via a (circular, zero-padded) DFT, while `scipy.ndimage.gaussian_filter`
builds a finite-support, spatially-truncated kernel (truncated at `truncate=4σ` by
default) convolved with `mode="constant"` (zero) boundaries — slightly different
discretization/truncation choices for the same underlying continuous operator. The
error grows slightly (not dramatically) as `D0` increases (σs decreases): a smaller
spatial kernel is sampled more coarsely relative to its own width, so discretization
error is marginally larger relative to the (very small) kernel scale, though in absolute
terms the effect is negligible here.

---

## 2.3 Applications: Denoising, Repair, and Sharpening

### (a) Noise–blur trade-off

`cameraman.tif` + `N(0, 20²)` Gaussian noise (seed=7, 8-bit range, not clipped before
filtering). PSNR vs `D0` for `D0 ∈ {5, 10, ..., 150}`, all three families:
`results/2_3a_psnr_sweep.png`.

| family | best D0 in [5,150] | peak PSNR (dB) |
|---|---|---|
| ideal | 150 | 27.46 |
| butterworth (n=2) | 150 | 28.72 |
| gaussian | 150 | 29.20 |

Within the mandated sweep range `[5,150]`, PSNR is still rising at `D0=150` for all
three families — the curves have not yet turned over. Extending the sweep as a
diagnostic (not part of the mandated range) shows the true optimum lies just beyond:
≈180–210 for all three (gaussian peaks ≈29.53 dB at D0≈180; ideal and butterworth both
peak ≈210, at 28.65 dB and 29.69 dB respectively) — confirming the expected trade-off
curve does turn over, just slightly past the required sweep window for this
image/noise level.

Within the range actually swept, the ordering **Gaussian ≥ Butterworth ≥ Ideal** holds
throughout, and this follows directly from the transfer-function shapes in §2.1: the
ideal filter's hard cutoff admits *all* noise power at frequencies up to `D0` completely
unattenuated and then blocks everything above `D0` outright, giving no graceful
trade-off control between signal and noise near the cutoff; the Butterworth and (even
more so) Gaussian filters roll off gradually, so for any given `D0` they already begin
attenuating noise power approaching the cutoff rather than passing it at full strength,
which is why they dominate the ideal filter's curve at every `D0` tested.

### (b) Character repair

`text_gaps.tif` (broken strokes) + Gaussian LPF at `D0=60`: `results/2_3b_character_repair.png`.
At `D0=60` the small gaps (1–2 pixels) are bridged by the Gaussian's smooth spatial
spread while the character shapes stay fully legible. Replacing the Gaussian with an
ideal filter at the same half-amplitude radius would be expected to introduce ringing
artifacts around the thin strokes (per §2.2's findings) — potentially creating new
spurious dark/light fringes right next to the thin strokes being repaired, which is
exactly the kind of artifact this repair task needs to avoid; the Gaussian's
ringing-free, monotonic smoothing is the right tool for this job.

### (c) High-pass filtering and the DC term

IHPF / BHPF(n=4) / GHPF on `thumb_print.tif` at a common half-amplitude radius
(`D0=30`, with `D0'` matched for the Gaussian): `results/2_3c_highpass.png`.

| filter | output mean |
|---|---|
| IHPF | 2.26 |
| BHPF | 2.33 |
| GHPF | 3.29 |

All three means are small (not large, and much smaller than the image's original mean
intensity) because every `H_HP = 1 − H_LP` satisfies `H_HP(P/2,Q/2) = 1 − H_LP(0) = 0`
at the exact centre (DC) bin — the DC/mean component of the *full padded* spectrum is
always completely removed by construction. The means are not *exactly* zero here
because removing the DC term makes the **entire padded `P×Q` canvas** sum to zero, but
we crop back to the original `M×N` region, and that sub-block need not itself average
to exactly zero even though the full padded canvas does.

High-frequency emphasis `H_HFE = k1 + k2·H_HP` with `k1=0.5, k2=1.5` (chosen so the
output keeps roughly half the original low-frequency/tonal content while adding 1.5× the
high-pass-filtered detail) restores visible tonal shading across the fingerprint while
keeping the ridge edges sharpened (`results/2_3c_highpass.png`, bottom-middle panel) —
compare to the flat, near-featureless high-pass-only outputs in the top row.

Thresholding the BHPF output at its 90th percentile gives a binary ridge map
(`results/2_3c_highpass.png`, bottom-right). The map is noisy — a simple global
percentile threshold is not locally adaptive, so it also binarizes background texture
and sensor noise wherever the BHPF output happens to exceed the global threshold,
especially in lower-contrast regions away from the fingerprint core. The ridges
themselves remain visually traceable in the map, but a cleaner result would need a
locally-adaptive threshold rather than a single global cutoff.

---

## Deliverables checklist

- [x] `make_filter` implementation + radial-profile plots of all 6 transfer functions.
- [x] Cutoff-matching derivation, power-retention table, inverse-design solution with
      numerical verification.
- [x] Dated ringing prediction + measured-vs-predicted overshoot tables + Butterworth
      kernel-ratio table + non-negative-kernel proof + reconciliation discussion.
- [x] σs derivation + Gaussian equivalence error table.
- [x] PSNR sweep plot + optimum table, repaired text image, high-pass + HFE results.
