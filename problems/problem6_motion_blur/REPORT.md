# Problem 6 — Motion Blur and Deblurring (20 pts)

Status: **done**. No dated pre-implementation prediction is required — the PDF's
Problem 6 text contains no "Pre-implementation Task" instruction.

Code: [`problem6.py`](problem6.py). Shared implementation:
[`src/dip_a2/motion_blur.py`](../../src/dip_a2/motion_blur.py). Figures/tables:
[`results/`](results/).

Data: `data/provided/problem5_6/reference_image.jpg` (3072×2048 RGB, shared with
Problem 5), downsampled to 768×512 (same practical rationale as Problems 3/5).
**Deliberate choice: processed in RGB, not grayscale** (unlike Problems 1–5) — the
PDF's own SNR formula for this problem is written with an explicit colour-channel sum
(`Σx,y,c`), signalling colour treatment is expected here. The motion OTF is identical
across channels (computed once) and applied independently to each of R/G/B; PSNR is
computed on the full RGB array directly; SSIM (2-D only, per `metrics.py::ssim`) is
computed on the luminance projection (`0.299R+0.587G+0.114B`) of both images — standard
practice for colour SSIM.

## Environment

Same machine as Problems 1–5 — CPU Intel(R) Core(TM) i5-9300H @ 2.40GHz (8 logical
CPUs), 14.96 GB RAM, NumPy 2.5.4 (see `results/machine_info.json`).

---

## 6.1 Model and Synthesize

### PSF derivation and discretization

For uniform linear motion of length `L` at angle `θ`, the continuous PSF is

```
h(x,y) = (1/L) ∫_{-L/2}^{L/2} δ(x − t cosθ, y − t sinθ) dt
```

i.e. a unit-mass line segment of length `L` through the origin at angle `θ`: the
camera/scene motion smears a point into this line, and the `1/L` factor normalizes
total energy to 1 (energy preservation — a PSF must integrate to 1 so the blurred
image keeps the same overall brightness as the original).

**Discretization** (`motion_blur.py::motion_psf_otf`): sample `t` finely (20×
oversampled relative to `L`) over `[-L/2, L/2]`, compute `(x,y)=(t cosθ, t sinθ)` for
each sample, round to the nearest integer pixel offset, and accumulate a count at that
offset in a `shape`-sized array (same size as the image being blurred); normalize so
the kernel sums to 1. The kernel is placed with the origin at array index `(0,0)`
(wrapping via modulo, not centred in the middle of the array) — i.e. built directly in
the layout `numpy.fft.fft2` expects for **circular convolution** (multiply by its own
DFT), the same "placed at the origin corner" convention used for kernels throughout
this repo since Problem 1.

### Boundary convention

**Circular**, as the PDF recommends, used identically for both forward blurring (image
FFT × kernel OTF, inverse FFT) and all deconvolution (naive inverse, Wiener) — no
zero-padding step in this problem (unlike Problems 2–3, which used `P=2N` zero-padding
for a different purpose: avoiding linear-vs-circular convolution mismatch for a
*small* spatial kernel against a large image; here the kernel and the deconvolution
operations are all meant to exactly invert each other under the *same* circular model,
so using one consistent native-resolution circular convention throughout is both
simpler and exactly self-consistent).

### Synthesis

`L=25px, θ=30°`. Blurred the clean RGB image (same `H` per channel), then added noise
at a stated `SNRdB = 10·log10(Σf²/Σn²) = 30dB` (not clipped before filtering) — solved
directly for the Gaussian `σ` achieving that ratio (`σ=3.982`, `motion_blur.py::add_noise_at_snr`).
Figure: `results/6_1_psf_and_blur.png` (PSF kernel, clean/blurred/blurred+noise).

### OTF derivation and numerical verification

Writing `(fx,fy)` for spatial frequency in cycles/pixel, the 1-D projection of the PSF
onto the motion axis is a rect function of width `L` and the Fourier transform of the
continuous PSF (a line integral / Radon-type projection) is the 1-D Fourier transform
of that rect sampled along the motion direction:

```
H(fx,fy) = sinc(L(fx cosθ + fy sinθ))
```

**Zero lines.** `sinc(z)=0` at `z=m, m∈ℤ\{0}`, so

```
fx cosθ + fy sinθ = m/L,  m ∈ ℤ\{0}
```

a family of parallel lines in frequency space, perpendicular to the motion direction
`(cosθ,sinθ)` (each line is a level set of the directional-projection coordinate
`fx cosθ+fy sinθ`, which is constant along the perpendicular direction and varies along
the motion direction itself) — adjacent zero lines are spaced `1/L` apart along that
projection coordinate. A discrete FFT grid samples `(fx,fy)=(u/M,v/N)` only at integer
`(u,v)`, so it generally lands *near*, not exactly *at, the continuous zero lines.

**Numerical verification**: compared the closed-form `sinc` prediction
(`motion_blur.py::motion_otf_analytic`, sampled directly on the DFT grid) against the
actual DFT of the rasterized spatial kernel from `motion_psf_otf`
(`results/6_1_otf_comparison.png`):

| statistic | value |
|---|---|
| median \|ΔH\| | 0.0136 |
| 99th percentile \|ΔH\| | 0.3646 |
| max \|ΔH\| | 0.5400 (at frequency ≈(−0.289, −0.500)) |

Agreement is excellent almost everywhere (median error 0.0136, on a quantity ranging
0–1) — the large outlier sits essentially exactly at `fy=−0.5`, the **Nyquist
frequency** of the 768-pixel axis. This is expected, not a bug: the rasterized kernel
is a genuinely discrete, periodic signal subject to aliasing, while the closed-form
`sinc` assumes a continuous, unaliased projection; the two can only disagree
significantly right at the single most alias-prone frequency (Nyquist), exactly what
is observed.

---

## 6.2 Naive Inverse Filtering and Why It Fails

Implemented `F_hat = G/H` with a numerical floor on `|H|` (`floor=1e-3`): wherever
`|H(u,v)| < floor`, that frequency's `|H|` is replaced by `floor` (preserving the phase
of `H`) before dividing, so the naive estimate stays finite everywhere.

`results/6_2_naive_inverse_spectra.png` shows `log|G|` (left) and the **raw** (pre-clip)
naive-inverse spectrum `log|F_hat|` (right), with the predicted OTF zero lines overlaid
in red. The right panel shows clear bright streaks precisely aligned with the
predicted zero lines — visible amplification exactly where the theory predicts it.

Quantified by comparing the mean `|F_hat|` at bins where `|H|` actually hit the floor
against *other*, non-floored high-frequency bins at a comparable frequency magnitude
(excluding the naturally-dominant low-frequency/DC core, which would otherwise swamp a
naive whole-spectrum average and understate the effect):

| | mean \|F_hat\| |
|---|---|
| at floored bins (\|H\|<1e-3) | 1.49×10⁶ |
| other high-frequency bins | 7.26×10⁴ |
| **ratio** | **20.5×** |

`results/6_2_naive_inverse_image.png` shows the resulting (display-clipped) spatial
image — dominated by noise, since the naive inverse filter inverts the noise just as
eagerly as the signal at every frequency.

**Why any nonzero noise can become arbitrarily large as `|H(u,v)|→0`:** the naive
estimate is `F_hat = G/H = F + N/H`; as `|H|→0`, the noise term `N/H` is a fixed
nonzero numerator divided by a denominator shrinking to zero, so it diverges without
bound — amplifying whatever noise happens to be present at that frequency arbitrarily,
regardless of how small the noise actually is.

---

## 6.3 Wiener Deconvolution

Derived/implemented the scalar-`K` regularized Wiener deconvolution filter
`F_hat = H*G / (|H|²+K)` (`motion_blur.py::wiener_deconvolve`) — this is exactly the
general Wiener result from Problem 5 (`W=Sff/(Sff+Snn)`) specialized to a *known,
deterministic* blur `H` rather than an unknown PSF, with `K` playing the role of a
constant, frequency-independent **noise-to-signal power ratio** `Snn/Sff` (the
simplification that makes `K` a single tunable scalar rather than a full function of
frequency). Image intensities are the raw 8-bit range (no [0,1] rescale needed here
since both `G` and `H` enter linearly/ratio-wise and `K`'s absolute scale is tuned
empirically against that range).

Swept `K` logarithmically (`10⁻⁶` to `1`, 25 points) at 4 SNR levels (10/20/30/40 dB):
`results/6_3_K_sweep.png` — every curve is a clean, classic U-shape: **low `K`** behaves
like the naive inverse filter (noise amplification dominates, PSNR collapses — visible
at `K=10⁻⁶` in `results/6_3_K_qualitative.png`, which looks like pure noise); **high
`K`** under-deconvolves (the filter degenerates toward `H*/|H|²`-scaled low-pass
behaviour, leaving visible residual motion blur — visible at `K=1` in the same figure,
still clearly streaked).

| SNR (dB) | optimal K | PSNR at optimum (dB) |
|---|---|---|
| 10 | 0.178 | 18.20 |
| 20 | 0.0562 | 21.75 |
| 30 | 0.0316 | 24.28 |
| 40 | 0.00562 | 26.45 |

The optimal `K` **decreases monotonically as SNR improves** (noise level drops) —
exactly as expected: less noise means less regularization is needed to suppress
amplification, so the optimum shifts toward smaller `K` (closer to the unregularized
inverse filter), while noisier data needs a larger `K` to damp the inevitable
near-zero-OTF amplification identified in §6.2.

---

## 6.4 Estimate the Blur, Then Remove It

**Photo**: `data/own_captures/motion_blur.jpeg` — a genuinely motion-blurred photo of
a hand swiped quickly in front of the camera, against an otherwise stationary desk
scene (bottle, plant, figurines, a handwritten name card confirming provenance).

**Estimation method** (`motion_blur.py::estimate_motion_blur`): computes the photo's
own centred log-magnitude spectrum; for a sweep of candidate angles, projects the
spectrum radially outward along that angle and scores how periodically it dips (via
the 1-D FFT power of the profile, excluding DC); takes the angle with the strongest
periodic dip pattern as the dark-stripe orientation (motion direction is perpendicular
to that, since the stripes themselves are perpendicular to the motion — derivable
directly from the zero-line geometry in §6.1: the zero lines run perpendicular to
`(cosθ,sinθ)`); recovers `L` from the dominant spacing of those dips via the same
`1/L` relationship derived in §6.1.

**Result**: `θ_est = 99.0°`, `L_est = 2.69px`, deconvolved with `K=0.01`.
`results/6_4_real_deblur.png`.

### This is itself the model-failure case study

`L_est≈2.7px` is clearly too small — by eye, the blur streak on the hand in the photo
spans tens of pixels, not 2–3. Looking at the deblurred result confirms this: the hand
is **barely sharpened at all**, while the already-sharp background (the bottle, the
chair, the desk edge) picks up a faint fine vertical ringing/over-sharpening texture
instead — the opposite of what a correctly-estimated deblur should do.

**Why the estimator failed, connected directly to the model's core assumption**: the
entire linear motion-blur model in §6.1–6.3 assumes the blur is **spatially
invariant** — the *same* PSF applies to every pixel in the frame. In this photo, that
assumption is badly violated: only the moving hand is blurred; the desk, bottle, and
background are sharp. The log-magnitude spectrum used for estimation is a property of
the **whole image at once** — and since most of the frame (everything except the hand)
is sharp, the spectrum is dominated by the sharp background's own high-frequency
content, not by the comparatively small, localized blur signature from the hand. The
angle-scoring procedure still found *some* best angle (there is always a maximum,
even a weak/spurious one), but the recovered length is essentially measuring noise in
the mostly-sharp spectrum rather than the real, much larger, localized blur extent.

This is a direct, concrete illustration of the limitation the PDF asks to connect to:
a single global `(θ,L)` and a single spatially-invariant Wiener deconvolution **cannot
correctly handle a photo where the blur itself varies spatially** (here, a moving
*subject* against a stationary background — one of the PDF's own named causes,
alongside camera rotation, changing velocity, occlusion, and clipped highlights). A
real fix would require *segmenting* the moving region first and estimating/deconvolving
locally — entirely outside the single-global-PSF model this problem builds.

Evaluated qualitatively, as required (no clean reference exists for a real photo):
the deblurred output is not a successful deblur of the hand, but it *is* a correctly
diagnosed and explained failure — arguably the most informative possible outcome for
this sub-task, since it makes the spatially-invariant assumption's real-world fragility
directly visible rather than just stated.

---

## Deliverables checklist

- [x] PSF derivation + OTF zero-spacing formula, numerically verified against the
      rasterized kernel's own DFT.
- [x] Synthetic inverse-filter spectrum with predicted zero locations marked.
- [x] Wiener K sweeps + optimum sensitivity to noise (4 SNR levels, clean monotonic
      trend).
- [x] Real, self-captured blur case study, with θ, L, deblurred output, and a
      documented model-failure mode (localized subject motion vs. the global
      spatially-invariant PSF assumption).
