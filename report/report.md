# DIP Assignment 2 — Report

## Environment

- CPU: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz (8 logical CPUs)
- RAM: 14.96 GB
- OS: Linux-7.0.0-38-generic-x86_64-with-glibc2.43
- Python: 3.14.4
- NumPy: 2.5.4

(Captured programmatically by `problems/problem1_dft/problem1.py::record_machine_info`,
saved at `problems/problem1_dft/results/machine_info.json`.)

---

## Problem 1 — DFT and the Machinery of Frequency (15 pts)

Code: [`problems/problem1_dft/problem1.py`](../problems/problem1_dft/problem1.py).
Figures/tables: `problems/problem1_dft/results/`. Dated pre-implementation prediction:
[`predictions/1.2_phase_magnitude_prediction.md`](../problems/problem1_dft/predictions/1.2_phase_magnitude_prediction.md).

The custom DFT (`src/dip_a2/dft.py`) builds the 1-D matrix `W_N[k,n] = exp(-j2πkn/N)`
and computes the 2-D transform by row-column separability, `F = W_M f W_N^T`, using only
NumPy array/matrix arithmetic — no call to `numpy.fft` anywhere inside it.

### 1.1 Build and Validate the DFT

**Validation against `numpy.fft.fft2`** on `data/provided/problem1/q1.png` (512x512):

| Metric | Value |
|---|---|
| max \|F_custom − F_numpy\| | 3.3700e-06 |
| relative to max\|F_numpy\| | 9.961e-14 |

The error is at float64 round-off level (~1e-14 relative), confirming the custom
separable-matrix DFT is correct, just computed less efficiently (O(N^3) via matrix
multiplication per row/column pass vs. O(N^2 log N) for the FFT).

#### (a) Conjugate symmetry proof

For real-valued `f(x,y)`,

```
F(u,v) = Σ_x Σ_y f(x,y) exp(-j2π(ux/M + vy/N))
F*(u,v) = Σ_x Σ_y f(x,y) exp(+j2π(ux/M + vy/N))   (since f is real, f* = f)
        = Σ_x Σ_y f(x,y) exp(-j2π((-u)x/M + (-v)y/N))
        = F(-u,-v)
```

so `F(u,v) = F*(-u,-v)`, i.e. **Hermitian symmetry**. On the DFT's periodic index grid,
`-u ≡ M-u (mod M)` and `-v ≡ N-v (mod N)`, so this pairs index `(u,v)` with
`(M-u, N-v)`. Exactly **half** of the `(u,v)` plane (one half-space, e.g. `u ∈ [0, M/2]`
together with all `v`, excluding the redundant mirrored half) determines the entire
spectrum — the other half is recoverable by conjugation and carries no extra
information.

Despite this redundancy, visualizers display the full, symmetric magnitude spectrum
because: (1) it preserves the natural interpretation of frequency as signed
(`u` ranging symmetrically about 0 after `fftshift`, matching positive/negative spatial
frequency, rather than an arbitrary half-plane cut that depends on orientation), and
(2) showing only half would break the visually intuitive radial/angular symmetry used
to read off orientation and cutoff radius of filters — redundant information is kept
for interpretability, not because it is informationally necessary.

Numerically verified: `max|F(u,v) − conj(F(-u,-v))|` = **3.3751e-06** (same round-off
level as the DFT validation above, i.e. the property holds up to floating-point error).

#### (b) Empirical verification of the 2-D Spatial Shift Theorem

Theorem: shifting `f` circularly by `(x0, y0)` multiplies its DFT by a linear phase
ramp, `F_shifted(u,v) = F(u,v) exp(-j2π(u x0/M + v y0/N))`. Verified on the test image
for five shifts:

| (x0, y0) | max abs error | relative error |
|---|---|---|
| (1, 0) | 6.249e-07 | 1.847e-14 |
| (0, 1) | 9.538e-07 | 2.819e-14 |
| (5, 3) | 1.170e-06 | 3.458e-14 |
| (17, 40) | 1.379e-06 | 4.075e-14 |
| (100, 200) | 5.067e-06 | 1.498e-13 |

All errors are at float64 round-off level, confirming the theorem holds exactly (up to
numerical precision) for the custom DFT.

#### Empirical verification of the 2-D Circular Convolution Theorem

A 3x3 Laplacian kernel was circularly convolved with the test image two independent
ways: (i) directly in the spatial domain via explicit shifted sums (`np.roll`-based,
ground truth, no FFT involved), and (ii) by zero-padding the kernel to image size,
taking both DFTs, multiplying, and inverting.

| Metric | Value |
|---|---|
| max abs error (spatial vs. frequency-domain) | 1.1433e-10 |
| relative error | 2.696e-13 |

Confirms `f ⊛ h ↔ F · H` (circular convolution in space = multiplication in frequency)
to numerical precision.

#### (c) Sinusoid parameter recovery

A 512x512 image was synthesized as a sum of 3 real 2-D cosines plus a constant 128
background, with ground truth recorded **before** any DFT was taken
(`results/sinusoid_ground_truth.json`):

| u | v | A | φ (rad) |
|---|---|---|---|
| 4 | 6 | 60.0 | 0.300 |
| 10 | −3 | 40.0 | 1.200 |
| −7 | 15 | 25.0 | −0.800 |

Recovered from the DFT's spectral peaks (locating the 3 strongest magnitude peaks
outside a small DC neighbourhood, reading off `(u,v)` from bin index, amplitude as
`2|F(u,v)|/N^2`, and phase as `angle(F(u,v))`):

| u | v | A | φ (rad) |
|---|---|---|---|
| 4 | 6 | 60.000 | 0.300 |
| 10 | −3 | 40.000 | 1.200 |
| −7 | 15 | 25.000 | −0.800 |

Recovery is exact to float64 precision — expected, since the synthetic frequencies were
deliberately chosen as integers (so they land exactly on DFT bins with no spectral
leakage).

**Frequency resolution limit.** The DFT only samples frequency at integer multiples of
`1/N` cycles/pixel (bin spacing), so adjacent frequency bins are separated by

```
Δf = 1/N cycles/pixel  =  1 cycle/image   (N = 512)
```

This follows directly from the DFT basis functions `exp(-j2πkn/N)`, which are periodic
in `k` with period `N` and complete exactly `k` cycles over the `N`-sample window — the
grid simply cannot resolve two frequencies closer together than one full cycle
difference across the whole image; this is the discrete analogue of the
uncertainty-principle trade-off between time/space extent and frequency resolution.

Figure: `problems/problem1_dft/results/1_1c_sinusoid_and_spectrum.png` shows the
synthetic image and its log-magnitude spectrum with 3 sharp peak pairs (plus their
Hermitian-symmetric mirrors), exactly as predicted.

---

### 1.2 Phase vs. Magnitude Information Isolation

**Pre-implementation prediction** (dated 2026-10-18, written before running this
section): see
[`predictions/1.2_phase_magnitude_prediction.md`](../problems/problem1_dft/predictions/1.2_phase_magnitude_prediction.md).
Summary of the prediction: by the Convolution Theorem, reconstructing with the test
image's phase and the noise's magnitude is equivalent to a circular convolution of the
flat-magnitude phase-only image with the inverse transform of the noise-magnitude
spectrum; since a white-noise magnitude spectrum is flat only *on average* (not exactly,
for one realization), the prediction was that the result would look like the
flat-magnitude reconstruction (clear structure) with extra fine speckle on top.

#### (a) Phase of test image + flat magnitude, vs. noise magnitude

Figure: `problems/problem1_dft/results/1_2_phase_magnitude.png`.

- **Phase + flat (constant) magnitude**: reconstructs a sharp, recognizable,
  high-pass-like rendition of the original image — essentially an edge map, since a
  perfectly flat magnitude spectrum removes all information about which frequencies
  carry more energy, leaving only the *alignment*/structure encoded by phase.
- **Phase + noise magnitude**: visually extremely similar to the flat-magnitude case —
  same recognizable structure — but with a layer of fine-grained speckle superimposed,
  exactly as predicted. This **confirms the prediction**: the measured outputs match the
  derived convolution-theorem argument (structure from phase dominates; speckle comes
  from the noise magnitude spectrum's deviation from perfectly flat).

**Why flat magnitude isolates phase more cleanly than swapping magnitudes between two
natural images:** a *natural* image's magnitude spectrum is strongly non-flat (concentrated
at low frequencies, see §2.1b's power-retention analysis), so swapping it onto another
image's phase still imposes a highly structured frequency weighting — i.e. a *second*
image's energy distribution is superimposed on the first image's phase, producing a
hybrid that contains large-scale shape leakage from the magnitude-donor image. A
perfectly flat magnitude applies *equal* weight to every frequency, so none of the
magnitude-donor's structure survives — only the phase's structural information remains,
giving a cleaner isolation of "what phase alone encodes."

#### (b) Magnitude of test image + zero phase

Figure: rightmost panel of `1_2_phase_magnitude.png` (log-scaled for display since the
result is dominated by one huge DC-like peak).

Setting `∠F(u,v) = 0` means `F(u,v) = |F(u,v)|`, a real, non-negative, even function of
`(u,v)` (since `|F(u,v)| = |F(-u,-v)|` always holds for any image). A real, even
spectrum inverse-transforms to a **real, even (centro-symmetric) spatial function**:
`g(x,y) = g(-x,-y)` (indices modulo N). This is confirmed numerically — the output's
maximum value sits exactly at the origin `(0,0)` (`g_zero_phase_peak_at_origin: true`),
and the image appears as a symmetric, blob-like pattern radiating from the origin with
no resemblance to the original scene's spatial layout, because all positional/alignment
information (which lived entirely in the phase) has been discarded; only the
energy-per-frequency (magnitude) survives, and a magnitude-only, phase-zero spectrum is,
by construction, the spectrum of an even function.

---

### 1.3 Padding, Wraparound, and Computational Overhead

A 5x5 Laplacian-type kernel was convolved with the test image three ways: (i) direct
spatial zero-padded linear convolution (ground truth, no FFT), (ii) correctly
zero-padded linear convolution via the custom DFT, and (iii) unpadded circular
convolution via the custom DFT at the image's native 512x512 resolution.

| Comparison | max abs error |
|---|---|
| zero-padded DFT conv vs. direct spatial linear conv | 7.851e-10 |
| circular DFT conv vs. direct spatial linear conv — whole image | 1608.0 |
| circular DFT conv vs. direct spatial linear conv — interior only | 5.384e-10 |
| circular DFT conv vs. direct spatial linear conv — border only | 1608.0 |

The zero-padded DFT result matches the direct spatial computation to numerical
precision everywhere. The unpadded circular convolution matches it exactly in the
*interior* (more than `⌊kernel_size/2⌋ = 2` pixels from any edge) but is **badly wrong
right at the border** (error up to 1608 intensity levels on a kernel whose center weight
is 16 — i.e. the convolution there is reading in wrapped-around pixels from the opposite
edge of the image instead of zeros). This is the expected signature of circular
wraparound corruption.

Figures: `1_3_padding_wraparound.png` (full images — wraparound is visible as a faint
dark/bright seam along the left edge of the circular-convolution result) and
`1_3_border_zoom.png` (zoomed top rows — the direct/padded result is near-black at the
very top edge while the circular result shows visibly lighter corruption, since it is
mixing in content from the bottom of the image).

#### (a) Minimal zero-padding derivation

For an `N×N` image and an `M×M` kernel, linear convolution produces a result supported
on indices `0 ... N+M-2` in each dimension (`N` possible kernel positions over the
image, extended by `M-1` on each side by the kernel's own extent) — i.e. **`N+M-1`**
taps per axis. Circular convolution on a grid of size `P` is only equal to this linear
convolution if the grid is large enough that no wraparound aliasing occurs, which
requires

```
P ≥ N + M - 1
```

in both dimensions. Hence the **minimal** padded size is exactly

```
N_x,pad = N_y,pad = N + M - 1
```

(here `512 + 5 - 1 = 516`, matching `min_padded_size: [516, 516]` in
`results/1_3_results.json`, and used for the "zero-padded DFT conv" result above, which
matches the ground truth to float64 precision — confirming the derivation.)

#### (b) Computational trade-off

With `P = N+M-1 = 516` against `N = 512`:

| Metric | Value |
|---|---|
| memory allocation factor, `(P/N)^2` | 1.0157 |
| runtime ratio, `(P^2 log2 P) / (N^2 log2 N)` | 1.0170 |

For a small kernel (`M ≪ N`), the padding overhead is minor (~1.6-1.7%) — but in
general the padded array is `P^2` instead of `N^2` floats (extra memory proportional to
`(P/N)^2 − 1`), and FFT-based convolution costs `O(P^2 log P)` instead of `O(N^2 log N)`
(extra runtime proportional to that same ratio, which grows faster than the memory
ratio alone because of the `log P` factor). This is exactly the trade-off that
`scipy.signal.fftconvolve` manages internally: it does not pad only to the *minimal*
`N+M-1`, but further rounds the padded size up to the next size with small prime
factors ("5-smooth"/fast-FFT-length), since an FFT of an awkward size (e.g. a large
prime) can be far slower than the small amount of *extra* padding needed to reach a fast
length — i.e. it deliberately trades a little more memory/padding for a potentially
large runtime win, the same memory-vs-runtime trade-off quantified above, just tuned
with an additional "FFT-friendly length" criterion on top of the `N+M-1` correctness
floor derived here.

---

### Deliverables checklist (Problem 1)

- [x] Quantitative error table vs. `numpy.fft.fft2` — §1.1.
- [x] Shift-theorem and convolution-theorem verification tables — §1.1(b).
- [x] Dated written prediction + reconstruction plots + reconciliation — §1.2.
- [x] Recovered sinusoid parameters + Δf derivation — §1.1(c).
- [x] Wraparound-corruption images + full padding-size derivation — §1.3.

---

## Problems 2–7

Scaffolded only (see `problems/problem{2..7}_*/README.md` for each problem's checklist).
Problem 7 (Lens Distortion) additionally needs instructor confirmation — see
`problems/problem7_lens_distortion/README.md`.
