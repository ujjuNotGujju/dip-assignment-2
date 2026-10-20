# Problem 3: Homomorphic filtering: illumination and reflectance (10)

Status: **done**. See [REPORT.pdf](REPORT.pdf) for the full write-up.

Data: `data/provided/problem3/tunnel_image.jpg` (unevenly lit tunnel photo, bright
light source + deep shadow in one frame).

## 3.1 The multiplicative model (4)
- [x] State the model I(x, y) = L(x, y)R(x, y); derive why the log converts this into
      an additive problem a linear frequency-domain filter can act on.
- [x] Explain precisely why the log must be taken before the Fourier transform, not
      after, with a concrete numerical demonstration of what goes wrong otherwise.

## 3.2 Implement and tune (6)
- [x] Implement the homomorphic high-frequency-emphasis filter and apply to log I,
      then exponentiate back.
- [x] Sweep γL, γH, D0 as an image grid; identify flattening vs. halo settings.
- [x] Compare against plain global histogram equalisation (implemented from scratch,
      no Assignment 1 code in this repo to reuse); local-contrast comparison table.

## Deliverables
Derivation of the log-domain argument; parameter sweep grid with halo cases
identified; local-contrast comparison table against histogram equalisation.
