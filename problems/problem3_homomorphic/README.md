# Problem 3 — Homomorphic Filtering: Illumination and Reflectance (10 pts)

Status: **not started**.

Data: `data/provided/problem3/tunnel_image.jpg` (unevenly lit photo).

## 3.1 The Multiplicative Model (4 pts)
- [ ] State `I = L*R`; derive why `log I = log L + log R` converts this to an additive
      problem a linear frequency-domain filter can act on.
- [ ] Explain why the log must precede the Fourier transform (not filtering in the
      log-magnitude spectrum after transforming `I` directly).

## 3.2 Implement and Tune (6 pts)
- [ ] Implement `H(u,v) = (γH-γL)[1 - exp(-cD²/D0²)] + γL` applied to `log I`, then
      exponentiate back.
- [ ] Sweep γL, γH, D0 as an image grid; identify flattening vs. halo settings.
- [ ] Compare vs. plain global histogram equalisation (reuse Assignment 1 code);
      local-contrast (local std) comparison table in a shadow region.

## Deliverables
Log-domain derivation; parameter sweep grid with halo cases identified; local-contrast
comparison table vs. histogram equalisation.
