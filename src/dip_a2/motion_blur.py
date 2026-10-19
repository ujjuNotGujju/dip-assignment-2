"""Linear motion-blur PSF/OTF construction and deconvolution helpers, from scratch
(NumPy array arithmetic + numpy.fft).
"""
import numpy as np


def motion_psf_otf(shape, L: float, theta_deg: float, oversample: int = 20):
    """Rasterize the continuous line PSF h(x,y) = (1/L) int_{-L/2}^{L/2} delta(x-t
    cos(theta), y-t sin(theta)) dt onto a `shape`-sized array, placed with its centre
    at the origin corner (index (0,0), wrapping) -- i.e. ready for CIRCULAR
    convolution via direct multiplication with an image's own DFT, the PDF's
    recommended boundary convention. Returns (psf, H) where H = numpy.fft.fft2(psf).
    """
    M, N = shape
    theta = np.deg2rad(theta_deg)
    n_samples = max(2, int(round(L * oversample)))
    t = np.linspace(-L / 2, L / 2, n_samples)
    dx = t * np.cos(theta)
    dy = t * np.sin(theta)
    # nearest-pixel accumulation, wrapped circularly (motion kernel centred at origin)
    ix = np.round(dx).astype(np.int64) % M
    iy = np.round(dy).astype(np.int64) % N
    psf = np.zeros((M, N), dtype=np.float64)
    np.add.at(psf, (ix, iy), 1.0)
    psf /= psf.sum()
    H = np.fft.fft2(psf)
    return psf, H


def motion_otf_analytic(shape, L: float, theta_deg: float):
    """Closed-form motion OTF H(fx,fy) = sinc(L*(fx*cos(theta) + fy*sin(theta))),
    sampled directly on the DFT frequency grid (fx=u/M, fy=v/N cycles/pixel, standard
    numpy.fft.fft2 bin order, not centred)."""
    M, N = shape
    theta = np.deg2rad(theta_deg)
    u = np.fft.fftfreq(M)  # cycles/pixel, bin order matching np.fft.fft2
    v = np.fft.fftfreq(N)
    fx = u.reshape(-1, 1)
    fy = v.reshape(1, -1)
    arg = L * (fx * np.cos(theta) + fy * np.sin(theta))
    return np.sinc(arg)  # numpy's sinc already includes the pi factor: sin(pi x)/(pi x)


def add_noise_at_snr(signal: np.ndarray, snr_db: float, rng: np.random.Generator):
    """Add zero-mean Gaussian noise achieving SNRdB = 10*log10(sum(f^2)/sum(n^2))."""
    signal_energy = np.sum(signal.astype(np.float64) ** 2)
    noise_energy_target = signal_energy / (10.0 ** (snr_db / 10.0))
    sigma = np.sqrt(noise_energy_target / signal.size)
    noise = rng.normal(0, sigma, size=signal.shape)
    return signal + noise, sigma


def naive_inverse_filter(G: np.ndarray, H: np.ndarray, floor: float = 1e-3):
    """F_hat = G / H, with a magnitude floor on H to avoid division blow-up beyond
    the stated numerical limit (phase of H is preserved; only |H| is floored)."""
    H_mag = np.abs(H)
    H_floored = np.where(H_mag < floor, floor * np.exp(1j * np.angle(H)), H)
    return G / H_floored


def wiener_deconvolve(G: np.ndarray, H: np.ndarray, K: float):
    """Scalar-K regularized Wiener deconvolution: F_hat = H* G / (|H|^2 + K)."""
    return np.conj(H) * G / (np.abs(H) ** 2 + K)


def estimate_motion_blur(image: np.ndarray, n_angles: int = 180):
    """Best-effort estimate of (theta_deg, L) from a motion-blurred image's own
    log-magnitude spectrum: dark stripes run perpendicular to the motion direction,
    spaced 1/L apart. For each candidate angle, projects the (centred) log-magnitude
    spectrum along that direction through the origin and scores how periodically it
    dips; the angle with the strongest periodic dip pattern is taken as the motion
    direction, and L is recovered from the dominant dip spacing via FFT of that 1-D
    profile. Not validated against real data (no captured photo available yet) --
    documented as best-effort in the report.
    """
    M, N = image.shape
    F = np.fft.fftshift(np.fft.fft2(image))
    mag = np.log1p(np.abs(F))
    cy, cx = M // 2, N // 2
    max_r = min(cy, cx) - 2
    r = np.arange(1, max_r)

    best_score, best_theta, best_profile = -np.inf, 0.0, None
    for theta_deg in np.linspace(0, 180, n_angles, endpoint=False):
        theta = np.deg2rad(theta_deg)
        xs = (cx + r * np.cos(theta)).astype(np.int64)
        ys = (cy + r * np.sin(theta)).astype(np.int64)
        valid = (xs >= 0) & (xs < N) & (ys >= 0) & (ys < M)
        profile = mag[ys[valid], xs[valid]]
        if len(profile) < 8:
            continue
        # periodicity score: power spectrum of the 1-D profile, excluding DC
        p = np.abs(np.fft.rfft(profile - profile.mean())) ** 2
        score = p[1:].max() if len(p) > 1 else 0.0
        if score > best_score:
            best_score, best_theta, best_profile = score, theta_deg, profile

    # recover dominant spacing (in pixels along the radial profile) from its FFT peak
    p = np.abs(np.fft.rfft(best_profile - best_profile.mean())) ** 2
    peak_k = int(np.argmax(p[1:])) + 1
    period_px = len(best_profile) / peak_k if peak_k > 0 else len(best_profile)
    L_est = max(M, N) / period_px if period_px > 0 else float("nan")
    # motion direction is perpendicular to the dark-stripe orientation measured above
    motion_theta_deg = (best_theta + 90.0) % 180.0
    diagnostics = {"stripe_angle_deg": float(best_theta), "stripe_spacing_px": float(period_px)}
    return motion_theta_deg, float(L_est), diagnostics
