"""
Problem 2 -- Frequency-Domain Filtering: Ideal, Butterworth, Gaussian (15 pts)

Runs 2.1, 2.2, 2.3 end to end and writes figures/tables into results/.
Transfer functions (make_filter) are built entirely from their formulas with NumPy.
numpy.fft is used for the forward/inverse transforms (explicitly allowed for this
problem). scipy.ndimage.gaussian_filter is used ONLY for the one PDF-sanctioned
comparison in 2.2d.
"""
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter  # sanctioned comparison, 2.2d only

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dip_a2.filters import make_filter, filter_pipeline, radial_distance  # noqa: E402
from dip_a2.metrics import psnr  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

rng = np.random.default_rng(seed=7)


def log(msg):
    print(f"[problem2] {msg}")


def record_machine_info():
    info = {"numpy_version": np.__version__}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        info["logical_cpus"] = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["ram_gb"] = round(int(line.split()[1]) / 1024 / 1024, 2)
                    break
    except Exception:
        pass
    info["platform"] = platform.platform()
    (RESULTS / "machine_info.json").write_text(json.dumps(info, indent=2))
    log(f"machine info: {info}")
    return info


def load(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


IMAGES = {
    "characters": load(DATA / "provided" / "problem2" / "Fig0441(a)(characters_test_pattern).tif"),
    "text_gaps": load(DATA / "provided" / "problem2" / "Fig0419(a)(text_gaps_of_1_and_2_pixels).tif"),
    "thumb_print": load(DATA / "provided" / "problem2" / "Fig0457(a)(thumb_print).tif"),
    "cameraman": load(DATA / "provided" / "problem2" / "cameraman.tif"),
}


# ---------------------------------------------------------------------------
# 2.1 Filter construction and transfer-function analysis
# ---------------------------------------------------------------------------
def section_2_1a():
    D0 = 40.0
    shape = (256, 256)
    H_ideal = make_filter("ideal", shape, D0)
    H_butter = make_filter("butterworth", shape, D0, n=2)
    H_gauss = make_filter("gaussian", shape, D0)

    D = radial_distance(shape)
    near = np.argmin(np.abs(D.ravel() - D0))
    val_ideal = H_ideal.ravel()[near]
    val_butter = H_butter.ravel()[near]
    val_gauss = H_gauss.ravel()[near]

    analytic = {
        "ideal_at_D0": 1.0,  # H=1 for D<=D0 (boundary included by our <=)
        "butterworth_at_D0": 0.5,  # 1/(1+1)
        "gaussian_at_D0": float(np.exp(-0.5)),  # e^-0.5
    }
    numeric = {
        "ideal_at_D0_numeric": float(val_ideal),
        "butterworth_at_D0_numeric": float(val_butter),
        "gaussian_at_D0_numeric": float(val_gauss),
    }
    log(f"2.1a analytic: {analytic}")
    log(f"2.1a numeric (nearest sample to D0={D0}): {numeric}")

    # Gaussian half-amplitude-matched D0'
    D0_prime = D0 / np.sqrt(2 * np.log(2))
    H_gauss_matched = make_filter("gaussian", shape, D0_prime)
    val_gauss_matched = H_gauss_matched.ravel()[near]
    log(f"2.1a matched D0' = {D0_prime:.4f}, H_gauss(D0) with matched D0' = {val_gauss_matched:.4f} (should be ~0.5)")

    # radial profile plot of all 6 transfer functions along one axis through the centre
    P, Q = shape
    row = P // 2
    d_axis = np.arange(Q) - Q // 2
    order = np.argsort(d_axis)
    d_axis = d_axis[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, H in [
        ("ILPF", make_filter("ideal", shape, D0)),
        ("BLPF n=2", make_filter("butterworth", shape, D0, n=2)),
        ("GLPF", make_filter("gaussian", shape, D0)),
        ("IHPF", make_filter("ideal", shape, D0, highpass=True)),
        ("BHPF n=2", make_filter("butterworth", shape, D0, n=2, highpass=True)),
        ("GHPF", make_filter("gaussian", shape, D0, highpass=True)),
    ]:
        ax.plot(d_axis, H[row, :][order], label=name)
    ax.axvline(D0, color="k", linestyle=":", linewidth=0.8)
    ax.axvline(-D0, color="k", linestyle=":", linewidth=0.8)
    ax.set_xlabel("D (signed radial index through centre row)")
    ax.set_ylabel("H")
    ax.set_title(f"Radial profiles of all 6 transfer functions (D0={D0})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "2_1_transfer_functions.png", dpi=150)
    plt.close(fig)

    result = {"D0": D0, "analytic": analytic, "numeric": numeric, "D0_prime_matched": float(D0_prime),
              "H_gauss_matched_at_D0": float(val_gauss_matched)}
    (RESULTS / "2_1a_result.json").write_text(json.dumps(result, indent=2))
    return result


def section_2_1b():
    img = IMAGES["characters"]
    M, N = img.shape
    P, Q = 2 * M, 2 * N
    padded = np.zeros((P, Q))
    padded[:M, :N] = img
    F = np.fft.fftshift(np.fft.fft2(padded))
    power = np.abs(F) ** 2
    total_power = power.sum()

    D = radial_distance((P, Q))
    max_r = int(np.ceil(D.max())) + 1
    radii = np.arange(0, max_r)
    # cumulative power within each integer radius, via sorting (efficient enough here)
    flat_D = D.ravel()
    flat_power = power.ravel()
    order = np.argsort(flat_D)
    sorted_D = flat_D[order]
    sorted_power = flat_power[order]
    cum_power = np.cumsum(sorted_power)
    cum_frac = 100.0 * cum_power / total_power

    targets = [90.0, 95.0, 99.0, 99.5]
    radii_for_targets = []
    for t in targets:
        idx = np.searchsorted(cum_frac, t)
        idx = min(idx, len(sorted_D) - 1)
        radii_for_targets.append(float(sorted_D[idx]))

    table = dict(zip([f"alpha_{t}" for t in targets], radii_for_targets))
    log(f"2.1b energy-retention radii: {table}")
    (RESULTS / "2_1b_result.json").write_text(json.dumps(table, indent=2))
    return table


def section_2_1c():
    # H(10) = 16/17, H(40) = 1/17 for Butterworth -> recover n, D0 analytically
    # (40/10)^(2n) = (1/17 ... )/(...) -> derivation done in report; result: n=2, D0=20
    n_recovered = 2
    D0_recovered = 20.0

    shape = (256, 256)
    H = make_filter("butterworth", shape, D0_recovered, n=n_recovered)
    D = radial_distance(shape)
    idx10 = np.unravel_index(np.argmin(np.abs(D - 10)), D.shape)
    idx40 = np.unravel_index(np.argmin(np.abs(D - 40)), D.shape)
    H_at_10 = H[idx10]
    H_at_40 = H[idx40]
    log(f"2.1c verify: H(D~10)={H_at_10:.6f} (target {16/17:.6f}), H(D~40)={H_at_40:.6f} (target {1/17:.6f})")

    # smallest integer n with H(2*D0) <= 0.01
    n_min = None
    for n_try in range(1, 50):
        H_try = 1.0 / (1.0 + 2.0 ** (2 * n_try))
        if H_try <= 0.01:
            n_min = n_try
            break
    log(f"2.1c smallest n with H(2D0)<=1%: n={n_min}")

    result = {
        "recovered_n": n_recovered,
        "recovered_D0": D0_recovered,
        "H_at_D10_verified": float(H_at_10),
        "H_at_D40_verified": float(H_at_40),
        "smallest_n_for_1pct_at_2D0": n_min,
    }
    (RESULTS / "2_1c_result.json").write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# 2.2 Ringing
# ---------------------------------------------------------------------------
def make_step_edge(N=512, B=200.0, A=50.0, side=256):
    img = np.full((N, N), A, dtype=np.float64)
    start = (N - side) // 2
    end = start + side
    img[start:end, start:end] = B
    return img


def sine_integral(z, n_steps=20000):
    """Si(z) = int_0^z sin(t)/t dt, via trapezoidal quadrature (from scratch, no scipy)."""
    if z == 0:
        return 0.0
    t = np.linspace(1e-12, z, n_steps)
    y = np.sin(t) / t
    y[0] = 1.0  # limit of sin(t)/t as t->0
    return float(np.trapezoid(y, t))


def measure_overshoot(profile, B, A, edge_pos):
    """Overshoot (%) and first-peak distance (px) to the right of a rising edge."""
    after = profile[edge_pos:edge_pos + 100]
    max_val = after.max()
    overshoot = 100.0 * (max_val - B) / (B - A)
    peak_idx = int(np.argmax(after))
    return overshoot, peak_idx


def section_2_2a(step_img):
    N = step_img.shape[0]
    P = 2 * N
    B, A = 200.0, 50.0
    side = 256
    edge_col = (N - side) // 2  # left edge of the square along the mid-row

    mid_row = N // 2
    corner_row = (N - side) // 2 + 2  # close to the top edge of the square

    rows = []
    for D0 in [10, 20, 40, 80]:
        H = make_filter("ideal", (P, P), D0)
        g = filter_pipeline(step_img, H)
        profile = g[mid_row, :]
        overshoot, peak_dist = measure_overshoot(profile, B, A, edge_col)
        Si_pi = sine_integral(np.pi)
        predicted_overshoot_pct = 100.0 * (Si_pi - np.pi / 2) / np.pi
        predicted_peak_dist = P / (2.0 * D0)
        rows.append({
            "D0": D0, "measured_overshoot_pct": float(overshoot),
            "predicted_overshoot_pct": float(predicted_overshoot_pct),
            "measured_peak_dist_px": peak_dist, "predicted_peak_dist_px": float(predicted_peak_dist),
        })
        log(f"2.2a D0={D0}: measured OS={overshoot:.2f}% (pred {predicted_overshoot_pct:.2f}%), "
            f"peak_dist={peak_dist}px (pred {predicted_peak_dist:.1f}px)")

        if D0 == 40:
            corner_profile = g[corner_row, :]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(profile)
            axes[0].set_title(f"Mid-row profile, D0={D0}")
            axes[0].axhline(B, color="gray", linestyle=":")
            axes[1].plot(corner_profile)
            axes[1].set_title(f"Near-corner row profile, D0={D0}")
            axes[1].axhline(B, color="gray", linestyle=":")
            fig.tight_layout()
            fig.savefig(RESULTS / "2_2a_profiles.png", dpi=150)
            plt.close(fig)

    (RESULTS / "2_2a_result.json").write_text(json.dumps(rows, indent=2))
    return rows


def section_2_2a_gaussian_check(step_img):
    """Reconcile the prediction's other explicit claim: the Gaussian LPF should show
    ~0% overshoot (non-negative, sum-to-one spatial kernel) at every D0, unlike the
    ideal filter's ~8.95% Gibbs overshoot measured in section_2_2a above."""
    N = step_img.shape[0]
    P = 2 * N
    B, A = 200.0, 50.0
    side = 256
    edge_col = (N - side) // 2
    mid_row = N // 2

    rows = []
    for D0 in [10, 20, 40, 80]:
        H = make_filter("gaussian", (P, P), float(D0))
        g = filter_pipeline(step_img, H)
        profile = g[mid_row, :]
        overshoot, peak_dist = measure_overshoot(profile, B, A, edge_col)
        rows.append({"D0": D0, "measured_overshoot_pct": float(overshoot)})
        log(f"2.2a Gaussian check D0={D0}: measured OS={overshoot:.4f}% (predicted ~0%)")

    (RESULTS / "2_2a_gaussian_result.json").write_text(json.dumps(rows, indent=2))
    return rows


def section_2_2b(step_img):
    N = step_img.shape[0]
    P = 2 * N
    B, A = 200.0, 50.0
    side = 256
    edge_col = (N - side) // 2
    mid_row = N // 2

    rows = []
    for n in [1, 2, 3, 5, 10, 20]:
        H = make_filter("butterworth", (P, P), 40.0, n=n)
        g = filter_pipeline(step_img, H)
        profile = g[mid_row, :]
        overshoot, peak_dist = measure_overshoot(profile, B, A, edge_col)

        # 1-D kernel h(x) = IDFT{H(u,0)} using the centred row through H (u axis at v=0)
        H_row = H[:, P // 2]  # centred column corresponds to v=0 after fftshift convention
        H_row_uncentered = np.fft.ifftshift(H_row)
        h = np.real(np.fft.ifft(H_row_uncentered))
        h = np.fft.fftshift(h)
        ratio = float(np.abs(h.min()) / h.max())

        rows.append({"n": n, "measured_overshoot_pct": float(overshoot), "kernel_ratio_minabs_over_max": ratio})
        log(f"2.2b n={n}: overshoot={overshoot:.3f}%, |min h|/max h={ratio:.4f}")

    (RESULTS / "2_2b_result.json").write_text(json.dumps(rows, indent=2))
    return rows


def section_2_2c(radii_2_1b):
    img = IMAGES["characters"]
    M, N = img.shape
    P, Q = 2 * M, 2 * N
    radii = list(radii_2_1b.values())

    fig, axes = plt.subplots(3, len(radii), figsize=(4 * len(radii), 12))
    for j, D0 in enumerate(radii):
        D0_prime = D0 / np.sqrt(2 * np.log(2))
        H_i = make_filter("ideal", (P, Q), D0)
        H_b = make_filter("butterworth", (P, Q), D0, n=2)
        H_g = make_filter("gaussian", (P, Q), D0_prime)
        for i, (name, H) in enumerate([("ILPF", H_i), ("BLPF n=2", H_b), ("GLPF (matched)", H_g)]):
            g = filter_pipeline(img, H)
            axes[i, j].imshow(g, cmap="gray")
            axes[i, j].set_title(f"{name}, D0={D0:.1f}")
            axes[i, j].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "2_2c_visual_grid.png", dpi=150)
    plt.close(fig)
    log("2.2c visual grid saved")


def section_2_2d():
    img = IMAGES["characters"]
    M, N = img.shape
    P, Q = 2 * M, 2 * N

    rows = []
    for D0 in [10, 40, 160]:
        H = make_filter("gaussian", (P, Q), D0)
        g_freq = filter_pipeline(img, H)
        sigma_s = P / (2 * np.pi * D0)
        g_spatial = gaussian_filter(img, sigma=sigma_s, mode="constant")
        max_err = float(np.max(np.abs(g_freq - g_spatial)))
        rows.append({"D0": D0, "sigma_s": float(sigma_s), "max_abs_err": max_err})
        log(f"2.2d D0={D0}: sigma_s={sigma_s:.4f}, max|g_freq-g_spatial|={max_err:.4f}")

    (RESULTS / "2_2d_result.json").write_text(json.dumps(rows, indent=2))
    return rows


# ---------------------------------------------------------------------------
# 2.3 Applications
# ---------------------------------------------------------------------------
def section_2_3a():
    img = IMAGES["cameraman"]
    M, N = img.shape
    P, Q = 2 * M, 2 * N
    noise = rng.normal(0, 20, size=img.shape)
    noisy = img + noise  # not clipped before filtering

    D0_values = np.arange(5, 151, 5)
    families = {"ideal": {}, "butterworth": {}, "gaussian": {}}
    curves = {k: [] for k in families}
    for D0 in D0_values:
        for kind, label in [("ideal", "ideal"), ("butterworth", "butterworth"), ("gaussian", "gaussian")]:
            n = 2 if kind == "butterworth" else None
            H = make_filter(kind, (P, Q), float(D0), n=n)
            g = filter_pipeline(noisy, H)
            g_clipped = np.clip(g, 0, 255)
            val = psnr(g_clipped, img, data_range=255.0)
            curves[kind].append(val)

    fig, ax = plt.subplots(figsize=(8, 5))
    best = {}
    for kind in families:
        vals = np.array(curves[kind])
        ax.plot(D0_values, vals, label=kind)
        best_idx = int(np.argmax(vals))
        best[kind] = {"D0": int(D0_values[best_idx]), "psnr": float(vals[best_idx])}
    ax.set_xlabel("D0")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs D0: noise-blur trade-off (cameraman, sigma=20 noise)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "2_3a_psnr_sweep.png", dpi=150)
    plt.close(fig)

    log(f"2.3a best per family: {best}")
    (RESULTS / "2_3a_result.json").write_text(json.dumps(best, indent=2))
    return best


def section_2_3b():
    img = IMAGES["text_gaps"]
    M, N = img.shape
    P, Q = 2 * M, 2 * N
    D0 = 60.0
    H = make_filter("gaussian", (P, Q), D0)
    g = filter_pipeline(img, H)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original (broken strokes)"); axes[0].axis("off")
    axes[1].imshow(g, cmap="gray"); axes[1].set_title(f"GLPF D0={D0}"); axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "2_3b_character_repair.png", dpi=150)
    plt.close(fig)
    log(f"2.3b character repair done with D0={D0}")
    (RESULTS / "2_3b_result.json").write_text(json.dumps({"D0": D0}, indent=2))


def section_2_3c():
    img = IMAGES["thumb_print"]
    M, N = img.shape
    P, Q = 2 * M, 2 * N
    D0 = 30.0  # common half-amplitude radius (ideal/ butterworth treat D0 as half-amp directly for HP)
    D0_gauss = D0 / np.sqrt(2 * np.log(2))

    H_i = make_filter("ideal", (P, Q), D0, highpass=True)
    H_b = make_filter("butterworth", (P, Q), D0, n=4, highpass=True)
    H_g = make_filter("gaussian", (P, Q), D0_gauss, highpass=True)

    g_i = filter_pipeline(img, H_i)
    g_b = filter_pipeline(img, H_b)
    g_g = filter_pipeline(img, H_g)

    means = {"IHPF": float(g_i.mean()), "BHPF": float(g_b.mean()), "GHPF": float(g_g.mean())}
    log(f"2.3c output means: {means}")

    # high-frequency emphasis using BHPF
    k1, k2 = 0.5, 1.5
    H_hfe = k1 + k2 * H_b
    g_hfe = filter_pipeline(img, H_hfe)

    # binary ridge map from BHPF output
    thresh = np.percentile(g_b, 90)
    ridge_map = (g_b > thresh).astype(np.float64)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(g_i, cmap="gray"); axes[0, 0].set_title("IHPF"); axes[0, 0].axis("off")
    axes[0, 1].imshow(g_b, cmap="gray"); axes[0, 1].set_title("BHPF n=4"); axes[0, 1].axis("off")
    axes[0, 2].imshow(g_g, cmap="gray"); axes[0, 2].set_title("GHPF"); axes[0, 2].axis("off")
    axes[1, 0].imshow(img, cmap="gray"); axes[1, 0].set_title("Original"); axes[1, 0].axis("off")
    axes[1, 1].imshow(g_hfe, cmap="gray"); axes[1, 1].set_title(f"HFE k1={k1},k2={k2}"); axes[1, 1].axis("off")
    axes[1, 2].imshow(ridge_map, cmap="gray"); axes[1, 2].set_title("Binary ridge map (BHPF)"); axes[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "2_3c_highpass.png", dpi=150)
    plt.close(fig)

    (RESULTS / "2_3c_result.json").write_text(json.dumps({"D0": D0, "means": means, "k1": k1, "k2": k2}, indent=2))
    return means


def main():
    record_machine_info()

    log("=== 2.1 ===")
    section_2_1a()
    radii = section_2_1b()
    section_2_1c()

    log("=== 2.2 ===")
    step_img = make_step_edge()
    np.save(DATA / "synthetic" / "step_edge.npy", step_img)
    section_2_2a(step_img)
    section_2_2a_gaussian_check(step_img)
    section_2_2b(step_img)
    section_2_2c(radii)
    section_2_2d()

    log("=== 2.3 ===")
    section_2_3a()
    section_2_3b()
    section_2_3c()

    log(f"All results written to {RESULTS}")


if __name__ == "__main__":
    main()
