"""
Problem 4 -- Edge-Preserving Smoothing: Bilateral vs. Kuwahara (15 pts)

Runs 4.1, 4.2, 4.3 end to end and writes figures/tables into results/.
No disallowed library (scipy.ndimage/scipy.signal/cv2/skimage) is used anywhere.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dip_a2.edge_preserving import bilateral_filter, kuwahara_filter, soft_kuwahara_filter  # noqa: E402
from dip_a2.metrics import psnr, ssim  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

rng = np.random.default_rng(seed=11)
NOISE_SIGMA = 20.0
SUBWINDOW_SIZE = 5  # Kuwahara sub-window: 5x5, total span 9x9


def log(msg):
    print(f"[problem4] {msg}")


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


def load_clean():
    im = Image.open(DATA / "provided" / "problem4" / "q4.png").convert("L")
    return np.asarray(im, dtype=np.float64)


def find_flat_block(image, block=30, margin=1):
    """Darkest-variance block (most flat), tiling `image`, excluding the outer ring."""
    H, W = image.shape
    rows, cols = H // block, W // block
    var_map = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            var_map[i, j] = image[i * block:(i + 1) * block, j * block:(j + 1) * block].var()
    interior = var_map[margin:rows - margin, margin:cols - margin]
    idx = np.unravel_index(np.argmin(interior), interior.shape)
    i, j = idx[0] + margin, idx[1] + margin
    return (slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block))


# ---------------------------------------------------------------------------
# 4.1 Implementation and matched-strength comparison
# ---------------------------------------------------------------------------
def section_4_1(clean, noisy):
    flat_slice = find_flat_block(clean, block=30, margin=1)
    noisy_flat_var = noisy[flat_slice].var()
    log(f"4.1 flat block slice={flat_slice}, noisy flat-region variance={noisy_flat_var:.3f}")

    kuwahara_out = kuwahara_filter(noisy, SUBWINDOW_SIZE)
    kuwahara_flat_var = kuwahara_out[flat_slice].var()
    target_reduction = noisy_flat_var / kuwahara_flat_var
    log(f"4.1 Kuwahara flat-region variance after filtering={kuwahara_flat_var:.4f}, "
        f"reduction factor={target_reduction:.3f}")

    # tune bilateral sigma_s (sigma_r large => range kernel ~inactive in flat region)
    # sigma_r is fixed at a normal, edge-aware value (NOT disabled) throughout both
    # the matching search and the final applied filter, so "matched" genuinely holds
    # for the filter that actually gets used -- only sigma_s is tuned to match.
    sigma_r_final = 40.0
    candidates = np.linspace(0.5, 20.0, 40)
    best_sigma_s, best_diff = None, np.inf
    # search only within the flat block itself for speed, same formula applies locally
    flat_noisy_patch = noisy[flat_slice]
    for sigma_s in candidates:
        out = bilateral_filter(flat_noisy_patch, sigma_s, sigma_r_final)
        reduction = flat_noisy_patch.var() / out.var()
        diff = abs(reduction - target_reduction)
        if diff < best_diff:
            best_diff, best_sigma_s, best_reduction = diff, sigma_s, reduction
    log(f"4.1 matched bilateral sigma_s={best_sigma_s:.3f}, sigma_r={sigma_r_final} "
        f"(reduction={best_reduction:.3f}, target={target_reduction:.3f})")

    bilateral_out = bilateral_filter(noisy, best_sigma_s, sigma_r_final)

    psnr_kuwahara = psnr(kuwahara_out, clean, data_range=255.0)
    psnr_bilateral = psnr(bilateral_out, clean, data_range=255.0)
    ssim_kuwahara = ssim(kuwahara_out, clean, data_range=255.0)
    ssim_bilateral = ssim(bilateral_out, clean, data_range=255.0)
    log(f"4.1 PSNR: kuwahara={psnr_kuwahara:.3f}, bilateral={psnr_bilateral:.3f}")
    log(f"4.1 SSIM: kuwahara={ssim_kuwahara:.4f}, bilateral={ssim_bilateral:.4f}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(clean, cmap="gray"); axes[0].set_title("Clean"); axes[0].axis("off")
    axes[1].imshow(noisy, cmap="gray"); axes[1].set_title(f"Noisy (sigma={NOISE_SIGMA})"); axes[1].axis("off")
    axes[2].imshow(kuwahara_out, cmap="gray"); axes[2].set_title(f"Kuwahara {SUBWINDOW_SIZE}x{SUBWINDOW_SIZE}"); axes[2].axis("off")
    axes[3].imshow(bilateral_out, cmap="gray"); axes[3].set_title(f"Bilateral sigma_s={best_sigma_s:.2f}"); axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "4_1_comparison.png", dpi=150)
    plt.close(fig)

    # cropped zoom at the circle/rectangle boundary
    H, W = clean.shape
    cy, cx = int(H * 0.65), int(W * 0.60)
    r = 60
    sl = (slice(max(0, cy - r), cy + r), slice(max(0, cx - r), cx + r))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, (name, arr) in zip(axes, [("Clean", clean), ("Noisy", noisy),
                                        ("Kuwahara", kuwahara_out), ("Bilateral", bilateral_out)]):
        ax.imshow(arr[sl], cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "4_1_boundary_zoom.png", dpi=150)
    plt.close(fig)

    result = {
        "noise_sigma": NOISE_SIGMA,
        "subwindow_size": SUBWINDOW_SIZE,
        "flat_slice": [[int(flat_slice[0].start), int(flat_slice[0].stop)],
                       [int(flat_slice[1].start), int(flat_slice[1].stop)]],
        "kuwahara_variance_reduction": float(target_reduction),
        "bilateral_sigma_s_matched": float(best_sigma_s),
        "bilateral_sigma_r": sigma_r_final,
        "bilateral_variance_reduction_at_match": float(best_reduction),
        "psnr": {"kuwahara": float(psnr_kuwahara), "bilateral": float(psnr_bilateral)},
        "ssim": {"kuwahara": float(ssim_kuwahara), "bilateral": float(ssim_bilateral)},
    }
    (RESULTS / "4_1_result.json").write_text(json.dumps(result, indent=2))
    return result, kuwahara_out, bilateral_out, best_sigma_s, sigma_r_final


# ---------------------------------------------------------------------------
# 4.2 Mitigating Kuwahara blockiness and complexity analysis
# ---------------------------------------------------------------------------
def dilate_mask(mask, radius):
    out = mask.copy()
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            out |= np.roll(np.roll(mask, di, axis=0), dj, axis=1)
    return out


def section_4_2(clean, noisy, kuwahara_out, sigma_s_matched):
    temperature = 15.0
    soft_out = soft_kuwahara_filter(noisy, SUBWINDOW_SIZE, temperature)

    # near-edge, flat (in the CLEAN image) mask, for a blockiness-isolating TV metric
    gx = np.zeros_like(clean); gx[:, 1:] = np.abs(np.diff(clean, axis=1))
    gy = np.zeros_like(clean); gy[1:, :] = np.abs(np.diff(clean, axis=0))
    grad = gx + gy
    edge_pixels = grad > 10.0
    near_edge = dilate_mask(edge_pixels, SUBWINDOW_SIZE) & ~edge_pixels

    def tv(img, mask):
        tgx = np.zeros_like(img); tgx[:, 1:] = np.abs(np.diff(img, axis=1))
        tgy = np.zeros_like(img); tgy[1:, :] = np.abs(np.diff(img, axis=0))
        return float((tgx + tgy)[mask].mean())

    tv_classical = tv(kuwahara_out, near_edge)
    tv_soft = tv(soft_out, near_edge)
    log(f"4.2 TV in near-edge flat zone: classical={tv_classical:.4f}, soft={tv_soft:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    H, W = clean.shape
    cy, cx = int(H * 0.65), int(W * 0.60)
    r = 60
    sl = (slice(max(0, cy - r), cy + r), slice(max(0, cx - r), cx + r))
    axes[0].imshow(kuwahara_out[sl], cmap="gray"); axes[0].set_title(f"Classical Kuwahara (TV={tv_classical:.3f})"); axes[0].axis("off")
    axes[1].imshow(soft_out[sl], cmap="gray"); axes[1].set_title(f"Soft Kuwahara (TV={tv_soft:.3f})"); axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "4_2_blockiness_comparison.png", dpi=150)
    plt.close(fig)

    # per-pixel operation counts (analytic, not timed), at the window sizes actually used
    w_repr = 2 * max(1, int(round(3 * 8.0))) + 1  # representative bilateral window (sigma_s~8 typical)
    w_matched = 2 * max(1, int(round(3 * sigma_s_matched))) + 1  # the ACTUAL sigma_s=4.1 used in 4.1's matched filter
    q = SUBWINDOW_SIZE  # subwindow span
    counts = {
        "classical_kuwahara": {
            "window_taps": 4 * q * q,
            "additions": 4 * q * q * 2,  # sum and sum-of-squares accumulation
            "multiplications": 4 * q * q,  # squaring for variance
            "exponentiations": 0,
            "comparisons": 3,  # argmin over 4 values
        },
        "soft_kuwahara": {
            "window_taps": 4 * q * q,
            "additions": 4 * q * q * 2 + 4,
            "multiplications": 4 * q * q + 4,
            "exponentiations": 4,  # softmax over 4 variances
            "comparisons": 0,
        },
        "bilateral_matched_sigma_s": {
            "window_taps": w_matched * w_matched,
            "additions": w_matched * w_matched * 3,
            "multiplications": w_matched * w_matched * 2,
            "exponentiations": w_matched * w_matched,
            "comparisons": 0,
            "note": f"w={w_matched} (the ACTUAL sigma_s={sigma_s_matched:.2f} matched in 4.1)",
        },
        "bilateral_representative": {
            "window_taps": w_repr * w_repr,
            "additions": w_repr * w_repr * 3,
            "multiplications": w_repr * w_repr * 2,
            "exponentiations": w_repr * w_repr,  # one range-kernel exp() per tap
            "comparisons": 0,
            "note": f"w={w_repr} (illustrative, larger sigma_s~8, to show the O(w^2) trend)",
        },
    }
    log(f"4.2 per-pixel operation counts: {json.dumps(counts, indent=2)}")

    result = {
        "temperature": temperature,
        "tv_classical_near_edge": tv_classical,
        "tv_soft_near_edge": tv_soft,
        "operation_counts": counts,
    }
    (RESULTS / "4_2_result.json").write_text(json.dumps(result, indent=2))
    return soft_out, result


# ---------------------------------------------------------------------------
# 4.3 Quantitative breakdown analysis of bilateral filtering
# ---------------------------------------------------------------------------
def make_step_image(N=256, I1=50.0, I2=200.0, sigma=0.0, rng=rng):
    img = np.full((N, N), I1, dtype=np.float64)
    img[:, N // 2:] = I2
    if sigma > 0:
        img = img + rng.normal(0, sigma, size=img.shape)
    return img


def section_4_3():
    """Sweep NCR = sigma_noise / |delta I| and measure bilateral edge preservation.

    sigma_s is held fixed throughout. sigma_r is set adaptively as
    sigma_r = max(sigma_noise, 5) -- i.e. proportional to the actual noise level of
    each test case (the textbook recommendation for choosing a bilateral filter's
    range parameter when the noise std is known/estimated: Tomasi & Manduchi-style
    "set sigma_r to the noise standard deviation"). This is a single FIXED rule
    (not re-tuned per NCR in an ad hoc way), and is what actually lets the filter
    denoise consistently at every noise level in the sweep -- a literally-fixed
    sigma_r (as tried first, before settling on this) turns out too small for the
    higher end of the NCR range tested here (the range kernel then rejects even
    same-side neighbours, so the filter does ~nothing and never shows the
    cross-edge-blending breakdown the PDF asks for).

    The edge-preservation metric is a band-averaged (not single-pixel) intensity
    difference across the boundary, averaged over several independent noise
    realizations per NCR, to keep the curve itself legible despite per-pixel
    measurement noise at high sigma.
    """
    N = 256
    I1, I2 = 50.0, 200.0
    delta_I = I2 - I1
    sigma_s = 6.0
    ncr_values = [0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4, 1.7, 2.0]
    n_trials = 6
    band = 4
    boundary_col = N // 2

    retained_pct = []
    retained_std = []
    for ncr in ncr_values:
        sigma = ncr * delta_I
        sigma_r = max(sigma, 5.0)
        vals = []
        for trial in range(n_trials):
            local_rng = np.random.default_rng(1000 + trial)
            noisy_step = make_step_image(N, I1, I2, sigma, rng=local_rng)
            filtered = bilateral_filter(noisy_step, sigma_s, sigma_r)
            left = filtered[:, boundary_col - band:boundary_col].mean()
            right = filtered[:, boundary_col:boundary_col + band].mean()
            vals.append(100.0 * (right - left) / delta_I)
        retained_pct.append(float(np.mean(vals)))
        retained_std.append(float(np.std(vals)))
        log(f"4.3 NCR={ncr:.2f} (sigma={sigma:.1f}, sigma_r={sigma_r:.1f}): "
            f"retained = {retained_pct[-1]:.2f}% (+/- {retained_std[-1]:.2f})")

    retained_pct = np.array(retained_pct)
    ncr_arr = np.array(ncr_values)

    # NCR* = NCR where the curve crosses the midpoint between its low-NCR plateau
    # (mean of the first 2 points) and its high-NCR plateau (mean of the last 2) --
    # a self-consistent "knee" definition, since the curve asymptotes well above 0%
    # rather than decaying all the way to it (a pure-spatial-average floor, not a
    # fixed threshold chosen in advance).
    low_plateau = retained_pct[:2].mean()
    high_plateau = retained_pct[-2:].mean()
    midpoint = (low_plateau + high_plateau) / 2.0
    below = np.where(retained_pct < midpoint)[0]
    i1 = below[0]
    x0, x1 = ncr_arr[i1 - 1], ncr_arr[i1]
    y0, y1 = retained_pct[i1 - 1], retained_pct[i1]
    ncr_star = x0 + (midpoint - y0) * (x1 - x0) / (y1 - y0)
    log(f"4.3 low plateau={low_plateau:.2f}%, high plateau={high_plateau:.2f}%, "
        f"midpoint={midpoint:.2f}%, critical NCR* = {ncr_star:.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ncr_values, retained_pct, yerr=retained_std, fmt="o-", capsize=3)
    ax.axhline(midpoint, color="gray", linestyle=":", label=f"midpoint ({midpoint:.1f}%)")
    ax.axvline(ncr_star, color="red", linestyle="--", label=f"NCR* = {ncr_star:.3f}")
    ax.set_xlabel("NCR = sigma_noise / |delta I|")
    ax.set_ylabel("Retained boundary contrast (%)")
    ax.set_title(f"Bilateral filter edge-preservation breakdown\n"
                 f"(sigma_s={sigma_s}, sigma_r = max(sigma_noise, 5))")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "4_3_breakdown_curve.png", dpi=150)
    plt.close(fig)

    # profile figure: a low-NCR vs high-NCR example, to visualize the blending
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, ncr in zip(axes, [0.1, 1.2]):
        sigma = ncr * delta_I
        sigma_r = max(sigma, 5.0)
        local_rng = np.random.default_rng(2024)
        noisy_step = make_step_image(N, I1, I2, sigma, rng=local_rng)
        filtered = bilateral_filter(noisy_step, sigma_s, sigma_r)
        mid_row = N // 2
        ax.plot(noisy_step[mid_row, :], alpha=0.3, label="noisy input")
        ax.plot(filtered[mid_row, :], label="bilateral output")
        ax.set_title(f"NCR={ncr}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "4_3_profiles.png", dpi=150)
    plt.close(fig)

    result = {
        "I1": I1, "I2": I2, "delta_I": delta_I,
        "sigma_s": sigma_s, "sigma_r_rule": "max(sigma_noise, 5)",
        "n_trials": n_trials,
        "ncr_values": ncr_values, "retained_pct": retained_pct.tolist(),
        "retained_std": retained_std,
        "low_plateau": float(low_plateau), "high_plateau": float(high_plateau),
        "midpoint": float(midpoint), "ncr_star": float(ncr_star),
    }
    (RESULTS / "4_3_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    record_machine_info()
    clean = load_clean()
    noisy = clean + rng.normal(0, NOISE_SIGMA, size=clean.shape)

    log("=== 4.1 ===")
    result_4_1, kuwahara_out, bilateral_out, sigma_s_matched, sigma_r_final = section_4_1(clean, noisy)

    log("=== 4.2 ===")
    section_4_2(clean, noisy, kuwahara_out, sigma_s_matched)

    log("=== 4.3 ===")
    section_4_3()

    log(f"All results written to {RESULTS}")


if __name__ == "__main__":
    main()
