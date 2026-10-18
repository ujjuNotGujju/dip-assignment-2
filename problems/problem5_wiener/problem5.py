"""
Problem 5 -- Denoising and the Wiener Filter (15 pts)

Runs 5.1 and 5.2 end to end, writing figures/tables to results/.
No disallowed library (scipy.ndimage/scipy.signal/cv2/skimage) is used anywhere.

Reuses Problem 2's make_filter/filter_pipeline (Gaussian/Butterworth LPF comparison
baselines) and Problem 4's bilateral_filter, plus the shared from-scratch PSNR/SSIM.
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
from dip_a2.filters import make_filter, filter_pipeline, radial_distance  # noqa: E402
from dip_a2.edge_preserving import bilateral_filter  # noqa: E402
from dip_a2.metrics import psnr, ssim  # noqa: E402
from dip_a2.wiener import wiener_denoise  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

SIGMAS = [5, 10, 20, 40]
MAX_DIM = 768


def log(msg):
    print(f"[problem5] {msg}")


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
    path = DATA / "provided" / "problem5_6" / "reference_image.jpg"
    im = Image.open(path).convert("L")
    w, h = im.size
    scale = MAX_DIM / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    im_small = im.resize(new_size, Image.LANCZOS)
    arr = np.asarray(im_small, dtype=np.float64)
    log(f"loaded reference_image.jpg: original {im.size}, downsampled to {im_small.size} (grayscale)")
    return arr


def oracle_sweep_lpf(kind, noisy, clean, n=None, D0_values=None):
    M, N = noisy.shape
    P, Q = 2 * M, 2 * N
    best = {"D0": None, "psnr": -np.inf, "out": None}
    for D0 in D0_values:
        H = make_filter(kind, (P, Q), float(D0), n=n)
        out = np.clip(filter_pipeline(noisy, H), 0, 255)
        val = psnr(out, clean, data_range=255.0)
        if val > best["psnr"]:
            best = {"D0": float(D0), "psnr": float(val), "out": out}
    return best


def oracle_sweep_bilateral(noisy, clean, sigma_r, sigma_s_values):
    best = {"sigma_s": None, "psnr": -np.inf, "out": None}
    for sigma_s in sigma_s_values:
        out = np.clip(bilateral_filter(noisy, sigma_s, sigma_r), 0, 255)
        val = psnr(out, clean, data_range=255.0)
        if val > best["psnr"]:
            best = {"sigma_s": float(sigma_s), "psnr": float(val), "out": out}
    return best


# ---------------------------------------------------------------------------
# 5.1 Wiener as MMSE denoiser + comparison
# ---------------------------------------------------------------------------
def section_5_1(clean):
    results_table = {}
    wiener_profiles = {}
    example_outputs = {}

    D0_values_gauss = np.linspace(10, 150, 10)
    D0_values_butter = np.linspace(10, 150, 10)
    sigma_s_values = np.linspace(0.5, 8.0, 8)

    for sigma in SIGMAS:
        local_rng = np.random.default_rng(seed=500 + sigma)
        noise = local_rng.normal(0, sigma, size=clean.shape)
        noisy = clean + noise  # not clipped before filtering

        wiener_out, W_profile = wiener_denoise(noisy, sigma_noise=sigma)
        wiener_out_c = np.clip(wiener_out, 0, 255)
        wiener_psnr = psnr(wiener_out_c, clean, data_range=255.0)
        wiener_ssim = ssim(wiener_out_c, clean, data_range=255.0)
        wiener_profiles[sigma] = W_profile.tolist()

        gauss_best = oracle_sweep_lpf("gaussian", noisy, clean, D0_values=D0_values_gauss)
        butter_best = oracle_sweep_lpf("butterworth", noisy, clean, n=2, D0_values=D0_values_butter)
        # sigma_r scaled to the noise level (2.5x), same lesson learned in Problem 4:
        # a fixed sigma_r starves the range kernel at high noise and badly
        # under-denoises, unfairly handicapping this comparison baseline.
        sigma_r_bilateral = max(2.5 * sigma, 10.0)
        bilat_best = oracle_sweep_bilateral(noisy, clean, sigma_r_bilateral, sigma_s_values)

        gauss_ssim = ssim(gauss_best["out"], clean, data_range=255.0)
        butter_ssim = ssim(butter_best["out"], clean, data_range=255.0)
        bilat_ssim = ssim(bilat_best["out"], clean, data_range=255.0)

        results_table[sigma] = {
            "wiener": {"psnr": float(wiener_psnr), "ssim": float(wiener_ssim), "oracle_tuned": False},
            "gaussian_lpf": {"psnr": gauss_best["psnr"], "ssim": float(gauss_ssim),
                              "D0": gauss_best["D0"], "oracle_tuned": True},
            "butterworth_lpf": {"psnr": butter_best["psnr"], "ssim": float(butter_ssim),
                                 "n": 2, "D0": butter_best["D0"], "oracle_tuned": True},
            "bilateral": {"psnr": bilat_best["psnr"], "ssim": float(bilat_ssim),
                          "sigma_s": bilat_best["sigma_s"], "sigma_r": sigma_r_bilateral, "oracle_tuned": True},
        }
        log(f"5.1 sigma={sigma}: " + json.dumps(results_table[sigma]))

        if sigma == 20:
            example_outputs = {
                "noisy": noisy, "wiener": wiener_out_c, "gaussian": gauss_best["out"],
                "butterworth": butter_best["out"], "bilateral": bilat_best["out"],
            }

    (RESULTS / "5_1_results_table.json").write_text(json.dumps(results_table, indent=2))

    # visual comparison at sigma=20
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.5))
    axes[0].imshow(clean, cmap="gray"); axes[0].set_title("Clean"); axes[0].axis("off")
    for ax, key in zip(axes[1:], ["noisy", "wiener", "gaussian", "butterworth", "bilateral"]):
        ax.imshow(example_outputs[key], cmap="gray")
        ax.set_title(key)
        ax.axis("off")
    fig.suptitle("sigma=20 comparison")
    fig.tight_layout()
    fig.savefig(RESULTS / "5_1_comparison_sigma20.png", dpi=150)
    plt.close(fig)

    # PSNR table plot across sigma
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = ["wiener", "gaussian_lpf", "butterworth_lpf", "bilateral"]
    for m in methods:
        vals = [results_table[s][m]["psnr"] for s in SIGMAS]
        ax.plot(SIGMAS, vals, "o-", label=m)
    ax.set_xlabel("noise sigma")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs noise level (wiener: not oracle-tuned; others: oracle-tuned)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "5_1_psnr_vs_sigma.png", dpi=150)
    plt.close(fig)

    # Wiener gain radial profile per sigma
    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma in SIGMAS:
        profile = np.array(wiener_profiles[sigma])
        r = np.arange(len(profile))
        max_r = min(len(profile), 200)
        ax.plot(r[:max_r], profile[:max_r], label=f"sigma={sigma}")
    ax.set_xlabel("radius (frequency index)")
    ax.set_ylabel("Wiener gain W(r)")
    ax.set_title("Wiener shrinkage vs frequency and noise power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "5_1_wiener_gain_profiles.png", dpi=150)
    plt.close(fig)

    return results_table, wiener_profiles


# ---------------------------------------------------------------------------
# 5.2 Test the assumptions
# ---------------------------------------------------------------------------
def section_5_2(clean):
    sigma = 20
    local_rng = np.random.default_rng(seed=520)
    noise = local_rng.normal(0, sigma, size=clean.shape)
    noisy = clean + noise

    scales = {"under (0.5x Snn)": 0.5, "correct (1x Snn)": 1.0, "over (2x Snn)": 2.0}
    outputs = {}
    table = {}
    for label, scale in scales.items():
        out, _ = wiener_denoise(noisy, sigma_noise=sigma, assumed_noise_scale=scale)
        out_c = np.clip(out, 0, 255)
        outputs[label] = out_c
        table[label] = {
            "scale": scale,
            "psnr": float(psnr(out_c, clean, data_range=255.0)),
            "ssim": float(ssim(out_c, clean, data_range=255.0)),
        }
    log(f"5.2 noise-power mis-specification: {json.dumps(table, indent=2)}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(noisy, cmap="gray"); axes[0].set_title(f"Noisy (sigma={sigma})"); axes[0].axis("off")
    for ax, label in zip(axes[1:], scales.keys()):
        ax.imshow(outputs[label], cmap="gray")
        ax.set_title(f"{label}\nPSNR={table[label]['psnr']:.2f}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "5_2_noise_misspecification.png", dpi=150)
    plt.close(fig)

    # edge-region crop: pick the highest-gradient region of the clean image
    gx = np.zeros_like(clean); gx[:, 1:] = np.abs(np.diff(clean, axis=1))
    gy = np.zeros_like(clean); gy[1:, :] = np.abs(np.diff(clean, axis=0))
    grad = gx + gy
    block = 40
    H, W = clean.shape
    rows, cols = H // block, W // block
    grad_block = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            grad_block[i, j] = grad[i * block:(i + 1) * block, j * block:(j + 1) * block].sum()
    bi, bj = np.unravel_index(np.argmax(grad_block), grad_block.shape)
    sl = (slice(bi * block, (bi + 1) * block), slice(bj * block, (bj + 1) * block))

    correct_out = outputs["correct (1x Snn)"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(clean[sl], cmap="gray"); axes[0].set_title("Clean (edge region)"); axes[0].axis("off")
    axes[1].imshow(noisy[sl], cmap="gray"); axes[1].set_title("Noisy"); axes[1].axis("off")
    axes[2].imshow(correct_out[sl], cmap="gray"); axes[2].set_title("Wiener (correctly specified)"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "5_2_edge_region_blur.png", dpi=150)
    plt.close(fig)

    # real noisy photo (own capture) -- graceful skip if absent
    own_dir = DATA / "own_captures"
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    own_photos = [p for p in own_dir.iterdir() if p.suffix.lower() in image_exts] if own_dir.exists() else []
    real_photo_result = None
    if own_photos:
        photo_path = own_photos[0]
        log(f"5.2 found real noisy photo: {photo_path.name} -- processing")
        real_img = np.asarray(Image.open(photo_path).convert("L"), dtype=np.float64)
        w, h = real_img.shape[1], real_img.shape[0]
        scale = MAX_DIM / max(w, h)
        if scale < 1.0:
            real_img = np.asarray(
                Image.open(photo_path).convert("L").resize(
                    (int(w * scale), int(h * scale)), Image.LANCZOS), dtype=np.float64)
        flat = real_img[:40, :40]
        est_sigma = float(flat.std())
        real_out, _ = wiener_denoise(real_img, sigma_noise=est_sigma)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(real_img, cmap="gray"); axes[0].set_title("Real noisy photo"); axes[0].axis("off")
        axes[1].imshow(np.clip(real_out, 0, 255), cmap="gray"); axes[1].set_title("Wiener-denoised"); axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(RESULTS / "5_2_real_noisy_photo.png", dpi=150)
        plt.close(fig)
        real_photo_result = {"file": photo_path.name, "estimated_sigma_from_corner_patch": est_sigma}
    else:
        log("5.2 PENDING: no real noisy photo found in data/own_captures/ -- "
            "this sub-deliverable is explicitly skipped per user instruction, not fabricated.")

    result = {
        "sigma": sigma,
        "misspecification_table": table,
        "edge_region_slice": [[int(sl[0].start), int(sl[0].stop)], [int(sl[1].start), int(sl[1].stop)]],
        "real_noisy_photo": real_photo_result,
        "real_noisy_photo_status": "done" if real_photo_result else "PENDING - needs a real captured photo",
    }
    (RESULTS / "5_2_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    record_machine_info()
    clean = load_clean()

    log("=== 5.1 ===")
    section_5_1(clean)

    log("=== 5.2 ===")
    section_5_2(clean)

    log(f"All results written to {RESULTS}")


if __name__ == "__main__":
    main()
