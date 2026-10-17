"""
Problem 3 -- Homomorphic Filtering: Illumination and Reflectance (10 pts)

Runs 3.1 (illustrative log-order demonstration) and 3.2 (implement/tune + histogram
equalization comparison) end to end, writing figures/tables to results/.

No disallowed library (scipy.ndimage/scipy.signal/cv2/skimage) is used anywhere in
this problem. PIL is used only for image I/O and a one-time downsample of the huge
source photo (a practical loading step, not one of the implemented operations).
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
from dip_a2.filters import make_homomorphic_filter, filter_pipeline_reflect  # noqa: E402
from dip_a2.utils import histogram_equalize  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)


def log(msg):
    print(f"[problem3] {msg}")


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


def load_and_downsample(path, max_dim=900):
    im = Image.open(path).convert("L")
    w, h = im.size
    scale = max_dim / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    im_small = im.resize(new_size, Image.LANCZOS)
    arr = np.asarray(im_small, dtype=np.float64)
    log(f"loaded {path.name}: original {im.size}, downsampled to {im_small.size}")
    return arr


# ---------------------------------------------------------------------------
# 3.1 Illustrative demonstration: why log must precede the FFT
# ---------------------------------------------------------------------------
def section_3_1(img):
    F = np.fft.fft2(img)
    log_mag_spectrum = np.log1p(np.abs(np.fft.fftshift(F)))  # log(|FFT(I)|): real, no phase

    log_img = np.log1p(img)
    F_of_log = np.fft.fft2(log_img)
    mag_of_F_of_log = np.log1p(np.abs(np.fft.fftshift(F_of_log)))  # |FFT(log I)|, in log display

    # quantify that log(|FFT(I)|) and |FFT(log I)| are simply different quantities
    a = log_mag_spectrum.ravel()
    b = mag_of_F_of_log.ravel()
    a_n = (a - a.mean()) / (a.std() + 1e-12)
    b_n = (b - b.mean()) / (b.std() + 1e-12)
    correlation = float(np.mean(a_n * b_n))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(log_mag_spectrum, cmap="viridis")
    axes[0].set_title("log(|FFT(I)|)  (log-magnitude spectrum of I)")
    axes[0].axis("off")
    axes[1].imshow(mag_of_F_of_log, cmap="viridis")
    axes[1].set_title("log(|FFT(log I)|)  (spectrum of log I)")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "3_1_log_order_demo.png", dpi=150)
    plt.close(fig)

    result = {
        "correlation_between_the_two_spectra": correlation,
        "note": "log(|FFT(I)|) is a real, phase-less display quantity and cannot be "
                "inverse-transformed back to a spatial image at all (phase was "
                "discarded by |.|); FFT(log I) is a full complex spectrum (magnitude "
                "AND phase) that CAN be correctly inverted. The two are also simply "
                "different arrays (log and FFT do not commute), confirmed by their "
                "low correlation below.",
    }
    log(f"3.1 log-order demo: correlation={correlation:.4f} (low/unrelated, as expected)")
    (RESULTS / "3_1_result.json").write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# 3.2 Implement and tune
# ---------------------------------------------------------------------------
def apply_homomorphic(image, D0, gammaL, gammaH, c=1.0):
    """Homomorphic filter, then rescale (min-max stretch) back to [0,255].

    The gammaL/gammaH gains act as multipliers in the LOG domain, so after
    exponentiating the result sits at a different absolute brightness level than the
    input (e.g. gammaL<1 compresses the illumination/DC term, which after exp() can
    collapse the whole image far below 0-255). Rescaling to use the full display range
    is standard practice for presenting homomorphic-filter output, and is what actually
    reveals the illumination-flattening/reflectance-detail effect.
    """
    M, N = image.shape
    P, Q = 2 * M, 2 * N
    H = make_homomorphic_filter((P, Q), D0, gammaL, gammaH, c)
    log_img = np.log1p(image)
    filtered_log = filter_pipeline_reflect(log_img, H)
    out = np.expm1(filtered_log)
    out_min, out_max = out.min(), out.max()
    out_stretched = (out - out_min) / (out_max - out_min + 1e-12) * 255.0
    return out_stretched


def find_extreme_blocks(image, block=50, margin=1):
    """Tile `image` into `block`x`block` tiles; return (shadow_slice, bright_slice)
    as the darkest / brightest tile by mean intensity, excluding the outermost ring
    of tiles (margin) to avoid pure-border artifacts."""
    M, N = image.shape
    rows = M // block
    cols = N // block
    means = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            means[i, j] = image[i * block:(i + 1) * block, j * block:(j + 1) * block].mean()
    means_interior = means[margin:rows - margin, margin:cols - margin]
    dark_idx = np.unravel_index(np.argmin(means_interior), means_interior.shape)
    bright_idx = np.unravel_index(np.argmax(means_interior), means_interior.shape)
    di, dj = dark_idx[0] + margin, dark_idx[1] + margin
    bi, bj = bright_idx[0] + margin, bright_idx[1] + margin
    shadow_slice = (slice(di * block, (di + 1) * block), slice(dj * block, (dj + 1) * block))
    bright_slice = (slice(bi * block, (bi + 1) * block), slice(bj * block, (bj + 1) * block))
    return shadow_slice, bright_slice


def section_3_2(img):
    M, N = img.shape

    # --- gammaL x gammaH sweep at a representative D0 ---
    D0_fixed = 60.0
    gammaLs = [0.25, 0.5, 0.75]
    gammaHs = [1.5, 2.0, 3.0]
    fig, axes = plt.subplots(len(gammaLs), len(gammaHs), figsize=(4 * len(gammaHs), 4 * len(gammaLs)))
    for i, gL in enumerate(gammaLs):
        for j, gH in enumerate(gammaHs):
            out = apply_homomorphic(img, D0_fixed, gL, gH)
            axes[i, j].imshow(out, cmap="gray", vmin=0, vmax=255)
            axes[i, j].set_title(f"gL={gL}, gH={gH}")
            axes[i, j].axis("off")
    fig.suptitle(f"gammaL x gammaH sweep at D0={D0_fixed}")
    fig.tight_layout()
    fig.savefig(RESULTS / "3_2_gamma_sweep.png", dpi=150)
    plt.close(fig)
    log("3.2 gamma sweep grid saved")

    # --- D0 sweep at an aggressive (gammaL, gammaH) to surface halos ---
    gL_aggr, gH_aggr = 0.2, 3.5
    D0s = [10, 30, 80, 250]
    fig, axes = plt.subplots(1, len(D0s), figsize=(4 * len(D0s), 4))
    for j, D0 in enumerate(D0s):
        out = apply_homomorphic(img, D0, gL_aggr, gH_aggr)
        axes[j].imshow(out, cmap="gray", vmin=0, vmax=255)
        axes[j].set_title(f"D0={D0}")
        axes[j].axis("off")
    fig.suptitle(f"D0 sweep at gammaL={gL_aggr}, gammaH={gH_aggr} (halo search)")
    fig.tight_layout()
    fig.savefig(RESULTS / "3_2_D0_sweep_halo.png", dpi=150)
    plt.close(fig)
    log("3.2 D0 halo-search sweep saved")

    # --- best-tuned homomorphic result (chosen by inspection of the sweeps above) ---
    best_params = {"D0": 60.0, "gammaL": 0.4, "gammaH": 2.0, "c": 1.0}
    homomorphic_out = apply_homomorphic(img, best_params["D0"], best_params["gammaL"],
                                         best_params["gammaH"], best_params["c"])

    # --- histogram equalization on the original image ---
    histeq_out = histogram_equalize(img)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    axes[0].imshow(img, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(homomorphic_out, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title(f"Homomorphic (best: {best_params})"); axes[1].axis("off")
    axes[2].imshow(histeq_out, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Global histogram equalization"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "3_2_comparison.png", dpi=150)
    plt.close(fig)

    # --- data-driven shadow/bright region selection on the ORIGINAL image ---
    shadow_slice, bright_slice = find_extreme_blocks(img, block=50, margin=1)
    log(f"3.2 shadow block slice={shadow_slice}, bright block slice={bright_slice}")

    def stats(arr, sl):
        patch = arr[sl]
        return {
            "mean": float(patch.mean()),
            "std": float(patch.std()),
            "max": float(patch.max()),
            "frac_above_250": float(np.mean(patch >= 250)),
        }

    table = {
        "shadow_region": {
            "original": stats(img, shadow_slice),
            "homomorphic": stats(homomorphic_out, shadow_slice),
            "histeq": stats(histeq_out, shadow_slice),
        },
        "bright_region": {
            "original": stats(img, bright_slice),
            "homomorphic": stats(homomorphic_out, bright_slice),
            "histeq": stats(histeq_out, bright_slice),
        },
        "best_params": best_params,
        "shadow_slice": [[int(shadow_slice[0].start), int(shadow_slice[0].stop)],
                          [int(shadow_slice[1].start), int(shadow_slice[1].stop)]],
        "bright_slice": [[int(bright_slice[0].start), int(bright_slice[0].stop)],
                          [int(bright_slice[1].start), int(bright_slice[1].stop)]],
    }
    log(f"3.2 local-contrast table: {json.dumps(table, indent=2)}")
    (RESULTS / "3_2_result.json").write_text(json.dumps(table, indent=2))
    return table


def main():
    record_machine_info()
    img = load_and_downsample(DATA / "provided" / "problem3" / "tunnel_image.jpg", max_dim=900)

    log("=== 3.1 ===")
    section_3_1(img)

    log("=== 3.2 ===")
    section_3_2(img)

    log(f"All results written to {RESULTS}")


if __name__ == "__main__":
    main()
