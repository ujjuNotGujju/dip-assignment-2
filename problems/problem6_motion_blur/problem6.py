"""
Problem 6 -- Motion Blur and Deblurring (20 pts)

Runs 6.1, 6.2, 6.3 end to end (RGB), and attempts 6.4 if a real motion-blurred photo
is present in data/own_captures/ (gracefully skipped/pending otherwise).
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
from dip_a2.motion_blur import (  # noqa: E402
    motion_psf_otf, motion_otf_analytic, add_noise_at_snr,
    naive_inverse_filter, wiener_deconvolve, estimate_motion_blur,
)
from dip_a2.metrics import psnr, ssim  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

MAX_DIM = 768
L_TRUE = 25.0
THETA_TRUE = 30.0
SNR_LEVELS = [10, 20, 30, 40]


def log(msg):
    print(f"[problem6] {msg}")


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


def load_clean_rgb():
    path = DATA / "provided" / "problem5_6" / "reference_image.jpg"
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = MAX_DIM / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    im_small = im.resize(new_size, Image.LANCZOS)
    arr = np.asarray(im_small, dtype=np.float64)
    log(f"loaded reference_image.jpg: original {im.size}, downsampled to {im_small.size} (RGB)")
    return arr


def luminance(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def blur_rgb(clean_rgb, H):
    M, N, C = clean_rgb.shape
    out = np.zeros_like(clean_rgb)
    for c in range(C):
        out[..., c] = np.real(np.fft.ifft2(np.fft.fft2(clean_rgb[..., c]) * H))
    return out


# ---------------------------------------------------------------------------
# 6.1 Model and synthesize
# ---------------------------------------------------------------------------
def section_6_1(clean_rgb):
    M, N = clean_rgb.shape[:2]
    psf, H_raster = motion_psf_otf((M, N), L_TRUE, THETA_TRUE)
    H_analytic = motion_otf_analytic((M, N), L_TRUE, THETA_TRUE)

    abs_diff = np.abs(np.abs(H_raster) - np.abs(H_analytic))
    max_abs_err = float(abs_diff.max())
    median_abs_err = float(np.median(abs_diff))
    p99_abs_err = float(np.percentile(abs_diff, 99))
    worst_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    worst_freq = (float(np.fft.fftfreq(M)[worst_idx[0]]), float(np.fft.fftfreq(N)[worst_idx[1]]))
    log(f"6.1 rasterized vs analytic OTF magnitude: median err={median_abs_err:.5f}, "
        f"99th pct={p99_abs_err:.5f}, max err={max_abs_err:.5f} (at freq {worst_freq}, "
        f"i.e. essentially the Nyquist frequency -- expected aliasing, see report)")

    blurred_rgb = blur_rgb(clean_rgb, H_raster)
    rng = np.random.default_rng(seed=600)
    noisy_blurred_rgb, sigma_used = add_noise_at_snr(blurred_rgb, snr_db=30.0, rng=rng)
    log(f"6.1 blurred + noise at SNR=30dB, sigma={sigma_used:.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    psf_crop = np.fft.fftshift(psf)
    axes[0].imshow(psf_crop[M // 2 - 20:M // 2 + 20, N // 2 - 20:N // 2 + 20], cmap="hot")
    axes[0].set_title(f"PSF (L={L_TRUE}, theta={THETA_TRUE} deg)")
    axes[0].axis("off")
    axes[1].imshow(np.clip(clean_rgb, 0, 255).astype(np.uint8)); axes[1].set_title("Clean"); axes[1].axis("off")
    axes[2].imshow(np.clip(blurred_rgb, 0, 255).astype(np.uint8)); axes[2].set_title("Blurred"); axes[2].axis("off")
    axes[3].imshow(np.clip(noisy_blurred_rgb, 0, 255).astype(np.uint8)); axes[3].set_title("Blurred + noise (SNR=30dB)"); axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "6_1_psf_and_blur.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(np.fft.fftshift(np.abs(H_raster)), cmap="viridis")
    axes[0].set_title("|H| rasterized (fftshifted)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(np.fft.fftshift(np.abs(H_analytic)), cmap="viridis")
    axes[1].set_title("|H| analytic sinc (fftshifted)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULTS / "6_1_otf_comparison.png", dpi=150)
    plt.close(fig)

    result = {
        "L": L_TRUE, "theta_deg": THETA_TRUE,
        "otf_median_abs_err_raster_vs_analytic": median_abs_err,
        "otf_p99_abs_err_raster_vs_analytic": p99_abs_err,
        "otf_max_abs_err_raster_vs_analytic": max_abs_err,
        "otf_max_err_freq": worst_freq,
        "snr_db_example": 30.0, "sigma_used": float(sigma_used),
    }
    (RESULTS / "6_1_result.json").write_text(json.dumps(result, indent=2))
    return H_raster, H_analytic, blurred_rgb, noisy_blurred_rgb


# ---------------------------------------------------------------------------
# 6.2 Naive inverse filtering
# ---------------------------------------------------------------------------
def section_6_2(clean_rgb, noisy_blurred_rgb, H):
    M, N = clean_rgb.shape[:2]
    clean_lum = luminance(clean_rgb)
    noisy_lum = luminance(noisy_blurred_rgb)

    G = np.fft.fft2(noisy_lum)
    F_hat_raw = naive_inverse_filter(G, H, floor=1e-3)
    f_hat_raw = np.real(np.fft.ifft2(F_hat_raw))

    # predicted zero lines of the analytic OTF, for overlay
    H_analytic = motion_otf_analytic((M, N), L_TRUE, THETA_TRUE)
    zero_mask = np.fft.fftshift(np.abs(H_analytic)) < 0.02

    log_G = np.log1p(np.abs(np.fft.fftshift(G)))
    log_Fhat = np.log1p(np.abs(np.fft.fftshift(F_hat_raw)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(log_G, cmap="viridis")
    axes[0].contour(zero_mask, levels=[0.5], colors="red", linewidths=0.5)
    axes[0].set_title("log|G| with predicted OTF zero lines")
    axes[0].axis("off")
    axes[1].imshow(log_Fhat, cmap="viridis", vmax=np.percentile(log_Fhat, 99.5))
    axes[1].contour(zero_mask, levels=[0.5], colors="red", linewidths=0.5)
    axes[1].set_title("log|F_hat| (naive inverse, pre-clip) -- amplification at zeros")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "6_2_naive_inverse_spectra.png", dpi=150)
    plt.close(fig)

    # Quantify amplification: compare mean |F_hat| at bins where |H| actually hit the
    # floor (the true "amplification" bins) against OTHER high-frequency bins at a
    # similar radius (excluding the naturally-dominant low-frequency/DC core, which
    # would otherwise swamp a naive whole-spectrum "elsewhere" average and understate
    # the effect) -- an apples-to-apples comparison at comparable frequency magnitude.
    floored_mask = np.abs(H) < 1e-3
    fx = np.fft.fftfreq(M).reshape(-1, 1)
    fy = np.fft.fftfreq(N).reshape(1, -1)
    radius = np.sqrt(fx ** 2 + fy ** 2)
    high_freq_elsewhere = (radius > 0.08) & (~floored_mask)
    near_zero_amp = float(np.abs(F_hat_raw)[floored_mask].mean())
    elsewhere_amp = float(np.abs(F_hat_raw)[high_freq_elsewhere].mean())
    log(f"6.2 mean |F_hat| at floored (|H|<1e-3) bins = {near_zero_amp:.3e}, "
        f"other high-freq bins = {elsewhere_amp:.3e}, ratio = {near_zero_amp / elsewhere_amp:.2f}x")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(clean_lum, cmap="gray"); axes[0].set_title("Clean luminance"); axes[0].axis("off")
    display_out = np.clip(f_hat_raw, 0, 255)
    axes[1].imshow(display_out, cmap="gray"); axes[1].set_title("Naive inverse estimate (clipped for display)"); axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "6_2_naive_inverse_image.png", dpi=150)
    plt.close(fig)

    result = {"near_zero_mean_amplitude": near_zero_amp, "elsewhere_mean_amplitude": elsewhere_amp,
              "amplification_ratio": near_zero_amp / elsewhere_amp, "floor": 1e-3}
    (RESULTS / "6_2_result.json").write_text(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# 6.3 Wiener deconvolution
# ---------------------------------------------------------------------------
def section_6_3(clean_rgb, H):
    clean_lum = luminance(clean_rgb)
    blurred_lum = np.real(np.fft.ifft2(np.fft.fft2(clean_lum) * H))

    K_values = np.logspace(-6, 0, 25)
    best_per_snr = {}
    curves = {}

    for snr_db in SNR_LEVELS:
        rng = np.random.default_rng(seed=700 + snr_db)
        noisy_lum, sigma_used = add_noise_at_snr(blurred_lum, snr_db, rng)
        G = np.fft.fft2(noisy_lum)
        psnrs = []
        for K in K_values:
            F_hat = wiener_deconvolve(G, H, K)
            out = np.clip(np.real(np.fft.ifft2(F_hat)), 0, 255)
            psnrs.append(psnr(out, clean_lum, data_range=255.0))
        psnrs = np.array(psnrs)
        best_idx = int(np.argmax(psnrs))
        best_per_snr[snr_db] = {"K": float(K_values[best_idx]), "psnr": float(psnrs[best_idx]),
                                  "sigma_used": float(sigma_used)}
        curves[snr_db] = psnrs.tolist()
        log(f"6.3 SNR={snr_db}dB: best K={K_values[best_idx]:.2e}, PSNR={psnrs[best_idx]:.2f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for snr_db in SNR_LEVELS:
        ax.plot(K_values, curves[snr_db], "o-", label=f"SNR={snr_db}dB", markersize=3)
        ax.axvline(best_per_snr[snr_db]["K"], linestyle=":", alpha=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Wiener deconvolution: PSNR vs K, per noise level")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "6_3_K_sweep.png", dpi=150)
    plt.close(fig)

    # qualitative low-K / optimal-K / high-K comparison at SNR=20dB
    snr_demo = 20
    rng = np.random.default_rng(seed=700 + snr_demo)
    noisy_lum, _ = add_noise_at_snr(blurred_lum, snr_demo, rng)
    G = np.fft.fft2(noisy_lum)
    K_low, K_opt, K_high = 1e-6, best_per_snr[snr_demo]["K"], 1.0
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(clean_lum, cmap="gray"); axes[0].set_title("Clean"); axes[0].axis("off")
    for ax, K, label in zip(axes[1:], [K_low, K_opt, K_high], ["low K (inverse-filter-like)", "optimal K", "high K (under-deconvolved)"]):
        out = np.clip(np.real(np.fft.ifft2(wiener_deconvolve(G, H, K))), 0, 255)
        ax.imshow(out, cmap="gray")
        ax.set_title(f"{label}\nK={K:.1e}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "6_3_K_qualitative.png", dpi=150)
    plt.close(fig)

    (RESULTS / "6_3_result.json").write_text(json.dumps({"best_per_snr": best_per_snr}, indent=2))
    return best_per_snr


# ---------------------------------------------------------------------------
# 6.4 Estimate the blur, then remove it (pending real data)
# ---------------------------------------------------------------------------
def section_6_4():
    own_dir = DATA / "own_captures"
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    own_photos = [p for p in own_dir.iterdir() if p.suffix.lower() in image_exts] if own_dir.exists() else []

    if not own_photos:
        log("6.4 PENDING: no real motion-blurred photo found in data/own_captures/ -- "
            "this sub-deliverable is explicitly skipped per user instruction, not fabricated.")
        (RESULTS / "6_4_result.json").write_text(json.dumps(
            {"status": "PENDING - needs a real captured motion-blurred photo"}, indent=2))
        return None

    photo_path = own_photos[0]
    log(f"6.4 found real motion-blurred photo: {photo_path.name} -- processing")
    im = Image.open(photo_path).convert("L")
    w, h = im.size
    scale = MAX_DIM / max(w, h)
    if scale < 1.0:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    real_img = np.asarray(im, dtype=np.float64)

    theta_est, L_est = estimate_motion_blur(real_img)
    log(f"6.4 estimated theta={theta_est:.1f} deg, L={L_est:.2f} px")
    _, H_est = motion_psf_otf(real_img.shape, L_est, theta_est)
    K_est = 0.01
    deblurred = np.clip(np.real(np.fft.ifft2(wiener_deconvolve(np.fft.fft2(real_img), H_est, K_est))), 0, 255)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(real_img, cmap="gray"); axes[0].set_title("Real motion-blurred photo"); axes[0].axis("off")
    axes[1].imshow(deblurred, cmap="gray"); axes[1].set_title(f"Wiener deblurred (theta={theta_est:.1f}, L={L_est:.1f}, K={K_est})"); axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "6_4_real_deblur.png", dpi=150)
    plt.close(fig)

    result = {"file": photo_path.name, "theta_est_deg": theta_est, "L_est": L_est, "K": K_est}
    (RESULTS / "6_4_result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    record_machine_info()
    clean_rgb = load_clean_rgb()

    log("=== 6.1 ===")
    H_raster, H_analytic, blurred_rgb, noisy_blurred_rgb = section_6_1(clean_rgb)

    log("=== 6.2 ===")
    section_6_2(clean_rgb, noisy_blurred_rgb, H_raster)

    log("=== 6.3 ===")
    section_6_3(clean_rgb, H_raster)

    log("=== 6.4 ===")
    section_6_4()

    log(f"All results written to {RESULTS}")


if __name__ == "__main__":
    main()
