"""
Problem 1 -- DFT and the Machinery of Frequency (15 pts)

Runs all of 1.1, 1.2, 1.3 end to end and writes figures/tables into results/.
Only NumPy array arithmetic is used for every operation the assignment asks us to
implement (the custom DFT, the shift/convolution-theorem checks, the spatial
convolution, the padding). numpy.fft is used only to *validate* our implementation,
and scipy/PIL are used only for image I/O / the named validation checks -- never to
perform an implemented operation.
"""
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dip_a2.dft import dft2d, idft2d, fftshift2d, ifftshift2d  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

rng = np.random.default_rng(seed=42)


def log(msg):
    print(f"[problem1] {msg}")


# ---------------------------------------------------------------------------
# Machine info
# ---------------------------------------------------------------------------
def record_machine_info():
    info = {
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        nproc = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
        info["logical_cpus"] = nproc
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / 1024 / 1024, 2)
                    break
    except Exception:
        pass
    (RESULTS / "machine_info.json").write_text(json.dumps(info, indent=2))
    log(f"machine info: {info}")
    return info


# ---------------------------------------------------------------------------
# 1.1 Build and validate the DFT
# ---------------------------------------------------------------------------
def load_test_image():
    im = Image.open(DATA / "provided" / "problem1" / "q1.png").convert("L")
    f = np.asarray(im, dtype=np.float64)
    assert f.shape == (512, 512), f"expected 512x512, got {f.shape}"
    return f


def section_1_1a_validate(f):
    """Validate custom DFT against numpy.fft.fft2; report max abs error."""
    F_custom = dft2d(f)
    F_numpy = np.fft.fft2(f)
    max_err = np.max(np.abs(F_custom - F_numpy))
    rel_err = max_err / np.max(np.abs(F_numpy))
    log(f"1.1 validation: max|F_custom - F_numpy| = {max_err:.6e} "
        f"(relative to max|F_numpy| = {rel_err:.3e})")
    return F_custom, F_numpy, max_err, rel_err


def section_1_1a_conjugate_symmetry(f, F_custom):
    """Numerically confirm F(u,v) = conj(F(-u,-v)) for real f."""
    M, N = f.shape
    F_neg = np.conj(np.roll(np.roll(F_custom[::-1, ::-1], 1, axis=0), 1, axis=1))
    max_err = np.max(np.abs(F_custom - F_neg))
    log(f"1.1a conjugate symmetry: max|F(u,v) - conj(F(-u,-v))| = {max_err:.6e}")
    return max_err


def section_1_1b_shift_theorem(f, F_custom):
    """Empirically verify the 2-D spatial shift theorem for several shifts."""
    M, N = f.shape
    u = np.arange(M).reshape(-1, 1)
    v = np.arange(N).reshape(1, -1)
    rows = []
    for (x0, y0) in [(1, 0), (0, 1), (5, 3), (17, 40), (100, 200)]:
        f_shifted = np.roll(np.roll(f, x0, axis=0), y0, axis=1)
        F_shifted_measured = dft2d(f_shifted)
        theoretical_factor = np.exp(-2j * np.pi * (u * x0 / M + v * y0 / N))
        F_shifted_theory = F_custom * theoretical_factor
        max_err = np.max(np.abs(F_shifted_measured - F_shifted_theory))
        rel_err = max_err / np.max(np.abs(F_shifted_theory))
        rows.append((x0, y0, max_err, rel_err))
        log(f"1.1b shift ({x0},{y0}): max abs err = {max_err:.6e}, rel err = {rel_err:.3e}")
    return rows


def circular_convolve_spatial(f, kernel):
    """Direct spatial circular convolution via np.roll (ground truth, from scratch)."""
    M, N = f.shape
    kM, kN = kernel.shape
    cy, cx = kM // 2, kN // 2
    out = np.zeros_like(f, dtype=np.float64)
    for i in range(kM):
        for j in range(kN):
            w = kernel[i, j]
            if w == 0:
                continue
            out += w * np.roll(np.roll(f, i - cy, axis=0), j - cx, axis=1)
    return out


def section_1_1b_convolution_theorem(f):
    """Empirically verify the 2-D circular convolution theorem."""
    kernel = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]], dtype=np.float64)  # 3x3 Laplacian
    M, N = f.shape
    kM, kN = kernel.shape
    K_padded = np.zeros((M, N), dtype=np.float64)
    cy, cx = kM // 2, kN // 2
    # place kernel so index (0,0) of K_padded aligns with the np.roll convention above
    K_padded[: kM, : kN] = kernel
    K_padded = np.roll(np.roll(K_padded, -cy, axis=0), -cx, axis=1)

    spatial_result = circular_convolve_spatial(f, kernel)
    F = dft2d(f)
    K = dft2d(K_padded)
    freq_result = np.real(idft2d(F * K))

    max_err = np.max(np.abs(spatial_result - freq_result))
    rel_err = max_err / np.max(np.abs(spatial_result))
    log(f"1.1b circular convolution theorem: max abs err = {max_err:.6e}, rel err = {rel_err:.3e}")
    return spatial_result, freq_result, max_err, rel_err


def section_1_1c_sinusoid_recovery():
    """Synthesize a sum of >=3 2-D sinusoids, recover params from DFT peaks."""
    N = 512
    x = np.arange(N).reshape(-1, 1)
    y = np.arange(N).reshape(1, -1)

    ground_truth = [
        {"u": 4, "v": 6, "A": 60.0, "phi": 0.3},
        {"u": 10, "v": -3, "A": 40.0, "phi": 1.2},
        {"u": -7, "v": 15, "A": 25.0, "phi": -0.8},
    ]
    (RESULTS / "sinusoid_ground_truth.json").write_text(json.dumps(ground_truth, indent=2))

    img = np.full((N, N), 128.0)
    for comp in ground_truth:
        phase_arg = 2 * np.pi * (comp["u"] * x / N + comp["v"] * y / N) + comp["phi"]
        img += comp["A"] * np.cos(phase_arg)

    F = dft2d(img)
    mag = np.abs(F)

    # exclude DC and its immediate neighbourhood when finding peaks
    mag_search = mag.copy()
    mag_search[0, 0] = 0
    for du in range(-2, 3):
        for dv in range(-2, 3):
            mag_search[du % N, dv % N] = 0

    recovered = []
    mag_work = mag_search.copy()
    n_peaks_needed = len(ground_truth)
    found = set()
    while len(recovered) < n_peaks_needed:
        idx = np.unravel_index(np.argmax(mag_work), mag_work.shape)
        u_idx, v_idx = idx
        u = u_idx if u_idx <= N // 2 else u_idx - N
        v = v_idx if v_idx <= N // 2 else v_idx - N
        # suppress the conjugate-symmetric twin and a small neighbourhood
        for du in range(-2, 3):
            for dv in range(-2, 3):
                mag_work[(u_idx + du) % N, (v_idx + dv) % N] = 0
                mag_work[(-u_idx + du) % N, (-v_idx + dv) % N] = 0
        if (-u, -v) in found:
            continue
        found.add((u, v))
        A_rec = 2 * mag[u_idx, v_idx] / (N * N)
        phi_rec = np.angle(F[u_idx, v_idx])
        recovered.append({"u": int(u), "v": int(v), "A": float(A_rec), "phi": float(phi_rec)})

    log(f"1.1c recovered sinusoid params: {recovered}")
    delta_f_cycles_per_pixel = 1.0 / N
    delta_f_cycles_per_image = 1.0
    (RESULTS / "sinusoid_recovered.json").write_text(json.dumps({
        "recovered": recovered,
        "delta_f_cycles_per_pixel": delta_f_cycles_per_pixel,
        "delta_f_cycles_per_image": delta_f_cycles_per_image,
    }, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Synthetic sinusoid-sum image")
    axes[0].axis("off")
    axes[1].imshow(np.log1p(fftshift2d(mag)), cmap="viridis")
    axes[1].set_title("log(1+|F|), shifted")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "1_1c_sinusoid_and_spectrum.png", dpi=150)
    plt.close(fig)

    return ground_truth, recovered


# ---------------------------------------------------------------------------
# 1.2 Phase vs magnitude isolation
# ---------------------------------------------------------------------------
def section_1_2(f):
    F = dft2d(f)
    mag = np.abs(F)
    phase = np.angle(F)

    # (a) phase of test image + flat (constant) magnitude
    flat_c = np.mean(mag)  # arbitrary constant scale, chosen as mean magnitude
    G_flat_phase = flat_c * np.exp(1j * phase)
    g_flat_phase = np.real(idft2d(G_flat_phase))

    # (a, noise-magnitude variant) phase of test image + magnitude of uniform white noise
    noise = rng.uniform(-128, 128, size=f.shape)
    N_noise = dft2d(noise)
    G_noise_mag = np.abs(N_noise) * np.exp(1j * phase)
    g_noise_mag = np.real(idft2d(G_noise_mag))

    # (b) magnitude of test image + zero phase
    G_zero_phase = mag * np.exp(1j * 0.0)
    g_zero_phase = np.real(idft2d(G_zero_phase))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(f, cmap="gray"); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(g_flat_phase, cmap="gray"); axes[1].set_title("Phase + flat magnitude"); axes[1].axis("off")
    axes[2].imshow(g_noise_mag, cmap="gray"); axes[2].set_title("Phase + noise magnitude"); axes[2].axis("off")
    axes[3].imshow(np.log1p(np.abs(g_zero_phase - g_zero_phase.min())), cmap="gray")
    axes[3].set_title("Magnitude + zero phase (log display)"); axes[3].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "1_2_phase_magnitude.png", dpi=150)
    plt.close(fig)

    stats = {
        "flat_magnitude_constant": float(flat_c),
        "g_flat_phase_range": [float(g_flat_phase.min()), float(g_flat_phase.max())],
        "g_noise_mag_range": [float(g_noise_mag.min()), float(g_noise_mag.max())],
        "g_zero_phase_range": [float(g_zero_phase.min()), float(g_zero_phase.max())],
        "g_zero_phase_peak_at_origin": bool(
            np.unravel_index(np.argmax(g_zero_phase), g_zero_phase.shape) == (0, 0)
        ),
    }
    (RESULTS / "1_2_stats.json").write_text(json.dumps(stats, indent=2))
    log(f"1.2 stats: {stats}")
    return stats


# ---------------------------------------------------------------------------
# 1.3 Padding, wraparound, computational overhead
# ---------------------------------------------------------------------------
def spatial_linear_convolve(f, kernel):
    """Direct zero-padded (linear) spatial convolution, from scratch."""
    M, N = f.shape
    kM, kN = kernel.shape
    cy, cx = kM // 2, kN // 2
    f_padded = np.zeros((M + 2 * cy, N + 2 * cx))
    f_padded[cy:cy + M, cx:cx + N] = f
    out = np.zeros_like(f)
    for i in range(kM):
        for j in range(kN):
            w = kernel[i, j]
            if w == 0:
                continue
            out += w * f_padded[i:i + M, j:j + N]
    return out


def section_1_3(f):
    kernel = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0],
    ], dtype=np.float64)  # 5x5 Laplacian-like kernel
    M, N = f.shape
    kM, kN = kernel.shape
    cy, cx = kM // 2, kN // 2

    # --- unpadded circular convolution via DFT (at image resolution -> wraparound) ---
    K_circ = np.zeros((M, N))
    K_circ[:kM, :kN] = kernel
    K_circ = np.roll(np.roll(K_circ, -cy, axis=0), -cx, axis=1)
    F = dft2d(f)
    Kc = dft2d(K_circ)
    circ_result = np.real(idft2d(F * Kc))

    # --- correctly zero-padded linear convolution via DFT ---
    Px, Py = M + kM - 1, N + kN - 1
    f_pad = np.zeros((Px, Py))
    f_pad[:M, :N] = f
    K_pad = np.zeros((Px, Py))
    K_pad[:kM, :kN] = kernel
    K_pad = np.roll(np.roll(K_pad, -cy, axis=0), -cx, axis=1)
    F_pad = dft2d(f_pad)
    K_pad_f = dft2d(K_pad)
    lin_full = np.real(idft2d(F_pad * K_pad_f))
    # crop back to the "same"-size linear-convolution result, aligned with spatial_linear_convolve
    lin_result = lin_full[:M, :N]

    # --- ground-truth direct spatial linear convolution (no FFT at all) ---
    direct_result = spatial_linear_convolve(f, kernel)

    err_padded_vs_direct = np.max(np.abs(lin_result - direct_result))
    err_circ_vs_direct = np.max(np.abs(circ_result - direct_result))
    border = max(cy, cx)
    err_circ_vs_direct_interior = np.max(np.abs(
        (circ_result - direct_result)[border:-border, border:-border]
    ))
    err_circ_vs_direct_border = np.max(np.abs(
        np.concatenate([
            (circ_result - direct_result)[:border, :].ravel(),
            (circ_result - direct_result)[-border:, :].ravel(),
            (circ_result - direct_result)[:, :border].ravel(),
            (circ_result - direct_result)[:, -border:].ravel(),
        ])
    ))

    log(f"1.3 padded-DFT vs direct spatial conv: max abs err = {err_padded_vs_direct:.6e}")
    log(f"1.3 circular-DFT vs direct spatial conv: max abs err (whole image) = {err_circ_vs_direct:.6e}")
    log(f"1.3 circular-DFT vs direct, interior only: {err_circ_vs_direct_interior:.6e}")
    log(f"1.3 circular-DFT vs direct, border only: {err_circ_vs_direct_border:.6e}")

    # memory / runtime trade-off
    mem_factor = (Px * Py) / (M * N)
    runtime_naive_ratio = (Px * Py * np.log2(Px * Py)) / (M * N * np.log2(M * N))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(direct_result, cmap="gray"); axes[0].set_title("Direct linear conv (ground truth)"); axes[0].axis("off")
    axes[1].imshow(lin_result, cmap="gray"); axes[1].set_title("Zero-padded DFT conv"); axes[1].axis("off")
    axes[2].imshow(circ_result, cmap="gray"); axes[2].set_title("Unpadded circular DFT conv (wraparound)"); axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "1_3_padding_wraparound.png", dpi=150)
    plt.close(fig)

    # zoomed border comparison
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(direct_result[:30, :], cmap="gray"); axes[0].set_title("Direct conv, top rows"); axes[0].axis("off")
    axes[1].imshow(circ_result[:30, :], cmap="gray"); axes[1].set_title("Circular conv, top rows (corrupted)"); axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "1_3_border_zoom.png", dpi=150)
    plt.close(fig)

    results = {
        "kernel_size": [kM, kN],
        "image_size": [M, N],
        "min_padded_size": [Px, Py],
        "err_padded_vs_direct": float(err_padded_vs_direct),
        "err_circ_vs_direct_whole": float(err_circ_vs_direct),
        "err_circ_vs_direct_interior": float(err_circ_vs_direct_interior),
        "err_circ_vs_direct_border": float(err_circ_vs_direct_border),
        "memory_factor": float(mem_factor),
        "runtime_ratio_nlogn": float(runtime_naive_ratio),
    }
    (RESULTS / "1_3_results.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    machine_info = record_machine_info()
    f = load_test_image()

    log("=== 1.1 ===")
    F_custom, F_numpy, max_err, rel_err = section_1_1a_validate(f)
    sym_err = section_1_1a_conjugate_symmetry(f, F_custom)
    shift_rows = section_1_1b_shift_theorem(f, F_custom)
    spatial_conv, freq_conv, conv_err, conv_rel_err = section_1_1b_convolution_theorem(f)
    ground_truth, recovered = section_1_1c_sinusoid_recovery()

    (RESULTS / "1_1_summary.json").write_text(json.dumps({
        "dft_validation_max_abs_err": float(max_err),
        "dft_validation_rel_err": float(rel_err),
        "conjugate_symmetry_max_err": float(sym_err),
        "shift_theorem_rows": [
            {"x0": x0, "y0": y0, "max_abs_err": float(e), "rel_err": float(re)}
            for (x0, y0, e, re) in shift_rows
        ],
        "convolution_theorem_max_abs_err": float(conv_err),
        "convolution_theorem_rel_err": float(conv_rel_err),
    }, indent=2))

    log("=== 1.2 ===")
    section_1_2(f)

    log("=== 1.3 ===")
    section_1_3(f)

    log(f"All results written to {RESULTS}")


if __name__ == "__main__":
    main()
