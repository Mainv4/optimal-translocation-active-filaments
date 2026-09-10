

import argparse
import os
import re

import matplotlib.pyplot as plt
import MDAnalysis as mda
import numpy as np
import numpy.fft as fft
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from tqdm import tqdm


def determine_and_validate_N(u, N_given=None, chain_length=40):
    total_atoms = len(u.atoms)

    if total_atoms % chain_length != 0:
        raise ValueError(
            f"Total number of atoms ({total_atoms}) is not divisible by chain length ({chain_length})"
        )

    N_calculated = total_atoms // chain_length

    if N_given is not None:
        if N_given != N_calculated:
            raise ValueError(
                f"Given number of polymers ({N_given}) does not match calculated value ({N_calculated}) "
                f"based on total atoms ({total_atoms}) and chain length ({chain_length})"
            )
        print(
            f"Validated: {N_given} polymers with {chain_length} atoms each (total: {total_atoms} atoms)"
        )
        return N_given
    else:
        print(
            f"Automatically determined: {N_calculated} polymers with {chain_length} atoms each (total: {total_atoms} atoms)"
        )
        return N_calculated


def set_plot_style():
    plt.style.use("ggplot")
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.size"] = 25
    plt.rcParams["legend.fontsize"] = 20
    plt.rcParams["axes.labelsize"] = 35
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["legend.labelspacing"] = 0.1
    plt.rcParams["legend.handlelength"] = 0.7
    plt.rcParams["legend.handletextpad"] = 0.25
    plt.rcParams["legend.borderpad"] = 0.1
    plt.rcParams["legend.borderaxespad"] = 0.5
    plt.rcParams["legend.columnspacing"] = 0.5
    plt.rcParams["xtick.labelsize"] = 35
    plt.rcParams["ytick.labelsize"] = 35
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["xtick.major.width"] = 2
    plt.rcParams["ytick.major.width"] = 2
    plt.rcParams["xtick.minor.width"] = 2
    plt.rcParams["ytick.minor.width"] = 2
    plt.rcParams["xtick.major.size"] = 10
    plt.rcParams["ytick.major.size"] = 10
    plt.rcParams["xtick.minor.size"] = 5
    plt.rcParams["ytick.minor.size"] = 5
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["axes.labelcolor"] = "black"
    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{bm}\usepackage{amsmath}",
        }
    )


def extract_parameters_from_path(path):
    path_parts = path.rstrip("/").split("/")
    param_dir = None

    for part in path_parts:
        if "Pe_" in part and "_T_" in part and "_k_" in part:
            param_dir = part
            break

    if param_dir is None:
        return None, None, None

    pattern = r"Pe_([0-9.]+)_T_([0-9.]+)_k_([0-9.]+)"
    match = re.search(pattern, param_dir)

    if match:
        pe = float(match.group(1))
        t = float(match.group(2))
        k = float(match.group(3))
        return pe, t, k
    else:
        return None, None, None


def read_trajectory(path, name_of_trajectory_file, name_of_topology_file=None):
    print("Reading the trajectory...")
    if name_of_topology_file:
        u = mda.Universe(
            path + "/" + name_of_topology_file,
            path + "/" + name_of_trajectory_file,
            topology_format="DATA",
            format="LAMMPSDUMP",
        )
    else:
        u = mda.Universe(
            path + "/" + name_of_trajectory_file,
            format="LAMMPSDUMP",
        )
    print("Trajectory read")
    return u


def exponential_decay(x, a, b):
    return a * np.exp(-x / b)


def oscillatory_exp(x, A, lp, q, phi):
    return A * np.exp(-x / lp) * np.cos(q * x + phi)


def linear_log_fit(x, neg_inv_lp):
    return neg_inv_lp * x


def calculate_r_squared(y_data, y_fit):
    mask = ~(np.isnan(y_data) | np.isnan(y_fit))
    if np.sum(mask) < 2:
        return np.nan

    y_data_clean = y_data[mask]
    y_fit_clean = y_fit[mask]

    ss_res = np.sum((y_data_clean - y_fit_clean) ** 2)
    ss_tot = np.sum((y_data_clean - np.mean(y_data_clean)) ** 2)

    if ss_tot == 0:
        return np.nan if ss_res != 0 else 1.0

    r_squared = 1 - (ss_res / ss_tot)
    return r_squared


def extract_parameter_uncertainty(popt, pcov, param_index):
    if pcov is None or np.any(np.isinf(pcov)) or param_index >= len(popt):
        return np.nan

    try:
        param_stderr = np.sqrt(np.diag(pcov))[param_index]
        return param_stderr
    except:
        return np.nan


def analyze_oscillations(x, y_data, L):
    try:
        dx = x[1] - x[0]
        freq = fft.fftfreq(len(x), dx)
        freq = freq[1 : len(freq) // 2]

        if y_data.ndim == 1:
            y_data = y_data[np.newaxis, :]

        n_curves = y_data.shape[0]
        power_spectra = np.zeros((n_curves, len(freq)))

        for i in range(n_curves):
            fft_vals = fft.fft(y_data[i])
            power_spectra[i] = np.abs(fft_vals[1 : len(freq) + 1]) ** 2

        mean_power = np.mean(power_spectra, axis=0)

        peaks, _ = find_peaks(mean_power, height=np.max(mean_power) / 10)
        dominant_freqs = freq[peaks] if len(peaks) > 0 else np.array([2 * np.pi / L])

        return freq, power_spectra, mean_power, dominant_freqs, peaks

    except Exception as e:
        print(f"Error in spectral analysis: {str(e)}")
        return None, None, None, None, None


def add_fit_parameters_text(
    ax, persistence_length, L, lp_osc=None, q=None, phi=None, r2_exp=None, r2_osc=None
):
    text = r"$\mathbf{Persistence\;lengths:}$" + "\n"
    text += r"$l_p^\mathrm{exp}/L = " + f"{persistence_length/L:.2f}$"
    if r2_exp is not None and not np.isnan(r2_exp):
        text += f" ($R^2 = {r2_exp:.3f}$)"

    if lp_osc is not None:
        text += r"$\newline l_p^\mathrm{osc}/L = " + f"{lp_osc/L:.2f}$"
        if r2_osc is not None and not np.isnan(r2_osc):
            text += f" ($R^2 = {r2_osc:.3f}$)"
        if q is not None:
            text += r"$\newline qL = " + f"{q*L:.2f}$"

    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=12,
        linespacing=1.5,
    )


def end_to_end_distance_sq(x, lp):
    return 3 * lp * x * (1 - 1.5 * (lp / x) * (1 - np.exp(-x / (1.5 * lp))))


def end_to_end_distance_sq_unnorm(x, lp):
    return 3 * lp * x * (1 - 1.5 * (lp / x) * (1 - np.exp(-x / (1.5 * lp))))


def plot_Rs_data(
    ax, x, y, Rs_data, n_frames, N, L, Lp=None, plot_type="linear", title=None
):
    for frame in tqdm(range(n_frames), desc="Plotting frames", leave=False):
        for polymer_idx in tqdm(
            range(N), desc=f"Frame {frame} - polymers", leave=False
        ):
            data = Rs_data[frame, polymer_idx] / (L * L)
            if plot_type == "loglog":
                mask = data > 0
                ax.loglog(x[mask], data[mask], color="blue", alpha=0.01, linewidth=0.1)
                ax.set_ylim(1e-4, 1)
            elif plot_type == "semilogx":
                ax.semilogx(x, data, color="blue", alpha=0.01, linewidth=0.1)
                ax.set_ylim(-0.01, 0.41)
            elif plot_type == "semilogy":
                mask = data > 0
                ax.semilogy(x, data[mask], color="blue", alpha=0.01, linewidth=0.1)
                ax.set_ylim(1e-4, 1)
            else:
                ax.plot(x, data, color="blue", alpha=0.01, linewidth=0.1)
                ax.set_ylim(-0.01, 0.41)

    if plot_type == "loglog":
        mask = y > 0
        ax.loglog(
            x[mask], y[mask], "k-", label=r"$\langle R^2(s) \rangle/L^2$", linewidth=2
        )
        ax.set_ylim(1e-4, 1)
    elif plot_type == "semilogx":
        ax.semilogx(x, y, "k-", label=r"$\langle R^2(s) \rangle/L^2$", linewidth=2)
        ax.set_ylim(-0.01, 0.41)
    elif plot_type == "semilogy":
        mask = y > 0
        ax.semilogy(
            x, y[mask], "k-", label=r"$\langle R^2(s) \rangle/L^2$", linewidth=2
        )
        ax.set_ylim(1e-4, 1)
    else:
        ax.plot(x, y, "k-", label=r"$\langle R^2(s) \rangle/L^2$", linewidth=2)
        ax.set_ylim(-0.01, 0.41)

    if Lp is not None:
        fit_curve = end_to_end_distance_sq(x, Lp)
        if plot_type == "loglog":
            ax.loglog(x, fit_curve, "r--", label=r"Fit", linewidth=2)
            ax.set_ylim(1e-4, 1)
        elif plot_type == "semilogx":
            ax.semilogx(x, fit_curve, "r--", label=r"Fit", linewidth=2)
            ax.set_ylim(-0.01, 0.41)
        elif plot_type == "semilogy":
            ax.semilogy(x, fit_curve, "r--", label=r"Fit", linewidth=2)
            ax.set_ylim(1e-4, 1)
        else:
            ax.plot(x, fit_curve, "r--", label=r"Fit", linewidth=2)
            ax.set_ylim(-0.01, 0.41)

    ax.set_xlabel(r"$s/L$")
    ax.set_ylabel(r"$\langle R^2(s) \rangle/L^2$")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=10)

    if Lp is not None:
        text = r"$\mathbf{Persistence\;length:}$" + "\n"
        text += r"$l_p/L = " + f"{Lp:.2f}$"
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=12,
        )


def calculate_Rs_correlation(u, path, N, L, has_topology=True):
    path_array = path.split("/")[:-1]
    output_base = "__".join(path_array[:])

    if output_base.startswith("FREE_SPACE__"):
        output_base = output_base[len("FREE_SPACE__") :]

    if output_base.startswith("ACTIVE__"):
        parts = output_base.split("__")
        if parts[0] == "ACTIVE":
            output_base = "__".join(parts[1:])
        elif len(parts) > 1 and parts[1].startswith("T"):
            output_base = "__".join(parts[2:])

    n_frames = len(u.trajectory) - 100
    n_segments = L - 1
    Rs_data = np.zeros((n_frames, N, n_segments))

    for ts in tqdm(u.trajectory[100:], desc="Computing R(s) - frames"):
        frame_idx = ts.frame - 100
        for polymer_idx in tqdm(
            range(N), desc=f"Frame {frame_idx+100} - polymers", leave=False
        ):
            start_idx = polymer_idx * L
            end_idx = (polymer_idx + 1) * L
            selection = u.atoms[start_idx:end_idx]
            positions = selection.positions[:, :2]

            for s in tqdm(
                range(1, L), desc=f"Polymer {polymer_idx} - segments", leave=False
            ):
                distances = positions[s:] - positions[:-s]
                Rs_data[frame_idx, polymer_idx, s - 1] = np.mean(
                    np.sum(distances**2, axis=1)
                )

    avg_Rs = np.mean(Rs_data, axis=(0, 1)) / (L * L)
    contour_length = np.arange(1, L)
    contour_length_normalized = contour_length / L

    try:
        popt, pcov = curve_fit(
            end_to_end_distance_sq, contour_length_normalized, avg_Rs, p0=[1 / 3]
        )
        Lp_normalized = popt[0]

        fit_data_Rs = end_to_end_distance_sq(contour_length_normalized, *popt)
        r2_Rs = calculate_r_squared(avg_Rs, fit_data_Rs)

        stderr_Rs = extract_parameter_uncertainty(popt, pcov, 0)

    except Exception as e:
        print(f"R(s) fitting failed: {str(e)}")
        Lp_normalized = float("nan")
        r2_Rs = float("nan")
        stderr_Rs = float("nan")

    output_file = f"DATA_TANGENT/correlation/{output_base}__Rs_correlation.dat"
    with open(output_file, "w") as f:
        f.write("# s/L R(s)^2/L^2\n")
        for i in range(len(avg_Rs)):
            f.write(f"{contour_length_normalized[i]:.6f} {avg_Rs[i]:.6f}\n")

    set_plot_style()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16))

    plot_Rs_data(
        ax1,
        contour_length_normalized,
        avg_Rs,
        Rs_data,
        n_frames,
        N,
        L,
        Lp_normalized,
        plot_type="linear",
        title="Linear scale",
    )

    plot_Rs_data(
        ax2,
        contour_length_normalized,
        avg_Rs,
        Rs_data,
        n_frames,
        N,
        L,
        Lp_normalized,
        plot_type="semilogx",
        title="Semi-log (x)",
    )

    plot_Rs_data(
        ax3,
        contour_length_normalized,
        avg_Rs,
        Rs_data,
        n_frames,
        N,
        L,
        Lp_normalized,
        plot_type="semilogy",
        title="Semi-log (y)",
    )

    plot_Rs_data(
        ax4,
        contour_length_normalized,
        avg_Rs,
        Rs_data,
        n_frames,
        N,
        L,
        Lp_normalized,
        plot_type="loglog",
        title="Log-log",
    )

    plt.tight_layout()
    plt.savefig(f"FIGURES/TANGENT_CORRELATION/{output_base}__Rs_correlation.png")
    plt.close()

    print(f"R(s) fit persistence length/L: {Lp_normalized:.2f}")

    try:
        popt_unnorm, pcov_unnorm = curve_fit(
            end_to_end_distance_sq_unnorm,
            contour_length,
            np.mean(Rs_data, axis=(0, 1)),
            p0=[L / 3],
        )
        Lp_unnorm = popt_unnorm[0]

        fit_data_Rs_unnorm = end_to_end_distance_sq_unnorm(contour_length, *popt_unnorm)
        r2_Rs_unnorm = calculate_r_squared(
            np.mean(Rs_data, axis=(0, 1)), fit_data_Rs_unnorm
        )

        stderr_Rs_unnorm = extract_parameter_uncertainty(popt_unnorm, pcov_unnorm, 0)

    except Exception as e:
        print(f"Unnormalized R(s) fitting failed: {str(e)}")
        Lp_unnorm = float("nan")
        r2_Rs_unnorm = float("nan")
        stderr_Rs_unnorm = float("nan")

    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    unnorm_Rs_data = np.mean(Rs_data, axis=(0, 1))
    for frame in tqdm(range(n_frames), desc="Plotting unnormalized R(s)", leave=False):
        for polymer_idx in tqdm(
            range(N), desc=f"Frame {frame} - polymers", leave=False
        ):
            ax.plot(
                contour_length,
                Rs_data[frame, polymer_idx],
                color="blue",
                alpha=0.01,
                linewidth=0.1,
            )

    ax.plot(
        contour_length,
        unnorm_Rs_data,
        "k-",
        label=r"$\langle R^2(s) \rangle$",
        linewidth=2,
    )

    if not np.isnan(Lp_unnorm):
        fit_curve = end_to_end_distance_sq_unnorm(contour_length, Lp_unnorm)
        ax.plot(contour_length, fit_curve, "r--", label=r"Fit", linewidth=2)

        text = r"$\mathbf{Persistence\;length:}$" + "\n"
        text += r"$l_p = " + f"{Lp_unnorm:.2f}$"
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            fontsize=12,
        )

    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$\langle R^2(s) \rangle$")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f"FIGURES/TANGENT_CORRELATION/{output_base}__Rs_correlation_unnorm.png")
    plt.close()

    print(f"Unnormalized R(s) fit persistence length: {Lp_unnorm:.2f}")

    return Lp_normalized, Lp_unnorm, r2_Rs, stderr_Rs, r2_Rs_unnorm, stderr_Rs_unnorm


def plot_single_polymer_analysis(
    positions,
    correlation,
    Rs_data,
    contour_length_normalized,
    L,
    frame_idx,
    polymer_idx,
):
    fig = plt.figure(figsize=(18, 6))

    ax1 = fig.add_subplot(131)
    ax1.plot(positions[:, 0], positions[:, 1], "b-", linewidth=4)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$y$")
    ax1.set_title(f"Configuration (frame {frame_idx}, polymer {polymer_idx})")

    ax2 = fig.add_subplot(132)
    ax2.plot(contour_length_normalized, correlation, "b.-", label="Correlation")
    ax2.set_xlabel(r"$s/L$")
    ax2.set_ylabel(r"$\cos \theta(s)$")
    ax2.set_title("Tangent correlation")

    ax3 = fig.add_subplot(133)
    s_values = np.arange(1, len(Rs_data) + 1) / L
    ax3.semilogy(s_values, Rs_data / (L * L), "r.-", label=r"$R^2(s)$")
    ax3.set_xlabel(r"$s/L$")
    ax3.set_ylabel(r"$R^2(s)/L^2$")
    ax3.set_title(r"$R^2(s)$")

    plt.tight_layout()
    return fig


def calculate_tangent_correlation(
    u,
    path,
    N,
    L,
    has_topology=True,
    analyze_individual=False,
    skip_rs=False,
    reduced_plotting=False,
    plot_sampling=10,
    max_plot_curves=1000,
):

    path_array = path.split("/")[:-1]
    output_base = "__".join(path_array[:])

    if output_base.startswith("FREE_SPACE__"):
        output_base = output_base[len("FREE_SPACE__") :]

    if output_base.startswith("ACTIVE__"):
        parts = output_base.split("__")
        if parts[0] == "ACTIVE":
            output_base = "__".join(parts[1:])
        elif len(parts) > 1 and parts[1].startswith("T"):
            output_base = "__".join(parts[2:])

    n_bonds = L - 1
    n_frames = len(u.trajectory) - 100

    correlation_sum = np.zeros(n_bonds, dtype=np.float32)
    correlation_count = np.zeros(n_bonds, dtype=np.float32)

    plot_data = []
    total_curves = n_frames * N
    sample_every = max(1, total_curves // max_plot_curves) if reduced_plotting else 1

    Lp_Rs_norm = Lp_Rs_unnorm = float("nan")

    print(f"Processing {n_frames} frames with {N} polymers each...")
    if reduced_plotting:
        print(
            f"Plotting mode: reduced (sampling every {sample_every} curves, max {max_plot_curves})"
        )

    curve_idx = 0
    for ts in tqdm(u.trajectory[100:], desc="Computing correlations - frames"):
        frame_idx = ts.frame - 100
        for polymer_idx in range(N):
            start_idx = polymer_idx * L
            end_idx = (polymer_idx + 1) * L
            selection = u.atoms[start_idx:end_idx]
            positions = selection.positions[:, :2]

            tangent_vectors = np.diff(positions, axis=0)
            tangent_vectors = (
                tangent_vectors / np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]
            )

            frame_correlation = np.zeros(n_bonds, dtype=np.float32)
            for i in range(n_bonds):
                correlation_sum_i = 0.0
                count_i = 0
                for j in range(n_bonds - i):
                    correlation = np.dot(tangent_vectors[j], tangent_vectors[j + i])
                    correlation_sum_i += correlation
                    count_i += 1
                if count_i > 0:
                    frame_correlation[i] = correlation_sum_i / count_i

            correlation_sum += frame_correlation
            correlation_count += 1

            if not reduced_plotting or (curve_idx % sample_every == 0):
                plot_data.append((frame_idx, polymer_idx, frame_correlation.copy()))

            curve_idx += 1

    avg_correlation = correlation_sum / correlation_count

    contour_length = np.arange(n_bonds)
    contour_length_normalized = contour_length / L

    persistence_length = lp_osc = lp_linear_log = q = phi = float("nan")
    r2_exp = r2_osc = r2_linear_log = float("nan")
    stderr_exp = stderr_osc = stderr_linear_log = float("nan")
    freq = power = dom_freqs = peaks = None

    if np.any(avg_correlation <= 0):
        zero_crossings = np.where(avg_correlation <= 0)[0]
        s_star_idx = zero_crossings[0]
        s_star = contour_length[s_star_idx]
        s_star_normalized = s_star / L
    else:
        s_star_idx = int(L / 4)
        s_star = s_star_idx
        s_star_normalized = s_star / L
        print(
            f"No zero crossings found in correlation function. Using fallback s* = L/4 = {s_star}"
        )

    fit_range_mask = contour_length <= s_star

    try:
        popt, pcov = curve_fit(
            exponential_decay,
            contour_length[fit_range_mask],
            avg_correlation[fit_range_mask],
            p0=[1.0, L / 3],
        )
        persistence_length = popt[1]

        fit_data_exp = exponential_decay(contour_length[fit_range_mask], *popt)
        r2_exp = calculate_r_squared(avg_correlation[fit_range_mask], fit_data_exp)

        stderr_exp = extract_parameter_uncertainty(popt, pcov, 1)

        log_mask = (avg_correlation > 0) & fit_range_mask
        if np.sum(log_mask) > 2:
            log_corr = np.log(avg_correlation[log_mask])
            contour_log_fit = contour_length[log_mask]
            try:
                popt_log, pcov_log = curve_fit(
                    linear_log_fit,
                    contour_log_fit,
                    log_corr,
                    p0=[-1.0 / persistence_length],
                )
                lp_linear_log = -1.0 / popt_log[0]

                fit_data_log = linear_log_fit(contour_log_fit, *popt_log)
                r2_linear_log = calculate_r_squared(log_corr, fit_data_log)

                slope_stderr = extract_parameter_uncertainty(popt_log, pcov_log, 0)
                if not np.isnan(slope_stderr) and popt_log[0] != 0:
                    stderr_linear_log = (
                        abs(lp_linear_log) ** 2 * slope_stderr / abs(popt_log[0])
                    )
                else:
                    stderr_linear_log = np.nan

            except Exception as e:
                print(f"Linear log fit failed: {str(e)}")
                lp_linear_log = float("nan")
        else:
            print("Not enough positive correlation values for linear log fit")
            lp_linear_log = float("nan")

        spectral_results = analyze_oscillations(contour_length, avg_correlation, L)
        if spectral_results[0] is not None:
            freq, power_spectra, mean_power, dom_freqs, peaks = spectral_results

            if len(dom_freqs) > 0:
                q_init = dom_freqs[0]
                p0 = [
                    1.0,
                    persistence_length,
                    q_init,
                    0,
                ]
                try:
                    popt_osc, pcov_osc = curve_fit(
                        oscillatory_exp, contour_length, avg_correlation, p0=p0
                    )
                    A, lp_osc, q, phi = popt_osc

                    fit_data_osc = oscillatory_exp(contour_length, *popt_osc)
                    r2_osc = calculate_r_squared(avg_correlation, fit_data_osc)

                    stderr_osc = extract_parameter_uncertainty(popt_osc, pcov_osc, 1)

                except Exception as e:
                    print(f"Oscillatory fit failed: {str(e)}")
                    lp_osc = q = phi = float("nan")
            else:
                print("No dominant frequencies found for oscillatory fit")
                lp_osc = q = phi = float("nan")
        else:
            print("Spectral analysis failed")
            freq = power_spectra = mean_power = dom_freqs = peaks = None
            lp_osc = q = phi = float("nan")

    except Exception as e:
        print(f"Exponential fitting failed: {str(e)}")
        persistence_length = lp_osc = lp_linear_log = q = phi = float("nan")
        r2_exp = r2_osc = r2_linear_log = float("nan")
        stderr_exp = stderr_osc = stderr_linear_log = float("nan")
        freq = power_spectra = mean_power = dom_freqs = peaks = None

    full_length = np.arange(n_bonds)
    if not np.isnan(persistence_length):
        normalized_fit = exponential_decay(full_length, *popt)

    output_file = f"DATA_TANGENT/correlation/{output_base}__tangent_correlation.dat"
    with open(output_file, "w") as f:
        f.write(f"# Contour_length/L Correlation (Fit range: 0 to {s_star:.1f})\n")
        for i in range(len(avg_correlation)):
            f.write(f"{contour_length_normalized[i]:.6f} {avg_correlation[i]:.6f}\n")

    if freq is not None and mean_power is not None:
        spectral_file = f"DATA_TANGENT/correlation/{output_base}__spectral.dat"
        with open(spectral_file, "w") as f:
            f.write("# qL Power\n")
            for i in range(len(freq)):
                f.write(f"{freq[i]*L:.6f} {mean_power[i]:.6f}\n")

    set_plot_style()

    fig, ax = plt.subplots(figsize=(8, 6))
    print(f"\nPlotting linear correlation plot with {len(plot_data)} curves...")
    for frame_idx, polymer_idx, frame_correlation in tqdm(
        plot_data, desc="Plotting individual curves"
    ):
        ax.plot(
            contour_length_normalized,
            frame_correlation,
            color="blue",
            alpha=0.01,
            linewidth=0.1,
        )

    ax.plot(
        contour_length_normalized,
        avg_correlation,
        "k-",
        label=r"$\langle \cos \theta \rangle$",
        alpha=1,
        linewidth=2,
    )

    if not np.isnan(persistence_length):
        ax.plot(contour_length_normalized, normalized_fit, "r--", label=r"$\text{Exp}$")

    if not np.isnan(lp_osc):
        osc_fit = oscillatory_exp(contour_length, *popt_osc)
        ax.plot(contour_length_normalized, osc_fit, "m--", label=r"$\text{Osc}$")

    ax.axvline(
        x=s_star_normalized,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label="Fit range (s*)",
    )
    ax.set_xlabel(r"$s/L$")
    ax.set_ylabel(r"$\langle \cos \theta(s) \rangle$")
    ax.legend(fontsize=10)
    add_fit_parameters_text(ax, persistence_length, L, lp_osc, q, phi, r2_exp, r2_osc)
    plt.tight_layout()
    plt.savefig(
        f"FIGURES/TANGENT_CORRELATION/{output_base}__tangent_correlation_lin.png"
    )
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    print(f"\nPlotting semi-log correlation plot with {len(plot_data)} curves...")
    for frame_idx, polymer_idx, frame_correlation in tqdm(
        plot_data, desc="Plotting semi-log curves"
    ):
        ax1.semilogx(
            contour_length_normalized,
            frame_correlation,
            color="blue",
            alpha=0.01,
            linewidth=0.1,
        )

    ax1.semilogx(
        contour_length_normalized,
        avg_correlation,
        "k-",
        label=r"$\langle \cos \theta \rangle$",
        alpha=1,
        linewidth=2,
    )

    if not np.isnan(persistence_length):
        ax1.semilogx(
            contour_length_normalized, normalized_fit, "r--", label=r"$\text{Exp}$"
        )

    if not np.isnan(lp_osc):
        osc_fit = oscillatory_exp(contour_length, *popt_osc)
        ax1.semilogx(contour_length_normalized, osc_fit, "m--", label=r"$\text{Osc}$")

    ax1.axvline(
        x=s_star_normalized,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label="Fit range (s*)",
    )
    ax1.set_xlabel(r"$s/L$")
    ax1.set_ylabel(r"$\langle \cos \theta \rangle$")
    ax1.legend(fontsize=10)
    ax1.set_title("Semi-log")
    add_fit_parameters_text(ax1, persistence_length, L, lp_osc, q, phi, r2_exp, r2_osc)

    print(f"\nPlotting log-log correlation plot with {len(plot_data)} curves...")
    for frame_idx, polymer_idx, frame_correlation in tqdm(
        plot_data, desc="Plotting log-log curves"
    ):
        mask = frame_correlation > 0
        if np.any(mask):
            ax2.loglog(
                contour_length_normalized[mask],
                frame_correlation[mask],
                color="blue",
                alpha=0.01,
                linewidth=0.1,
            )

    mask = avg_correlation > 0
    ax2.loglog(
        contour_length_normalized[mask],
        avg_correlation[mask],
        "k-",
        label=r"$\langle \cos \theta \rangle$",
        alpha=1,
        linewidth=2,
    )

    if not np.isnan(persistence_length):
        fit_mask = normalized_fit > 0
        ax2.loglog(
            contour_length_normalized[fit_mask],
            normalized_fit[fit_mask],
            "r--",
            label=r"$\text{Exp}$",
        )

    if not np.isnan(lp_osc):
        fit_mask = osc_fit > 0
        ax2.loglog(
            contour_length_normalized[fit_mask],
            osc_fit[fit_mask],
            "m--",
            label=r"$\text{Osc}$",
        )

    ax2.axvline(
        x=s_star_normalized,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label="Fit range (s*)",
    )
    ax2.set_xlabel(r"$s/L$")
    ax2.set_ylabel(r"$\langle \cos \theta \rangle$")
    ax2.legend(fontsize=10)
    ax2.set_title("Log-log")
    add_fit_parameters_text(ax2, persistence_length, L, lp_osc, q, phi, r2_exp, r2_osc)

    plt.tight_layout()
    plt.savefig(
        f"FIGURES/TANGENT_CORRELATION/{output_base}__tangent_correlation_log.png"
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    print(f"\nPlotting semi-log y correlation plot with {len(plot_data)} curves...")
    for frame_idx, polymer_idx, frame_correlation in tqdm(
        plot_data, desc="Plotting semi-log y curves"
    ):
        mask = frame_correlation > 0
        if np.any(mask):
            ax.semilogy(
                contour_length_normalized[mask],
                frame_correlation[mask],
                color="blue",
                alpha=0.01,
                linewidth=0.1,
            )

    mask = avg_correlation > 0
    ax.semilogy(
        contour_length_normalized[mask],
        avg_correlation[mask],
        "k-",
        label=r"$\langle \cos \theta \rangle$",
        alpha=1,
        linewidth=2,
    )

    if not np.isnan(persistence_length):
        fit_mask = normalized_fit > 0
        ax.semilogy(
            contour_length_normalized[fit_mask],
            normalized_fit[fit_mask],
            "r--",
            label=r"$\text{Exp}$",
        )

    if not np.isnan(lp_osc):
        fit_mask = osc_fit > 0
        ax.semilogy(
            contour_length_normalized[fit_mask],
            osc_fit[fit_mask],
            "m--",
            label=r"$\text{Osc}$",
        )

    ax.axvline(
        x=s_star_normalized,
        color="gray",
        linestyle=":",
        alpha=0.5,
        label="Fit range (s*)",
    )
    ax.set_xlabel(r"$s/L$")
    ax.set_ylabel(r"$\langle \cos \theta \rangle$")
    ax.legend(fontsize=10)
    add_fit_parameters_text(ax, persistence_length, L, lp_osc, q, phi, r2_exp, r2_osc)
    plt.tight_layout()
    plt.savefig(
        f"FIGURES/TANGENT_CORRELATION/{output_base}__tangent_correlation_semilogy.png"
    )
    plt.close()

    if freq is not None and mean_power is not None:
        plt.figure(figsize=(8, 6))

        print(f"\nPlotting spectral analysis with {len(plot_data)} curves...")
        successful_spectra = 0
        for frame_idx, polymer_idx, frame_correlation in tqdm(
            plot_data, desc="Plotting spectra"
        ):
            f, ps, _, _, _ = analyze_oscillations(contour_length, frame_correlation, L)
            if f is not None and ps is not None:
                plt.plot(f * L, ps[0], color="blue", alpha=0.01, linewidth=0.1)
                successful_spectra += 1

        if successful_spectra > 0:
            plt.plot(freq * L, mean_power, "k-", label="Mean", linewidth=2)

            if peaks is not None and len(peaks) > 0:
                plt.plot(dom_freqs * L, mean_power[peaks], "ro", label="Peaks")

            plt.xlabel(r"$qL$")
            plt.ylabel(r"Power")
            plt.yscale("log")
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.savefig(f"FIGURES/TANGENT_CORRELATION/{output_base}__spectrum.png")
        else:
            print("No successful spectral analyses to plot")
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))

    mask = avg_correlation > 0
    log_correlation = np.log(avg_correlation[mask])
    s_over_L_masked = contour_length_normalized[mask]

    log_cos_over_s = log_correlation / (
        s_over_L_masked * L
    )

    ax.plot(
        s_over_L_masked,
        log_cos_over_s,
        "bo-",
        label=r"$\log \langle \cos \theta(s) \rangle / s$",
    )

    target_idx = np.abs(s_over_L_masked - 0.5).argmin()
    s_over_L_at_point = s_over_L_masked[target_idx]
    log_cos_over_s_at_point = log_cos_over_s[target_idx]

    lp_log_over_s = -1.0 / log_cos_over_s_at_point
    lp_log_over_s_L = lp_log_over_s / L

    ax.plot(
        s_over_L_at_point,
        log_cos_over_s_at_point,
        "ro",
        markersize=10,
        label=f"Value at s/L = {s_over_L_at_point:.3f}",
    )

    text = r"$\mathbf{Persistence\;length\;(log/s):}$" + "\n"
    text += r"$l_p/L = " + f"{lp_log_over_s_L:.3f}$" + "\n"
    text += r"$\mathrm{at}\;s/L = " + f"{s_over_L_at_point:.3f}$"

    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=12,
    )

    print(
        f"log/s analysis persistence length/L at s/L={s_over_L_at_point:.3f}: {lp_log_over_s_L:.3f}"
    )

    ax.set_xlabel(r"$s/L$")
    ax.set_ylabel(r"$\log \langle \cos \theta(s) \rangle / s$")
    ax.set_title("Log over s analysis")
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(
        f"FIGURES/TANGENT_CORRELATION/{output_base}__tangent_correlation_log_over_s.png"
    )
    plt.close()

    if not np.isnan(lp_linear_log):
        fig, ax = plt.subplots(figsize=(8, 6))

        log_mask = (avg_correlation > 0) & fit_range_mask

        if np.sum(log_mask) > 0:
            ax.semilogy(
                contour_length_normalized[log_mask],
                avg_correlation[log_mask],
                "ko-",
                label=r"$\langle \cos \theta \rangle$ (0 to s*)",
                linewidth=2,
                markersize=4,
            )

            log_fit_curve = np.exp(
                linear_log_fit(contour_length[log_mask], popt_log[0])
            )
            ax.semilogy(
                contour_length_normalized[log_mask],
                log_fit_curve,
                "g--",
                label=r"Linear log fit",
                linewidth=2,
            )

            ax.axvline(
                x=s_star_normalized,
                color="gray",
                linestyle=":",
                alpha=0.5,
                label=f"s* = {s_star_normalized:.2f}L",
            )

            text = r"$\mathbf{Linear\;log\;fit:}$" + "\n"
            text += r"$\log C(s) = -s/l_p$" + "\n"
            text += r"$l_p/L = " + f"{lp_linear_log/L:.2f}$" + "\n"
            text += r"$\mathrm{Fit\;range:}\;0 \to s*$"

            ax.text(
                0.98,
                0.98,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                fontsize=12,
            )

            ax.set_xlabel(r"$s/L$")
            ax.set_ylabel(r"$\langle \cos \theta(s) \rangle$")
            ax.set_title("Linear log fit (0 to s*)")
            ax.legend(fontsize=10)
            ax.set_xlim(0, s_star_normalized * 1.1)

            plt.tight_layout()
            plt.savefig(
                f"FIGURES/TANGENT_CORRELATION/{output_base}__tangent_correlation_linear_log.png"
            )
            plt.close()

            print(f"Linear log fit persistence length/L: {lp_linear_log/L:.2f}")

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        contour_length_normalized,
        avg_correlation,
        "ko-",
        label=r"$\langle \cos \theta \rangle$",
        linewidth=1,
        markersize=4,
    )

    if not np.isnan(persistence_length):
        normalized_fit = exponential_decay(contour_length, *popt)
        ax.plot(
            contour_length_normalized,
            normalized_fit,
            "r-",
            label=r"Exp. fit (s*)",
            linewidth=2,
        )

    if not np.isnan(lp_linear_log):
        log_mask = (avg_correlation > 0) & fit_range_mask
        if np.sum(log_mask) > 0:
            linear_log_fit_curve = np.exp(
                linear_log_fit(contour_length[log_mask], popt_log[0])
            )
            ax.plot(
                contour_length_normalized[log_mask],
                linear_log_fit_curve,
                "g--",
                label=r"Linear log fit (s*)",
                linewidth=2,
            )

    if not np.isnan(lp_osc):
        osc_fit = oscillatory_exp(contour_length, *popt_osc)
        ax.plot(
            contour_length_normalized, osc_fit, "m-.", label=r"Osc. fit", linewidth=2
        )

    ax.axvline(
        x=s_star_normalized, color="red", linestyle=":", alpha=0.5, label=r"$s*$"
    )

    ax.set_xlabel(r"$s/L$")
    ax.set_ylabel(r"$\langle \cos \theta(s) \rangle$")
    ax.set_title("Comparison of fitting methods")
    ax.legend(fontsize=10, loc="best")

    all_lp_text = r"$\mathbf{Persistence\;lengths\;(l_p/L):}$" + "\n"
    all_lp_text += (
        r"$\mathrm{Exp.\;fit\;(s*)} = " + f"{persistence_length/L:.2f}$" + "\n"
    )
    if not np.isnan(lp_linear_log):
        all_lp_text += (
            r"$\mathrm{Linear\;log\;fit\;(s*)} = " + f"{lp_linear_log/L:.2f}$" + "\n"
        )
    if not np.isnan(lp_log_over_s):
        all_lp_text += r"$\mathrm{Log/s\;limit} = " + f"{lp_log_over_s/L:.2f}$" + "\n"
    if not np.isnan(lp_osc):
        all_lp_text += r"$\mathrm{Osc.\;fit} = " + f"{lp_osc/L:.2f}$" + "\n"
    if not np.isnan(Lp_Rs_norm):
        all_lp_text += r"$\mathrm{R(s)\;analysis} = " + f"{Lp_Rs_norm:.2f}$"

    ax.text(
        0.98,
        0.02,
        all_lp_text,
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(
        f"FIGURES/TANGENT_CORRELATION/{output_base}__tangent_correlation_methods_comparison.png"
    )
    plt.close()

    print(f"First zero-crossing at s* = {s_star_normalized:.2f}L")
    print(
        f"Exponential fit persistence length/L (fit range [0,s*]): {persistence_length/L:.2f}"
    )
    if not np.isnan(lp_linear_log):
        print(
            f"Linear log fit persistence length/L (fit range [0,s*]): {lp_linear_log/L:.2f}"
        )
    print(f"Oscillatory fit persistence length/L: {lp_osc/L:.2f}")
    if freq is not None and len(dom_freqs) > 0:
        print(f"Dominant oscillation wavevector (qL): {q*L:.2f}")

    if not skip_rs:
        print("\nPerforming R(s) analysis...")
        Lp_Rs_norm, Lp_Rs_unnorm, r2_Rs, stderr_Rs, r2_Rs_unnorm, stderr_Rs_unnorm = (
            calculate_Rs_correlation(u, path, N, L, has_topology)
        )
    else:
        print("\nSkipping R(s) analysis (--skip-rs flag set)")
        Lp_Rs_norm = Lp_Rs_unnorm = r2_Rs = stderr_Rs = r2_Rs_unnorm = (
            stderr_Rs_unnorm
        ) = float("nan")
    print(f"R(s) fit persistence length/L: {Lp_Rs_norm:.2f}")
    print(f"R(s) fit persistence length: {Lp_Rs_unnorm:.2f}")

    if analyze_individual:
        individual_dir = (
            f"FIGURES/TANGENT_CORRELATION/{output_base}/individual_polymers"
        )
        os.makedirs(individual_dir, exist_ok=True)

        n_samples = min(10, n_frames * N)
        random_indices = np.random.choice(n_frames * N, n_samples, replace=False)
        random_frames = random_indices // N
        random_polymers = random_indices % N

        for i, (frame_idx, polymer_idx) in enumerate(
            tqdm(
                zip(random_frames, random_polymers),
                desc="Analyzing individual polymers",
                total=len(random_frames),
            )
        ):
            ts = u.trajectory[frame_idx + 100]
            start_idx = polymer_idx * L
            end_idx = (polymer_idx + 1) * L
            selection = u.atoms[start_idx:end_idx]
            positions = selection.positions[:, :2]

            tangent_vectors = np.diff(positions, axis=0)
            tangent_vectors = (
                tangent_vectors / np.linalg.norm(tangent_vectors, axis=1)[:, np.newaxis]
            )

            single_correlation = np.zeros(n_bonds)
            for j in tqdm(
                range(n_bonds),
                desc=f"Computing correlations for polymer {polymer_idx}",
                leave=False,
            ):
                for k in range(n_bonds - j):
                    single_correlation[j] += np.dot(
                        tangent_vectors[k], tangent_vectors[k + j]
                    )

            single_Rs = np.zeros(n_bonds)
            for s in tqdm(
                range(1, L),
                desc=f"Computing R(s) for polymer {polymer_idx}",
                leave=False,
            ):
                distances = positions[s:] - positions[:-s]
                single_Rs[s - 1] = np.mean(np.sum(distances**2, axis=1))

            fig = plot_single_polymer_analysis(
                positions,
                single_correlation,
                single_Rs,
                contour_length_normalized,
                L,
                frame_idx + 100,
                polymer_idx,
            )

            plt.savefig(
                f"{individual_dir}/polymer_{i+1}_frame_{frame_idx+100}_id_{polymer_idx}.png"
            )
            plt.close()

    return (
        persistence_length / L,
        lp_osc / L,
        q,
        (Lp_Rs_norm, Lp_Rs_unnorm),
        lp_log_over_s / L,
        lp_linear_log / L,
        r2_exp,
        r2_osc,
        r2_linear_log,
        r2_Rs,
        stderr_exp / L,
        stderr_osc / L,
        stderr_linear_log / L,
        stderr_Rs,
    )


def main():
    os.makedirs("DATA_TANGENT/correlation", exist_ok=True)
    os.makedirs("DATA_TANGENT/plots", exist_ok=True)
    os.makedirs("FIGURES/TANGENT_CORRELATION", exist_ok=True)

    parser = argparse.ArgumentParser(
        description="Calculate tangent vectors correlation and persistence length."
    )
    parser.add_argument(
        "--path",
        help="Path to the directory containing the trajectory files",
        required=True,
    )
    parser.add_argument("--topology", help="Name of the topology file", required=False)
    parser.add_argument(
        "--trajectory", help="Name of the trajectory file", required=True
    )
    parser.add_argument("-N", help="Number of polymers", required=False)
    parser.add_argument(
        "-L", help="Length of each polymer", required=False, default=40, type=int
    )
    parser.add_argument(
        "--analyze-individual",
        action="store_true",
        help="Analyze and plot 10 random individual polymers",
    )

    parser.add_argument(
        "--skip-rs", action="store_true", help="Skip R(s) analysis to save memory"
    )
    parser.add_argument(
        "--reduced-plotting",
        action="store_true",
        help="Use subsampled plotting to reduce memory usage",
    )
    parser.add_argument(
        "--plot-sampling",
        type=int,
        default=1000,
        help="Plot every Nth frame/polymer (default: 1000)",
    )
    parser.add_argument(
        "--max-plot-curves",
        type=int,
        default=100,
        help="Maximum number of individual curves to plot (default: 100)",
    )

    args = parser.parse_args()

    trajectory_path = args.path
    topology_file = args.topology.split("/")[-1] if args.topology else None
    name_of_trajectory_file = args.trajectory.split("/")[-1]
    L = args.L

    u = read_trajectory(trajectory_path, name_of_trajectory_file, topology_file)

    N = determine_and_validate_N(u, int(args.N) if args.N is not None else None, L)

    results = calculate_tangent_correlation(
        u,
        trajectory_path,
        N,
        L,
        has_topology=(topology_file is not None),
        analyze_individual=args.analyze_individual,
        skip_rs=args.skip_rs,
        reduced_plotting=args.reduced_plotting,
        plot_sampling=args.plot_sampling,
        max_plot_curves=args.max_plot_curves,
    )
    (
        Lp_exp,
        Lp_osc,
        q,
        (Lp_Rs_norm, Lp_Rs_unnorm),
        Lp_log_over_s,
        Lp_linear_log,
        r2_exp,
        r2_osc,
        r2_linear_log,
        r2_Rs,
        stderr_exp,
        stderr_osc,
        stderr_linear_log,
        stderr_Rs,
    ) = results

    print("\nSummary of persistence length calculations:")
    print(f"Exponential fit (s*) (l_p/L): {Lp_exp:.2f}")
    print(f"Linear log fit (s*) (l_p/L): {Lp_linear_log:.2f}")
    print(f"Log/s limit (l_p/L): {Lp_log_over_s:.2f}")
    print(f"Oscillatory fit (l_p/L): {Lp_osc:.2f}")
    print(f"R(s) analysis (l_p/L): {Lp_Rs_norm:.2f}")
    print(f"R(s) analysis (l_p): {Lp_Rs_unnorm:.2f}")

    pe, t, k = extract_parameters_from_path(trajectory_path)

    if pe is not None and t is not None and k is not None:
        csv_filename = f"DATA_LP/Pe_{pe}_T_{t}_k_{k}.csv"

        data = {
            "Pe": [pe],
            "T": [t],
            "k": [k],
            "l_p_exp": [Lp_exp * L],
            "l_p_linear_log": [Lp_linear_log * L],
            "l_p_osc": [Lp_osc * L],
            "q": [q if not np.isnan(q) else np.nan],
            "l_p_log_over_s": [Lp_log_over_s * L],
            "l_p_exp_normalized": [Lp_exp],
            "l_p_linear_log_normalized": [Lp_linear_log],
            "l_p_osc_normalized": [Lp_osc],
            "l_p_log_over_s_normalized": [Lp_log_over_s],
            "qL": [q * L if not np.isnan(q) else np.nan],
            "r2_exp": [r2_exp if not np.isnan(r2_exp) else np.nan],
            "r2_osc": [r2_osc if not np.isnan(r2_osc) else np.nan],
            "r2_linear_log": [r2_linear_log if not np.isnan(r2_linear_log) else np.nan],
            "r2_Rs": [r2_Rs if not np.isnan(r2_Rs) else np.nan],
            "stderr_exp_normalized": [
                stderr_exp if not np.isnan(stderr_exp) else np.nan
            ],
            "stderr_osc_normalized": [
                stderr_osc if not np.isnan(stderr_osc) else np.nan
            ],
            "stderr_linear_log_normalized": [
                stderr_linear_log if not np.isnan(stderr_linear_log) else np.nan
            ],
            "stderr_Rs_normalized": [stderr_Rs if not np.isnan(stderr_Rs) else np.nan],
            "stderr_exp": [stderr_exp * L if not np.isnan(stderr_exp) else np.nan],
            "stderr_osc": [stderr_osc * L if not np.isnan(stderr_osc) else np.nan],
            "stderr_linear_log": [
                stderr_linear_log * L if not np.isnan(stderr_linear_log) else np.nan
            ],
            "stderr_Rs": [stderr_Rs * L if not np.isnan(stderr_Rs) else np.nan],
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_filename, index=False)

        print(f"Results saved to {csv_filename}")
        print(f"Parameters: Pe={pe}, T={t}, k={k}")
        print(
            f"l_p_exp={Lp_exp * L:.3f}, l_p_linear_log={Lp_linear_log * L:.3f}, l_p_osc={Lp_osc * L:.3f}, l_p_log_over_s={Lp_log_over_s * L:.3f}, q={q:.3f}"
        )
    else:
        print("Could not extract parameters from path. CSV not saved.")


if __name__ == "__main__":
    main()
