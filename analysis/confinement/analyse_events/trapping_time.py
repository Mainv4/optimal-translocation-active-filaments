import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from analyse_events.utilities import (read_mass_center_head_tail_x,
                                      set_plot_style)


def calculate_trapping_times(x_positions, time, cavity_bounds):
    trapping_times = []
    start_times = []
    end_times = []
    current_trap_start = None

    for i in range(len(time)):
        is_in_cavity = cavity_bounds[0] <= x_positions[i] <= cavity_bounds[1]

        if is_in_cavity and current_trap_start is None:
            current_trap_start = time[i]
        elif not is_in_cavity and current_trap_start is not None:
            trap_duration = time[i] - current_trap_start
            trapping_times.append(trap_duration)
            start_times.append(current_trap_start)
            end_times.append(time[i])
            current_trap_start = None

    return np.array(trapping_times), np.array(start_times), np.array(end_times)


def merge_nearby_events(times, starts, ends, min_gap=0.1):
    if len(times) <= 1:
        return times, starts, ends

    merged_times = []
    merged_starts = []
    merged_ends = []

    current_start = starts[0]
    current_end = ends[0]

    for i in range(1, len(times)):
        gap = starts[i] - current_end
        if gap < min_gap:
            current_end = ends[i]
        else:
            merged_times.append(current_end - current_start)
            merged_starts.append(current_start)
            merged_ends.append(current_end)
            current_start = starts[i]
            current_end = ends[i]

    merged_times.append(current_end - current_start)
    merged_starts.append(current_start)
    merged_ends.append(current_end)

    return np.array(merged_times), np.array(merged_starts), np.array(merged_ends)


def exponential_decay(x, a, tau):
    return a * np.exp(-x / tau)


def linear_fit(x, a, b):
    return a * x + b


def save_trapping_times(all_trapping_times, polymer_events, path):
    data_dir = "DATA_trapping_time/"
    os.makedirs(data_dir, exist_ok=True)

    filename = path.replace("/", "__") + "__trapping_times.txt"
    filepath = os.path.join(data_dir, filename)

    with open(filepath, "w") as f:
        f.write("# Trapping times (minutes)\n")
        f.write("# Format: time,polymer_id,cavity\n")

        for polymer in polymer_events:
            polymer_id = polymer["polymer_id"]

            for i, duration in enumerate(polymer["times_left"]):
                f.write(f"{duration:.6f},{polymer_id},left\n")

            for i, duration in enumerate(polymer["times_right"]):
                f.write(f"{duration:.6f},{polymer_id},right\n")

        f.write(f"\n# Summary Statistics\n")
        f.write(f"# Total events: {len(all_trapping_times)}\n")
        f.write(f"# Mean time: {np.mean(all_trapping_times):.6f} min\n")
        f.write(f"# Median time: {np.median(all_trapping_times):.6f} min\n")
        f.write(f"# Max time: {np.max(all_trapping_times):.6f} min\n")

    print(f"Trapping times saved to {filepath}")


def plot_trapping_time_distribution(path, x1=None, x2=None):
    massCenter_data, _, _ = read_mass_center_head_tail_x(path)
    time, x_cm, _, N = massCenter_data

    dump_freq = 100000
    dt = 0.001
    tau = 1e-2
    time = time * dump_freq * dt * tau / 60

    x_cm = x_cm / 2 + 14.2 / 2

    left_cavity = (0, 8)
    right_cavity = (88, 96)

    all_trapping_times = []

    polymer_stats = []
    polymer_events = []

    for i in range(N):
        times_left, starts_left, ends_left = calculate_trapping_times(
            x_cm[:, i], time, left_cavity
        )
        times_right, starts_right, ends_right = calculate_trapping_times(
            x_cm[:, i], time, right_cavity
        )

        times_left, starts_left, ends_left = merge_nearby_events(
            times_left, starts_left, ends_left
        )
        times_right, starts_right, ends_right = merge_nearby_events(
            times_right, starts_right, ends_right
        )

        polymer_event_data = {
            "polymer_id": i + 1,
            "times_left": times_left,
            "times_right": times_right,
            "starts_left": starts_left,
            "starts_right": starts_right,
            "ends_left": ends_left,
            "ends_right": ends_right,
        }
        polymer_events.append(polymer_event_data)

        print(f"\n=== Detailed events for Polymer {i + 1} ===")
        print("\nLeft cavity events:")
        if len(times_left) > 0:
            for j, (duration, start, end) in enumerate(
                zip(times_left, starts_left, ends_left), 1
            ):
                print(
                    f"Event {j}: Duration = {duration:.2f} min (t = {start:.2f} to {end:.2f} min)"
                )
        else:
            print("No trapping events")

        print("\nRight cavity events:")
        if len(times_right) > 0:
            for j, (duration, start, end) in enumerate(
                zip(times_right, starts_right, ends_right), 1
            ):
                print(
                    f"Event {j}: Duration = {duration:.2f} min (t = {start:.2f} to {end:.2f} min)"
                )
        else:
            print("No trapping events")

        polymer_times = np.concatenate([times_left, times_right])
        all_trapping_times.extend(polymer_times)

        stats = {
            "polymer_id": i + 1,
            "n_events_left": len(times_left),
            "n_events_right": len(times_right),
            "mean_time": np.mean(polymer_times) if len(polymer_times) > 0 else 0,
            "median_time": np.median(polymer_times) if len(polymer_times) > 0 else 0,
            "max_time": np.max(polymer_times) if len(polymer_times) > 0 else 0,
        }
        polymer_stats.append(stats)

        print(f"\nSummary for Polymer {i + 1}:")
        print(f"Left cavity events: {stats['n_events_left']}")
        print(f"Right cavity events: {stats['n_events_right']}")
        print(f"Mean trapping time: {stats['mean_time']:.2f} min")
        print(f"Median trapping time: {stats['median_time']:.2f} min")
        print(f"Maximum trapping time: {stats['max_time']:.2f} min")
        print("=" * 50)

    save_trapping_times(all_trapping_times, polymer_events, path)

    set_plot_style()

    plt.hist(all_trapping_times, bins=30, density=True, alpha=0.7)
    plt.xlabel(r"$\tau$ (min)", fontsize=14)
    plt.ylabel(r"$P(\tau)$", fontsize=14)
    plt.grid(False)

    stats_text = f"N events: {len(all_trapping_times)}\n"
    stats_text += f"Mean: {np.mean(all_trapping_times):.2f} min\n"
    stats_text += f"Median: {np.median(all_trapping_times):.2f} min\n"
    stats_text += f"Max: {np.max(all_trapping_times):.2f} min"

    plt.text(
        0.95,
        0.95,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=12,
    )
    plt.xlim(-0.5, 15)

    print(f"\nOverall statistics:")
    print(f"Total number of trapping events: {len(all_trapping_times)}")
    print(f"Mean trapping time: {np.mean(all_trapping_times):.2f} min")
    print(
        f"Median trapping time: {np.median(all_trapping_times):.2f} min"
    )
    print(f"Maximum trapping time: {np.max(all_trapping_times):.2f} min")

    fig_lin = plt.figure(figsize=(8, 6))
    fig_log = plt.figure(figsize=(8, 6))
    fig_logfit = plt.figure(figsize=(8, 6))
    fig_linfit = plt.figure(figsize=(8, 6))

    fit_min_time = 0
    fit_max_time = 12
    BINS = 20

    base_path = "FIGURES/trapping_times/" + path + "/"
    os.makedirs(base_path, exist_ok=True)
    base_name = path.replace("/", "__") + "__TrappingTimeDistribution"

    counts, bin_edges = np.histogram(all_trapping_times, bins=BINS, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    mask_nonzero = counts > 0
    fit_centers = bin_centers[mask_nonzero]
    log_counts = np.log(counts[mask_nonzero])

    fit_mask = (fit_centers >= fit_min_time) & (fit_centers <= fit_max_time)

    if np.sum(fit_mask) > 2:
        try:
            exp_fit_mask = (bin_centers >= fit_min_time) & (bin_centers <= fit_max_time)
            fit_centers_exp = bin_centers[exp_fit_mask]
            fit_counts_exp = counts[exp_fit_mask]

            nonzero_mask_exp = fit_counts_exp > 0
            if np.sum(nonzero_mask_exp) > 2:
                popt_exp, pcov_exp = optimize.curve_fit(
                    exponential_decay,
                    fit_centers_exp[nonzero_mask_exp],
                    fit_counts_exp[nonzero_mask_exp],
                    p0=[max(counts), 1.0],
                )

                x_fit_exp = np.linspace(0, 15, 100)
                y_fit_exp = exponential_decay(x_fit_exp, *popt_exp)

                plt.figure(fig_lin.number)
                plt.plot(
                    x_fit_exp,
                    y_fit_exp,
                    "--r",
                    label=rf"$\tau_{{\mathrm{{exp}}}}={popt_exp[1]:.2f}$ min",
                )

                plt.figure(fig_log.number)
                plt.plot(
                    x_fit_exp,
                    y_fit_exp,
                    "--r",
                    label=rf"$\tau_{{\mathrm{{exp}}}}={popt_exp[1]:.2f}$ min",
                )

                plt.figure(fig_linfit.number)
                plt.plot(
                    x_fit_exp,
                    y_fit_exp,
                    "--r",
                    label=rf"$\tau_{{\mathrm{{exp}}}}={popt_exp[1]:.2f}$ min",
                )

            fit_x = fit_centers[fit_mask]
            fit_y = log_counts[fit_mask]
            popt_linear, pcov_linear = optimize.curve_fit(
                linear_fit, fit_x, fit_y, p0=[-1, 0]
            )
            tau_from_linear = -1 / popt_linear[0]

            x_fit = np.linspace(0, 15, 100)
            y_fit = linear_fit(x_fit, *popt_linear)
            y_fit_exp = np.exp(y_fit)

            plt.figure(fig_lin.number)
            plt.hist(all_trapping_times, bins=BINS, density=True, alpha=0.7)
            plt.plot(
                x_fit, y_fit_exp, "--k", label=rf"$\tau={tau_from_linear:.2f}$ min"
            )
            plt.xlabel(r"$\tau$ (min)")
            plt.ylabel(r"$P(\tau)$")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-0.5, 15)

            plt.figure(fig_log.number)
            plt.hist(all_trapping_times, bins=BINS, density=True, alpha=0.7)
            plt.plot(
                x_fit, y_fit_exp, "--k", label=rf"$\tau={tau_from_linear:.2f}$ min"
            )
            plt.yscale("log")
            plt.xlabel(r"$\tau$ (min)")
            plt.ylabel(r"$P(\tau)$")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-0.5, 15)
            plt.ylim(1e-5, 1.2)

            plt.figure(fig_logfit.number)
            plt.scatter(
                fit_centers, log_counts, color="gray", alpha=0.3, label="all points"
            )
            plt.scatter(fit_x, fit_y, color="blue", alpha=0.7, label="fitted points")
            plt.plot(x_fit, y_fit, "--k", label=rf"$\tau={tau_from_linear:.2f}$ min")
            plt.xlabel(r"$\tau$ (min)")
            plt.ylabel(r"$\ln(P(\tau))$")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-0.5, 15)

            log_ticks = np.array([-5, -4, -3, -2, -1, 0])
            plt.yticks(log_ticks, [f"$10^{{{int(t)}}}$" for t in log_ticks])
            plt.ylim(-5.5, 0.5)

            plt.figure(fig_linfit.number)
            plt.scatter(
                fit_centers,
                np.exp(log_counts),
                color="gray",
                alpha=0.3,
                label="all points",
            )
            plt.scatter(
                fit_x, np.exp(fit_y), color="blue", alpha=0.7, label="fitted points"
            )
            plt.plot(
                x_fit, y_fit_exp, "--k", label=rf"$\tau={tau_from_linear:.2f}$ min"
            )
            plt.xlabel(r"$\tau$ (min)")
            plt.ylabel(r"$P(\tau)$")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.xlim(-0.5, 15)
            plt.ylim(-0.1, 1.2)

            stats_text = f"N events: {len(all_trapping_times)}\n"
            stats_text += f"Mean: {np.mean(all_trapping_times):.2f} min\n"
            stats_text += f"Median: {np.median(all_trapping_times):.2f} min\n"
            stats_text += f"Max: {np.max(all_trapping_times):.2f} min\n"
            stats_text += f"Slope = {popt_linear[0]:.2f}\n"
            stats_text += r"$\tau_{\mathrm{lin}} = " + f"{tau_from_linear:.2f}$ min\n"
            stats_text += r"$\tau_{\mathrm{exp}} = " + f"{popt_exp[1]:.2f}$ min"

            plt.figure(fig_lin.number)
            plt.plot(
                x_fit, y_fit_exp, "--k", label=r"$\tau=" + f"{tau_from_linear:.2f}$ min"
            )

            plt.figure(fig_log.number)
            plt.plot(
                x_fit, y_fit_exp, "--k", label=r"$\tau=" + f"{tau_from_linear:.2f}$ min"
            )

            plt.figure(fig_logfit.number)
            plt.plot(
                x_fit, y_fit, "--k", label=r"$\tau=" + f"{tau_from_linear:.2f}$ min"
            )

            plt.figure(fig_linfit.number)
            plt.plot(
                x_fit, y_fit_exp, "--k", label=r"$\tau=" + f"{tau_from_linear:.2f}$ min"
            )

            for fig in [fig_lin, fig_log, fig_logfit, fig_linfit]:
                plt.figure(fig.number)
                plt.text(
                    0.95,
                    0.95,
                    stats_text,
                    transform=plt.gca().transAxes,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(facecolor="white", alpha=0.8),
                    fontsize=12,
                )

            slope_error = np.sqrt(pcov_linear[0, 0])
            tau_error = tau_from_linear * slope_error / abs(popt_linear[0])
            tau_exp_error = np.sqrt(pcov_exp[1, 1])

            with open(f"{base_path}trapping_time.txt", "w") as f:
                f.write(
                    f"Lin trapping time: {tau_from_linear:.4f} ± {tau_error:.4f} min\n"
                )
                f.write(
                    f"Exp trapping time: {popt_exp[1]:.4f} ± {tau_exp_error:.4f} min\n"
                )
                f.write("\nFit parameters:\n")
                f.write(f"Slope: {popt_linear[0]:.4f} ± {slope_error:.4f}\n")
                f.write(f"A_exp: {popt_exp[0]:.4f}\n")

        except Exception as e:
            print(f"Fitting failed: {str(e)}")

    fig_lin.savefig(f"{base_path}{base_name}_linear.png")
    fig_log.savefig(f"{base_path}{base_name}_log.png")
    fig_logfit.savefig(f"{base_path}{base_name}_logfit.png")
    fig_linfit.savefig(f"{base_path}{base_name}_scatter.png")
    plt.close("all")
