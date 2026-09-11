import matplotlib.pyplot as plt
import numpy as np

from common import (LAB_P_TTRAP, LAB_TTRAP, exp_trapping_events, fit_exponential_log_space, log_bins, record, save,
                    set_style_framed)

MAX_TRAP_TIME = 10.0
N_BINS = 30
LENGTH_CLASSES = [(10, 21, r"$\ell_c \in [10, 21]$ mm", "#A3CDC9", "o"),
                  (23, 27, r"$\ell_c \in [23, 27]$ mm", "#73A6A3", "s"),
                  (27, 34, r"$\ell_c \in [27, 34]$ mm", "#3D7572", "^")]


def scale_fonts():
    plt.rcParams.update({"font.size": 15 * 1.4, "axes.labelsize": 18 * 1.4,
                         "xtick.labelsize": 13.5 * 1.4, "ytick.labelsize": 13.5 * 1.4})


def draw_distribution(ax, times, bins, label, color, marker, panel=None, series=None):
    counts, edges = np.histogram(times, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    tau, _, A = fit_exponential_log_space(times, bins)
    if np.isfinite(tau):
        label = rf"{label}, $\tau_{{\mathrm{{tr}}}} = {tau:.1f}$ min"
        x = np.linspace(0.01, bins[-1], 200)
        ax.plot(x, A * np.exp(-x / tau), "--", color=color, lw=1.5, zorder=2)
    pos = counts > 0
    ax.scatter(centers[pos], counts[pos], s=80, color=color, marker=marker, label=label, alpha=0.85,
               edgecolors="black", linewidths=0.5, zorder=3)
    if panel is not None:
        record(panel, series, tau_tr_min=centers[pos], probability_density_per_min=counts[pos],
               fitted_tau_tr_min=tau)


def finish(ax, max_time):
    ax.set_yscale("log")
    ax.set_xlabel(LAB_TTRAP)
    ax.set_ylabel(LAB_P_TTRAP)
    ax.set_xlim(0, max_time)
    ax.legend(fontsize=13, loc="lower left")


def main():
    set_style_framed()
    scale_fonts()
    df = exp_trapping_events(20, MAX_TRAP_TIME)
    bins = log_bins(MAX_TRAP_TIME, N_BINS)
    fig, ax = plt.subplots()
    for lo, hi, label, color, marker in LENGTH_CLASSES:
        times = df.loc[(df["length_mm"] >= lo) & (df["length_mm"] < hi), "time_min"].values
        draw_distribution(ax, times, bins, label, color, marker, "A", f"living worms lc {lo}-{hi} mm")
    finish(ax, MAX_TRAP_TIME)
    fig.tight_layout()
    save(fig, "fig3_trapping_time_distributions_exp")


if __name__ == "__main__":
    main()
