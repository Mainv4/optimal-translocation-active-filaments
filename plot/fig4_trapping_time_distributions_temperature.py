import matplotlib.pyplot as plt
import numpy as np

from common import (LAB_P_TTRAP, LAB_TTRAP, TEMP_COLORS, TEMPERATURES, exp_trapping_events, fit_exponential_log_space,
                    log_bins, save, set_style_framed)

TRAP_CUTOFF = 15.0
TRAP_NBINS = 28
MARKER_AREA = 40
LABELS = {T: rf"$T = {T}\,^{{\circ}}$C" for T in TEMPERATURES}


def main():
    set_style_framed()
    bins = log_bins(TRAP_CUTOFF, TRAP_NBINS)
    centers = 0.5 * (bins[:-1] + bins[1:])
    x_fit = np.linspace(0.01, TRAP_CUTOFF, 200)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for T in TEMPERATURES:
        times = exp_trapping_events(T, TRAP_CUTOFF)["time_min"].values
        counts, _ = np.histogram(times, bins=bins, density=True)
        pos = counts > 0
        ax.scatter(centers[pos], counts[pos], s=MARKER_AREA, color=TEMP_COLORS[T], marker="s", label=LABELS[T],
                   alpha=0.85, edgecolors="black", linewidths=0.5, zorder=3)
        tau, _, A = fit_exponential_log_space(times, bins)
        ax.plot(x_fit, A * np.exp(-x_fit / tau), "--", color=TEMP_COLORS[T], lw=1.5, zorder=2)
    ax.set_yscale("log")
    ax.set_xlabel(LAB_TTRAP)
    ax.set_ylabel(LAB_P_TTRAP)
    ax.set_xlim(0, TRAP_CUTOFF)
    ax.set_ylim(1e-2, 2)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "fig4_trapping_time_distributions_temperature")


if __name__ == "__main__":
    main()
