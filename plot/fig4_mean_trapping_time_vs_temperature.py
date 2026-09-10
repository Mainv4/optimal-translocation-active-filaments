import matplotlib.pyplot as plt
import numpy as np

from common import (CALIBRATED, LAB_T_C, TEMP_COLORS, TEMPERATURES, exp_trapping_events, fit_exponential_log_space,
                    log_bins, save, set_style_framed, sim_trapping_events)
from fig4_trapping_time_distributions_temperature import TRAP_CUTOFF, TRAP_NBINS

MARKER_PT = 7


def main():
    set_style_framed()
    bins = log_bins(TRAP_CUTOFF, TRAP_NBINS)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for T in TEMPERATURES:
        times = exp_trapping_events(T, TRAP_CUTOFF)["time_min"].values
        tau, _, _ = fit_exponential_log_space(times, bins)
        ax.errorbar(T, np.mean(times), yerr=np.std(times) / np.sqrt(len(times)), marker="s", color=TEMP_COLORS[T],
                    markersize=MARKER_PT, capsize=4, lw=1.5, zorder=3)
        ax.plot(T, tau, marker="s", color=TEMP_COLORS[T], markersize=MARKER_PT, markerfacecolor="white",
                markeredgewidth=1.5, zorder=3)
        sim_times = sim_trapping_events(40, *CALIBRATED[T], TRAP_CUTOFF)
        tau_sim, tau_sim_err, _ = fit_exponential_log_space(sim_times, bins)
        ax.errorbar(T, tau_sim, yerr=tau_sim_err, marker="D", color=TEMP_COLORS[T], markersize=MARKER_PT,
                    markerfacecolor="none", markeredgewidth=1.5, capsize=2, lw=1.0, ls="none", zorder=4)
    ax.set_xlabel(LAB_T_C)
    ax.set_ylabel(r"$\langle\tau_{\mathrm{tr}}\rangle,\;\tau_c$ (min)")
    ax.set_xlim(5, 35)
    ax.set_ylim(0.5, 3.5)
    ax.set_xticks(TEMPERATURES)
    fig.tight_layout()
    save(fig, "fig4_mean_trapping_time_vs_temperature")


if __name__ == "__main__":
    main()
