import matplotlib.pyplot as plt
import numpy as np

from common import (CALIBRATED, LAB_T_C, TEMP_COLORS, TEMPERATURES, exp_trapping_events, fit_exponential_log_space,
                    log_bins, record, save, set_style_framed, sim_trapping_events)
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
        record("D", "living worms", temperature_C=T, mean_tau_tr_min=np.mean(times),
               standard_error_min=np.std(times) / np.sqrt(len(times)), fitted_tau_c_min=tau, n_events=len(times))
        sim_times = sim_trapping_events(40, *CALIBRATED[T], TRAP_CUTOFF)
        tau_sim, tau_sim_err, _ = fit_exponential_log_space(sim_times, bins)
        record("D", "model", temperature_C=T, fitted_tau_c_min=tau_sim, fit_error_min=tau_sim_err,
               n_events=len(sim_times))
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
