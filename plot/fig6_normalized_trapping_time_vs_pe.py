import matplotlib.pyplot as plt
import numpy as np

from common import (CMAP_ACTIVITY, F_SIM_TO_NN, LAB_FA_NN, LAB_PE_EFF, LAB_RATIO_INF, TEMPERATURES, activity_bins,
                    add_exp_markers, exp_msd, exp_saturation_times, exp_table, record, save, set_log_ticks, set_style,
                    sim_table, timescale_filter)

EXP_RATIO_INF = {10: 1.5, 20: 1.4, 30: 1.4}


def main():
    set_style()
    sim = timescale_filter(sim_table())
    exp = exp_table()
    _, tau_sat_exp = exp_saturation_times(exp_msd())
    trot, tsat, ttrap = sim["tau_rot"].values, sim["tau_trans"].values, sim["ttrap"].values
    pe_eff = trot / tsat
    c = sim["Pe"].values * F_SIM_TO_NN
    ratio = np.full(len(sim), np.nan)
    for b in activity_bins(sim):
        if np.isfinite(b["tau_inf"]):
            ratio[b["in_bin"]] = ttrap[b["in_bin"]] / b["tau_inf"]
    show = np.isfinite(ratio) & ~((trot < 0.1) & (ttrap > 5))
    fig = plt.figure(figsize=(5.4, 4.2))
    ax = fig.add_axes([0.14, 0.14, 0.70, 0.82])
    sc = ax.scatter(pe_eff[show], ratio[show], c=c[show], cmap=CMAP_ACTIVITY, vmin=c.min(), vmax=c.max(),
                    s=24, marker="D", alpha=0.75, edgecolors="none", zorder=2)
    pe_eff_exp = exp["tau_rot"].values / np.array([tau_sat_exp[T] for T in TEMPERATURES])
    add_exp_markers(ax, pe_eff_exp, [EXP_RATIO_INF[T] for T in TEMPERATURES], exp["T_celsius"])
    record("F", "model", pe_number=pe_eff[show], tau_tr_over_tau_tr_inf=ratio[show], fa_nN=c[show],
           kappa_over_uE=sim["kappa"].values[show], T_star=sim["T"].values[show])
    record("F", "living worms", pe_number=pe_eff_exp, tau_tr_over_tau_tr_inf=[EXP_RATIO_INF[T] for T in TEMPERATURES],
           temperature_C=exp["T_celsius"])
    x_lo, x_hi = pe_eff.min() * 0.5, pe_eff.max() * 2.0
    ax.plot([x_lo, x_hi], [1.0, 1.0], ls="--", lw=1.3, color="0.35", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(1e-1, 1e2)
    set_log_ticks(ax, "y", [1e-1, 1e0, 1e1, 1e2], powers=True)
    ax.set_xlabel(LAB_PE_EFF)
    ax.set_ylabel(LAB_RATIO_INF)
    fig.colorbar(sc, cax=fig.add_axes([0.86, 0.14, 0.03, 0.82])).set_label(LAB_FA_NN)
    save(fig, "fig6_normalized_trapping_time_vs_pe")


if __name__ == "__main__":
    main()
