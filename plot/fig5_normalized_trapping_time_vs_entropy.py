import matplotlib.pyplot as plt
import numpy as np

from common import (CMAP_ACTIVITY, F_SIM_TO_NN, LAB_FA_NN, LAB_H, LAB_TTRAP_OVER_TAU_E0,
                    add_exp_markers, exp_table, paper_filter, record, save, set_style, sim_table)


def main():
    set_style()
    sim = paper_filter(sim_table()).dropna(subset=["H_conf", "tau_decorr_free_unit"])
    sim["ttrap_norm"] = sim["ttrap"] / sim["tau_decorr_free_unit"]
    exp = exp_table()
    fig = plt.figure(figsize=(5.2, 4.3))
    ax = fig.add_axes([0.15, 0.15, 0.66, 0.80])
    sc = ax.scatter(sim["H_conf"], sim["ttrap_norm"], c=sim["Pe"] * F_SIM_TO_NN,
                    cmap=CMAP_ACTIVITY, s=30, marker="D", alpha=0.85, edgecolors="none", zorder=2)
    add_exp_markers(ax, exp["H_conf"], exp["ttrap"] / exp["tau_decorr_free_unit"], exp["T_celsius"])
    positive = sim["ttrap_norm"].values > 0
    slope, intercept = np.polyfit(sim["H_conf"].values[positive],
                                  np.log10(sim["ttrap_norm"].values[positive]), 1)
    x_line = np.linspace(4.75, 5.8, 100)
    ax.plot(x_line, 10 ** (slope * x_line + intercept), ls="--", lw=1.3, color="0.25", zorder=4)
    record("C", "model", shannon_entropy_H=sim["H_conf"], tau_tr_over_tau_e0=sim["ttrap_norm"],
           fa_nN=sim["Pe"] * F_SIM_TO_NN, kappa_over_uE=sim["kappa"], T_star=sim["T"])
    record("C", "living worms", shannon_entropy_H=exp["H_conf"],
           tau_tr_over_tau_e0=exp["ttrap"] / exp["tau_decorr_free_unit"], temperature_C=exp["T_celsius"])
    record("C", "log-linear fit", fit_slope_per_H=slope, fit_intercept=intercept)
    ax.set_yscale("log")
    ax.set_ylim(1e0, 1e2)
    ax.set_xlim(4.5, 6.25)
    ax.set_xticks([4.5, 5.0, 5.5, 6.0])
    ax.set_xlabel(LAB_H)
    ax.set_ylabel(LAB_TTRAP_OVER_TAU_E0)
    ax.set_box_aspect(1)
    fig.colorbar(sc, cax=fig.add_axes([0.84, 0.15, 0.03, 0.80])).set_label(LAB_FA_NN)
    save(fig, "fig5_normalized_trapping_time_vs_entropy")


if __name__ == "__main__":
    main()
