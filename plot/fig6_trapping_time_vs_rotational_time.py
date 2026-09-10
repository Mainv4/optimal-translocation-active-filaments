import matplotlib.pyplot as plt
import numpy as np

from common import (CMAP_ACTIVITY, F_SIM_TO_NN, LAB_FA_NN, LAB_TROT, LAB_TTRAP, YLIM_TRAP, add_exp_markers,
                    add_slope_guide, exp_table, save, set_log_ticks, set_style, sim_table, timescale_filter)


def main():
    set_style()
    sim = timescale_filter(sim_table())
    exp = exp_table()
    trot, ttrap = sim["tau_rot"].values, sim["ttrap"].values
    c = sim["Pe"].values * F_SIM_TO_NN
    show = ~((trot < 0.1) & (ttrap > 5))
    fig = plt.figure(figsize=(5.4, 4.2))
    ax = fig.add_axes([0.14, 0.14, 0.70, 0.82])
    sc = ax.scatter(trot[show], ttrap[show], c=c[show], cmap=CMAP_ACTIVITY, vmin=c.min(), vmax=c.max(),
                    s=24, marker="D", alpha=0.75, edgecolors="none", zorder=2)
    x_anc, y_anc = 10 ** np.median(np.log10(trot)), 10 ** np.median(np.log10(ttrap))
    add_slope_guide(ax, +1, x_anc, y_anc, (trot.min() * 0.7, trot.max() * 1.5))
    x_prop = trot.max() * 0.9
    y_prop = 10 ** (np.log10(y_anc) + np.log10(x_prop) - np.log10(x_anc))
    ax.text(x_prop * 0.6, y_prop * 1.4, r"$\propto \tau_{\theta}$", fontsize=13, color="0.25", ha="right", va="bottom")
    add_exp_markers(ax, exp["tau_rot"], exp["ttrap"], exp["T_celsius"], legend=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*YLIM_TRAP)
    set_log_ticks(ax, "x", [0.01, 0.1, 0.4])
    ax.set_xlabel(LAB_TROT)
    ax.set_ylabel(LAB_TTRAP)
    fig.colorbar(sc, cax=fig.add_axes([0.86, 0.14, 0.03, 0.82])).set_label(LAB_FA_NN)
    save(fig, "fig6_trapping_time_vs_rotational_time")


if __name__ == "__main__":
    main()
