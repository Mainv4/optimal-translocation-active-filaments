import matplotlib.pyplot as plt
import numpy as np

from common import (CMAP_ACTIVITY, CMAP_KAPPA, EXP_EDGE_LW, EXP_MARKER, F_SIM_TO_NN, LAB_FA_NN, LAB_KAPPA, LAB_TSAT,
                    LAB_TTRAP, TEMP_COLORS, TEMPERATURES, YLIM_TRAP, activity_bins, activity_color, add_exp_markers,
                    add_slope_guide, exp_msd, exp_saturation_times, exp_table, save, set_style, sim_table,
                    timescale_filter)

FIT_LOG_A = -15.007
FIT_ALPHA = -0.948
EXP_TAU_INF = {10: (62.2, 3.0), 20: (254.9, 0.8), 30: (80.2, 2.2)}
PLATEAU_LINE_X = (0.3, 3.0)


def tau_inf_fit(f_nN):
    return np.exp(FIT_LOG_A) * (np.asarray(f_nN) * 1e-9) ** FIT_ALPHA


def draw(ax, sim, exp, tau_sat_exp, color_by="Pe", decorations=True):
    tsat, ttrap, Pe = sim["tau_trans"].values, sim["ttrap"].values, sim["Pe"].values
    if color_by == "kappa":
        c, cmap, label = sim["kappa"].values, CMAP_KAPPA, LAB_KAPPA
    else:
        c, cmap, label = Pe * F_SIM_TO_NN, CMAP_ACTIVITY, LAB_FA_NN
    show = ~((tsat < 0.1) & (ttrap > 5))
    sc = ax.scatter(tsat[show], ttrap[show], c=c[show], cmap=cmap, vmin=c.min(), vmax=c.max(),
                    s=24, marker="D", alpha=0.75, edgecolors="none", zorder=2)
    low = (tsat <= 0.3) & (tsat > 0) & (ttrap > 0)
    x_anc, y_anc = 10 ** np.median(np.log10(tsat[low])), 10 ** np.median(np.log10(ttrap[low]))
    add_slope_guide(ax, +1, x_anc, y_anc, (tsat.min() * 0.5, 2.0))
    y_prop = 10 ** (np.log10(y_anc) + np.log10(2.0) - np.log10(x_anc))
    ax.text(2.0 * 0.95, y_prop * 1.4, r"$\propto \tau_d$", fontsize=13, color="0.25", ha="right", va="bottom")
    if decorations:
        bins = [b for b in activity_bins(sim) if b["on_plateau"].any()]
        for b in bins:
            ax.plot(list(PLATEAU_LINE_X), [b["tau_inf"]] * 2, ls="--", lw=1.2,
                    color=activity_color(b["Pe_med"], Pe), alpha=0.85, zorder=4)
        ins = ax.inset_axes([0.15, 0.64, 0.36, 0.34])
        for b in bins:
            ins.scatter([b["Pe_med"] * F_SIM_TO_NN], [b["tau_inf"]], marker="o", s=45,
                        c=[activity_color(b["Pe_med"], Pe)], edgecolors="black", linewidths=0.6, zorder=6)
        f_line = np.linspace(5, 230, 400)
        ins.plot(f_line, tau_inf_fit(f_line), ls="--", lw=1.0, color="0.25", zorder=3)
        for T, (f_nN, tau_min) in EXP_TAU_INF.items():
            ins.scatter([f_nN], [tau_min], marker=EXP_MARKER, s=60, c=TEMP_COLORS[T],
                        edgecolors="black", linewidths=EXP_EDGE_LW, zorder=4)
        ins.set_xlabel(LAB_FA_NN, fontsize=11, labelpad=1)
        ins.set_ylabel(r"$\tau_{\mathrm{tr},\infty}$", fontsize=11, labelpad=1)
        ins.set_xticks([0, 100, 200])
        ins.set_yticks([0, 2.5, 5])
        ins.set_ylim(0, 5.5)
        ins.tick_params(labelsize=10)
    add_exp_markers(ax, [tau_sat_exp[T] for T in TEMPERATURES], exp["ttrap"], exp["T_celsius"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*YLIM_TRAP)
    ax.set_xlabel(LAB_TSAT)
    ax.set_ylabel(LAB_TTRAP)
    return sc, label


def main(color_by="Pe", decorations=True, name="fig6_trapping_time_vs_translational_time"):
    set_style()
    sim = timescale_filter(sim_table())
    exp = exp_table()
    _, tau_sat_exp = exp_saturation_times(exp_msd())
    fig = plt.figure(figsize=(5.4, 4.2))
    ax = fig.add_axes([0.14, 0.14, 0.70, 0.82])
    sc, label = draw(ax, sim, exp, tau_sat_exp, color_by, decorations)
    fig.colorbar(sc, cax=fig.add_axes([0.86, 0.14, 0.03, 0.82])).set_label(label)
    save(fig, name)


if __name__ == "__main__":
    main()
