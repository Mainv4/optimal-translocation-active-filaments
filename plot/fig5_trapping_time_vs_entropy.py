import matplotlib.pyplot as plt

from common import (CMAP_ACTIVITY, F_SIM_TO_NN, LAB_FA_NN, LAB_H, LAB_TTRAP, add_exp_markers,
                    exp_table, paper_filter, record, save, set_style, sim_table)


def main():
    set_style()
    sim = paper_filter(sim_table()).dropna(subset=["H_conf"])
    exp = exp_table()
    fig = plt.figure(figsize=(5.2, 4.3))
    ax = fig.add_axes([0.15, 0.15, 0.66, 0.80])
    sc = ax.scatter(sim["H_conf"], sim["ttrap"], c=sim["Pe"] * F_SIM_TO_NN, cmap=CMAP_ACTIVITY,
                    s=30, marker="D", alpha=0.85, edgecolors="none", zorder=2)
    add_exp_markers(ax, exp["H_conf"], exp["ttrap"], exp["T_celsius"])
    record("B", "model", shannon_entropy_H=sim["H_conf"], tau_tr_min=sim["ttrap"],
           fa_nN=sim["Pe"] * F_SIM_TO_NN, kappa_over_uE=sim["kappa"], T_star=sim["T"])
    record("B", "living worms", shannon_entropy_H=exp["H_conf"], tau_tr_min=exp["ttrap"],
           temperature_C=exp["T_celsius"])
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e1)
    ax.set_xlim(4.5, 6.25)
    ax.set_xticks([4.5, 5.0, 5.5, 6.0])
    ax.set_xlabel(LAB_H)
    ax.set_ylabel(LAB_TTRAP)
    ax.set_box_aspect(1)
    fig.colorbar(sc, cax=fig.add_axes([0.84, 0.15, 0.03, 0.80])).set_label(LAB_FA_NN)
    save(fig, "fig5_trapping_time_vs_entropy")


if __name__ == "__main__":
    main()
