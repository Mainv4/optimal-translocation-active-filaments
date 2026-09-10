import matplotlib.pyplot as plt

from common import (CMAP_ACTIVITY, F_SIM_TO_NN, LAB_FA_NN, LAB_H, LAB_KAPPA, paper_filter, save,
                    set_style, sim_table)


def main():
    set_style()
    sim = paper_filter(sim_table()).dropna(subset=["H_conf", "kappa"])
    fig = plt.figure(figsize=(5.2, 4.3))
    ax = fig.add_axes([0.15, 0.15, 0.66, 0.80])
    sc = ax.scatter(sim["kappa"], sim["H_conf"], c=sim["Pe"] * F_SIM_TO_NN, cmap=CMAP_ACTIVITY,
                    s=34, marker="D", alpha=0.80, edgecolors="none", zorder=2)
    ax.set_xlabel(LAB_KAPPA)
    ax.set_ylabel(LAB_H)
    ax.set_box_aspect(1)
    fig.colorbar(sc, cax=fig.add_axes([0.84, 0.15, 0.03, 0.80])).set_label(LAB_FA_NN)
    save(fig, "figS_entropy_vs_kappa")


if __name__ == "__main__":
    main()
