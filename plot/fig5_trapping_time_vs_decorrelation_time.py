import matplotlib.pyplot as plt

from common import (CMAP_ACTIVITY, CMAP_KAPPA, F_SIM_TO_NN, LAB_FA_NN, LAB_KAPPA, LAB_TAU_EC,
                    LAB_TTRAP, add_exp_markers, exp_table, paper_filter, record, save, set_style,
                    sim_table)


def colouring(sim, color_by):
    if color_by == "kappa":
        return sim["kappa"], CMAP_KAPPA, LAB_KAPPA
    return sim["Pe"] * F_SIM_TO_NN, CMAP_ACTIVITY, LAB_FA_NN


def draw(ax, sim, exp, color_by="Pe", panel="A"):
    c, cmap, label = colouring(sim, color_by)
    record(panel, "model", tau_ec_min=sim["tau_decorr_cavity_unit"], tau_tr_min=sim["ttrap"],
           fa_nN=sim["Pe"] * F_SIM_TO_NN, kappa_over_uE=sim["kappa"], T_star=sim["T"])
    record(panel, "living worms", tau_ec_min=exp["tau_decorr_cavity_unit"], tau_tr_min=exp["ttrap"],
           temperature_C=exp["T_celsius"])
    sc = ax.scatter(sim["tau_decorr_cavity_unit"], sim["ttrap"], c=c, cmap=cmap,
                    s=30, marker="D", alpha=0.85, edgecolors="none", zorder=2)
    add_exp_markers(ax, exp["tau_decorr_cavity_unit"], exp["ttrap"], exp["T_celsius"], legend=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-1, 1e1)
    ax.set_xlabel(LAB_TAU_EC)
    ax.set_ylabel(LAB_TTRAP)
    ax.set_box_aspect(1)
    return sc, label


def main(color_by="Pe", name="fig5_trapping_time_vs_decorrelation_time", panel="A"):
    set_style()
    sim = paper_filter(sim_table()).dropna(subset=["tau_decorr_cavity_unit"])
    exp = exp_table()
    fig = plt.figure(figsize=(5.2, 4.3))
    ax = fig.add_axes([0.15, 0.15, 0.66, 0.80])
    sc, label = draw(ax, sim, exp, color_by, panel)
    fig.colorbar(sc, cax=fig.add_axes([0.84, 0.15, 0.03, 0.80])).set_label(label)
    save(fig, name)


if __name__ == "__main__":
    main()
