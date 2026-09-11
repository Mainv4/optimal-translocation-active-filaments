import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import (DATA, LAB_P_RGRE, LAB_RE, LAB_RG, RG_RANGE, SMOOTH_SIGMA, draw_confmap, filter_by_envelope,
                    load_boundaries, read_confmap_npy, read_rere_csv, save, set_style_framed)
from fig2_com_trajectories import C_EXP, C_SIM, K, PE, T_SIM


def main():
    set_style_framed()
    env = load_boundaries()
    fig = plt.figure(figsize=(7, 3.5))
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0)
    ax_exp = fig.add_subplot(gs[0, 0])
    ax_sim = fig.add_subplot(gs[0, 1], sharey=ax_exp)
    cax = fig.add_subplot(gs[0, 2])
    re, rg = read_rere_csv(DATA / "exp" / "cavity" / "ReRg_20.csv")
    draw_confmap(ax_exp, re, rg, env, smooth_sigma=SMOOTH_SIGMA, boundary_color=C_EXP, label=r"Exp $20\,^\circ$C conf.",
                 panel="F", series="living worms 20 C")
    re, rg = filter_by_envelope(*read_confmap_npy(DATA / "sim" / "confinement" / "conformations" / f"Pe_{PE}_T_{T_SIM}_k_{K}.npy"), env)
    mesh = draw_confmap(ax_sim, re, rg, env, boundary_color=C_SIM, label=r"Sim $20\,^\circ$C conf.",
                        panel="F", series=f"model fa={PE} T={T_SIM} kappa={K}")
    ax_exp.set_xlabel(LAB_RG)
    ax_exp.set_ylabel(LAB_RE)
    ax_exp.set_xticks([t for t in ax_exp.get_xticks() if t < RG_RANGE[1] - 0.01])
    ax_sim.set_xlabel(LAB_RG)
    plt.setp(ax_sim.get_yticklabels(), visible=False)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(LAB_P_RGRE)
    cbar.ax.yaxis.set_ticks([1e-4, 1e-3, 1e-2])
    cbar.ax.yaxis.set_ticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$"])
    save(fig, "fig2_conformational_map_confined")


if __name__ == "__main__":
    main()
