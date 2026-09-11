import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from common import (DATA, LAB_P_RGRE, LAB_RE, LAB_RG, SMOOTH_SIGMA, TEMP_COLORS, TEMP_LABELS, TEMPERATURES,
                    draw_confmap, load_boundaries, read_rere_csv, save, set_style_framed)

def colorbar(fig, mesh, cax):
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(LAB_P_RGRE)
    cbar.ax.yaxis.set_ticks([1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    cbar.ax.yaxis.set_ticklabels(["$10^{-4}$", "", "", "$10^{-3}$", "", "", "$10^{-2}$"])


def row_figure():
    fig = plt.figure(figsize=(10, 3.5))
    gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 0.03, 0.08], wspace=0)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    for c, ax in enumerate(axes):
        ax.set_xlabel(LAB_RG)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3] if c == 0 else [0.1, 0.2, 0.3])
        if c == 0:
            ax.set_ylabel(LAB_RE)
        else:
            ax.set_yticklabels([])
    return fig, axes, fig.add_subplot(gs[0, 4])


def main():
    set_style_framed()
    env = load_boundaries()
    fig, axes, cax = row_figure()
    for ax, T in zip(axes, TEMPERATURES):
        re, rg = read_rere_csv(DATA / "exp" / "free_space" / f"ReRg_{T}.csv")
        mesh = draw_confmap(ax, re, rg, env, smooth_sigma=SMOOTH_SIGMA, boundary_color=TEMP_COLORS[T],
                            label=f"Exp {TEMP_LABELS[T]}", panel="B", series=f"living worms {T} C")
    colorbar(fig, mesh, cax)
    save(fig, "fig1_conformational_maps_exp")


if __name__ == "__main__":
    main()
