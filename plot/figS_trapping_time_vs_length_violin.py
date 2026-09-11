import matplotlib.pyplot as plt
import numpy as np

from common import LAB_LC, LAB_TTRAP, TEMP_COLORS, TEMPERATURES, exp_trapping_events, record, save, set_style_framed
from fig3_trapping_time_vs_length import group_by_length

MAX_TRAP_TIME = 15.0


def main():
    set_style_framed()
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    fig.subplots_adjust(hspace=0)
    for ax, T in zip(axes, TEMPERATURES):
        groups = group_by_length(exp_trapping_events(T, MAX_TRAP_TIME))
        positions = [g[0] for g in groups]
        for lc, times in groups:
            record(f"{T}C", "living worms", contour_length_mm=lc, tau_tr_min=times)
        parts = ax.violinplot([g[1] for g in groups], positions=positions, widths=1.5,
                              showmeans=True, showmedians=True, showextrema=True)
        for body in parts["bodies"]:
            body.set_facecolor(TEMP_COLORS[T])
            body.set_alpha(0.5)
            body.set_edgecolor("black")
            body.set_linewidth(1)
        parts["cmeans"].set_edgecolor("darkred")
        parts["cmeans"].set_linewidth(2)
        parts["cmedians"].set_edgecolor("darkblue")
        parts["cmedians"].set_linewidth(2)
        ax.plot(positions, [np.mean(g[1]) for g in groups], "k--", lw=1.5, zorder=4)
        ax.set_ylabel(LAB_TTRAP)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(-1, MAX_TRAP_TIME + 1)
    axes[-1].set_xlabel(LAB_LC)
    save(fig, "figS_trapping_time_vs_length_violin")


if __name__ == "__main__":
    main()
