import matplotlib.pyplot as plt

from common import LAB_MSD_ROT, LAB_T_MIN, TEMP_COLORS, TEMP_LABELS, TEMPERATURES, exp_msd, fit_rot_exp, save, set_style


def main():
    set_style()
    msd = exp_msd()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for T in TEMPERATURES:
        m = msd[T]["rot"]
        ax.plot(m[:, 0], m[:, 1], lw=1.8, color=TEMP_COLORS[T], label=TEMP_LABELS[T], zorder=3)
        _, t_line, line = fit_rot_exp(m)
        ax.plot(t_line, line, ls="--", lw=1.0, color="0.25", zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(LAB_T_MIN)
    ax.set_ylabel(LAB_MSD_ROT)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9, edgecolor="0.7")
    fig.tight_layout()
    save(fig, "fig6_rotational_msd")


if __name__ == "__main__":
    main()
