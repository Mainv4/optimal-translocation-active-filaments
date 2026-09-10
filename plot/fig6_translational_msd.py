import matplotlib.pyplot as plt

from common import (LAB_MSD_TR, LAB_T_MIN, PLATEAU_WINDOW_MIN, TEMP_COLORS, TEMP_LABELS, TEMPERATURES,
                    exp_msd, exp_saturation_times, save, set_log_ticks, set_style)


def main():
    set_style()
    msd = exp_msd()
    plateau, tau_sat = exp_saturation_times(msd)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for T in TEMPERATURES:
        m = msd[T]["trans"]
        ax.plot(m[:, 0], m[:, 1], lw=1.8, color=TEMP_COLORS[T], label=TEMP_LABELS[T], zorder=3)
        ax.plot(list(PLATEAU_WINDOW_MIN), [plateau[T]] * 2, ls=":", lw=1.2, color=TEMP_COLORS[T], zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(LAB_T_MIN)
    ax.set_ylabel(LAB_MSD_TR)
    set_log_ticks(ax, "y", [6, 10, 15])
    y0 = ax.get_ylim()[0]
    for T in TEMPERATURES:
        ax.plot([tau_sat[T]] * 2, [y0, plateau[T]], ls="--", lw=0.8, color=TEMP_COLORS[T], alpha=0.85, zorder=4)
    fig.tight_layout()
    save(fig, "fig6_translational_msd")


if __name__ == "__main__":
    main()
