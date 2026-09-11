import matplotlib.pyplot as plt
import numpy as np

from common import (LAB_MSD_TR, LAB_T_MIN, PLATEAU_WINDOW_MIN, TEMP_COLORS, TEMP_LABELS, TEMPERATURES, crossing_time,
                    exp_msd, plateau_in_window, record, save, set_style)


def main():
    set_style()
    msd = exp_msd()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True, gridspec_kw={"wspace": 0.0})
    for ax, T in zip(axes, TEMPERATURES):
        m = msd[T]["trans"]
        t, y = m[:, 0], m[:, 1]
        plateau = plateau_in_window(m)
        tau_d = crossing_time(m, plateau)
        window = (t >= PLATEAU_WINDOW_MIN[0]) & (t <= PLATEAU_WINDOW_MIN[1])
        spread = float(np.nanstd(y[window]))
        record(f"{T}C", "living worms", t_min=t, translational_msd_mm2=y, plateau_mm2=plateau,
               plateau_sd_mm2=spread, tau_d_min=tau_d, window_start_min=PLATEAU_WINDOW_MIN[0],
               window_end_min=PLATEAU_WINDOW_MIN[1])
        ax.axvspan(*PLATEAU_WINDOW_MIN, color="0.85", lw=0, zorder=0)
        ax.axhline(plateau, ls="--", lw=1.4, color=TEMP_COLORS[T], zorder=2)
        ax.axhspan(plateau - spread, plateau + spread, color=TEMP_COLORS[T], alpha=0.25, lw=0, zorder=1)
        ax.plot(t, y, lw=1.8, color=TEMP_COLORS[T], zorder=3)
        ax.plot(t[window], y[window], lw=2.6, color=TEMP_COLORS[T], zorder=4)
        ax.axvline(tau_d, ls=":", lw=1.4, color=TEMP_COLORS[T], zorder=2)
        ax.plot([tau_d], [plateau], ls="none", marker="o", ms=7, mfc="white", mec="0.30", mew=1.4, zorder=5)
        ax.set_xscale("log")
        ax.set_xlabel(LAB_T_MIN)
        ax.set_title(TEMP_LABELS[T], fontsize=13, pad=6)
    axes[0].set_ylabel(LAB_MSD_TR)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0)
    save(fig, "figS_plateau_identification")


if __name__ == "__main__":
    main()
