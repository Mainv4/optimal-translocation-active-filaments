import matplotlib.pyplot as plt
import numpy as np

from common import LAB_T_C, TEMPERATURES, exp_msd, exp_saturation_times, exp_table, save, set_style

COLOR_ROT = "#4C72B0"
COLOR_SAT = "#C44E52"


def main():
    set_style()
    exp = exp_table()
    _, tau_sat = exp_saturation_times(exp_msd())
    T = exp["T_celsius"].values.astype(float)
    tau_rot_s = exp["tau_rot"].values.astype(float) * 60.0
    tau_sat_s = np.array([tau_sat[int(t)] for t in T]) * 60.0
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.plot(T, tau_rot_s, marker="s", ms=9, lw=1.2, color=COLOR_ROT, zorder=3)
    ax.set_xlabel(LAB_T_C)
    ax.set_ylabel(r"$\tau_{\theta}$ (s)", color=COLOR_ROT)
    ax.tick_params(axis="y", labelcolor=COLOR_ROT)
    ax.set_xticks(TEMPERATURES)
    ax.set_xticklabels([rf"${t}$" for t in TEMPERATURES])
    ax2 = ax.twinx()
    ax2.plot(T, tau_sat_s, marker="s", ms=9, lw=1.2, color=COLOR_SAT, zorder=3)
    ax2.set_ylabel(r"$\tau_d$ (s)", color=COLOR_SAT)
    ax2.tick_params(axis="y", labelcolor=COLOR_SAT)
    ax.text(0.04, 0.06, r"$\leftarrow \tau_{\theta}$", transform=ax.transAxes, color=COLOR_ROT,
            fontsize=16, fontweight="bold", va="bottom", ha="left")
    ax.text(0.96, 0.94, r"$\tau_d \rightarrow$", transform=ax.transAxes, color=COLOR_SAT,
            fontsize=16, fontweight="bold", va="top", ha="right")
    fig.tight_layout()
    save(fig, "fig6_timescales_vs_temperature")


if __name__ == "__main__":
    main()
