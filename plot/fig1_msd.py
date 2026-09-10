import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import CALIBRATED, DATA, TEMP_COLORS, TEMP_COLORS_SIM, TEMP_LABELS, TEMPERATURES, save, set_style_framed

MATCH_INTERVAL = (10, 50)
S = 2.2 / 1.5


def main():
    set_style_framed()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axvspan(MATCH_INTERVAL[0], 100.0, color="#DDDDDD", alpha=0.5, zorder=0)
    handles = []
    for T in TEMPERATURES:
        pe, t, k = CALIBRATED[T]
        msd = np.loadtxt(DATA / "sim" / "free_space" / "msd" / f"Pe_{pe}_T_{t}_k_{k}_MSD_CM.dat", comments="#")
        times = np.arange(len(msd))
        v = (times > 0) & (msd > 0)
        ax.plot(times[v], msd[v], color=TEMP_COLORS_SIM[T], lw=1.5 * S, ls="--", zorder=4)
        exp = pd.read_csv(DATA / "exp" / "free_space" / f"{T}C_msd.csv")
        te, me = exp["t"].values, exp["msd"].values
        v = np.isfinite(me) & (me > 0) & (te > 0)
        ax.plot(te[v], me[v], color=TEMP_COLORS[T], lw=2.0 * S, ls="-", zorder=3)
        handles.append(Line2D([0], [0], color=TEMP_COLORS[T], lw=2 * S, ls="-", label=TEMP_LABELS[T] + " exp"))
        handles.append(Line2D([0], [0], color=TEMP_COLORS_SIM[T], lw=1.5 * S, ls="--",
                              label=TEMP_LABELS[T] + rf" sim ($f^a\!=\!{pe},\;T\!=\!{t},\;\kappa\!=\!{k}$)"))
    ax.legend(handles=handles, fontsize=5.5 * S, loc="upper left", frameon=True, fancybox=False,
              edgecolor="none", facecolor="white", framealpha=0.85, handlelength=2.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (s)", fontsize=12 * S)
    ax.set_ylabel(r"$\langle r^2 \rangle$ (mm$^2$)", fontsize=12 * S)
    ax.tick_params(labelsize=9 * S, width=1.2 * S, length=4 * S, pad=6 * S)
    ax.set_xlim(0.5, 250)
    ax.set_ylim(1e-1, 1e4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2 * S)
    save(fig, "fig1_msd")


if __name__ == "__main__":
    main()
