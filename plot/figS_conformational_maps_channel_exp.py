import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec

from common import DATA, LAB_RE, LAB_RG, RE_RANGE, RG_RANGE, TEMP_LABELS, TEMPERATURES, load_boundaries, save, set_style_framed

CAVITY_EDGE = 40.0
L_B = 80.0
BIN_EDGES = [0.2, 0.3, 0.4]
CAT_LABELS = ["Trapped", r"$<0.2$", r"$0.2$–$0.3$", r"$0.3$–$0.4$", r"$>0.4$"]
CAT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ffff33", "#984ea3"]
CMAP_CAT = ListedColormap(CAT_COLORS)
NORM_CAT = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), CMAP_CAT.N)


def position_category(x_centered):
    a = np.abs(np.asarray(x_centered, dtype=float))
    channel = np.digitize(a / L_B, BIN_EDGES).astype(int) + 1
    return channel * (1 - (a >= CAVITY_EDGE).astype(int))


def envelope_keep(re, rg, env):
    circ_rg, circ_re, min_rg, min_re = env
    i = np.argsort(circ_re)
    upper = np.interp(re, circ_re[i], circ_rg[i], left=np.nan, right=np.nan)
    i = np.argsort(min_re)
    lower = np.interp(re, min_re[i], min_rg[i], left=np.nan, right=np.nan)
    return ((rg <= upper) | np.isnan(upper)) & ((rg >= lower) | np.isnan(lower))


def exp_scatter(T, env):
    df = pd.read_csv(DATA / "exp" / "cavity" / f"ReRg_{T}.csv")
    re, rg, x = df["Ree"].to_numpy(float), df["Rgy"].to_numpy(float), df["com (x)"].to_numpy(float)
    valid = np.isfinite(re) & np.isfinite(rg) & np.isfinite(x)
    re, rg, x = re[valid], rg[valid], x[valid]
    keep = envelope_keep(re, rg, env)
    return re[keep], rg[keep], position_category(x)[keep]


def row_figure(panels, env, name):
    circ_rg, circ_re, min_rg, min_re = env
    fig = plt.figure(figsize=(11, 3.6))
    gs = GridSpec(1, 4, width_ratios=[1, 1, 1, 0.06], wspace=0)
    for col, (label, re, rg, cat) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        for c in range(5):
            m = cat == c
            if m.any():
                ax.scatter(rg[m], re[m], c=CAT_COLORS[c], s=3, alpha=0.35, linewidths=0, rasterized=True)
        ax.plot(circ_rg, circ_re, "-", color="k", lw=2.0, alpha=0.7)
        ax.plot(min_rg, min_re, "-", color="k", lw=2.0, alpha=0.7)
        ax.set_xlim(RG_RANGE)
        ax.set_ylim(RE_RANGE)
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=13.5, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set_xlabel(LAB_RG)
        ax.set_xticks([0.0, 0.1, 0.2, 0.3] if col == 0 else [0.1, 0.2, 0.3])
        if col == 0:
            ax.set_ylabel(LAB_RE)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP_CAT, norm=NORM_CAT), cax=fig.add_subplot(gs[0, 3]), ticks=range(5))
    cbar.ax.set_yticklabels(CAT_LABELS)
    cbar.set_label(r"$x_{\mathrm{cm}}/L_b$")
    save(fig, name)


def main():
    set_style_framed()
    env = load_boundaries()
    panels = [(f"Exp {TEMP_LABELS[T]}", *exp_scatter(T, env)) for T in TEMPERATURES]
    row_figure(panels, env, "figS_conformational_maps_channel_exp")


if __name__ == "__main__":
    main()
