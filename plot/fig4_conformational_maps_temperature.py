import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec

from common import (DATA, LAB_RE, LAB_RG, RE_RANGE, RG_RANGE, SMOOTH_SIGMA, TEMPERATURES, confmap_density,
                    filter_by_envelope, load_boundaries, read_rere_csv, save, set_style_framed)

VMAX_RATIO = 0.8
LABELS = {T: rf"$T = {T}\,^{{\circ}}$C" for T in TEMPERATURES}


def main():
    set_style_framed()
    env = load_boundaries()
    density, mask = {}, None
    for T in TEMPERATURES:
        re, rg = filter_by_envelope(*read_rere_csv(DATA / "exp" / "cavity" / f"ReRg_{T}.csv"), env)
        density[T], mask = confmap_density(re, rg, env, SMOOTH_SIGMA)
    fig = plt.figure(figsize=(5, 7))
    gs = GridSpec(2, 3, width_ratios=[1, 0.03, 0.08], wspace=0, hspace=0.05)
    ax_top, ax_bot = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[:, 2])
    norm = TwoSlopeNorm(vmin=-VMAX_RATIO, vcenter=0, vmax=VMAX_RATIO)
    cmap = plt.cm.PuOr_r.copy()
    cmap.set_bad(color="white")
    circ_rg, circ_re, min_rg, min_re = env
    for ax, T in ((ax_top, 10), (ax_bot, 30)):
        valid = (density[T] > 0) & (density[20] > 0)
        ratio = np.full_like(density[T], np.nan)
        ratio[valid] = np.log10(density[T][valid] / density[20][valid])
        ratio[~mask] = np.nan
        ratio = np.where(np.isnan(ratio), np.nan, np.clip(ratio, -VMAX_RATIO, VMAX_RATIO))
        mesh = ax.imshow(ratio.T, origin="lower", extent=[*RG_RANGE, *RE_RANGE], aspect="auto", cmap=cmap,
                         norm=norm, interpolation="nearest")
        ax.plot(circ_rg, circ_re, "-", color="0.4", lw=1.5)
        ax.plot(min_rg, min_re, "-", color="0.4", lw=1.5)
        ax.set_xlim(RG_RANGE)
        ax.set_ylim(RE_RANGE)
        ax.set_xticks([0, 0.1, 0.2, 0.3])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_ylabel(LAB_RE)
        ax.text(0.95, 0.08, LABELS[T], transform=ax.transAxes, fontsize=12, va="bottom", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none"))
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_bot.set_xlabel(LAB_RG)
    fig.colorbar(mesh, cax=cax).set_label(r"$\log_{10}(P_T\,/\,P_{20^\circ\mathrm{C}})$")
    save(fig, "fig4_conformational_maps_temperature")


if __name__ == "__main__":
    main()
