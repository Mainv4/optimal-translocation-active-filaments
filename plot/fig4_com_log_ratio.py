import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import TEMPERATURES, exp_cavity_files, record, save, set_style_framed

DEVICE_LENGTH = 96.0
HALF_DEVICE = 48.0
CAVITY_BOUNDARY = 8.0
CAVITY_THRESHOLD = 40.0
N_BINS = 60
PANEL_COLORS = {10: "#2060B0", 30: "#B02020"}
LABELS = {T: rf"$T = {T}\,^{{\circ}}$C" for T in TEMPERATURES}


def folded_com_positions():
    out = {T: [] for T in TEMPERATURES}
    for path in exp_cavity_files():
        df = pd.read_csv(path)
        T = int(df["T"].dropna().iloc[0])
        x = df["com (x)"].values
        x = x[np.isfinite(x)]
        candidates = [x + (HALF_DEVICE - x.max()), -x + (HALF_DEVICE - (-x).max())]
        aligned = max(candidates, key=lambda a: np.sum(a > CAVITY_THRESHOLD))
        x_panel = HALF_DEVICE - aligned
        out[T].extend(np.where(x_panel <= HALF_DEVICE, x_panel, DEVICE_LENGTH - x_panel).tolist())
    return {T: np.asarray(v) for T, v in out.items()}


def main():
    set_style_framed()
    com = folded_com_positions()
    bins = np.linspace(0, HALF_DEVICE, N_BINS)
    centers = 0.5 * (bins[:-1] + bins[1:])
    h20, _ = np.histogram(com[20], bins=bins, density=True)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(4.5, 5), sharex=True)
    for ax, T in ((ax_top, 10), (ax_bot, 30)):
        h, _ = np.histogram(com[T], bins=bins, density=True)
        valid = (h > 0) & (h20 > 0)
        ratio = np.full_like(h, np.nan)
        ratio[valid] = np.log(h[valid] / h20[valid])
        ax.bar(centers, ratio, centers[1] - centers[0], color=PANEL_COLORS[T], edgecolor="white", linewidth=0.5)
        record("A", f"living worms {T} C over 20 C", x_cm_mm=centers, log_probability_ratio=ratio)
        ax.axvline(0, color="0.7", ls="--", lw=1.5)
        ax.axvline(CAVITY_BOUNDARY, color="0.7", ls="--", lw=1.5)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_ylim(-1.2, 1.2)
        ax.text(0.95, 0.92, LABELS[T], transform=ax.transAxes, fontsize=12, va="top", ha="right")
    ax_bot.set_xlabel(r"$x_{\mathrm{cm}}$ (mm)")
    fig.supylabel(r"$\log\bigl(P_T(x_{\mathrm{cm}})\,/\,P_{20^\circ\mathrm{C}}(x_{\mathrm{cm}})\bigr)$", fontsize=18, x=-0.02)
    fig.subplots_adjust(hspace=0.05)
    save(fig, "fig4_com_log_ratio")


if __name__ == "__main__":
    main()
