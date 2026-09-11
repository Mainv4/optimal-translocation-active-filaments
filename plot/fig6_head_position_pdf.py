import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from common import DATA, record, save, set_style

CAVITY_RADIUS = 4.0
OPENING_HALF_WIDTH = 1.0
CAVITY_CENTER_X = 44.0


def head_positions():
    dx, dy = [], []
    for path in sorted((DATA / "exp" / "cavity").glob("*/worm_*.csv")):
        df = pd.read_csv(path)
        ex, ey = df["end_1 (x)"].to_numpy(), df["end_1 (y)"].to_numpy()
        valid = np.isfinite(ex) & np.isfinite(ey)
        ex, ey = ex[valid], ey[valid]
        for x in (ex, -ex):
            x_max = float(np.max(x))
            if x_max < CAVITY_CENTER_X - 1.0:
                continue
            x_local = x + (CAVITY_RADIUS - x_max)
            inside = x_local ** 2 + ey ** 2 <= CAVITY_RADIUS ** 2
            dx.append(x_local[inside])
            dy.append(ey[inside])
    return np.concatenate(dx), np.concatenate(dy)


def main():
    set_style()
    hx, hy = head_positions()
    r_pad = CAVITY_RADIUS * 1.05
    H, xe, ye = np.histogram2d(hx, hy, bins=40, range=[[-r_pad, r_pad], [-r_pad, r_pad]])
    H = H.T
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
    XC, YC = np.meshgrid(xc, yc)
    H = np.ma.masked_where((XC ** 2 + YC ** 2 > CAVITY_RADIUS ** 2) | (H == 0), H)
    keep = ~np.ma.getmaskarray(H)
    record("B inset", "living worms, all temperatures", x_mm=XC[keep], y_mm=YC[keep], counts=H.data[keep])
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.imshow(H, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis",
              norm=LogNorm(), aspect="equal", interpolation="nearest", zorder=1)
    theta_open = np.arcsin(OPENING_HALF_WIDTH / CAVITY_RADIUS)
    arc = np.linspace(-np.pi + theta_open, np.pi - theta_open, 200)
    ax.plot(CAVITY_RADIUS * np.cos(arc), CAVITY_RADIUS * np.sin(arc), color="0.3", lw=1.2, zorder=3)
    x_open, x_end = -CAVITY_RADIUS * np.cos(theta_open), -CAVITY_RADIUS - 1.5
    for sign in (1, -1):
        ax.plot([x_open, x_end], [sign * OPENING_HALF_WIDTH] * 2, color="0.3", lw=1.2, zorder=3)
    ax.set_xlim(x_end - 0.3, CAVITY_RADIUS * 1.15)
    ax.set_ylim(-CAVITY_RADIUS * 1.15, CAVITY_RADIUS * 1.15)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig6_head_position_pdf")


if __name__ == "__main__":
    main()
