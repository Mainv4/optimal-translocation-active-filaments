import matplotlib.pyplot as plt
import numpy as np

from common import record, save, set_style_framed
from fig2_com_trajectories import C_EXP, C_SIM, HALF_DEVICE, exp_trajectories, sim_trajectories

DEVICE_LENGTH = 96.0
CAVITY_DEPTH = 8.0
Y_MAX = 0.07


def main():
    set_style_framed()
    _, exp_coms = exp_trajectories()
    sim_coms = np.concatenate([tr[:, 1] for tr in sim_trajectories()])
    bins = np.linspace(0, DEVICE_LENGTH, 40)
    centers = 0.5 * (bins[1:] + bins[:-1])
    width = centers[1] - centers[0]
    h_exp, _ = np.histogram(-(exp_coms - HALF_DEVICE), bins=bins, density=True)
    h_sim, _ = np.histogram(sim_coms + HALF_DEVICE, bins=bins, density=True)
    fig, (ax_exp, ax_sim) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, h, color, series in ((ax_exp, h_exp, C_EXP, "living worms 20 C"), (ax_sim, h_sim, C_SIM, "model 20 C")):
        ax.bar(centers, h, width, color=color, edgecolor="white", linewidth=0.5)
        record("E", series, x_cm_mm=centers, probability_density_per_mm=h)
        for x in (0, DEVICE_LENGTH, CAVITY_DEPTH, DEVICE_LENGTH - CAVITY_DEPTH):
            ax.vlines(x, 0, Y_MAX, colors="gray", linestyles="dashed", lw=4)
        ax.set_xlabel(r"$x_{cm}$ (mm)", size=22)
        ax.tick_params(axis="both", labelsize=22)
        for sp in ax.spines.values():
            sp.set_linewidth(2.5)
        ax.set_ylim(0, Y_MAX)
    ax_exp.set_ylabel(r"$P(x_{cm})$", size=22)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    save(fig, "fig2_com_distribution")


if __name__ == "__main__":
    main()
