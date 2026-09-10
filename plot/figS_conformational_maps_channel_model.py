import numpy as np

from common import CALIBRATED, CONTOUR_LENGTH, DATA, TEMPERATURES, load_boundaries, set_style_framed
from fig2_com_trajectories import HALF_DEVICE, SIM_X_OFFSET
from figS_conformational_maps_channel_exp import envelope_keep, position_category, row_figure

STEMS = {10: "Pe_0.3_T_0.2_k_0.5", 20: "Pe_0.5_T_0.2_k_0.8", 30: "Pe_0.8_T_0.1_k_0.5"}


def sim_scatter(stem, env):
    data = np.load(DATA / "sim" / "confinement" / "conformations" / f"{stem}.npy")
    com = np.loadtxt(DATA / "sim" / "confinement" / "center_of_mass" / f"{stem}__CenterOfMassOverTime.dat", skiprows=1)
    rg, re = data[:, :, 0], data[:, :, 1]
    n_frames, n_poly = min(rg.shape[0], com.shape[0]), min(rg.shape[1], (com.shape[1] - 1) // 2)
    x = (com[:n_frames, 1:1 + 2 * n_poly:2] + SIM_X_OFFSET) / 2.0 - HALF_DEVICE
    rg, re, x = rg[:n_frames, :n_poly].flatten() / CONTOUR_LENGTH, re[:n_frames, :n_poly].flatten() / CONTOUR_LENGTH, x.flatten()
    valid = np.isfinite(rg) & np.isfinite(re) & np.isfinite(x)
    re, rg, x = re[valid], rg[valid], x[valid]
    keep = envelope_keep(re, rg, env)
    return re[keep], rg[keep], position_category(x)[keep]


def main():
    set_style_framed()
    env = load_boundaries()
    panels = []
    for T in TEMPERATURES:
        pe, t, k = CALIBRATED[T]
        panels.append((rf"$f^a\!=\!{pe},\ T\!=\!{t},\ \kappa\!=\!{k}$", *sim_scatter(STEMS[T], env)))
    row_figure(panels, env, "figS_conformational_maps_channel_model")


if __name__ == "__main__":
    main()
