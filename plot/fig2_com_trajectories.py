import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, exp_cavity_files, record, save, set_style_framed

PE, T_SIM, K = 0.5, 0.2, 0.8
TARGET_T = 20
SIM_X_OFFSET = 16.0
HALF_DEVICE = 48.0
CAVITY_LEFT, CAVITY_RIGHT = 10.0, 86.0
L_MAX = 38.0
SIM_FRAME_TO_MIN = 1.0 / 60.0
EXP_FRAME_TO_MIN = 2.0 / 60.0
MATCH_DURATION_MIN = 60.0
C_EXP = "#73A6A3"
C_SIM = "#CE7E5A"


def sim_trajectories():
    data = np.loadtxt(DATA / "sim" / "confinement" / "center_of_mass" / f"Pe_{PE}_T_{T_SIM}_k_{K}__CenterOfMassOverTime.dat",
                      skiprows=1)
    n_poly = (data.shape[1] - 1) // 2
    x = (data[:, 1:1 + 2 * n_poly:2] + SIM_X_OFFSET) / 2.0 - HALF_DEVICE
    return [np.column_stack([data[:, 0], x[:, i]]) for i in range(n_poly)]


def exp_trajectories():
    trajs, coms = [], []
    for path in exp_cavity_files():
        df = pd.read_csv(path)
        if int(df["T"].dropna().iloc[0]) != TARGET_T:
            continue
        x = df["com (x)"].values
        keep = np.isfinite(x)
        trajs.append(np.column_stack([df.iloc[:, 0].values[keep], x[keep]]))
        coms.extend(x[keep].tolist())
    return trajs, np.asarray(coms)


def break_trajectory(traj):
    segments, start, in_bridge, n_ch, n_tr = [], 0, False, 0, 0
    for j, x in enumerate(traj[:, 1]):
        if -L_MAX < x < L_MAX and not in_bridge:
            n_ch += 1
        if (x < -L_MAX or x > L_MAX) and in_bridge:
            n_tr += 1
        if n_ch == 3:
            in_bridge, n_ch = True, 0
            segments.append(traj[start:j])
            start = j
        if n_tr == 3:
            in_bridge, n_tr = False, 0
            segments.append(traj[start:j])
            start = j
    if start < traj.shape[0]:
        segments.append(traj[start:])
    return segments


def fingerprint(segments, frame_to_min):
    total = sum(s.shape[0] for s in segments)
    n_left = sum(np.sum(s[:, 1] < -L_MAX) for s in segments)
    n_right = sum(np.sum(s[:, 1] > L_MAX) for s in segments)
    rate = len(segments) / (total * frame_to_min)
    return rate, n_left / total, (total - n_left - n_right) / total, n_right / total


def best_pair(sim_segmented, exp_segmented):
    sim_fp = [fingerprint(s, SIM_FRAME_TO_MIN) for s in sim_segmented]
    exp_fp = [fingerprint(s, EXP_FRAME_TO_MIN) for s in exp_segmented]
    best, best_dist = (0, 0), np.inf
    for si, sfp in enumerate(sim_fp):
        for ei, efp in enumerate(exp_fp):
            if sfp[0] == 0 or efp[0] == 0:
                continue
            dist = abs(sfp[0] - efp[0]) / max(sfp[0], efp[0]) + np.sqrt(sum((a - b) ** 2 for a, b in zip(sfp[1:], efp[1:])))
            if dist < best_dist:
                best, best_dist = (si, ei), dist
    return best


def draw_trajectory(ax, segments, color, frame_to_min, t_min_cutoff, panel=None, series=None):
    for seg in segments:
        t = seg[:, 0] * frame_to_min
        keep = t >= t_min_cutoff
        if not keep.any():
            continue
        t, x = t[keep], seg[keep, 1] + HALF_DEVICE
        ax.plot(t, x, lw=3, color=color)
        record(panel, series, t_min=t, x_cm_mm=x) if panel else None
        mean_x = np.mean(seg[keep, 1])
        if mean_x < -L_MAX:
            ax.fill_between(t, -2, CAVITY_LEFT, color="silver", alpha=0.3)
        elif mean_x > L_MAX:
            ax.fill_between(t, CAVITY_RIGHT, 98, color="silver", alpha=0.3)
        else:
            ax.fill_between(t, CAVITY_LEFT, CAVITY_RIGHT, color="silver", alpha=0.3)
    ax.axhline(CAVITY_LEFT, color="gray", ls="--", lw=2)
    ax.axhline(CAVITY_RIGHT, color="gray", ls="--", lw=2)
    ax.set_ylabel(r"$x_{\mathrm{cm}}$ (mm)")
    ax.set_xlabel(r"$t$ (min)")
    ax.set_xlim(0, 61)
    ax.set_ylim(-2, 98)


def selected_pair():
    sim = [tr[:int(MATCH_DURATION_MIN * 60)] for tr in sim_trajectories()]
    exp, coms = exp_trajectories()
    sim_segmented = [break_trajectory(tr) for tr in sim]
    exp_segmented = [break_trajectory(tr) for tr in exp]
    si, ei = best_pair(sim_segmented, exp_segmented)
    return sim_segmented[si], exp_segmented[ei]


def main():
    set_style_framed()
    sim_segments, exp_segments = selected_pair()
    layout = (("fig2_com_trajectory_exp", exp_segments, C_EXP, EXP_FRAME_TO_MIN, 4.0, "C", "living worm 20 C"),
              ("fig2_com_trajectory_sim", sim_segments, C_SIM, SIM_FRAME_TO_MIN, 3.0, "D", "model 20 C"))
    for name, segments, color, scale, cutoff, panel, series in layout:
        fig, ax = plt.subplots(figsize=(9, 3.5))
        draw_trajectory(ax, segments, color, scale, cutoff, panel, series)
        fig.tight_layout()
        save(fig, name)


if __name__ == "__main__":
    main()
