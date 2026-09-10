
import re
import sqlite3
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy import optimize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "exp_analyse_scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "figure_2_translocation"))

import plot_fig2_panels as fig2
import msd_matching as msd_mod

BASE_DIR = Path(__file__).parent.parent
OUTPUT_BASE = Path(__file__).parent / "output" / "gallery"
TABLES_DIR = BASE_DIR / "active-polymer-worms-viz"

COM_DIR = BASE_DIR.parent / "CONFINEMENT" / "N_40" / "DATA_CenterOfMass"
CONFMAP_DIR = BASE_DIR / "DATA_Corr_Rg_Re"
EXP_CONFMAP_DIR = BASE_DIR / "experimental_conformational_maps" / "data" / "free_space"
BOUNDARY_DIR = BASE_DIR / "experimental_conformational_maps" / "data" / "boundaries"

DB_SIM_TRAP = TABLES_DIR / "trapping_times.db"
DB_EXP_TRAP = TABLES_DIR / "exp_trapping_times.db"

TRAP_CUTOFF = 15.0
TRAP_NBINS = 28
MIN_CONFMAP_POINTS = 100_000
MIN_TRAP_EVENTS = 10
MIN_TRAJ_DURATION_MIN = 55
CONTOUR_LENGTH = 39.0
TOP_N = 10

BINS_CONF = 50
RG_RANGE = [0.04, 0.3]
RE_RANGE = [-0.05, 1.05]
VMIN, VMAX = 1e-4, 1e-2

EXP_TEMPS = [10, 20, 30]
TEMP_TO_MSD_LABEL = {10: "10C", 20: "20C", 30: "30C"}
TEMP_TO_CONFMAP_FILE = {10: "ReRg_5.csv", 20: "ReRg_20.csv", 30: "ReRg_30.csv"}
TEMP_TO_CONFMAP_LABEL = {10: r"5$^\circ$C", 20: r"20$^\circ$C", 30: r"30$^\circ$C"}
TEMP_COLORS_EXP = {10: "#75ABDD", 20: "#84D495", 30: "#D77567"}
TEMP_COLORS_SIM = {10: "#4A85B8", 20: "#4DA067", 30: "#B84E3F"}


def compute_pareto_front(objectives):
    n = len(objectives)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        diff = objectives - objectives[i]
        le_all = np.all(diff <= 0, axis=1)
        lt_any = np.any(diff < 0, axis=1)
        dominated_by = le_all & lt_any
        dominated_by[i] = False
        if dominated_by.any():
            is_pareto[i] = False
    return is_pareto


def select_top_n(objectives, n=TOP_N):
    front_mask = compute_pareto_front(objectives)

    lo = objectives.min(axis=0)
    hi = objectives.max(axis=0)
    rng = hi - lo
    rng[rng == 0] = 1.0
    norm_all = (objectives - lo) / rng
    dist_all = np.linalg.norm(norm_all, axis=1)

    all_order = np.argsort(dist_all)
    top = min(n, len(objectives))
    return all_order[:top], front_mask


def get_production_sims():
    sims = {}
    for npy_path in sorted(CONFMAP_DIR.glob("*.npy")):
        m = re.search(r"Pe_([0-9.]+)_T_([0-9.]+)_k_([0-9.]*)", npy_path.stem)
        if not m:
            continue
        pe = float(m.group(1))
        t = float(m.group(2))
        k = float(m.group(3)) if m.group(3) else 0.0
        data = np.load(npy_path, mmap_mode="r")
        if data.shape[0] * data.shape[1] < MIN_CONFMAP_POINTS:
            continue
        if find_com_file(pe, t, k) is None:
            continue
        sims[(pe, t, k)] = npy_path
    return sims


def load_sim_trapping_all():
    conn = sqlite3.connect(DB_SIM_TRAP)
    df = pd.read_sql(
        "SELECT Pe, T, kappa, time_min FROM trapping_events WHERE N = 40", conn
    )
    conn.close()
    datasets = {}
    for (pe, t, k), grp in df.groupby(["Pe", "T", "kappa"]):
        times = grp["time_min"].values
        if len(times) >= MIN_TRAP_EVENTS:
            datasets[(float(pe), float(t), float(k))] = times
    return datasets


def load_exp_trapping(temp_c):
    conn = sqlite3.connect(DB_EXP_TRAP)
    cur = conn.cursor()
    cur.execute(
        "SELECT time_min FROM exp_trapping_events WHERE T_exp = ?", (temp_c,)
    )
    times = np.array([r[0] for r in cur.fetchall()])
    conn.close()
    return times


def load_exp_confmap(temp_c):
    fname = TEMP_TO_CONFMAP_FILE[temp_c]
    fpath = EXP_CONFMAP_DIR / fname
    if not fpath.exists():
        return None, None
    df = pd.read_csv(fpath)
    re_col = "Ree" if "Ree" in df.columns else "Re"
    rg_col = "Rgy" if "Rgy" in df.columns else "Rg"
    Re = df[re_col].values
    Rg = df[rg_col].values
    valid = np.isfinite(Re) & np.isfinite(Rg)
    return Re[valid], Rg[valid]


def _fit_logspace(times, bins):
    counts, edges = np.histogram(times, bins=bins, density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    mask = (counts > 0) & (centers > 0)
    if mask.sum() < 3:
        return None, None, None
    log_c = np.log(counts[mask])
    try:
        popt, _ = optimize.curve_fit(
            lambda x, a, b: a * x + b, centers[mask], log_c, p0=[-1, 0]
        )
        slope, intercept = popt
        if slope >= 0:
            return None, None, None
        tau = -1.0 / slope
        A = np.exp(intercept)
        if tau <= 0 or A <= 0:
            return None, None, None
        predicted = slope * centers[mask] + intercept
        ss_res = np.sum((log_c - predicted) ** 2)
        ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return tau, A, r2
    except Exception:
        return None, None, None


def compute_m1(times):
    times_cut = times[times <= TRAP_CUTOFF]
    if len(times_cut) < MIN_TRAP_EVENTS:
        return np.nan
    bins = np.logspace(np.log10(0.05), np.log10(TRAP_CUTOFF), TRAP_NBINS + 1)
    tau_fit, _, _ = _fit_logspace(times_cut, bins)
    if tau_fit is None:
        return np.nan
    tau_mean = float(np.mean(times_cut))
    return abs(tau_fit - tau_mean) / tau_mean


def compute_rmse_for_sim(pe, t, k, exp_msd):
    msd_file = msd_mod.SIM_DATA_DIR / f"Pe_{pe}_T_{t}_k_{k}_MSD_CM.dat"
    if not msd_file.exists():
        for f in msd_mod.SIM_DATA_DIR.glob(f"Pe_{pe}*T_{t}*k_{k}*MSD_CM*"):
            msd_file = f
            break
    if not msd_file.exists():
        return np.inf
    times_s, msd_s = msd_mod.load_simulation_msd(msd_file)
    if times_s is None:
        return np.inf
    return msd_mod.compute_rmse_log(
        exp_msd["times"], exp_msd["msd"], times_s, msd_s, *msd_mod.MATCH_INTERVAL
    )


def find_com_file(pe, t, k):
    for f in COM_DIR.glob(f"*Pe_{pe}*T_{t}*k_{k}*CenterOfMass*"):
        return f
    for f in COM_DIR.glob(f"*Pe_{pe:.1f}*T_{t}*k_{k}*CenterOfMass*"):
        return f
    if k == int(k):
        for f in COM_DIR.glob(f"*Pe_{pe}*T_{t}*k_{int(k)}.*CenterOfMass*"):
            return f
    return None


def find_confmap_file(pe, t, k):
    exact = CONFMAP_DIR / f"Pe_{pe}_T_{t}_k_{k}.npy"
    if exact.exists():
        return exact
    if k == int(k):
        alt = CONFMAP_DIR / f"Pe_{pe}_T_{t}_k_{int(k)}.npy"
        if alt.exists():
            return alt
    for f in CONFMAP_DIR.glob(f"Pe_{pe}*T_{t}*k_{k}*.npy"):
        return f
    return None


def load_sim_trajs(filepath):
    data = np.loadtxt(filepath, skiprows=1)
    n_poly = (data.shape[1] - 1) // 2
    trajs = []
    for i in range(n_poly):
        x_col = 1 + 2 * i
        t = data[:, 0]
        x_mm = (data[:, x_col] + fig2.SIM_X_OFFSET) / 2.0 - 48.0
        trajs.append(np.column_stack([t, x_mm]))
    return trajs


def load_sim_confmap(npy_path):
    data = np.load(npy_path)
    rg_l = data[:, :, 0].flatten() / CONTOUR_LENGTH
    re_l = data[:, :, 1].flatten() / CONTOUR_LENGTH
    valid = np.isfinite(re_l) & np.isfinite(rg_l)
    return re_l[valid], rg_l[valid]


def load_boundaries():
    circ_path = BOUNDARY_DIR / "Rg_Re_Circle.txt"
    min_path = BOUNDARY_DIR / "empirical_min_boundary.txt"
    circ_Rg, circ_Re = (
        (np.loadtxt(circ_path)[:, 0], np.loadtxt(circ_path)[:, 1])
        if circ_path.exists()
        else (None, None)
    )
    min_Rg, min_Re = (
        (np.loadtxt(min_path)[:, 0], np.loadtxt(min_path)[:, 1])
        if min_path.exists()
        else (None, None)
    )
    return circ_Rg, circ_Re, min_Rg, min_Re


def filter_by_envelope(Re, Rg, circ_Rg, circ_Re, min_Rg, min_Re):
    valid = np.ones(len(Re), dtype=bool)
    if circ_Re is not None:
        idx = np.argsort(circ_Re)
        max_rg = np.interp(Re, circ_Re[idx], circ_Rg[idx], left=np.nan, right=np.nan)
        valid &= (Rg <= max_rg) | np.isnan(max_rg)
    if min_Re is not None:
        idx = np.argsort(min_Re)
        min_rg = np.interp(Re, min_Re[idx], min_Rg[idx], left=np.nan, right=np.nan)
        valid &= (Rg >= min_rg) | np.isnan(min_rg)
    return Re[valid], Rg[valid]


def make_heatmap(ax, Re, Rg, label, circ_Rg, circ_Re, min_Rg, min_Re,
                 smooth_sigma=0.0):
    h, xedges, yedges = np.histogram2d(
        Rg, Re, bins=BINS_CONF, range=[RG_RANGE, RE_RANGE]
    )
    density = h / h.sum()
    if smooth_sigma > 0:
        density = gaussian_filter(density, sigma=smooth_sigma)
        rg_centers = 0.5 * (xedges[:-1] + xedges[1:])
        re_centers = 0.5 * (yedges[:-1] + yedges[1:])
        rg_grid, re_grid = np.meshgrid(rg_centers, re_centers, indexing='ij')
        mask = (re_grid >= 0) & (re_grid <= 1.0) & (rg_grid > 0)
        if circ_Rg is not None and circ_Re is not None:
            idx = np.argsort(circ_Re)
            max_rg = np.interp(re_grid, circ_Re[idx], circ_Rg[idx], left=0, right=0)
            mask &= (rg_grid <= max_rg)
        if min_Rg is not None and min_Re is not None:
            idx = np.argsort(min_Re)
            min_rg_interp = np.interp(re_grid, min_Re[idx], min_Rg[idx],
                                      left=np.inf, right=np.inf)
            mask &= (rg_grid >= min_rg_interp)
        density[~mask] = 0
    ax.imshow(
        density.T,
        origin="lower",
        extent=[RG_RANGE[0], RG_RANGE[1], RE_RANGE[0], RE_RANGE[1]],
        aspect="auto",
        norm=LogNorm(vmin=VMIN, vmax=VMAX),
        cmap="viridis",
    )
    if circ_Rg is not None:
        ax.plot(circ_Rg, circ_Re, "r-", lw=1, alpha=0.8)
    if min_Rg is not None:
        ax.plot(min_Rg, min_Re, "b-", lw=1, alpha=0.8)
    ax.text(
        0.05, 0.95, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
        va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="none"),
    )
    ax.set_xlim(RG_RANGE)
    ax.set_ylim(RE_RANGE)
    ax.set_xlabel(r"$R_g/L$")
    ax.set_ylabel(r"$R_e/L$")


def load_sim_trapping(pe, t, k):
    conn = sqlite3.connect(DB_SIM_TRAP)
    cur = conn.cursor()
    cur.execute(
        """SELECT time_min FROM trapping_events
           WHERE N = 40 AND ABS(Pe - ?) < 0.001
             AND ABS(T - ?) < 0.001 AND ABS(kappa - ?) < 0.001""",
        (pe, t, k),
    )
    times = np.array([r[0] for r in cur.fetchall()])
    conn.close()
    return times


def plot_msd_panel(ax, pe, t, k, exp_msd, rmse_val, temp_c):
    c_exp = TEMP_COLORS_EXP[temp_c]
    c_sim = TEMP_COLORS_SIM[temp_c]

    ax.axvspan(*msd_mod.MATCH_INTERVAL, color="#DDDDDD", alpha=0.5, zorder=0)

    v = ~np.isnan(exp_msd["msd"]) & (exp_msd["msd"] > 0) & (exp_msd["times"] > 0)
    ax.plot(exp_msd["times"][v], exp_msd["msd"][v], color=c_exp, lw=2.0, ls="-", zorder=3)

    msd_file = msd_mod.SIM_DATA_DIR / f"Pe_{pe}_T_{t}_k_{k}_MSD_CM.dat"
    if not msd_file.exists():
        for f in msd_mod.SIM_DATA_DIR.glob(f"Pe_{pe}*T_{t}*k_{k}*MSD_CM*"):
            msd_file = f
            break
    if msd_file.exists():
        times_s, msd_s = msd_mod.load_simulation_msd(msd_file)
        if times_s is not None:
            vs = (times_s > 0) & (msd_s > 0) & ~np.isnan(msd_s)
            ax.plot(times_s[vs], msd_s[vs], color=c_sim, lw=1.5, ls="--", zorder=4)

    rmse_str = f"RMSE={rmse_val:.3f}" if np.isfinite(rmse_val) else "no data"
    handles = [
        Line2D([0], [0], color=c_exp, lw=2, ls="-",
               label=rf"{temp_c}$^\circ$C exp"),
        Line2D([0], [0], color=c_sim, lw=1.5, ls="--",
               label=f"sim ({rmse_str})"),
    ]
    ax.legend(handles=handles, fontsize=6, loc="upper left", frameon=True,
              fancybox=False, edgecolor="none", facecolor="white", framealpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$\langle r^2 \rangle$ (mm$^2$)")
    ax.set_xlim(msd_mod.PLOT_T_MIN, msd_mod.PLOT_T_MAX)
    ax.set_ylim(msd_mod.MSD_YLIM)


def plot_trapping_panel(ax, sim_times, exp_times, temp_c):
    bins = np.logspace(np.log10(0.05), np.log10(TRAP_CUTOFF), TRAP_NBINS + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    x_fit = np.linspace(0.01, TRAP_CUTOFF, 200)
    c_exp = TEMP_COLORS_EXP[temp_c]
    c_sim = TEMP_COLORS_SIM[temp_c]

    if len(exp_times) > 0:
        exp_clip = exp_times[exp_times <= TRAP_CUTOFF]
        counts_e, _ = np.histogram(exp_clip, bins=bins, density=True)
        mask_e = counts_e > 0
        ax.plot(centers[mask_e], counts_e[mask_e], "o", color=c_exp, ms=7,
                markeredgecolor="k", markeredgewidth=0.5, zorder=5,
                label=rf"Exp {temp_c}$^\circ$C")
        tau_e, A_e, _ = _fit_logspace(exp_clip, bins)
        if tau_e is not None:
            ax.plot(x_fit, A_e * np.exp(-x_fit / tau_e), "--", color=c_exp,
                    lw=1.5, zorder=3, label=rf"$\tau_e$={tau_e:.2f}")
        mu_e = float(np.mean(exp_clip))
        ax.plot(x_fit, (1.0 / mu_e) * np.exp(-x_fit / mu_e), "-.",
                color=c_exp, lw=1.2, zorder=2)

    if len(sim_times) > 0:
        sim_clip = sim_times[sim_times <= TRAP_CUTOFF]
        counts_s, _ = np.histogram(sim_clip, bins=bins, density=True)
        mask_s = counts_s > 0
        ax.plot(centers[mask_s], counts_s[mask_s], "o", color=c_sim, ms=7,
                markerfacecolor="none", markeredgecolor=c_sim, markeredgewidth=1.5,
                zorder=4, label="Sim")
        tau_s, A_s, _ = _fit_logspace(sim_clip, bins)
        if tau_s is not None:
            ax.plot(x_fit, A_s * np.exp(-x_fit / tau_s), "--", color=c_sim,
                    lw=1.5, zorder=3, label=rf"$\tau_s$={tau_s:.2f}")
        mu_s = float(np.mean(sim_clip))
        ax.plot(x_fit, (1.0 / mu_s) * np.exp(-x_fit / mu_s), "-.",
                color=c_sim, lw=1.2, zorder=2)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ (min)")
    ax.set_ylabel(r"$P(\tau)$")
    ax.set_xlim(0, TRAP_CUTOFF)
    ax.legend(fontsize=6, loc="upper right", frameon=True,
              fancybox=False, edgecolor="none", facecolor="white", framealpha=0.85)


MATCH_DURATION_MIN = 60.0
MATCH_DURATION_SIM_FRAMES = int(MATCH_DURATION_MIN * 60)


def truncate_traj(traj, max_frames):
    if traj.shape[0] <= max_frames:
        return traj
    return traj[:max_frames]


def traj_state_fractions(segments, frame_to_min):
    lmax = 38
    total_frames = sum(seg.shape[0] for seg in segments)
    if total_frames == 0:
        return 0.0, 0.0, 0.0, 0.0

    n_left = sum(np.sum(seg[:, 1] < -lmax) for seg in segments)
    n_right = sum(np.sum(seg[:, 1] > lmax) for seg in segments)
    n_bridge = total_frames - n_left - n_right

    rate = len(segments) / (total_frames * frame_to_min) if total_frames > 0 else 0.0
    return rate, n_left / total_frames, n_bridge / total_frames, n_right / total_frames


def find_best_traj_pair(sim_trajs, exp_all_segments, exp_frame_to_min):
    sim_truncated = [truncate_traj(tr, MATCH_DURATION_SIM_FRAMES) for tr in sim_trajs]
    sim_segmented = fig2.break_trajectory(sim_truncated)

    sim_fps = []
    for segs in sim_segmented:
        sim_fps.append(traj_state_fractions(segs, fig2.SIM_FRAME_TO_MIN))

    exp_fps = []
    for segs in exp_all_segments:
        exp_fps.append(traj_state_fractions(segs, exp_frame_to_min))

    best_dist = np.inf
    best_si, best_ei = 0, 0
    for si, sfp in enumerate(sim_fps):
        s_rate, s_left, s_bridge, s_right = sfp
        if s_rate == 0:
            continue
        for ei, efp in enumerate(exp_fps):
            e_rate, e_left, e_bridge, e_right = efp
            if e_rate == 0:
                continue
            rate_diff = abs(s_rate - e_rate) / max(s_rate, e_rate)
            state_diff = np.sqrt(
                (s_left - e_left) ** 2
                + (s_bridge - e_bridge) ** 2
                + (s_right - e_right) ** 2
            )
            dist = rate_diff + state_diff
            if dist < best_dist:
                best_dist = dist
                best_si, best_ei = si, ei

    sim_rate = sim_fps[best_si][0]
    exp_rate = exp_fps[best_ei][0]
    return (best_si, best_ei, sim_segmented[best_si], exp_all_segments[best_ei],
            sim_rate, exp_rate)


def make_gallery_figure(pe, t, k, temp_c, m1_val, rmse_val, rank,
                        exp_msd, exp_trap, exp_re, exp_rg,
                        boundaries, exp_all_segments,
                        confmap_path, out_dir):
    circ_Rg, circ_Re, min_Rg, min_Re = boundaries
    tag = f"Pe{pe}_T{t}_k{k}".replace(".", "p")

    fig = plt.figure(figsize=(18, 9))
    gs_outer = gridspec.GridSpec(
        3, 2, width_ratios=[1, 1.6], height_ratios=[1, 1, 1],
        hspace=0.35, wspace=0.25,
    )

    ax_msd = fig.add_subplot(gs_outer[0, 0])
    plot_msd_panel(ax_msd, pe, t, k, exp_msd, rmse_val, temp_c)

    gs_conf = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[1, 0], wspace=0)
    ax_csim = fig.add_subplot(gs_conf[0, 0])
    if confmap_path is not None:
        sim_re, sim_rg = load_sim_confmap(confmap_path)
        sim_re_f, sim_rg_f = filter_by_envelope(
            sim_re, sim_rg, circ_Rg, circ_Re, min_Rg, min_Re
        )
        make_heatmap(ax_csim, sim_re_f, sim_rg_f, "Sim",
                     circ_Rg, circ_Re, min_Rg, min_Re)
    else:
        ax_csim.text(0.5, 0.5, "No data", transform=ax_csim.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")

    ax_cexp = fig.add_subplot(gs_conf[0, 1])
    if exp_re is not None:
        conf_label = f"Exp {TEMP_TO_CONFMAP_LABEL[temp_c]}"
        make_heatmap(ax_cexp, exp_re, exp_rg, conf_label,
                     circ_Rg, circ_Re, min_Rg, min_Re, smooth_sigma=1.0)
    else:
        ax_cexp.text(0.5, 0.5, "No data", transform=ax_cexp.transAxes,
                     ha="center", va="center", fontsize=10, color="gray")
    ax_cexp.set_ylabel("")
    ax_cexp.set_yticklabels([])

    ax_trap = fig.add_subplot(gs_outer[2, 0])
    sim_trap = load_sim_trapping(pe, t, k)
    plot_trapping_panel(ax_trap, sim_trap, exp_trap, temp_c)

    gs_right = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[:, 1], hspace=0.3)

    ax_sim_traj = fig.add_subplot(gs_right[0])
    ax_exp_traj = fig.add_subplot(gs_right[1])

    com_file = find_com_file(pe, t, k)
    if com_file is not None and len(exp_all_segments) > 0:
        trajs = load_sim_trajs(com_file)
        (sim_idx, exp_idx, sim_segs, exp_segs,
         sim_rate, exp_rate) = find_best_traj_pair(
            trajs, exp_all_segments, fig2.EXP_FRAME_TO_MIN)

        fig2.plot_trajectory(ax_sim_traj, sim_segs,
                             fig2.C_SIM, fig2.SIM_FRAME_TO_MIN)
        ax_sim_traj.set_title(
            f"Simulation (rate={sim_rate:.1f}/min)", fontsize=9, loc="left")

        fig2.plot_trajectory(ax_exp_traj, exp_segs,
                             fig2.C_EXP, fig2.EXP_FRAME_TO_MIN)
        ax_exp_traj.set_title(
            rf"Experiment worm \#{exp_idx} (rate={exp_rate:.1f}/min)",
            fontsize=9, loc="left")
    else:
        ax_sim_traj.text(0.5, 0.5, "No COM data", transform=ax_sim_traj.transAxes,
                         ha="center", va="center", fontsize=12, color="gray")
        ax_exp_traj.text(0.5, 0.5, "No exp data", transform=ax_exp_traj.transAxes,
                         ha="center", va="center", fontsize=12, color="gray")

    rmse_str = f"RMSE={rmse_val:.3f}" if np.isfinite(rmse_val) else "no MSD"
    fig.suptitle(
        rf"$f^a\!=\!{pe},\;T\!=\!{t},\;\kappa\!=\!{k}$ --- "
        rf"{rmse_str}, M1={m1_val:.1%} --- "
        rf"Rank {rank} @ {temp_c}$^\circ$C",
        fontsize=11, y=0.99,
    )

    out_path = out_dir / f"{tag}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def plot_pareto_fronts(results_by_temp, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, temp_c in zip(axes, EXP_TEMPS):
        res = results_by_temp[temp_c]
        m1_all = res["m1_all"]
        rmse_all = res["rmse_all"]
        front_mask = res["front_mask"]
        selected_idx = res["selected_idx"]

        c_exp = TEMP_COLORS_EXP[temp_c]

        ax.scatter(m1_all, rmse_all, s=15, c="#CCCCCC", edgecolors="none",
                   alpha=0.6, zorder=1, label="All sims")
        front_idx = np.where(front_mask)[0]
        ax.scatter(m1_all[front_idx], rmse_all[front_idx], s=40,
                   facecolors="none", edgecolors=c_exp, linewidths=1.5,
                   zorder=3, label=f"Pareto front ({len(front_idx)})")
        ax.scatter(m1_all[selected_idx], rmse_all[selected_idx], s=80,
                   c=c_exp, edgecolors="black", linewidths=0.8,
                   zorder=4, label=f"Top {len(selected_idx)}")

        for rank_i, si in enumerate(selected_idx):
            ax.annotate(str(rank_i + 1),
                        (m1_all[si], rmse_all[si]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, fontweight="bold")

        ax.set_xlabel("M1 (trapping)")
        ax.set_ylabel("RMSE MSD (log-space)")
        ax.set_title(rf"{temp_c}$^\circ$C", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto fronts → {out_path}")


def main():
    fig2.set_plot_style()

    print("Scanning production simulations (confmap >= 100k points + COM)...")
    prod_sims = get_production_sims()
    prod_keys = list(prod_sims.keys())
    print(f"  {len(prod_keys)} simulations pass quality filter")

    print("Loading simulation trapping times...")
    trap_data = load_sim_trapping_all()
    print(f"  {len(trap_data)} parameter sets with >= {MIN_TRAP_EVENTS} events")

    print("Computing M1 for each simulation...")
    m1_cache = {}
    for key in tqdm(prod_keys, desc="  M1"):
        if key in trap_data:
            m1_cache[key] = compute_m1(trap_data[key])
        else:
            m1_cache[key] = np.nan

    n_valid_m1 = sum(1 for v in m1_cache.values() if np.isfinite(v))
    print(f"  {n_valid_m1}/{len(prod_keys)} have valid M1")

    print("Loading conformational boundaries...")
    boundaries = load_boundaries()

    all_rows = []
    results_by_temp = {}

    for temp_c in EXP_TEMPS:
        msd_label = TEMP_TO_MSD_LABEL[temp_c]
        print(f"\n{'='*60}")
        print(f"  Temperature: {temp_c}°C (MSD: {msd_label})")
        print(f"{'='*60}")

        exp_msd = msd_mod.load_experimental_msd(msd_label)
        if exp_msd is None:
            print(f"  WARNING: no experimental MSD for {msd_label}, skipping")
            continue
        exp_trap = load_exp_trapping(temp_c)
        print(f"  Exp trapping: {len(exp_trap)} events")
        exp_re, exp_rg = load_exp_confmap(temp_c)
        if exp_re is not None:
            print(f"  Exp confmap: {len(exp_re)} points ({TEMP_TO_CONFMAP_FILE[temp_c]})")

        print(f"  Loading experimental trajectories ({temp_c}°C)...")
        exp_trajs, _ = fig2.load_exp_trajectories(target_T=temp_c)
        if len(exp_trajs) == 0:
            print(f"  WARNING: no exp trajectories for {temp_c}°C, using 20°C fallback")
            exp_trajs, _ = fig2.load_exp_trajectories(target_T=20)

        raw_segments = fig2.break_trajectory(exp_trajs)

        exp_all_segments = []
        for segs in raw_segments:
            if not segs:
                continue
            t_min = segs[0][0, 0]
            t_max = segs[-1][-1, 0]
            duration_min = (t_max - t_min) * fig2.EXP_FRAME_TO_MIN
            if duration_min >= MIN_TRAJ_DURATION_MIN:
                exp_all_segments.append(segs)
        print(f"  Exp worms: {len(raw_segments)} total, "
              f"{len(exp_all_segments)} with >= {MIN_TRAJ_DURATION_MIN} min")

        if len(exp_all_segments) == 0:
            print(f"  WARNING: no exp worms >= {MIN_TRAJ_DURATION_MIN} min, keeping all")
            exp_all_segments = raw_segments

        print(f"  Exp worms available for trajectory matching: {len(exp_all_segments)}")

        print("  Computing RMSE_MSD...")
        rmse_cache = {}
        for key in tqdm(prod_keys, desc=f"  RMSE {temp_c}°C"):
            pe, t, k = key
            rmse_cache[key] = compute_rmse_for_sim(pe, t, k, exp_msd)

        n_valid_rmse = sum(1 for v in rmse_cache.values() if np.isfinite(v))
        print(f"  {n_valid_rmse}/{len(prod_keys)} have valid RMSE")

        valid_keys = []
        m1_vals = []
        rmse_vals = []
        for key in prod_keys:
            m1 = m1_cache[key]
            rmse = rmse_cache[key]
            if np.isfinite(m1) and np.isfinite(rmse):
                valid_keys.append(key)
                m1_vals.append(m1)
                rmse_vals.append(rmse)

        m1_arr = np.array(m1_vals)
        rmse_arr = np.array(rmse_vals)
        print(f"  {len(valid_keys)} sims with both valid M1 and RMSE")

        if len(valid_keys) < 2:
            print(f"  Too few valid sims, skipping {temp_c}°C")
            continue

        objectives = np.column_stack([m1_arr, rmse_arr])
        selected_idx, front_mask = select_top_n(objectives, TOP_N)
        n_front = int(front_mask.sum())
        print(f"  Pareto front: {n_front} non-dominated points")
        print(f"  Selected: {len(selected_idx)} candidates")

        results_by_temp[temp_c] = {
            "m1_all": m1_arr, "rmse_all": rmse_arr,
            "front_mask": front_mask, "selected_idx": selected_idx,
        }

        out_dir = OUTPUT_BASE / f"T{temp_c}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Generating gallery figures in {out_dir.name}/...")
        for rank_i, si in enumerate(selected_idx):
            pe, t, k = valid_keys[si]
            m1 = m1_arr[si]
            rmse = rmse_arr[si]

            cpath = prod_sims[(pe, t, k)]
            out = make_gallery_figure(
                pe, t, k, temp_c, m1, rmse, rank_i + 1,
                exp_msd, exp_trap, exp_re, exp_rg,
                boundaries, exp_all_segments,
                cpath, out_dir,
            )
            print(f"    #{rank_i+1}: Pe={pe}, T={t}, k={k} "
                  f"(M1={m1:.3f}, RMSE={rmse:.3f}) → {out.name}")

            all_rows.append({
                "temp_exp": temp_c, "rank": rank_i + 1,
                "Pe": pe, "T_sim": t, "kappa": k,
                "M1": m1, "RMSE_MSD": rmse,
            })

    if all_rows:
        df_out = pd.DataFrame(all_rows)
        csv_path = OUTPUT_BASE / "selection_summary.csv"
        df_out.to_csv(csv_path, index=False)
        print(f"\nSummary CSV → {csv_path}")

    if results_by_temp:
        plot_pareto_fronts(results_by_temp, OUTPUT_BASE / "pareto_fronts.png")

    print("\n** DONE **")


if __name__ == "__main__":
    main()
