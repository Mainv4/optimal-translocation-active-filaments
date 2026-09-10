
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_DIR = Path(__file__).parent.parent

SIM_DATA_DIR = BASE_DIR / "DATA_MSD"
EXP_DATA_DIR = BASE_DIR / "FREE_SPACE_EXP"
CONFMAP_DATA_DIR = BASE_DIR / "DATA_Corr_Rg_Re"
OUTPUT_DIR = BASE_DIR / "FIGURES_EXP" / "msd-com"
CONFMAP_OUTPUT_DIR = BASE_DIR / "experimental_conformational_maps" / "output"

MIN_CONFMAP_POINTS = 100_000

TEMPERATURES = ["10C", "20C", "30C"]
TEMP_COLORS_EXP = {"10C": "#75ABDD", "20C": "#84D495", "30C": "#D77567"}
TEMP_COLORS_SIM = {"10C": "#4A85B8", "20C": "#4DA067", "30C": "#B84E3F"}
TEMP_LABELS = {
    "10C": r"$10\,^{\circ}$C",
    "20C": r"$20\,^{\circ}$C",
    "30C": r"$30\,^{\circ}$C",
}

MATCH_INTERVAL = (10, 50)

PLOT_T_MIN = 0.5
PLOT_T_MAX = 250
MSD_YLIM = (1e-1, 1e4)

DPI = 300


def set_plot_style():
    plt.rcParams["figure.dpi"] = 200
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["axes.grid"] = False
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["axes.edgecolor"] = "black"
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.major.size"] = 4
    plt.rcParams["ytick.major.size"] = 4
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })


def parse_filename(filename):
    pattern = r"Pe_([0-9.]+)_T_([0-9.]+)_k_([0-9.]+)_MSD_CM\.dat"
    match = re.match(pattern, filename)
    if match:
        return {"Pe": float(match.group(1)), "T": float(match.group(2)), "k": float(match.group(3))}
    return None


def load_simulation_msd(filepath):
    try:
        data = np.loadtxt(filepath, comments="#")
        return np.arange(len(data)), data
    except Exception as e:
        print(f"Error loading {filepath.name}: {e}")
        return None, None


def build_confmap_point_counts():
    counts = {}
    for npy_path in CONFMAP_DATA_DIR.glob("*.npy"):
        match = re.search(r"Pe_([0-9.]+)_T_([0-9.]+)_k_([0-9.]*)", npy_path.stem)
        if match:
            pe, t = float(match.group(1)), float(match.group(2))
            k = float(match.group(3)) if match.group(3) else 0.0
            data = np.load(npy_path, mmap_mode='r')
            counts[(pe, t, k)] = data.shape[0] * data.shape[1]
    return counts


def load_all_simulations():
    sim_files = sorted(SIM_DATA_DIR.glob("*_MSD_CM.dat"))
    print(f"  Found {len(sim_files)} files in {SIM_DATA_DIR.name}")

    confmap_counts = build_confmap_point_counts()
    print(f"  Found {len(confmap_counts)} conformational map files")

    sim_data = []
    n_excluded = 0
    for filepath in tqdm(sim_files, desc="  Loading simulations"):
        params = parse_filename(filepath.name)
        if params is None:
            continue
        key = (params["Pe"], params["T"], params["k"])
        if confmap_counts.get(key, 0) < MIN_CONFMAP_POINTS:
            n_excluded += 1
            continue
        times, msd = load_simulation_msd(filepath)
        if times is None:
            continue
        sim_data.append({"Pe": params["Pe"], "T": params["T"], "k": params["k"],
                         "times": times, "msd": msd})

    print(f"  Loaded {len(sim_data)} valid simulations ({n_excluded} excluded)")
    return sim_data


def load_experimental_msd(temp):
    csv_path = EXP_DATA_DIR / f"{temp}_msd.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return {"times": df["t"].values, "msd": df["msd"].values}


def compute_rmse_log(time_exp, msd_exp, time_sim, msd_sim, t_min, t_max):
    mask_exp = (time_exp >= t_min) & (time_exp <= t_max) & ~np.isnan(msd_exp) & (msd_exp > 0)
    if not np.any(mask_exp):
        return np.inf
    t_common = time_exp[mask_exp]
    msd_exp_valid = msd_exp[mask_exp]

    valid_sim = ~np.isnan(msd_sim) & (msd_sim > 0) & (time_sim > 0)
    if not np.any(valid_sim):
        return np.inf
    if time_sim[valid_sim].max() < t_max or time_sim[valid_sim].min() > t_min:
        return np.inf

    msd_sim_interp = np.interp(t_common, time_sim[valid_sim], msd_sim[valid_sim])
    return np.sqrt(np.mean((np.log10(msd_exp_valid) - np.log10(msd_sim_interp)) ** 2))


def find_best_matches(sim_data, exp_data):
    t_min, t_max = MATCH_INTERVAL
    best_matches = {}

    for temp in TEMPERATURES:
        if temp not in exp_data:
            continue
        exp = exp_data[temp]
        best_sim, best_rmse = None, np.inf

        for sim in sim_data:
            rmse = compute_rmse_log(exp["times"], exp["msd"], sim["times"], sim["msd"], t_min, t_max)
            if rmse < best_rmse:
                best_rmse = rmse
                best_sim = sim

        if best_sim is not None:
            best_matches[temp] = {**best_sim, "rmse": best_rmse}
            print(f"  {temp}: Pe={best_sim['Pe']}, T={best_sim['T']}, k={best_sim['k']}, "
                  f"RMSE={best_rmse:.4f}")

    return best_matches


def estimate_D(times, msd, t_min=None, t_max=None):
    if t_min is None:
        t_min = MATCH_INTERVAL[0]
    if t_max is None:
        t_max = MATCH_INTERVAL[1]
    mask = (times >= t_min) & (times <= t_max) & np.isfinite(msd) & (msd > 0) & (times > 0)
    if np.sum(mask) < 5:
        return np.nan, np.nan
    D_eff = msd[mask] / (4 * times[mask])
    return np.mean(D_eff), np.std(D_eff)


def compute_local_slope(lag_times, msd, window_decades=0.5):
    valid = (lag_times > 0) & (msd > 0) & ~np.isnan(msd)
    if np.sum(valid) < 10:
        return np.full_like(msd, np.nan)

    log_t = np.log10(lag_times)
    log_msd = np.log10(msd)
    slope = np.full_like(msd, np.nan)

    for idx in np.where(valid)[0]:
        half_w = window_decades / 2
        in_window = (log_t >= log_t[idx] - half_w) & (log_t <= log_t[idx] + half_w) & valid
        widx = np.where(in_window)[0]
        if len(widx) >= 3:
            x, y = log_t[widx], log_msd[widx]
            xm, ym = np.mean(x), np.mean(y)
            denom = np.sum((x - xm) ** 2)
            if denom > 0:
                slope[idx] = np.sum((x - xm) * (y - ym)) / denom
    return slope


def _build_legend(exp_data, best_matches):
    handles = []
    for temp in TEMPERATURES:
        if temp not in exp_data:
            continue
        lbl_exp = TEMP_LABELS[temp] + " exp"
        handles.append(Line2D([0], [0], color=TEMP_COLORS_EXP[temp], lw=2, ls="-",
                              label=lbl_exp))
        if temp in best_matches:
            sim = best_matches[temp]
            lbl_sim = (TEMP_LABELS[temp] +
                       rf" sim ($f^a\!=\!{sim['Pe']},\;T\!=\!{sim['T']},\;\kappa\!=\!{sim['k']}$)")
            handles.append(Line2D([0], [0], color=TEMP_COLORS_SIM[temp], lw=1.5, ls="--",
                                  label=lbl_sim))
    return handles


def create_msd_figure(exp_data, best_matches):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axvspan(*MATCH_INTERVAL, color="#DDDDDD", alpha=0.5, zorder=0)

    for temp in TEMPERATURES:
        if temp in best_matches:
            sim = best_matches[temp]
            v = (sim["times"] > 0) & (sim["msd"] > 0) & ~np.isnan(sim["msd"])
            ax.plot(sim["times"][v], sim["msd"][v],
                    color=TEMP_COLORS_SIM[temp], lw=1.5, ls="--", zorder=4)
    for temp in TEMPERATURES:
        if temp not in exp_data:
            continue
        exp = exp_data[temp]
        v = ~np.isnan(exp["msd"]) & (exp["msd"] > 0) & (exp["times"] > 0)
        ax.plot(exp["times"][v], exp["msd"][v],
                color=TEMP_COLORS_EXP[temp], lw=2.0, ls="-", zorder=3)

    ax.legend(handles=_build_legend(exp_data, best_matches), fontsize=5.5,
              loc="upper left", frameon=True, fancybox=False, edgecolor="none",
              facecolor="white", framealpha=0.85, handlelength=2.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$\langle r^2 \rangle$ (mm$^2$)")
    ax.set_xlim(PLOT_T_MIN, PLOT_T_MAX); ax.set_ylim(MSD_YLIM)

    fig_path = OUTPUT_DIR / "msd_matching.png"
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"  Saved: {fig_path}")

    CONFMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path2 = CONFMAP_OUTPUT_DIR / "msd_matching.png"
    fig.savefig(fig_path2, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    print(f"  Saved: {fig_path2}")

    plt.close()


def create_slope_figure(exp_data, best_matches):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axvspan(*MATCH_INTERVAL, color="#DDDDDD", alpha=0.5, zorder=0)

    for temp in TEMPERATURES:
        if temp in best_matches:
            sim = best_matches[temp]
            v = (sim["times"] > 0) & (sim["msd"] > 0) & ~np.isnan(sim["msd"])
            slope_s = compute_local_slope(sim["times"][v], sim["msd"][v])
            ax.plot(sim["times"][v], slope_s,
                    color=TEMP_COLORS_SIM[temp], lw=1.5, ls="--", zorder=4)
    for temp in TEMPERATURES:
        if temp not in exp_data:
            continue
        exp = exp_data[temp]
        v = ~np.isnan(exp["msd"]) & (exp["msd"] > 0) & (exp["times"] > 0)
        slope_e = compute_local_slope(exp["times"][v], exp["msd"][v])
        ax.plot(exp["times"][v], slope_e,
                color=TEMP_COLORS_EXP[temp], lw=2.0, ls="-", zorder=3)

    ax.legend(handles=_build_legend(exp_data, best_matches), fontsize=5.5,
              loc="upper right", frameon=True, fancybox=False, edgecolor="none",
              facecolor="white", framealpha=0.85, handlelength=2.0)
    ax.set_xscale("log")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$\alpha(t)$")
    ax.set_xlim(PLOT_T_MIN, PLOT_T_MAX); ax.set_ylim(0, 2.5)

    fig_path = OUTPUT_DIR / "slope.png"
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"  Saved: {fig_path}")


def create_diffusion_figure(exp_data, best_matches):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.axvspan(*MATCH_INTERVAL, color="#DDDDDD", alpha=0.5, zorder=0)

    for temp in TEMPERATURES:
        if temp in best_matches:
            sim = best_matches[temp]
            v = (sim["times"] > 0) & (sim["msd"] > 0) & ~np.isnan(sim["msd"])
            ax.plot(sim["times"][v], sim["msd"][v] / (4 * sim["times"][v]),
                    color=TEMP_COLORS_SIM[temp], lw=1.5, ls="--", zorder=4)
    for temp in TEMPERATURES:
        if temp not in exp_data:
            continue
        exp = exp_data[temp]
        v = ~np.isnan(exp["msd"]) & (exp["msd"] > 0) & (exp["times"] > 0)
        ax.plot(exp["times"][v], exp["msd"][v] / (4 * exp["times"][v]),
                color=TEMP_COLORS_EXP[temp], lw=2.0, ls="-", zorder=3)

    ax.legend(handles=_build_legend(exp_data, best_matches), fontsize=5.5,
              loc="lower right", frameon=True, fancybox=False, edgecolor="none",
              facecolor="white", framealpha=0.85, handlelength=2.0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$D_{\mathrm{eff}}$ (mm$^2$/s)")
    ax.set_xlim(PLOT_T_MIN, PLOT_T_MAX)

    fig_path = OUTPUT_DIR / "diffusion.png"
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"  Saved: {fig_path}")


def main():
    set_plot_style()

    print("=" * 60)
    print("Loading simulation MSD data...")
    sim_data = load_all_simulations()

    print("\nLoading experimental MSD data...")
    exp_data = {}
    for temp in TEMPERATURES:
        exp = load_experimental_msd(temp)
        if exp is not None:
            exp_data[temp] = exp
            print(f"  {temp}: t=[{exp['times'][0]:.2f}, {exp['times'][-1]:.2f}]s, {len(exp['times'])} pts")

    print(f"\nFinding best matches (medium range {MATCH_INTERVAL})...")
    best_matches = find_best_matches(sim_data, exp_data)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")
    create_msd_figure(exp_data, best_matches)
    create_slope_figure(exp_data, best_matches)
    create_diffusion_figure(exp_data, best_matches)

    print("\nDone.")


if __name__ == "__main__":
    main()
