from pathlib import Path

import cmocean.cm as cmo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "plot" / "output"
DPI = 300

F_SIM_TO_NN = 100.0
CONTOUR_LENGTH = 39.0

TEMPERATURES = [10, 20, 30]
TEMP_COLORS = {10: "#75ABDD", 20: "#84D495", 30: "#D77567"}
TEMP_COLORS_SIM = {10: "#4A85B8", 20: "#4DA067", 30: "#B84E3F"}
TEMP_LABELS = {T: rf"{T}$^\circ$C" for T in TEMPERATURES}
CALIBRATED = {10: (0.3, 0.2, 0.5), 20: (0.5, 0.2, 0.8), 30: (0.8, 0.1, 0.5)}

CMAP_ACTIVITY = cmo.matter
CMAP_KAPPA = "viridis"
EXP_MARKER = "s"
EXP_SIZE = 100
EXP_EDGE_LW = 0.8

KAPPA_RANGE = (0.3, 2.3)
T_RANGE = (0.08, 0.31)
PE_RANGE = (0.3, 2.0)
PLATEAU_WINDOW_MIN = (0.8, 1.0)
YLIM_TRAP = (1e-1, 1e2)

LAB_T_MIN = r"$t$ (min)"
LAB_T_C = r"$T$ ($^\circ$C)"
LAB_MSD_ROT = r"$\langle (\theta(t)-\theta_0)^2 \rangle$ (rad$^2$)"
LAB_MSD_TR = r"$\langle (r(t)-r_0)^2 \rangle$ (mm$^2$)"
LAB_TROT = r"$\tau_{\theta}$ (min)"
LAB_TSAT = r"$\tau_d$ (min)"
LAB_TTRAP = r"$\tau_{\mathrm{tr}}$ (min)"
LAB_RATIO_INF = r"$\tau_{\mathrm{tr}} \,/\, \tau_{\mathrm{tr},\infty}$"
LAB_PE_EFF = r"$\mathrm{Pe} = \tau_{\theta} \,/\, \tau_d$"
LAB_FA_NN = r"$f^a$ (nN)"
LAB_KAPPA = r"$\kappa \,/\, u_E$"
LAB_H = r"$H$"
LAB_TAU_EC = r"$\tau_{e,c}$ (min)"
LAB_TTRAP_OVER_TAU_E0 = r"$\tau_{\mathrm{tr}} / \tau_{e,0}$"


def set_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Computer Modern Roman"],
        "figure.dpi": 150,
        "lines.linewidth": 2,
        "font.size": 13,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
    })


def fmt_tick(x, pos=None):
    if abs(x - round(x)) < 1e-9:
        return rf"${int(round(x))}$"
    return rf"${x:g}$"


def fmt_pow10(x, pos=None):
    return rf"$10^{{{int(round(np.log10(x)))}}}$"


def set_log_ticks(ax, axis, ticks, powers=False):
    loc = FixedLocator(ticks)
    fmt = FuncFormatter(fmt_pow10 if powers else fmt_tick)
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(loc)
    a.set_major_formatter(fmt)
    a.set_minor_locator(NullLocator())


def save(fig, name):
    if _source_data is not None:
        plt.close(fig)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUTPUT / name}.png")


def sim_table():
    df = pd.read_csv(DATA / "tables" / "data_confinement.csv", comment="#")
    for c in ("Pe", "T", "kappa"):
        df[c] = df[c].round(3)
    return df


def paper_filter(df):
    keep = (df["kappa"].between(*KAPPA_RANGE)
            & df["T"].between(*T_RANGE)
            & df["Pe"].between(*PE_RANGE)
            & df["ttrap"].notna())
    return df[keep].copy()


def timescale_filter(df):
    keep = df[["ttrap", "tau_rot", "tau_trans", "lp_free", "Pe"]].notna().all(axis=1)
    keep &= (df["tau_rot"] > 0) & (df["tau_trans"] > 0)
    return df[keep].copy()


def exp_table():
    df = pd.read_csv(DATA / "tables" / "data_exp.csv", comment="#")
    return df.sort_values("T_celsius").reset_index(drop=True)


def exp_msd():
    out = {}
    for T in TEMPERATURES:
        d = np.load(DATA / "exp" / "cavity" / "msd" / f"EXP_T{T}.npz", allow_pickle=True)
        out[T] = {"rot": np.asarray(d["msd_rot"]), "trans": np.asarray(d["msd_trans"])}
    return out


def plateau_in_window(msd, window=PLATEAU_WINDOW_MIN):
    t, m = msd[:, 0], msd[:, 1]
    mask = (t >= window[0]) & (t <= window[1])
    return float(np.nanmean(m[mask]))


def crossing_time(msd, threshold):
    t, m = msd[:, 0], msd[:, 1]
    above = m >= threshold
    if not above.any():
        return None
    i = int(np.argmax(above))
    if i == 0:
        return float(t[0])
    m0, m1 = m[i - 1], m[i]
    if m1 == m0:
        return float(t[i])
    return float(t[i - 1] + (threshold - m0) / (m1 - m0) * (t[i] - t[i - 1]))


def fit_rot_exp(msd_rot, t_start_min=2.0 / 60, t_end_min=20.0 / 60):
    t, m = msd_rot[:, 0], msd_rot[:, 1]
    mask = (t >= t_start_min) & (t <= t_end_min)
    popt, _ = curve_fit(lambda x, a, b: a * x + b, t[mask], m[mask])
    t_line = np.linspace(t[mask][0], t[mask][-1], 50)
    return popt[0] / 2.0, t_line, popt[0] * t_line + popt[1]


def exp_saturation_times(msd):
    plateau, tau_sat = {}, {}
    for T in TEMPERATURES:
        plateau[T] = plateau_in_window(msd[T]["trans"])
        tau_sat[T] = crossing_time(msd[T]["trans"], plateau[T])
    return plateau, tau_sat


def add_exp_markers(ax, x, y, temperatures, legend=False, **kwargs):
    for xi, yi, T in zip(np.asarray(x), np.asarray(y), np.asarray(temperatures, dtype=int)):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.scatter(xi, yi, marker=EXP_MARKER, s=EXP_SIZE, c=TEMP_COLORS[T],
                       edgecolors="black", linewidths=EXP_EDGE_LW, zorder=5, **kwargs)
    if legend:
        handles = [Line2D([0], [0], marker=EXP_MARKER, ls="", color=TEMP_COLORS[T],
                          markeredgecolor="black", markeredgewidth=EXP_EDGE_LW,
                          markersize=8, label=rf"$T={T}\,^{{\circ}}$C")
                   for T in TEMPERATURES]
        ax.legend(handles=handles, fontsize=10, loc="lower right", framealpha=0.95,
                  edgecolor="0.6", fancybox=False, borderpad=0.4, handletextpad=0.3)


def add_slope_guide(ax, slope, x_anchor, y_anchor, x_span, color="0.35", lw=1.3, ls="--"):
    x_line = np.logspace(np.log10(x_span[0]), np.log10(x_span[1]), 100)
    log_y = np.log10(y_anchor) + slope * (np.log10(x_line) - np.log10(x_anchor))
    ax.plot(x_line, 10 ** log_y, ls=ls, lw=lw, color=color, zorder=1)


PLATEAU_THRESH_MIN = 0.3
N_BINS = 5
HIGH_PE_SPLIT = 1.5


def activity_bins(df):
    Pe = df["Pe"].values.astype(float)
    tsat = df["tau_trans"].values.astype(float)
    ttrap = df["ttrap"].values.astype(float)
    plateau = tsat > PLATEAU_THRESH_MIN
    _, edges = pd.qcut(Pe[plateau], q=N_BINS, duplicates="drop", retbins=True)
    if edges[-1] > HIGH_PE_SPLIT > edges[-2]:
        edges = np.insert(edges, len(edges) - 1, HIGH_PE_SPLIT)
    bins = []
    for j in range(len(edges) - 1):
        lo, hi = edges[j], edges[j + 1]
        in_bin = (Pe >= lo) & (Pe <= hi) if j == 0 else (Pe > lo) & (Pe <= hi)
        on_plateau = in_bin & plateau
        tau_inf = float(np.nanmean(ttrap[on_plateau])) if on_plateau.any() else np.nan
        bins.append({"lo": lo, "hi": hi, "in_bin": in_bin, "on_plateau": on_plateau,
                     "Pe_med": float(np.median(Pe[in_bin])), "tau_inf": tau_inf})
    return bins


def activity_color(Pe_value, Pe_all):
    lo, hi = float(np.nanmin(Pe_all)), float(np.nanmax(Pe_all))
    return CMAP_ACTIVITY(np.clip((Pe_value - lo) / max(hi - lo, 1e-12), 0.0, 1.0))


def set_style_framed():
    set_style()
    plt.rcParams.update({
        "figure.dpi": 200,
        "font.size": 15,
        "axes.labelsize": 18,
        "xtick.labelsize": 13.5,
        "ytick.labelsize": 13.5,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "text.latex.preamble": r"\usepackage{amsmath}",
    })


RG_RANGE = (0.0, 0.3)
RE_RANGE = (-0.05, 1.05)
CONFMAP_BINS = 50
CONFMAP_VMIN, CONFMAP_VMAX = 1e-4, 1e-2
CONFMAP_CMAP = "viridis"
SMOOTH_SIGMA = 1.0


def load_boundaries():
    circ = np.loadtxt(DATA / "boundaries" / "Rg_Re_Circle.txt")
    low = np.loadtxt(DATA / "boundaries" / "empirical_min_boundary.txt")
    return circ[:, 0], circ[:, 1], low[:, 0], low[:, 1]


def read_rere_csv(path):
    df = pd.read_csv(path)
    re = df["Ree" if "Ree" in df.columns else "Re"].values
    rg = df["Rgy" if "Rgy" in df.columns else "Rg"].values
    valid = np.isfinite(re) & np.isfinite(rg)
    return re[valid], rg[valid]


def read_confmap_npy(path):
    data = np.load(path)
    rg = data[:, :, 0].flatten() / CONTOUR_LENGTH
    re = data[:, :, 1].flatten() / CONTOUR_LENGTH
    valid = np.isfinite(re) & np.isfinite(rg)
    return re[valid], rg[valid]


def filter_by_envelope(re, rg, env):
    circ_rg, circ_re, min_rg, min_re = env
    keep = np.ones(len(re), dtype=bool)
    i = np.argsort(circ_re)
    upper = np.interp(re, circ_re[i], circ_rg[i], left=np.nan, right=np.nan)
    keep &= (rg <= upper) | np.isnan(upper)
    i = np.argsort(min_re)
    lower = np.interp(re, min_re[i], min_rg[i], left=np.nan, right=np.nan)
    keep &= (rg >= lower) | np.isnan(lower)
    return re[keep], rg[keep]


def confmap_density(re, rg, env, smooth_sigma=0.0):
    from scipy.ndimage import gaussian_filter
    h, xe, ye = np.histogram2d(rg, re, bins=CONFMAP_BINS, range=[RG_RANGE, RE_RANGE])
    density = h / h.sum()
    mask = np.ones_like(density, dtype=bool)
    if smooth_sigma > 0:
        density = gaussian_filter(density, sigma=smooth_sigma)
        circ_rg, circ_re, min_rg, min_re = env
        rg_c, re_c = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])
        rg_grid, re_grid = np.meshgrid(rg_c, re_c, indexing="ij")
        mask = (re_grid >= 0) & (re_grid <= 1.0) & (rg_grid > 0)
        i = np.argsort(circ_re)
        mask &= rg_grid <= np.interp(re_grid, circ_re[i], circ_rg[i], left=0, right=0)
        i = np.argsort(min_re)
        mask &= rg_grid >= np.interp(re_grid, min_re[i], min_rg[i], left=np.inf, right=np.inf)
        density[~mask] = 0
    return density, mask


def confmap_centres():
    rg_e = np.linspace(*RG_RANGE, CONFMAP_BINS + 1)
    re_e = np.linspace(*RE_RANGE, CONFMAP_BINS + 1)
    return np.meshgrid(0.5 * (rg_e[:-1] + rg_e[1:]), 0.5 * (re_e[:-1] + re_e[1:]), indexing="ij")


def draw_confmap(ax, re, rg, env, smooth_sigma=0.0, boundary_color="0.5", label=None,
                 panel=None, series=None):
    from matplotlib.colors import LogNorm
    density, _ = confmap_density(re, rg, env, smooth_sigma)
    if panel is not None:
        RG, RE = confmap_centres()
        keep = density > 0
        record(panel, series, Rg_over_lc=RG[keep], Re_over_lc=RE[keep], probability=density[keep])
    im = ax.imshow(density.T, origin="lower", extent=[*RG_RANGE, *RE_RANGE], aspect="auto",
                   norm=LogNorm(vmin=CONFMAP_VMIN, vmax=CONFMAP_VMAX), cmap=CONFMAP_CMAP)
    circ_rg, circ_re, min_rg, min_re = env
    ax.plot(circ_rg, circ_re, "-", color=boundary_color, lw=2.0, alpha=0.6)
    ax.plot(min_rg, min_re, "-", color=boundary_color, lw=2.0, alpha=0.6)
    if label is not None:
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=13.5, fontweight="bold",
                va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                               alpha=0.85, edgecolor="none"))
    ax.set_xlim(RG_RANGE)
    ax.set_ylim(RE_RANGE)
    return im


LAB_RG = r"$R_g/\ell_c$"
LAB_RE = r"$R_e/\ell_c$"
LAB_P_RGRE = r"$P(R_g, R_e)$"
LAB_P_TTRAP = r"$P(\tau_{\mathrm{tr}})$"
LAB_LC = r"$\ell_c$ (mm)"

MIN_TRAP_TIME = 10 / 60


def exp_trapping_events(temperature, max_time):
    import sqlite3
    with sqlite3.connect(DATA / "tables" / "exp_trapping_times.db") as conn:
        df = pd.read_sql("SELECT length_mm, time_min FROM exp_trapping_events WHERE T_exp=?",
                         conn, params=(temperature,))
    return df[(df["time_min"] >= MIN_TRAP_TIME) & (df["time_min"] <= max_time)]


def sim_trapping_events(N, Pe, T, kappa, max_time):
    import sqlite3
    with sqlite3.connect(DATA / "tables" / "trapping_times.db") as conn:
        df = pd.read_sql("SELECT time_min FROM trapping_events WHERE N=? AND ABS(Pe-?)<0.01 "
                         "AND ABS(T-?)<0.01 AND ABS(kappa-?)<0.01", conn, params=(N, Pe, T, kappa))
    times = df["time_min"].values
    return times[(times >= MIN_TRAP_TIME) & (times <= max_time)]


def log_bins(max_time, n_bins):
    return np.logspace(np.log10(0.05), np.log10(max_time), n_bins + 1)


def fit_exponential_log_space(times, bins):
    counts, edges = np.histogram(times, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (counts > 0) & (centers > 0)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    popt, pcov = curve_fit(lambda x, a, b: a * x + b, centers[mask], np.log(counts[mask]), p0=[-1, 0])
    tau = -1.0 / popt[0]
    tau_err = np.sqrt(pcov[0, 0]) / popt[0] ** 2 if pcov[0, 0] > 0 else 0.0
    return tau, tau_err, np.exp(popt[1])


def exp_cavity_files():
    return sorted((DATA / "exp" / "cavity").glob("*/worm_*.csv"))


_source_data = None


def source_data_begin():
    global _source_data
    _source_data = []


def record(panel, series, **columns):
    if _source_data is None:
        return
    cols = {}
    for key, value in columns.items():
        arr = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
        cols[key] = arr
    n = max(len(arr) for arr in cols.values())
    cols = {k: (v if len(v) == n else np.full(n, v[0])) for k, v in cols.items()}
    frame = pd.DataFrame(cols)
    frame.insert(0, "series", series)
    frame.insert(0, "panel", panel)
    _source_data.append(frame)


def source_data_frame():
    global _source_data
    frame = pd.concat(_source_data, ignore_index=True)
    _source_data = None
    return frame
