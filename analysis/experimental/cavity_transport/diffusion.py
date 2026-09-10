
import numpy as np
from scipy.optimize import curve_fit

from .config import CAVITY_RADIUS, CAVITY_RADIUS_15


def find_diffusive_regime(msd_rot, max_slope=1.5, min_slope=0.99, min_points=3, smooth_window=5):
    if len(msd_rot) < min_points:
        return 0, len(msd_rot)

    time = msd_rot[:, 0]
    msd = msd_rot[:, 1]
    valid = (time > 0) & (msd > 0)

    if np.sum(valid) < min_points:
        return 0, min_points

    time_valid = time[valid]
    log_t = np.log10(time_valid)
    log_msd = np.log10(msd[valid])
    local_slope = np.gradient(log_msd, log_t)

    if smooth_window > 1 and len(local_slope) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        local_slope_smooth = np.convolve(local_slope, kernel, mode='valid')
        offset = smooth_window // 2
        time_smooth = time_valid[offset:offset + len(local_slope_smooth)]
    else:
        local_slope_smooth = local_slope
        time_smooth = time_valid
        offset = 0

    start_idx_smooth = None
    for i in range(len(local_slope_smooth)):
        if local_slope_smooth[i] < max_slope:
            start_idx_smooth = i
            break

    if start_idx_smooth is None:
        print(f"  No point with slope < {max_slope} found")
        return 0, min_points

    if start_idx_smooth == 0:
        print(f"  Band starts at beginning (first smoothed slope = {local_slope_smooth[0]:.3f} < {max_slope})")
        start_idx = 0
    else:
        print(f"  Entered band at t = {time_smooth[start_idx_smooth]:.4f}, slope = {local_slope_smooth[start_idx_smooth]:.3f}")
        start_idx = start_idx_smooth + offset

    end_idx_smooth = len(local_slope_smooth)
    for i in range(start_idx_smooth, len(local_slope_smooth)):
        if local_slope_smooth[i] < min_slope:
            end_idx_smooth = i
            print(f"  Exited band at t = {time_smooth[i]:.4f}, slope = {local_slope_smooth[i]:.3f} < {min_slope})")
            break

    end_idx = end_idx_smooth + offset

    n_points = end_idx - start_idx
    if n_points < min_points:
        print(f"  Warning: only {n_points} points, need at least {min_points}")

    print(f"  Diffusive regime: indices {start_idx} to {end_idx} ({n_points} points)")
    print(f"  Time range: t = {time_valid[start_idx]:.4f} to {time_valid[end_idx-1]:.4f}")

    return start_idx, end_idx


def fit_power_law(msd_rot, start_idx=5, end_fraction=0.5):
    if len(msd_rot) == 0:
        return None, None, None, None

    time = msd_rot[:, 0]
    msd = msd_rot[:, 1]

    end_idx = int(len(msd) * end_fraction)
    end_idx = max(end_idx, start_idx + 5)

    t_fit = time[start_idx:end_idx]
    msd_fit = msd[start_idx:end_idx]

    valid = (t_fit > 0) & (msd_fit > 0)
    if np.sum(valid) < 3:
        return None, None, None, None

    log_t = np.log10(t_fit[valid])
    log_msd = np.log10(msd_fit[valid])

    try:
        coeffs, cov = np.polyfit(log_t, log_msd, 1, cov=True)
        alpha = coeffs[0]
        log_A = coeffs[1]
        A = 10**log_A

        alpha_error = np.sqrt(cov[0, 0])
        log_A_error = np.sqrt(cov[1, 1])
        A_error = A * log_A_error * np.log(10)

        print(f"  Power law fit: MSD = {A:.4e} * t^{alpha:.3f}")
        print(f"    α = {alpha:.3f} ± {alpha_error:.3f}")

        return A, alpha, A_error, alpha_error

    except Exception as e:
        print(f"  Power law fit failed: {e}")
        return None, None, None, None


def fit_rotational_diffusion(msd_rot, start_idx=None, end_idx=None,
                             auto_detect=True, max_slope=1.5):
    if len(msd_rot) == 0:
        print("  Cannot fit empty MSD data")
        return None, None, None, None, None

    time = msd_rot[:, 0]
    msd = msd_rot[:, 1]

    if auto_detect:
        fit_start_idx, fit_end_idx = find_diffusive_regime(msd_rot, max_slope=max_slope)
    else:
        fit_start_idx = start_idx if start_idx is not None else 5
        fit_end_idx = end_idx if end_idx is not None else int(len(msd) * 0.3)
        fit_end_idx = max(fit_end_idx, fit_start_idx + 5)

    def linear_func(t, a, b):
        return a * t + b

    t_fit = time[fit_start_idx:fit_end_idx]
    msd_fit = msd[fit_start_idx:fit_end_idx]

    if len(t_fit) < 3:
        print(f"  Warning: Not enough points for fitting ({len(t_fit)})")
        return None, None, None, fit_start_idx, fit_end_idx

    print(f"  Fitting rotational MSD from t={t_fit[0]:.3f} to t={t_fit[-1]:.3f} ({len(t_fit)} points)")

    try:
        popt, pcov = curve_fit(linear_func, t_fit, msd_fit)
        perr = np.sqrt(np.diag(pcov))

        residuals = msd_fit - linear_func(t_fit, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((msd_fit - np.mean(msd_fit))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        print(f"  Fit quality: R² = {r_squared:.4f}")

        slope = popt[0]
        slope_error = perr[0]

        D_r = slope / 2
        D_r_error = slope_error / 2
        tau_rot = 1 / D_r if D_r > 0 else np.inf

        print(f"  D_r = {D_r:.4e} ± {D_r_error:.4e} rad²/min")
        print(f"  τ_rot = 1/D_r = {tau_rot:.4f} min")

        return D_r, D_r_error, tau_rot, fit_start_idx, fit_end_idx

    except Exception as e:
        print(f"  Fitting failed: {e}")
        return None, None, None, fit_start_idx, fit_end_idx


def find_saturation_time(msd_trans, R=None):
    if R is None:
        R = CAVITY_RADIUS

    if len(msd_trans) == 0:
        print("  Cannot find saturation time for empty MSD")
        return None

    time = msd_trans[:, 0]
    msd = msd_trans[:, 1]

    R_squared = R**2
    print(f"  Looking for MSD reaching R² = {R_squared:.2f} (R = {R})")

    idx = np.argmax(msd >= R_squared)

    if msd[idx] >= R_squared:
        tau_trans = time[idx]
        print(f"  τ_trans = {tau_trans:.4f} min (MSD = {msd[idx]:.2f} at index {idx})")
        return tau_trans
    else:
        max_idx = np.argmax(msd)
        tau_trans = time[max_idx]
        print(f"  Warning: MSD never reaches R² = {R_squared:.2f}")
        print(f"  Max MSD = {msd[max_idx]:.2f} at t = {tau_trans:.4f} min")
        print(f"  Using fallback τ_trans = {tau_trans:.4f} min")
        return tau_trans


def find_saturation_time_15(msd_trans):
    print(f"  Finding saturation time for R² = 15 (R = {CAVITY_RADIUS_15:.3f}):")
    tau_15 = find_saturation_time(msd_trans, R=CAVITY_RADIUS_15)
    return tau_15


def find_saturation_time_plateau(msd_trans, tail_cutoff=0.2, slope_threshold=0.3,
                                  smooth_window=5):
    if len(msd_trans) == 0:
        print("  Cannot find plateau for empty MSD")
        return None, None, None

    time = msd_trans[:, 0]
    msd = msd_trans[:, 1]

    n_total = len(time)
    n_keep = int(n_total * (1 - tail_cutoff))
    if n_keep < 10:
        print(f"  Too few points after truncation ({n_keep})")
        return None, None, None

    time_trunc = time[:n_keep]
    msd_trunc = msd[:n_keep]

    valid = (time_trunc > 0) & (msd_trunc > 0)
    if np.sum(valid) < 10:
        print("  Too few valid points for plateau detection")
        return None, None, None

    log_t = np.log10(time_trunc[valid])
    log_msd = np.log10(msd_trunc[valid])
    local_slope = np.gradient(log_msd, log_t)

    if smooth_window > 1 and len(local_slope) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        local_slope = np.convolve(local_slope, kernel, mode='same')

    half_idx = len(local_slope) // 2
    plateau_mask = np.abs(local_slope[half_idx:]) < slope_threshold
    msd_valid = msd_trunc[valid]
    time_valid = time_trunc[valid]

    if np.sum(plateau_mask) >= 3:
        plateau_values = msd_valid[half_idx:][plateau_mask]
        print(f"  Plateau detected: {np.sum(plateau_mask)} points with |slope| < {slope_threshold}")
    else:
        fallback_start = int(len(msd_valid) * 0.7)
        plateau_values = msd_valid[fallback_start:]
        print(f"  No clear plateau (|slope| < {slope_threshold}), using last 30% as fallback")

    if len(plateau_values) == 0:
        print("  Empty plateau region")
        return None, None, None

    plateau_mean = float(np.mean(plateau_values))
    plateau_std = float(np.std(plateau_values))
    print(f"  Plateau mean = {plateau_mean:.4f}, std = {plateau_std:.4f}")

    for i in range(1, len(msd_valid)):
        if msd_valid[i - 1] < plateau_mean <= msd_valid[i]:
            frac = (plateau_mean - msd_valid[i - 1]) / (msd_valid[i] - msd_valid[i - 1])
            tau_sat = time_valid[i - 1] + frac * (time_valid[i] - time_valid[i - 1])
            print(f"  τ_sat = {tau_sat:.4f} (first crossing of plateau mean)")
            return tau_sat, plateau_mean, plateau_std

    if msd_valid[0] >= plateau_mean:
        tau_sat = time_valid[0]
        print(f"  τ_sat = {tau_sat:.4f} (MSD starts above plateau mean)")
        return tau_sat, plateau_mean, plateau_std

    print("  MSD never reaches plateau mean")
    return None, plateau_mean, plateau_std


def calculate_mobility(msd, tau):
    if len(msd) == 0 or tau is None:
        return None

    time = msd[:, 0]
    msd_values = msd[:, 1]

    idx = np.argmin(np.abs(time - tau))
    msd_at_tau = msd_values[idx]

    mobility = msd_at_tau / tau if tau > 0 else np.inf

    print(f"  Mobility μ = MSD(τ)/τ = {msd_at_tau:.4f}/{tau:.4f} = {mobility:.4e}")
    return mobility


def find_first_maximum(msd_trans, smooth_window=5):
    if len(msd_trans) == 0:
        print("  Cannot find first maximum for empty MSD")
        return None, None

    time = msd_trans[:, 0]
    msd = msd_trans[:, 1]

    dmsd_dt = np.gradient(msd, time)

    if smooth_window > 1 and len(dmsd_dt) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        dmsd_dt = np.convolve(dmsd_dt, kernel, mode='same')

    for i in range(1, len(dmsd_dt)):
        if dmsd_dt[i-1] > 0 and dmsd_dt[i] <= 0:
            tau_max = time[i]
            msd_max = msd[i]
            R_max = np.sqrt(msd_max)
            D_max = 2 * R_max
            print(f"  First maximum found at τ_max = {tau_max:.4f} min")
            print(f"    MSD_max = {msd_max:.2f}")
            print(f"    R = √MSD = {R_max:.2f}")
            print(f"    D = 2R = {D_max:.2f}")
            return tau_max, msd_max

    idx = np.argmax(msd)
    tau_max = time[idx]
    msd_max = msd[idx]
    R_max = np.sqrt(msd_max)
    D_max = 2 * R_max
    print(f"  No zero-crossing found, using global max at τ_max = {tau_max:.4f} min")
    print(f"    MSD_max = {msd_max:.2f}")
    print(f"    R = √MSD = {R_max:.2f}")
    print(f"    D = 2R = {D_max:.2f}")
    return tau_max, msd_max
