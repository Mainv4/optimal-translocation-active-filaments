

import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem
from scipy.signal import savgol_filter


def filter_velocity_outliers(time, com_x, com_y, v_max):
    if len(time) < 2:
        return time, com_x, com_y

    dx = np.diff(com_x)
    dy = np.diff(com_y)
    dt = np.diff(time)

    valid_dt = dt > 0
    velocities = np.zeros(len(dt))
    velocities[valid_dt] = np.sqrt(dx[valid_dt]**2 + dy[valid_dt]**2) / dt[valid_dt]

    keep_mask = np.concatenate([[True], velocities < v_max])

    time_filt = time[keep_mask]
    com_x_filt = com_x[keep_mask]
    com_y_filt = com_y[keep_mask]

    n_removed = len(time) - len(time_filt)
    if n_removed > 0:
        frac_removed = 100.0 * n_removed / len(time)
        if frac_removed > 5.0:
            warnings.warn(f"Removed {n_removed}/{len(time)} points ({frac_removed:.1f}%) with v > {v_max} mm/s")

    return time_filt, com_x_filt, com_y_filt


def load_worm_trajectory(filepath, v_max=None):
    try:
        df = pd.read_csv(filepath)

        time = df["time"].values
        com_x = df["com (x)"].values
        com_y = df["com (y)"].values

        valid_mask = ~np.isnan(com_x) & ~np.isnan(com_y) & ~np.isnan(time)

        time = time[valid_mask]
        com_x = com_x[valid_mask]
        com_y = com_y[valid_mask]

        if len(time) == 0:
            warnings.warn(f"No valid frames in {filepath}")
            return None

        if v_max is not None:
            time, com_x, com_y = filter_velocity_outliers(time, com_x, com_y, v_max)

            if len(time) < 2:
                warnings.warn(f"Too few frames remaining after velocity filtering in {filepath}")
                return None

        return time, com_x, com_y

    except Exception as e:
        warnings.warn(f"Failed to load {filepath}: {e}")
        return None


def calculate_msd_single_worm(time, com_x, com_y):
    n_frames = len(time)

    dt = np.median(np.diff(time))

    max_lag = n_frames // 2

    lag_times = []
    msd_values = []

    for lag in range(1, max_lag + 1):
        dx = com_x[lag:] - com_x[:-lag]
        dy = com_y[lag:] - com_y[:-lag]

        msd = np.mean(dx**2 + dy**2)

        lag_time = lag * dt
        lag_times.append(lag_time)
        msd_values.append(msd)

    return np.array(lag_times), np.array(msd_values)


def compute_msd_ensemble(worm_files, min_worms=None, v_max=None):
    if min_worms is None:
        min_worms = max(5, int(0.3 * len(worm_files)))

    print(f"Processing {len(worm_files)} worm trajectories...")
    if v_max is not None:
        print(f"  Filtering velocities > {v_max} mm/s")
    print(f"  Using min_worms = {min_worms} ({100*min_worms/len(worm_files):.1f}% of total)")

    all_lag_times = []
    all_msd_values = []

    for i, filepath in enumerate(worm_files):
        result = load_worm_trajectory(filepath, v_max=v_max)
        if result is None:
            continue

        time, com_x, com_y = result

        lag_times, msd_values = calculate_msd_single_worm(time, com_x, com_y)

        all_lag_times.append(lag_times)
        all_msd_values.append(msd_values)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(worm_files)} worms")

    if len(all_msd_values) == 0:
        raise ValueError("No valid worm trajectories found!")

    print(f"  Successfully processed {len(all_msd_values)} worms")

    all_times_concat = np.concatenate(all_lag_times)
    t_min = all_times_concat.min()
    t_max = all_times_concat.max()

    dt_median = np.median(np.diff(all_lag_times[0]))
    time_grid = np.arange(t_min, t_max + dt_median/2, dt_median)

    print(f"  Time grid: {t_min:.3f} - {t_max:.3f} s ({len(time_grid)} points)")

    msd_matrix = np.full((len(all_msd_values), len(time_grid)), np.nan)

    for i, (lt, msd) in enumerate(zip(all_lag_times, all_msd_values)):
        valid_mask = (time_grid >= lt[0]) & (time_grid <= lt[-1])

        if np.any(valid_mask):
            msd_matrix[i, valid_mask] = np.interp(
                time_grid[valid_mask],
                lt,
                msd
            )

    n_worms_array = np.sum(~np.isnan(msd_matrix), axis=0)

    valid_indices = np.where(n_worms_array >= min_worms)[0]

    if len(valid_indices) == 0:
        raise ValueError(f"No time points have at least {min_worms} worms!")

    cutoff_idx = valid_indices[-1] + 1
    time_grid = time_grid[:cutoff_idx]
    msd_matrix = msd_matrix[:, :cutoff_idx]
    n_worms_array = n_worms_array[:cutoff_idx]

    msd_mean = np.nanmean(msd_matrix, axis=0)
    msd_std = np.nanstd(msd_matrix, axis=0)
    msd_err = msd_std / np.sqrt(n_worms_array)

    print(f"  Final data range: {time_grid[0]:.3f} - {time_grid[-1]:.3f} s")
    print(f"  N_worms: {n_worms_array[0]:.0f} → {n_worms_array[-1]:.0f}")

    return time_grid, msd_mean, msd_err, n_worms_array


def compute_msd_ensemble_direct(worm_files):
    print(f"\n[Method: Direct averaging] Processing {len(worm_files)} worm trajectories...")

    all_lag_times = []
    all_msd_values = []

    for i, filepath in enumerate(worm_files):
        result = load_worm_trajectory(filepath)
        if result is None:
            continue

        time, com_x, com_y = result
        lag_times, msd_values = calculate_msd_single_worm(time, com_x, com_y)

        all_lag_times.append(lag_times)
        all_msd_values.append(msd_values)

    if len(all_msd_values) == 0:
        raise ValueError("No valid worm trajectories found!")

    print(f"  Successfully processed {len(all_msd_values)} worms")

    min_length = min(len(lt) for lt in all_lag_times)

    time_grid = all_lag_times[0][:min_length]
    msd_matrix = np.array([msd[:min_length] for msd in all_msd_values])

    msd_mean = np.mean(msd_matrix, axis=0)
    msd_std = np.std(msd_matrix, axis=0)
    msd_err = msd_std / np.sqrt(len(all_msd_values))

    print(f"  Time range: {time_grid[0]:.3f} - {time_grid[-1]:.3f} s ({len(time_grid)} points)")

    return time_grid, msd_mean, msd_err


def compute_msd_ensemble_smoothed(worm_files, window_length=11, polyorder=3):
    print(f"\n[Method: Smoothed positions] Processing {len(worm_files)} worm trajectories...")
    print(f"  Using Savitzky-Golay filter (window={window_length}, order={polyorder})")

    all_lag_times = []
    all_msd_values = []

    for i, filepath in enumerate(worm_files):
        result = load_worm_trajectory(filepath)
        if result is None:
            continue

        time, com_x, com_y = result

        if len(time) >= window_length:
            com_x = savgol_filter(com_x, window_length, polyorder)
            com_y = savgol_filter(com_y, window_length, polyorder)

        lag_times, msd_values = calculate_msd_single_worm(time, com_x, com_y)

        all_lag_times.append(lag_times)
        all_msd_values.append(msd_values)

    if len(all_msd_values) == 0:
        raise ValueError("No valid worm trajectories found!")

    print(f"  Successfully processed {len(all_msd_values)} worms")

    min_length = min(len(lt) for lt in all_lag_times)

    time_grid = all_lag_times[0][:min_length]
    msd_matrix = np.array([msd[:min_length] for msd in all_msd_values])

    msd_mean = np.mean(msd_matrix, axis=0)
    msd_std = np.std(msd_matrix, axis=0)
    msd_err = msd_std / np.sqrt(len(all_msd_values))

    print(f"  Time range: {time_grid[0]:.3f} - {time_grid[-1]:.3f} s ({len(time_grid)} points)")

    return time_grid, msd_mean, msd_err


def plot_methods_comparison(temp_label, methods_data, original_data, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    colors = {
        "Default (union+interp)": "red",
        "Filtered v<200 mm/s": "orange",
        "Filtered v<100 mm/s": "green",
        "Direct avg (intersection)": "blue",
        "Smoothed positions (Savgol)": "purple",
    }

    time_orig, msd_orig, err_orig = original_data

    ax1.loglog(time_orig, msd_orig, 'ko-', linewidth=3, markersize=6,
               label='Original', alpha=0.8, zorder=10)

    for method_name, (time, msd, err) in methods_data.items():
        ax1.loglog(time, msd, 'o-', color=colors.get(method_name, 'gray'),
                   linewidth=2, markersize=4, label=method_name, alpha=0.7)

    ax1.set_xlabel('Time (s)', fontsize=14)
    ax1.set_ylabel('MSD (mm²)', fontsize=14)
    ax1.set_title(f'{temp_label} - MSD Calculation Methods Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')

    for method_name, (time, msd, err) in methods_data.items():
        msd_interp = np.interp(time_orig, time, msd)
        ratio = msd_interp / msd_orig

        ax2.semilogx(time_orig, ratio, 'o-', color=colors.get(method_name, 'gray'),
                     linewidth=2, markersize=3, label=method_name, alpha=0.7)

    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax2.set_xlabel('Time (s)', fontsize=14)
    ax2.set_ylabel('Ratio (Method / Original)', fontsize=14)
    ax2.set_title(f'{temp_label} - Ratio to Original', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"methods_comparison_{temp_label}.png")
    plt.savefig(filepath, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Methods comparison plot saved: {filepath}")


def process_temperature(temp_dir, temp_label, output_dir):
    print("\n" + "=" * 70)
    print(f"Processing {temp_label}")
    print("=" * 70)

    worm_files = sorted(glob.glob(os.path.join(temp_dir, "*.csv")))

    if len(worm_files) == 0:
        raise ValueError(f"No worm files found in {temp_dir}")

    print(f"Found {len(worm_files)} worm files")

    print("\n[Method: Default (union+interp)]")
    time, msd, msd_err, n_worms = compute_msd_ensemble(worm_files)

    output_file = os.path.join(output_dir, f"RECALCULATED_{temp_label}_msd.csv")
    df = pd.DataFrame({
        "time": time,
        "msd": msd,
        "msd_err": msd_err,
        "n_worms": n_worms,
    })
    df.to_csv(output_file, index=False)

    print(f"\nSaved: {output_file}")
    print(f"  Time range: {time.min():.3f} - {time.max():.3f} s")
    print(f"  MSD range: {msd.min():.4e} - {msd.max():.4e} mm²")
    print(f"  Number of points: {len(time)}")


    print("\n[Method: Default with v_max = 200 mm/s]")
    time_v200, msd_v200, err_v200, n_worms_v200 = compute_msd_ensemble(worm_files, v_max=200.0)

    output_file_v200 = os.path.join(output_dir, f"RECALCULATED_{temp_label}_msd_v200.csv")
    df_v200 = pd.DataFrame({
        "time": time_v200,
        "msd": msd_v200,
        "msd_err": err_v200,
        "n_worms": n_worms_v200,
    })
    df_v200.to_csv(output_file_v200, index=False)
    print(f"Saved: {output_file_v200}")

    print("\n[Method: Default with v_max = 100 mm/s]")
    time_v100, msd_v100, err_v100, n_worms_v100 = compute_msd_ensemble(worm_files, v_max=100.0)

    output_file_v100 = os.path.join(output_dir, f"RECALCULATED_{temp_label}_msd_v100.csv")
    df_v100 = pd.DataFrame({
        "time": time_v100,
        "msd": msd_v100,
        "msd_err": err_v100,
        "n_worms": n_worms_v100,
    })
    df_v100.to_csv(output_file_v100, index=False)
    print(f"Saved: {output_file_v100}")


    methods_data = {}
    methods_data["Default (union+interp)"] = (time, msd, msd_err)
    methods_data["Filtered v<200 mm/s"] = (time_v200, msd_v200, err_v200)
    methods_data["Filtered v<100 mm/s"] = (time_v100, msd_v100, err_v100)

    try:
        time_direct, msd_direct, err_direct = compute_msd_ensemble_direct(worm_files)
        methods_data["Direct avg (intersection)"] = (time_direct, msd_direct, err_direct)
    except Exception as e:
        print(f"  Direct method failed: {e}")

    try:
        time_smooth, msd_smooth, err_smooth = compute_msd_ensemble_smoothed(worm_files)
        methods_data["Smoothed positions (Savgol)"] = (time_smooth, msd_smooth, err_smooth)
    except Exception as e:
        print(f"  Smoothed method failed: {e}")

    original_file = os.path.join(output_dir, f"{temp_label}_msd.csv")
    if os.path.exists(original_file):
        print(f"\nComparing with original: {original_file}")
        df_orig = pd.read_csv(original_file)
        time_orig = df_orig["time"].values
        msd_orig = df_orig["msd"].values
        err_orig = df_orig["msd_err"].values
        original_data = (time_orig, msd_orig, err_orig)

        plot_methods_comparison(temp_label, methods_data, original_data, output_dir)
    else:
        print(f"\nNo original file found for comparison: {original_file}")

    return time, msd, msd_err, n_worms


def plot_comparison(recalc_data, existing_data, temp_label, output_dir):
    time_recalc, msd_recalc, err_recalc, _ = recalc_data
    time_exist, msd_exist, err_exist = existing_data

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.loglog(time_recalc, msd_recalc, 'o-', label='Recalculated', markersize=4, alpha=0.7)
    ax1.loglog(time_exist, msd_exist, 's-', label='Existing', markersize=4, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('MSD (mm²)')
    ax1.set_title(f'{temp_label} - MSD Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    common_time = time_recalc
    msd_exist_interp = np.interp(common_time, time_exist, msd_exist)
    ratio = msd_recalc / msd_exist_interp

    ax2.semilogx(common_time, ratio, 'o-', markersize=4, alpha=0.7)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.5, label='Perfect match')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Ratio (Recalc / Existing)')
    ax2.set_title(f'{temp_label} - Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(output_dir, f"comparison_{temp_label}.png")
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  Comparison plot saved: {output_file}")


def main():
    print("=" * 70)
    print("Experimental MSD Recalculation from Raw Trajectories")
    print("=" * 70)

    base_dir = "EXP_FREE_SPACE_MULTI_T"
    output_dir = "EXP_FREE_SPACE_MULTI_T"
    os.makedirs(output_dir, exist_ok=True)

    temp_configs = [
        ("E10C", "5C"),
        ("E20C", "20C"),
        ("E30C", "30C"),
    ]

    results = {}

    for temp_dir_name, temp_label in temp_configs:
        temp_dir = os.path.join(base_dir, temp_dir_name)

        try:
            time, msd, msd_err, n_worms = process_temperature(temp_dir, temp_label, output_dir)
            results[temp_label] = (time, msd, msd_err, n_worms)
        except Exception as e:
            print(f"ERROR processing {temp_label}: {e}")
            continue

    print("\n" + "=" * 70)
    print("COMPARING WITH EXISTING MSD FILES")
    print("=" * 70)

    for temp_label in results.keys():
        existing_file = os.path.join(base_dir, f"{temp_label}_msd.csv")

        if os.path.exists(existing_file):
            print(f"\n{temp_label}:")
            try:
                df_exist = pd.read_csv(existing_file)
                time_exist = df_exist["time"].values
                msd_exist = df_exist["msd"].values
                err_exist = df_exist["msd_err"].values

                plot_comparison(
                    results[temp_label],
                    (time_exist, msd_exist, err_exist),
                    temp_label,
                    output_dir
                )
            except Exception as e:
                print(f"  ERROR comparing: {e}")
        else:
            print(f"\n{temp_label}: No existing file found at {existing_file}")

    print("\n" + "=" * 70)
    print("COMPLETED")
    print("=" * 70)
    print(f"\nRecalculated MSD files saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    exit(main())
