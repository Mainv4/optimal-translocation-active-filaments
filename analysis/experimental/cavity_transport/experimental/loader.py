
import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

EXP_DATA_DIR = "EXP_DATA_Cavity"

LONG_TRAJ_DIRS = [
    "241112", "241114", "241118", "241121_1", "241121_2",
    "241122_1_5worms", "241122_2", "241125_1", "241125_2",
    "241127_1", "241127_2_5worms", "241128_1", "241128_2"
]

DT_SHORT = 0.04

SHORT_DYNAMICS_CAVITY_DIRS = [
    "10C_cavity", "20C_cavity", "30C_cavity",
    "241112", "241114", "241118"
]


def find_csv_files(base_dir=None, include_short=False):
    if base_dir is None:
        base_dir = EXP_DATA_DIR

    csv_files = []

    for subdir in LONG_TRAJ_DIRS:
        pattern = os.path.join(base_dir, subdir, "worm_*.csv")
        csv_files.extend(glob.glob(pattern))

    if include_short:
        pattern = os.path.join(base_dir, "short_dynamics", "*", "*.csv")
        csv_files.extend(glob.glob(pattern))

    return sorted(csv_files)


def load_single_trajectory(csv_path):
    try:
        df = pd.read_csv(csv_path)

        required = ['com (x)', 'com (y)', 'time', 'T']
        for col in required:
            if col not in df.columns:
                print(f"  Warning: Missing column '{col}' in {csv_path}")
                return None

        time_s = df['time'].values
        x = df['com (x)'].values
        y = df['com (y)'].values
        temperature = df['T'].values[0]

        time_min = time_s / 60.0

        has_endpoints = 'end_1 (x)' in df.columns and 'end_2 (x)' in df.columns

        if has_endpoints:
            end1_x = df['end_1 (x)'].values
            end1_y = df['end_1 (y)'].values
            end2_x = df['end_2 (x)'].values
            end2_y = df['end_2 (y)'].values
        else:
            end1_x = end1_y = end2_x = end2_y = None

        worm_id = os.path.basename(csv_path).replace('.csv', '')
        parent_dir = os.path.basename(os.path.dirname(csv_path))
        worm_id = f"{parent_dir}_{worm_id}"

        valid = ~np.isnan(x) & ~np.isnan(y)
        if has_endpoints:
            valid_endpoints = ~np.isnan(end1_x) & ~np.isnan(end1_y) & \
                             ~np.isnan(end2_x) & ~np.isnan(end2_y)
        else:
            valid_endpoints = np.ones_like(valid)

        traj = {
            'time': time_min[valid],
            'x': x[valid],
            'y': y[valid],
            'temperature': temperature,
            'worm_id': worm_id,
            'path': csv_path,
        }

        if has_endpoints:
            combined_valid = valid & valid_endpoints
            traj['end1_x'] = end1_x[combined_valid]
            traj['end1_y'] = end1_y[combined_valid]
            traj['end2_x'] = end2_x[combined_valid]
            traj['end2_y'] = end2_y[combined_valid]
            traj['time_endpoints'] = time_min[combined_valid]

        return traj

    except Exception as e:
        print(f"  Error loading {csv_path}: {e}")
        return None


def load_trajectories_by_temperature(base_dir=None, include_short=False, verbose=True):
    csv_files = find_csv_files(base_dir, include_short)

    if verbose:
        print(f"Found {len(csv_files)} CSV files")

    trajectories_by_temp = defaultdict(list)

    for csv_path in csv_files:
        traj = load_single_trajectory(csv_path)
        if traj is not None:
            temp = float(traj['temperature'])
            trajectories_by_temp[temp].append(traj)

    trajectories_by_temp = dict(sorted(trajectories_by_temp.items()))

    if verbose:
        print("Trajectories by temperature:")
        for temp, trajs in trajectories_by_temp.items():
            total_frames = sum(len(t['time']) for t in trajs)
            print(f"  T = {temp}°C: {len(trajs)} worms, {total_frames} total frames")

    return trajectories_by_temp


def convert_to_segments(trajectories, min_length=10):
    segments = []

    for traj in trajectories:
        time = traj['time']
        x = traj['x']
        y = traj['y']

        if len(time) < min_length:
            continue

        segment = np.column_stack([time, x, y, np.zeros(len(time))])
        segments.append(segment)

    return segments


def _extract_temperature_from_dirname(dirname):
    if dirname.endswith("_cavity"):
        temp_str = dirname.split("C")[0]
        return float(temp_str)
    elif dirname.startswith("24"):
        return 20.0
    else:
        raise ValueError(f"Unknown short dynamics directory: {dirname}")


def find_short_dynamics_files(base_dir=None):
    if base_dir is None:
        base_dir = EXP_DATA_DIR

    csv_files = []

    for subdir in SHORT_DYNAMICS_CAVITY_DIRS:
        pattern = os.path.join(base_dir, "short_dynamics", subdir, "*.csv")
        csv_files.extend(glob.glob(pattern))

    return sorted(csv_files)


def load_short_dynamics_trajectory(csv_path):
    try:
        df = pd.read_csv(csv_path)

        required = ['com (x)', 'com (y)']
        for col in required:
            if col not in df.columns:
                print(f"  Warning: Missing column '{col}' in {csv_path}")
                return None

        x = df['com (x)'].values
        y = df['com (y)'].values

        n_frames = len(df)
        time_s = np.arange(n_frames) * DT_SHORT
        time_min = time_s / 60.0

        parent_dir = os.path.basename(os.path.dirname(csv_path))
        try:
            temperature = _extract_temperature_from_dirname(parent_dir)
        except ValueError as e:
            print(f"  Warning: {e}")
            return None

        has_endpoints = 'end_1 (x)' in df.columns and 'end_2 (x)' in df.columns

        if has_endpoints:
            end1_x = df['end_1 (x)'].values
            end1_y = df['end_1 (y)'].values
            end2_x = df['end_2 (x)'].values
            end2_y = df['end_2 (y)'].values
        else:
            end1_x = end1_y = end2_x = end2_y = None

        worm_id = os.path.basename(csv_path).replace('.csv', '')
        worm_id = f"short_{parent_dir}_{worm_id}"

        valid = ~np.isnan(x) & ~np.isnan(y)
        if has_endpoints:
            valid_endpoints = ~np.isnan(end1_x) & ~np.isnan(end1_y) & \
                             ~np.isnan(end2_x) & ~np.isnan(end2_y)
        else:
            valid_endpoints = np.ones_like(valid)

        traj = {
            'time': time_min[valid],
            'x': x[valid],
            'y': y[valid],
            'temperature': temperature,
            'worm_id': worm_id,
            'path': csv_path,
            'is_short_dynamics': True,
            'dt_seconds': DT_SHORT,
        }

        if has_endpoints:
            combined_valid = valid & valid_endpoints
            traj['end1_x'] = end1_x[combined_valid]
            traj['end1_y'] = end1_y[combined_valid]
            traj['end2_x'] = end2_x[combined_valid]
            traj['end2_y'] = end2_y[combined_valid]
            traj['time_endpoints'] = time_min[combined_valid]

        return traj

    except Exception as e:
        print(f"  Error loading short dynamics {csv_path}: {e}")
        return None


def load_short_dynamics_by_temperature(base_dir=None, verbose=True):
    csv_files = find_short_dynamics_files(base_dir)

    if verbose:
        print(f"Found {len(csv_files)} short dynamics CSV files")

    trajectories_by_temp = defaultdict(list)

    for csv_path in csv_files:
        traj = load_short_dynamics_trajectory(csv_path)
        if traj is not None:
            temp = float(traj['temperature'])
            trajectories_by_temp[temp].append(traj)

    trajectories_by_temp = dict(sorted(trajectories_by_temp.items()))

    if verbose:
        print("Short dynamics trajectories by temperature:")
        for temp, trajs in trajectories_by_temp.items():
            total_frames = sum(len(t['time']) for t in trajs)
            duration_s = total_frames * DT_SHORT
            print(f"  T = {temp}°C: {len(trajs)} worms, {total_frames} total frames ({duration_s:.1f} s)")

    return trajectories_by_temp
