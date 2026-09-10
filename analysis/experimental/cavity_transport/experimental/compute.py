
import argparse
import os
import numpy as np
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

from .loader import (
    load_trajectories_by_temperature,
    load_short_dynamics_by_temperature,
    DT_SHORT,
)

LEFT_CAVITY_X = (-48, -32)
RIGHT_CAVITY_X = (32, 48)
CHANNEL_X = (-32, 32)
CAVITY_Y = (-4, 4)

LEFT_CAVITY_CENTER = (-44.0, 0.0)
RIGHT_CAVITY_CENTER = (44.0, 0.0)

EXP_CAVITY_RADIUS = 4.0
EXP_CAVITY_RADIUS_GEOM = 4.0
EXP_CAVITY_RADIUS_EFF = 3.0
EXP_CAVITY_RADIUS_STAR = 3.75
EXP_R_SQUARED_THRESHOLD = 15.0

MIN_SEGMENT_LENGTH = 50

DT_NOMINAL = 2.0 / 60.0

DT_SHORT_MIN = DT_SHORT / 60.0

CROSSOVER_TIME_SEC = 2.0


def interpolate_gaps(time, x, y, dt_nominal=DT_NOMINAL, max_gap_factor=3):
    if len(time) < 2:
        return [(time.copy(), x.copy(), y.copy())]

    dt = np.diff(time)
    gap_threshold = max_gap_factor * dt_nominal

    large_gap_indices = np.where(dt > gap_threshold)[0]

    split_points = [0] + list(large_gap_indices + 1) + [len(time)]

    segments = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]

        if end - start < 2:
            continue

        t_seg = time[start:end]
        x_seg = x[start:end]
        y_seg = y[start:end]

        dt_seg = np.diff(t_seg)
        small_gap_indices = np.where((dt_seg > 1.5 * dt_nominal) & (dt_seg <= gap_threshold))[0]

        if len(small_gap_indices) == 0:
            segments.append((t_seg.copy(), x_seg.copy(), y_seg.copy()))
        else:
            new_time = [t_seg[0]]
            new_x = [x_seg[0]]
            new_y = [y_seg[0]]

            for j in range(len(t_seg) - 1):
                dt_j = t_seg[j + 1] - t_seg[j]

                if dt_j > 1.5 * dt_nominal and dt_j <= gap_threshold:
                    n_interp = int(np.round(dt_j / dt_nominal)) - 1
                    if n_interp > 0:
                        t_interp = np.linspace(t_seg[j], t_seg[j + 1], n_interp + 2)[1:-1]
                        x_interp = np.interp(t_interp, [t_seg[j], t_seg[j + 1]], [x_seg[j], x_seg[j + 1]])
                        y_interp = np.interp(t_interp, [t_seg[j], t_seg[j + 1]], [y_seg[j], y_seg[j + 1]])
                        new_time.extend(t_interp)
                        new_x.extend(x_interp)
                        new_y.extend(y_interp)

                new_time.append(t_seg[j + 1])
                new_x.append(x_seg[j + 1])
                new_y.append(y_seg[j + 1])

            segments.append((np.array(new_time), np.array(new_x), np.array(new_y)))

    return segments if segments else [(time.copy(), x.copy(), y.copy())]


def extract_cavity_segments_exp(trajectories, verbose=True, interpolate=True, return_metadata=False):
    segments = []
    metadata = []
    n_interpolated = 0

    for traj_idx, traj in enumerate(trajectories):
        time_raw = traj['time']
        x_raw = traj['x']
        y_raw = traj['y']
        worm_id = traj.get('worm_id', f'worm_{traj_idx:03d}')

        has_endpoints = 'end1_x' in traj and traj['end1_x'] is not None

        time_ep = time_raw
        end1_x_raw = np.full(len(time_raw), np.nan)
        end1_y_raw = np.full(len(time_raw), np.nan)
        end2_x_raw = np.full(len(time_raw), np.nan)
        end2_y_raw = np.full(len(time_raw), np.nan)
        ep_aligned = True

        if has_endpoints:
            time_ep = traj.get('time_endpoints', time_raw)
            end1_x_raw = traj['end1_x']
            end1_y_raw = traj['end1_y']
            end2_x_raw = traj['end2_x']
            end2_y_raw = traj['end2_y']

            if len(time_ep) == len(time_raw):
                ep_aligned = True
            else:
                ep_aligned = False

        if interpolate:
            sub_trajs = interpolate_gaps(time_raw, x_raw, y_raw)
            if len(sub_trajs) > 1:
                n_interpolated += 1
        else:
            sub_trajs = [(time_raw, x_raw, y_raw)]

        for time, x, y in sub_trajs:
            if has_endpoints:
                if ep_aligned and not interpolate:
                    end1_x = end1_x_raw
                    end1_y = end1_y_raw
                    end2_x = end2_x_raw
                    end2_y = end2_y_raw
                else:
                    end1_x = np.interp(time, time_ep, end1_x_raw)
                    end1_y = np.interp(time, time_ep, end1_y_raw)
                    end2_x = np.interp(time, time_ep, end2_x_raw)
                    end2_y = np.interp(time, time_ep, end2_y_raw)
            else:
                end1_x = np.full(len(time), np.nan)
                end1_y = np.full(len(time), np.nan)
                end2_x = np.full(len(time), np.nan)
                end2_y = np.full(len(time), np.nan)

            in_left = x < LEFT_CAVITY_X[1]
            in_right = x > RIGHT_CAVITY_X[0]
            in_cavity = in_left | in_right

            segment_start = None
            current_cavity = None

            for i in range(len(time)):
                if in_cavity[i]:
                    cavity_id = -1 if in_left[i] else 1
                    if segment_start is None:
                        segment_start = i
                        current_cavity = cavity_id
                    elif cavity_id != current_cavity:
                        if i - segment_start >= MIN_SEGMENT_LENGTH:
                            seg = np.column_stack([
                                time[segment_start:i],
                                x[segment_start:i],
                                y[segment_start:i],
                                np.full(i - segment_start, current_cavity),
                                end1_x[segment_start:i],
                                end1_y[segment_start:i],
                                end2_x[segment_start:i],
                                end2_y[segment_start:i],
                            ])
                            segments.append(seg)
                            metadata.append({
                                'worm_id': worm_id,
                                'cavity': 'left' if current_cavity < 0 else 'right',
                                'traj_idx': traj_idx,
                            })
                        segment_start = i
                        current_cavity = cavity_id
                else:
                    if segment_start is not None:
                        if i - segment_start >= MIN_SEGMENT_LENGTH:
                            seg = np.column_stack([
                                time[segment_start:i],
                                x[segment_start:i],
                                y[segment_start:i],
                                np.full(i - segment_start, current_cavity),
                                end1_x[segment_start:i],
                                end1_y[segment_start:i],
                                end2_x[segment_start:i],
                                end2_y[segment_start:i],
                            ])
                            segments.append(seg)
                            metadata.append({
                                'worm_id': worm_id,
                                'cavity': 'left' if current_cavity < 0 else 'right',
                                'traj_idx': traj_idx,
                            })
                        segment_start = None
                        current_cavity = None

            if segment_start is not None:
                if len(time) - segment_start >= MIN_SEGMENT_LENGTH:
                    seg = np.column_stack([
                        time[segment_start:],
                        x[segment_start:],
                        y[segment_start:],
                        np.full(len(time) - segment_start, current_cavity),
                        end1_x[segment_start:],
                        end1_y[segment_start:],
                        end2_x[segment_start:],
                        end2_y[segment_start:],
                    ])
                    segments.append(seg)
                    metadata.append({
                        'worm_id': worm_id,
                        'cavity': 'left' if current_cavity < 0 else 'right',
                        'traj_idx': traj_idx,
                    })

    if verbose:
        n_left = sum(1 for s in segments if s[0, 3] < 0)
        n_right = sum(1 for s in segments if s[0, 3] > 0)
        print(f"  Extracted {len(segments)} cavity segments ({n_left} left, {n_right} right)")
        if interpolate and n_interpolated > 0:
            print(f"  ({n_interpolated} trajectories had gaps interpolated/split)")

        if segments:
            all_x = np.concatenate([s[:, 1] for s in segments])
            all_y = np.concatenate([s[:, 2] for s in segments])
            lengths = [len(s) for s in segments]
            print(f"  Spatial extent: x ∈ [{all_x.min():.1f}, {all_x.max():.1f}] mm, y ∈ [{all_y.min():.1f}, {all_y.max():.1f}] mm")
            print(f"  Segment lengths: min={min(lengths)}, median={np.median(lengths):.0f}, max={max(lengths)} points")

    if return_metadata:
        return segments, metadata
    return segments


ALIGNED_CAVITY_CENTER = (44.0, 0.0)
ALIGNED_CAVITY_THRESHOLD = 40.0


def extract_aligned_segments(trajectories, tracking_point='end1', verbose=True, return_metadata=False):
    if tracking_point not in ['end1', 'end2']:
        raise ValueError(f"tracking_point must be 'end1' or 'end2', got '{tracking_point}'")

    segments = []
    metadata = []
    n_aligned_A = 0
    n_aligned_B = 0

    for traj_idx, traj in enumerate(trajectories):
        worm_id = traj.get('worm_id', f'worm_{traj_idx:03d}')

        if tracking_point == 'end1':
            if 'end1_x' not in traj or traj['end1_x'] is None:
                continue
            x_raw = traj['end1_x']
            y_raw = traj['end1_y']
            time_raw = traj.get('time_endpoints', traj['time'])
        else:
            if 'end2_x' not in traj or traj['end2_x'] is None:
                continue
            x_raw = traj['end2_x']
            y_raw = traj['end2_y']
            time_raw = traj.get('time_endpoints', traj['time'])

        valid = ~np.isnan(x_raw) & ~np.isnan(y_raw)
        if np.sum(valid) < MIN_SEGMENT_LENGTH:
            continue

        x_valid = x_raw[valid]
        y_valid = y_raw[valid]
        time_valid = time_raw[valid]

        for version, x_v in [('A', x_valid), ('B', -x_valid)]:
            x_max = np.max(x_v)
            if x_max < 43.0:
                continue

            dx = 48.0 - x_max
            x_aligned = x_v + dx

            if version == 'A':
                n_aligned_A += 1
            else:
                n_aligned_B += 1

            in_cavity = x_aligned > ALIGNED_CAVITY_THRESHOLD

            segment_start = None
            for i in range(len(time_valid)):
                if in_cavity[i]:
                    if segment_start is None:
                        segment_start = i
                else:
                    if segment_start is not None:
                        if i - segment_start >= MIN_SEGMENT_LENGTH:
                            seg = np.column_stack([
                                time_valid[segment_start:i],
                                x_aligned[segment_start:i],
                                y_valid[segment_start:i],
                            ])
                            segments.append(seg)
                            metadata.append({
                                'worm_id': worm_id,
                                'traj_idx': traj_idx,
                                'version': version,
                            })
                        segment_start = None

            if segment_start is not None:
                if len(time_valid) - segment_start >= MIN_SEGMENT_LENGTH:
                    seg = np.column_stack([
                        time_valid[segment_start:],
                        x_aligned[segment_start:],
                        y_valid[segment_start:],
                    ])
                    segments.append(seg)
                    metadata.append({
                        'worm_id': worm_id,
                        'traj_idx': traj_idx,
                        'version': version,
                    })

    if verbose:
        print(f"  Extracted {len(segments)} aligned segments ({tracking_point})")
        print(f"  Aligned trajectories: {n_aligned_A} original + {n_aligned_B} mirror")

        if segments:
            all_x = np.concatenate([s[:, 1] for s in segments])
            all_y = np.concatenate([s[:, 2] for s in segments])
            lengths = [len(s) for s in segments]
            print(f"  Spatial extent: x ∈ [{all_x.min():.1f}, {all_x.max():.1f}] mm, y ∈ [{all_y.min():.1f}, {all_y.max():.1f}] mm")
            print(f"  Segment lengths: min={min(lengths)}, median={np.median(lengths):.0f}, max={max(lengths)} points")

    if return_metadata:
        return segments, metadata
    return segments


def calculate_rotational_msd_aligned(trajectories, tracking_point='end1', max_lag_fraction=0.5, verbose=True):
    if tracking_point not in ['end1', 'end2']:
        raise ValueError(f"tracking_point must be 'end1' or 'end2', got '{tracking_point}'")

    if verbose:
        print(f"Calculating rotational MSD (aligned, {tracking_point})...")

    if not trajectories:
        print("  No trajectories found")
        return np.array([]), []

    segments = extract_aligned_segments(trajectories, tracking_point=tracking_point, verbose=verbose)

    if not segments:
        print("  No aligned segments found")
        return np.array([]), []

    all_individual_msds = []
    all_dt = []
    center_x, center_y = ALIGNED_CAVITY_CENTER

    for segment in tqdm(segments, desc="  Processing segments", disable=not verbose):
        time = segment[:, 0]
        x = segment[:, 1]
        y = segment[:, 2]

        if len(time) < MIN_SEGMENT_LENGTH:
            continue

        dt = np.median(np.diff(time))
        all_dt.append(dt)

        x_centered = x - center_x
        y_centered = y - center_y

        theta_raw = np.arctan2(y_centered, x_centered)
        theta = np.unwrap(theta_raw)

        max_lag = int(len(theta) * max_lag_fraction)
        max_lag = max(max_lag, 1)
        msd_seg = np.full(max_lag, np.nan)

        for lag in range(1, max_lag + 1):
            displacements = (theta[lag:] - theta[:-lag]) ** 2
            if len(displacements) > 0:
                msd_seg[lag - 1] = np.mean(displacements)

        all_individual_msds.append(msd_seg)

    if not all_individual_msds:
        print("  No valid segments")
        return np.array([]), []

    dt_avg = np.mean(all_dt)
    if verbose:
        print(f"  Average dt = {dt_avg:.4f} min ({dt_avg*60:.1f} s)")

    MIN_SEGMENTS_FOR_AVERAGE = max(5, len(all_individual_msds) // 4)

    max_len = max(len(m) for m in all_individual_msds)
    padded = np.full((len(all_individual_msds), max_len), np.nan)
    for i, m in enumerate(all_individual_msds):
        padded[i, :len(m)] = m

    average_msd = np.nanmean(padded, axis=0)
    std_msd = np.nanstd(padded, axis=0)
    count = np.sum(~np.isnan(padded), axis=0)
    std_error = std_msd / np.sqrt(np.maximum(count, 1))

    time_lags = np.arange(1, max_len + 1) * dt_avg

    valid = (~np.isnan(average_msd)) & (count >= MIN_SEGMENTS_FOR_AVERAGE)
    if not np.any(valid):
        print("  Warning: Not enough segments for reliable rotational MSD")
        valid = ~np.isnan(average_msd)

    if not np.any(valid):
        print("  Warning: All MSD values are NaN")
        return np.array([]), all_individual_msds

    last_valid = np.where(valid)[0][-1] + 1
    msd_result = np.column_stack((
        time_lags[:last_valid],
        average_msd[:last_valid],
        std_error[:last_valid]
    ))

    if verbose:
        print(f"  {len(all_individual_msds)} segments, truncated at lag {last_valid} ({time_lags[last_valid-1]*60:.0f} s)")

    return msd_result, all_individual_msds


def calculate_translational_msd_aligned(trajectories, tracking_point='end1', max_lag_fraction=0.5, verbose=True):
    if tracking_point not in ['end1', 'end2']:
        raise ValueError(f"tracking_point must be 'end1' or 'end2', got '{tracking_point}'")

    if verbose:
        print(f"Calculating translational MSD (aligned, {tracking_point})...")

    if not trajectories:
        print("  No trajectories found")
        return np.array([]), []

    segments = extract_aligned_segments(trajectories, tracking_point=tracking_point, verbose=verbose)

    if not segments:
        print("  No aligned segments found")
        return np.array([]), []

    all_individual_msds = []
    all_dt = []

    for segment in tqdm(segments, desc="  Processing segments", disable=not verbose):
        time = segment[:, 0]
        x = segment[:, 1]
        y = segment[:, 2]

        if len(time) < MIN_SEGMENT_LENGTH:
            continue

        dt = np.median(np.diff(time))
        all_dt.append(dt)

        max_lag = int(len(x) * max_lag_fraction)
        max_lag = max(max_lag, 1)
        msd_seg = np.full(max_lag, np.nan)

        for lag in range(1, max_lag + 1):
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            displacements = dx**2 + dy**2
            if len(displacements) > 0:
                msd_seg[lag - 1] = np.mean(displacements)

        all_individual_msds.append(msd_seg)

    if not all_individual_msds:
        print("  No valid segments")
        return np.array([]), []

    dt_avg = np.mean(all_dt)
    if verbose:
        print(f"  Average dt = {dt_avg:.4f} min ({dt_avg*60:.1f} s)")

    MIN_SEGMENTS_FOR_AVERAGE = max(5, len(all_individual_msds) // 4)

    max_len = max(len(m) for m in all_individual_msds)
    padded = np.full((len(all_individual_msds), max_len), np.nan)
    for i, m in enumerate(all_individual_msds):
        padded[i, :len(m)] = m

    average_msd = np.nanmean(padded, axis=0)
    std_msd = np.nanstd(padded, axis=0)
    count = np.sum(~np.isnan(padded), axis=0)
    std_error = std_msd / np.sqrt(np.maximum(count, 1))

    time_lags = np.arange(1, max_len + 1) * dt_avg

    valid = (~np.isnan(average_msd)) & (count >= MIN_SEGMENTS_FOR_AVERAGE)
    if not np.any(valid):
        print("  Warning: Not enough segments for reliable MSD")
        valid = ~np.isnan(average_msd)

    if not np.any(valid):
        print("  Warning: All MSD values are NaN")
        return np.array([]), all_individual_msds

    last_valid = np.where(valid)[0][-1] + 1
    msd_result = np.column_stack((
        time_lags[:last_valid],
        average_msd[:last_valid],
        std_error[:last_valid]
    ))

    if verbose:
        print(f"  {len(all_individual_msds)} segments, truncated at lag {last_valid} ({time_lags[last_valid-1]*60:.0f} s)")

    return msd_result, all_individual_msds


def calculate_rotational_msd_exp(trajectories, tracking_point='com', max_lag_fraction=0.5, verbose=True):
    col_map = {
        'com': (1, 2),
        'end1': (4, 5),
        'end2': (6, 7),
    }
    if tracking_point not in col_map:
        raise ValueError(f"tracking_point must be one of {list(col_map.keys())}")
    x_col, y_col = col_map[tracking_point]

    if verbose:
        print(f"Calculating rotational MSD (experimental, {tracking_point})...")

    if not trajectories:
        print("  No trajectories found")
        return np.array([]), []

    segments = extract_cavity_segments_exp(trajectories, verbose=verbose)

    if not segments:
        print("  No cavity segments found")
        return np.array([]), []

    all_individual_msds = []
    all_dt = []

    for segment in tqdm(segments, desc="  Processing segments", disable=not verbose):
        time = segment[:, 0]
        x = segment[:, x_col]
        y = segment[:, y_col]
        cavity_id = segment[0, 3]

        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            continue

        if len(time) < MIN_SEGMENT_LENGTH:
            continue

        dt = np.median(np.diff(time))
        all_dt.append(dt)

        if cavity_id < 0:
            center_x, center_y = LEFT_CAVITY_CENTER
        else:
            center_x, center_y = RIGHT_CAVITY_CENTER

        x_centered = x - center_x
        y_centered = y - center_y

        if cavity_id > 0:
            x_centered = -x_centered

        theta_raw = np.arctan2(y_centered, x_centered)
        theta = np.unwrap(theta_raw)

        max_lag = int(len(theta) * max_lag_fraction)
        max_lag = max(max_lag, 1)
        msd_seg = np.full(max_lag, np.nan)

        for lag in range(1, max_lag + 1):
            displacements = (theta[lag:] - theta[:-lag]) ** 2
            if len(displacements) > 0:
                msd_seg[lag - 1] = np.mean(displacements)

        all_individual_msds.append(msd_seg)

    if not all_individual_msds:
        print("  No valid segments")
        return np.array([]), []

    dt_avg = np.mean(all_dt)
    if verbose:
        print(f"  Average dt = {dt_avg:.4f} min ({dt_avg*60:.1f} s)")

    MIN_SEGMENTS_FOR_AVERAGE = max(5, len(all_individual_msds) // 4)

    max_len = max(len(m) for m in all_individual_msds)
    padded = np.full((len(all_individual_msds), max_len), np.nan)
    for i, m in enumerate(all_individual_msds):
        padded[i, :len(m)] = m

    average_msd = np.nanmean(padded, axis=0)
    std_msd = np.nanstd(padded, axis=0)
    count = np.sum(~np.isnan(padded), axis=0)
    std_error = std_msd / np.sqrt(np.maximum(count, 1))

    time_lags = np.arange(1, max_len + 1) * dt_avg

    valid = (~np.isnan(average_msd)) & (count >= MIN_SEGMENTS_FOR_AVERAGE)
    if not np.any(valid):
        print("  Warning: Not enough segments for reliable rotational MSD")
        valid = ~np.isnan(average_msd)

    if not np.any(valid):
        print("  Warning: All MSD values are NaN")
        return np.array([]), all_individual_msds

    last_valid = np.where(valid)[0][-1] + 1
    msd_result = np.column_stack((
        time_lags[:last_valid],
        average_msd[:last_valid],
        std_error[:last_valid]
    ))

    if verbose:
        print(f"  Min segments for average: {MIN_SEGMENTS_FOR_AVERAGE}, truncated at lag {last_valid} ({time_lags[last_valid-1]*60:.0f} s)")
        print(f"  Rotational MSD computed: {len(msd_result)} time points")

    return msd_result, all_individual_msds


def calculate_translational_msd_exp(trajectories, tracking_point='com', max_lag_fraction=0.5, verbose=True):
    col_map = {
        'com': (1, 2),
        'end1': (4, 5),
        'end2': (6, 7),
    }
    if tracking_point not in col_map:
        raise ValueError(f"tracking_point must be one of {list(col_map.keys())}")
    x_col, y_col = col_map[tracking_point]

    if verbose:
        print(f"Calculating translational MSD (experimental, {tracking_point})...")

    if not trajectories:
        print("  No trajectories found")
        return np.array([]), []

    segments = extract_cavity_segments_exp(trajectories, verbose=verbose)

    if not segments:
        print("  No cavity segments found")
        return np.array([]), []

    all_individual_msds = []
    all_dt = []

    for segment in tqdm(segments, desc="  Processing segments", disable=not verbose):
        time = segment[:, 0]
        x = segment[:, x_col]
        y = segment[:, y_col]

        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            continue

        if len(time) < MIN_SEGMENT_LENGTH:
            continue

        dt = np.median(np.diff(time))
        all_dt.append(dt)

        max_lag = int(len(x) * max_lag_fraction)
        max_lag = max(max_lag, 1)
        msd_seg = np.full(max_lag, np.nan)

        for lag in range(1, max_lag + 1):
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            displacements = dx**2 + dy**2
            if len(displacements) > 0:
                msd_seg[lag - 1] = np.mean(displacements)

        all_individual_msds.append(msd_seg)

    if not all_individual_msds:
        print("  No valid segments")
        return np.array([]), []

    dt_avg = np.mean(all_dt)
    if verbose:
        print(f"  Average dt = {dt_avg:.4f} min ({dt_avg*60:.1f} s)")

    MIN_SEGMENTS_FOR_AVERAGE = max(5, len(all_individual_msds) // 4)

    max_len = max(len(m) for m in all_individual_msds)
    padded = np.full((len(all_individual_msds), max_len), np.nan)
    for i, m in enumerate(all_individual_msds):
        padded[i, :len(m)] = m

    average_msd = np.nanmean(padded, axis=0)
    std_msd = np.nanstd(padded, axis=0)
    count = np.sum(~np.isnan(padded), axis=0)
    std_error = std_msd / np.sqrt(np.maximum(count, 1))

    time_lags = np.arange(1, max_len + 1) * dt_avg

    valid = (~np.isnan(average_msd)) & (count >= MIN_SEGMENTS_FOR_AVERAGE)
    if not np.any(valid):
        print("  Warning: Not enough segments for reliable MSD")
        valid = ~np.isnan(average_msd)

    if not np.any(valid):
        print("  Warning: All MSD values are NaN")
        return np.array([]), all_individual_msds

    last_valid = np.where(valid)[0][-1] + 1
    msd_result = np.column_stack((
        time_lags[:last_valid],
        average_msd[:last_valid],
        std_error[:last_valid]
    ))

    if verbose:
        print(f"  Min segments for average: {MIN_SEGMENTS_FOR_AVERAGE}, truncated at lag {last_valid} ({time_lags[last_valid-1]*60:.0f} s)")

    if verbose:
        print(f"  Translational MSD computed: {len(msd_result)} time points")

    return msd_result, all_individual_msds


def extract_e2e_aligned_segments(trajectories, verbose=True):
    segments = []
    n_aligned_A = 0
    n_aligned_B = 0

    for traj_idx, traj in enumerate(trajectories):
        if 'end1_x' not in traj or traj['end1_x'] is None:
            continue
        if 'end2_x' not in traj or traj['end2_x'] is None:
            continue

        end1_x_raw = traj['end1_x']
        end1_y_raw = traj['end1_y']
        end2_x_raw = traj['end2_x']
        end2_y_raw = traj['end2_y']
        time_raw = traj.get('time_endpoints', traj['time'])

        valid = (~np.isnan(end1_x_raw) & ~np.isnan(end1_y_raw) &
                 ~np.isnan(end2_x_raw) & ~np.isnan(end2_y_raw))
        if np.sum(valid) < MIN_SEGMENT_LENGTH:
            continue

        end1_x = end1_x_raw[valid]
        end1_y = end1_y_raw[valid]
        end2_x = end2_x_raw[valid]
        end2_y = end2_y_raw[valid]
        time_valid = time_raw[valid]

        for version, (e1x, e2x) in [('A', (end1_x, end2_x)),
                                     ('B', (-end1_x, -end2_x))]:
            x_max = np.max(e1x)
            if x_max < 43.0:
                continue

            dx = 48.0 - x_max
            e1x_aligned = e1x + dx
            e2x_aligned = e2x + dx

            if version == 'A':
                n_aligned_A += 1
            else:
                n_aligned_B += 1

            in_cavity = e1x_aligned > ALIGNED_CAVITY_THRESHOLD

            segment_start = None
            for i in range(len(time_valid)):
                if in_cavity[i]:
                    if segment_start is None:
                        segment_start = i
                else:
                    if segment_start is not None:
                        if i - segment_start >= MIN_SEGMENT_LENGTH:
                            e2e_x = e2x_aligned[segment_start:i] - e1x_aligned[segment_start:i]
                            e2e_y = end2_y[segment_start:i] - end1_y[segment_start:i]
                            seg = np.column_stack([
                                time_valid[segment_start:i],
                                e2e_x,
                                e2e_y,
                            ])
                            segments.append(seg)
                        segment_start = None

            if segment_start is not None:
                if len(time_valid) - segment_start >= MIN_SEGMENT_LENGTH:
                    e2e_x = e2x_aligned[segment_start:] - e1x_aligned[segment_start:]
                    e2e_y = end2_y[segment_start:] - end1_y[segment_start:]
                    seg = np.column_stack([
                        time_valid[segment_start:],
                        e2e_x,
                        e2e_y,
                    ])
                    segments.append(seg)

    if verbose:
        print(f"  Extracted {len(segments)} e2e segments (aligned)")
        print(f"  Aligned trajectories: {n_aligned_A} original + {n_aligned_B} mirror")

        if segments:
            lengths = [len(s) for s in segments]
            print(f"  Segment lengths: min={min(lengths)}, median={np.median(lengths):.0f}, max={max(lengths)} points")

    return segments


def compute_e2e_autocorr_segment(segment, dt_nominal=DT_NOMINAL):
    time = segment[:, 0]
    e2e_x = segment[:, 1]
    e2e_y = segment[:, 2]

    norms = np.sqrt(e2e_x**2 + e2e_y**2)
    valid = norms > 0
    if np.sum(valid) < 10:
        return None, None, None

    ux = np.zeros_like(e2e_x)
    uy = np.zeros_like(e2e_y)
    ux[valid] = e2e_x[valid] / norms[valid]
    uy[valid] = e2e_y[valid] / norms[valid]

    mean_norm_sq = np.mean(norms[valid]**2)

    n = len(time)
    max_lag = n // 2

    autocorr_full = np.zeros(max_lag)
    autocorr_unit = np.zeros(max_lag)

    for lag in range(max_lag):
        valid_pairs = valid[:n-lag] & valid[lag:]
        if np.sum(valid_pairs) > 0:
            dot_unit = ux[:n-lag] * ux[lag:] + uy[:n-lag] * uy[lag:]
            autocorr_unit[lag] = np.mean(dot_unit[valid_pairs])
            dot_full = e2e_x[:n-lag] * e2e_x[lag:] + e2e_y[:n-lag] * e2e_y[lag:]
            autocorr_full[lag] = np.mean(dot_full[valid_pairs])

    if autocorr_unit[0] > 0:
        autocorr_unit = autocorr_unit / autocorr_unit[0]

    if mean_norm_sq > 0:
        autocorr_full = autocorr_full / mean_norm_sq

    time_lags = np.arange(max_lag) * dt_nominal

    return time_lags, autocorr_full, autocorr_unit


def _find_tau_decorr(time_grid, mean_autocorr):
    threshold = 1.0 / np.e
    below_threshold = mean_autocorr < threshold
    if np.any(below_threshold):
        idx = np.where(below_threshold)[0][0]
        if idx > 0:
            t0, t1 = time_grid[idx-1], time_grid[idx]
            c0, c1 = mean_autocorr[idx-1], mean_autocorr[idx]
            return t0 + (t1 - t0) * (c0 - threshold) / (c0 - c1)
        else:
            return time_grid[0]
    return np.nan


def _aggregate_autocorr(all_data, all_times, label, verbose):
    if len(all_data) == 0:
        return None, np.nan
    min_len = min(len(ac) for ac in all_data)
    time_grid = all_times[0][:min_len]
    matrix = np.array([ac[:min_len] for ac in all_data])
    mean_ac = np.mean(matrix, axis=0)
    std_ac = np.std(matrix, axis=0)
    std_err = std_ac / np.sqrt(len(all_data))
    tau = _find_tau_decorr(time_grid, mean_ac)
    if verbose:
        if np.isfinite(tau):
            print(f"  {label}: τ_decorr = {tau:.4f} min = {tau*60:.2f} s")
        else:
            print(f"  {label}: τ_decorr NOT REACHED")
    return np.column_stack([time_grid, mean_ac, std_err]), tau


def calculate_e2e_autocorr_aligned(trajectories, verbose=True):
    if verbose:
        print("\nComputing e2e vector autocorrelation (aligned, both observables)...")

    segments = extract_e2e_aligned_segments(trajectories, verbose=verbose)

    if len(segments) == 0:
        print("  No valid segments found")
        return {'full': (None, None), 'unit': (None, None)}

    all_full = []
    all_unit = []
    all_times = []

    for seg in segments:
        time_lags, ac_full, ac_unit = compute_e2e_autocorr_segment(seg)
        if time_lags is not None and len(time_lags) > 10:
            all_full.append(ac_full)
            all_unit.append(ac_unit)
            all_times.append(time_lags)

    if len(all_full) == 0:
        print("  No valid autocorrelations computed")
        return {'full': (None, None), 'unit': (None, None)}

    if verbose:
        print(f"  Computed autocorrelation for {len(all_full)} segments")

    result_full, tau_full = _aggregate_autocorr(all_full, all_times, "Full vector", verbose)
    result_unit, tau_unit = _aggregate_autocorr(all_unit, all_times, "Unit vector", verbose)

    return {'full': (result_full, tau_full), 'unit': (result_unit, tau_unit)}


MIN_SEGMENT_LENGTH_SHORT = 10


def extract_e2e_aligned_segments_short(trajectories, verbose=True):
    segments = []
    n_aligned_A = 0
    n_aligned_B = 0

    for traj_idx, traj in enumerate(trajectories):
        if 'end1_x' not in traj or traj['end1_x'] is None:
            continue
        if 'end2_x' not in traj or traj['end2_x'] is None:
            continue

        end1_x_raw = traj['end1_x']
        end1_y_raw = traj['end1_y']
        end2_x_raw = traj['end2_x']
        end2_y_raw = traj['end2_y']
        time_raw = traj.get('time_endpoints', traj['time'])

        valid = (~np.isnan(end1_x_raw) & ~np.isnan(end1_y_raw) &
                 ~np.isnan(end2_x_raw) & ~np.isnan(end2_y_raw))
        if np.sum(valid) < MIN_SEGMENT_LENGTH_SHORT:
            continue

        end1_x = end1_x_raw[valid]
        end1_y = end1_y_raw[valid]
        end2_x = end2_x_raw[valid]
        end2_y = end2_y_raw[valid]
        time_valid = time_raw[valid]

        for version, (e1x, e2x) in [('A', (end1_x, end2_x)),
                                     ('B', (-end1_x, -end2_x))]:
            x_max = np.max(e1x)
            if x_max < 43.0:
                continue

            dx = 48.0 - x_max
            e1x_aligned = e1x + dx
            e2x_aligned = e2x + dx

            if version == 'A':
                n_aligned_A += 1
            else:
                n_aligned_B += 1

            in_cavity = e1x_aligned > ALIGNED_CAVITY_THRESHOLD

            segment_start = None
            for i in range(len(time_valid)):
                if in_cavity[i]:
                    if segment_start is None:
                        segment_start = i
                else:
                    if segment_start is not None:
                        if i - segment_start >= MIN_SEGMENT_LENGTH_SHORT:
                            e2e_x = e2x_aligned[segment_start:i] - e1x_aligned[segment_start:i]
                            e2e_y = end2_y[segment_start:i] - end1_y[segment_start:i]
                            seg = np.column_stack([
                                time_valid[segment_start:i],
                                e2e_x,
                                e2e_y,
                            ])
                            segments.append(seg)
                        segment_start = None

            if segment_start is not None:
                if len(time_valid) - segment_start >= MIN_SEGMENT_LENGTH_SHORT:
                    e2e_x = e2x_aligned[segment_start:] - e1x_aligned[segment_start:]
                    e2e_y = end2_y[segment_start:] - end1_y[segment_start:]
                    seg = np.column_stack([
                        time_valid[segment_start:],
                        e2e_x,
                        e2e_y,
                    ])
                    segments.append(seg)

    if verbose:
        print(f"  Extracted {len(segments)} short-dynamics e2e segments (aligned)")
        print(f"  Aligned trajectories: {n_aligned_A} original + {n_aligned_B} mirror")

        if segments:
            lengths = [len(s) for s in segments]
            durations = [len(s) * DT_SHORT for s in segments]
            print(f"  Segment lengths: min={min(lengths)}, median={np.median(lengths):.0f}, max={max(lengths)} points")
            print(f"  Segment durations: min={min(durations):.2f}s, max={max(durations):.2f}s")

    return segments


def compute_e2e_autocorr_segment_short(segment):
    time = segment[:, 0]
    e2e_x = segment[:, 1]
    e2e_y = segment[:, 2]

    norms = np.sqrt(e2e_x**2 + e2e_y**2)
    valid = norms > 0
    if np.sum(valid) < 10:
        return None, None, None

    ux = np.zeros_like(e2e_x)
    uy = np.zeros_like(e2e_y)
    ux[valid] = e2e_x[valid] / norms[valid]
    uy[valid] = e2e_y[valid] / norms[valid]

    mean_norm_sq = np.mean(norms[valid]**2)

    n = len(time)
    max_lag = n // 2

    autocorr_full = np.zeros(max_lag)
    autocorr_unit = np.zeros(max_lag)

    for lag in range(max_lag):
        valid_pairs = valid[:n-lag] & valid[lag:]
        if np.sum(valid_pairs) > 0:
            dot_unit = ux[:n-lag] * ux[lag:] + uy[:n-lag] * uy[lag:]
            autocorr_unit[lag] = np.mean(dot_unit[valid_pairs])
            dot_full = e2e_x[:n-lag] * e2e_x[lag:] + e2e_y[:n-lag] * e2e_y[lag:]
            autocorr_full[lag] = np.mean(dot_full[valid_pairs])

    if autocorr_unit[0] > 0:
        autocorr_unit = autocorr_unit / autocorr_unit[0]
    if mean_norm_sq > 0:
        autocorr_full = autocorr_full / mean_norm_sq

    time_lags = np.arange(max_lag) * DT_SHORT_MIN

    return time_lags, autocorr_full, autocorr_unit


def calculate_e2e_autocorr_short(trajectories, verbose=True):
    if verbose:
        print("\nComputing short dynamics e2e vector autocorrelation (both observables)...")

    segments = extract_e2e_aligned_segments_short(trajectories, verbose=verbose)

    if len(segments) == 0:
        print("  No valid short dynamics segments found")
        return {'full': (None, None), 'unit': (None, None)}

    all_full = []
    all_unit = []
    all_times = []

    for seg in segments:
        time_lags, ac_full, ac_unit = compute_e2e_autocorr_segment_short(seg)
        if time_lags is not None and len(time_lags) > 5:
            all_full.append(ac_full)
            all_unit.append(ac_unit)
            all_times.append(time_lags)

    if len(all_full) == 0:
        print("  No valid short dynamics autocorrelations computed")
        return {'full': (None, None), 'unit': (None, None)}

    if verbose:
        print(f"  Computed autocorrelation for {len(all_full)} short segments")
        max_time_s = all_times[0][-1] * 60
        print(f"  Short dynamics autocorr computed up to τ = {max_time_s:.2f} s")

    result_full, tau_full = _aggregate_autocorr(all_full, all_times, "Full vector (short)", verbose)
    result_unit, tau_unit = _aggregate_autocorr(all_unit, all_times, "Unit vector (short)", verbose)

    return {'full': (result_full, tau_full), 'unit': (result_unit, tau_unit)}


def merge_autocorrelations(short_autocorr, long_autocorr, crossover_sec=CROSSOVER_TIME_SEC, verbose=True):
    crossover_min = crossover_sec / 60.0

    t_short = short_autocorr[:, 0]
    c_short = short_autocorr[:, 1]
    e_short = short_autocorr[:, 2]

    t_long = long_autocorr[:, 0]
    c_long = long_autocorr[:, 1]
    e_long = long_autocorr[:, 2]

    if t_short[-1] * 60 < crossover_sec:
        if verbose:
            print(f"  Warning: Short dynamics don't reach crossover ({t_short[-1]*60:.2f}s < {crossover_sec}s)")
            print(f"  Using last short dynamics point as crossover")
        c_short_at_crossover = c_short[-1]
        crossover_min = t_short[-1]
    else:
        c_short_at_crossover = np.interp(crossover_min, t_short, c_short)

    c_long_at_crossover = np.interp(crossover_min, t_long, c_long)

    if c_long_at_crossover > 0:
        scale_factor = c_short_at_crossover / c_long_at_crossover
    else:
        scale_factor = 1.0

    if verbose:
        print(f"  Crossover at τ = {crossover_sec:.1f} s ({crossover_min:.4f} min)")
        print(f"  C_short(crossover) = {c_short_at_crossover:.4f}")
        print(f"  C_long(crossover) = {c_long_at_crossover:.4f}")
        print(f"  Scale factor = {scale_factor:.4f}")

    mask_short = t_short < crossover_min
    t_merged = list(t_short[mask_short])
    c_merged = list(c_short[mask_short])
    e_merged = list(e_short[mask_short])

    mask_long = t_long >= crossover_min
    t_merged.extend(t_long[mask_long])
    c_merged.extend(c_long[mask_long] * scale_factor)
    e_merged.extend(e_long[mask_long] * scale_factor)

    merged_autocorr = np.column_stack([
        np.array(t_merged),
        np.array(c_merged),
        np.array(e_merged)
    ])

    if verbose:
        print(f"  Merged curve: {len(merged_autocorr)} points, τ = {merged_autocorr[0,0]*60:.3f}s to {merged_autocorr[-1,0]*60:.1f}s")

    return merged_autocorr, scale_factor


def fit_rotational_diffusion_exp(msd_rot, t_start_sec=2.0, t_end_sec=20.0):
    if len(msd_rot) == 0:
        print("  Cannot fit empty MSD")
        return None, None, None, None

    time_min = msd_rot[:, 0]
    time_sec = time_min * 60
    msd = msd_rot[:, 1]

    mask = (time_sec >= t_start_sec) & (time_sec <= t_end_sec) & (msd > 0)

    if np.sum(mask) < 3:
        print(f"  Warning: Not enough points in range [{t_start_sec}, {t_end_sec}] s ({np.sum(mask)} points)")
        return None, None, None, None

    t_fit = time_min[mask]
    msd_fit = msd[mask]

    print(f"  Fitting MSD = 2*D_r*t (slope=1): t = {t_fit[0]*60:.1f} to {t_fit[-1]*60:.1f} s ({len(t_fit)} points)")

    log_t = np.log(t_fit)
    log_msd = np.log(msd_fit)

    log_2Dr = np.mean(log_msd - log_t)
    D_r = np.exp(log_2Dr) / 2

    residuals = log_msd - (log_2Dr + log_t)
    std_residuals = np.std(residuals)
    D_r_error = D_r * std_residuals / np.sqrt(len(t_fit))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_msd - np.mean(log_msd))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    tau_rot = 1 / D_r if D_r > 0 else np.inf

    D_r_per_sec = D_r / 60
    tau_rot_sec = tau_rot * 60
    print(f"  D_r = {D_r_per_sec:.4f} rad²/s (R² = {r_squared:.4f})")
    print(f"  tau_rot = 1/D_r = {tau_rot_sec:.1f} s")

    fit_params = {
        't_start_sec': t_start_sec,
        't_end_sec': t_end_sec,
        'slope': 1.0,
        'log_2Dr': log_2Dr,
        'r_squared': r_squared,
    }

    return D_r, D_r_error, tau_rot, fit_params


def find_saturation_times_exp(msd_trans, R_sq_threshold=None):
    if R_sq_threshold is None:
        R_sq_threshold = EXP_R_SQUARED_THRESHOLD

    if len(msd_trans) == 0:
        print("  Cannot find saturation time for empty MSD")
        return {'tau_trans': None, 'R_sq_threshold': R_sq_threshold}

    time = msd_trans[:, 0]
    msd = msd_trans[:, 1]

    print(f"  MSD trans range: {msd.min():.1f} to {msd.max():.1f} mm²")

    tau_trans = None
    for i in range(1, len(msd)):
        if msd[i-1] < R_sq_threshold <= msd[i]:
            t0, t1 = time[i-1], time[i]
            m0, m1 = msd[i-1], msd[i]
            tau_trans = t0 + (R_sq_threshold - m0) * (t1 - t0) / (m1 - m0)
            break

    if tau_trans is None and msd[0] >= R_sq_threshold:
        tau_trans = time[0]

    tau_s = f"{tau_trans*60:.2f}" if tau_trans else "N/A"
    print(f"  Saturation: R² = {R_sq_threshold:.0f} mm² → τ_trans = {tau_s} s (interpolated)")

    return {
        'tau_trans': tau_trans,
        'R_sq_threshold': R_sq_threshold,
    }


def find_saturation_time_plateau_exp(msd_trans, tail_cutoff=0.2, slope_threshold=0.3,
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
        print(f"  No clear plateau, using last 30% as fallback")

    if len(plateau_values) == 0:
        print("  Empty plateau region")
        return None, None, None

    plateau_mean = float(np.mean(plateau_values))
    plateau_std = float(np.std(plateau_values))
    print(f"  Plateau mean = {plateau_mean:.2f} mm², std = {plateau_std:.2f} mm²")

    for i in range(1, len(msd_valid)):
        if msd_valid[i - 1] < plateau_mean <= msd_valid[i]:
            frac = (plateau_mean - msd_valid[i - 1]) / (msd_valid[i] - msd_valid[i - 1])
            tau_sat = time_valid[i - 1] + frac * (time_valid[i] - time_valid[i - 1])
            print(f"  τ_sat = {tau_sat*60:.2f} s (first crossing of plateau mean)")
            return tau_sat, plateau_mean, plateau_std

    if msd_valid[0] >= plateau_mean:
        tau_sat = time_valid[0]
        print(f"  τ_sat = {tau_sat*60:.2f} s (MSD starts above plateau mean)")
        return tau_sat, plateau_mean, plateau_std

    print("  MSD never reaches plateau mean")
    return None, plateau_mean, plateau_std


def find_first_maximum_exp(msd_trans, smooth_window=5):
    if len(msd_trans) == 0:
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
            print(f"  First maximum at tau = {tau_max:.4f} min, MSD = {msd_max:.2f} mm²")
            return tau_max, msd_max

    idx = np.argmax(msd)
    tau_max = time[idx]
    msd_max = msd[idx]
    print(f"  No zero-crossing, using global max: tau = {tau_max:.4f} min, MSD = {msd_max:.2f} mm²")
    return tau_max, msd_max


def process_temperature(trajectories, temperature, plot=False, save_data=True, show=True):
    print(f"\n{'='*60}")
    print(f"Processing T = {temperature}°C ({len(trajectories)} worms)")
    print(f"{'='*60}")

    tracking_points = ['end1', 'end2']
    msd_data_by_tp = {}

    for tp in tracking_points:
        print(f"\n  --- Tracking point: {tp} (aligned methodology) ---")
        msd_rot, _ = calculate_rotational_msd_aligned(trajectories, tracking_point=tp)
        msd_trans, _ = calculate_translational_msd_aligned(trajectories, tracking_point=tp)

        D_r, D_r_error, tau_rot, fit_params = fit_rotational_diffusion_exp(msd_rot)

        if D_r is not None and fit_params is not None:
            t_start = fit_params['t_start_sec']
            t_end = fit_params['t_end_sec']
            time_sec = msd_rot[:, 0] * 60
            msd_r = msd_rot[:, 1]
            mask = (time_sec >= t_start) & (time_sec <= t_end) & (msd_r > 0)
            if np.sum(mask) >= 3:
                log_t = np.log10(time_sec[mask])
                log_msd = np.log10(msd_r[mask])
                alpha_rot = np.polyfit(log_t, log_msd, 1)[0]
                print(f"  Power law exponent α = {alpha_rot:.3f}")
                if not (0.8 <= alpha_rot <= 1.3):
                    print(f"  No diffusive regime (α = {alpha_rot:.2f}, not in [0.8, 1.3]) → D_r = None")
                    D_r, D_r_error, tau_rot = None, None, None

        print("\n  Finding saturation time from plateau mean:")
        tau_trans, plateau_mean, plateau_std = find_saturation_time_plateau_exp(msd_trans)

        N_rot = None
        if tau_trans is not None and tau_rot is not None and tau_trans > 0:
            N_rot = tau_rot / tau_trans

        if tau_rot:
            tau_rot_s = tau_rot * 60
            print(f"\n  N_rot (plateau-based):")
            if tau_trans is not None:
                tau_trans_s = tau_trans * 60
                print(f"    τ_rot = {tau_rot_s:.1f} s, τ_trans = {tau_trans_s:.1f} s")
                print(f"    N_rot = {tau_rot_s:.1f} / {tau_trans_s:.1f} = {N_rot:.2f}")
            else:
                print(f"    τ_trans = N/A (no plateau found)")

        msd_data_by_tp[tp] = {
            'msd_rot': msd_rot,
            'msd_trans': msd_trans,
            'results': {
                'temperature': temperature,
                'n_worms': len(trajectories),
                'D_r': D_r,
                'D_r_error': D_r_error,
                'tau_rot': tau_rot,
                'fit_params': fit_params,
                'tau_trans': tau_trans,
                'plateau_mean': plateau_mean,
                'plateau_std': plateau_std,
                'N_rot': N_rot,
            }
        }

    results = msd_data_by_tp['end1']['results']
    msd_rot_primary = msd_data_by_tp['end1']['msd_rot']
    msd_trans_primary = msd_data_by_tp['end1']['msd_trans']

    if save_data:
        output_dir = "DATA_EXP"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"EXP_T{int(temperature)}.npz")

        np.savez(
            output_file,
            msd_rot=msd_rot_primary,
            msd_trans=msd_trans_primary,
            msd_rot_end1=msd_data_by_tp['end1']['msd_rot'],
            msd_trans_end1=msd_data_by_tp['end1']['msd_trans'],
            msd_rot_end2=msd_data_by_tp['end2']['msd_rot'],
            msd_trans_end2=msd_data_by_tp['end2']['msd_trans'],
            **results
        )
        print(f"  Saved: {output_file}")

    if plot:
        plot_msd_analysis_exp(msd_rot_primary, msd_trans_primary, results, temperature, show=show)

    return results, msd_data_by_tp


def plot_msd_analysis_exp(msd_rot, msd_trans, results, temperature, show=True):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tau_rot = results.get('tau_rot')
    tau_rot_s = tau_rot * 60 if tau_rot else None
    tau_trans = results.get('tau_trans')
    tau_trans_s = tau_trans * 60 if tau_trans else None
    N_rot = results.get('N_rot')
    plateau_mean = results.get('plateau_mean')
    plateau_std = results.get('plateau_std')

    ax = axes[0]
    if len(msd_trans) > 0:
        t_trans = msd_trans[:, 0]
        msd_t = msd_trans[:, 1]
        valid = (t_trans > 0) & (msd_t > 0)
        t_sec = t_trans[valid] * 60
        msd_v = msd_t[valid]

        ax.loglog(t_sec, msd_v, 'k-', linewidth=2.5, zorder=10)

        if plateau_mean is not None and plateau_std is not None:
            ax.axhspan(plateau_mean - plateau_std, plateau_mean + plateau_std,
                       alpha=0.2, color='red', zorder=1)
            ax.axhline(plateau_mean, color='red', linestyle='--', linewidth=2, alpha=0.8,
                       label=f'plateau = {plateau_mean:.1f} mm²')

        if tau_rot_s:
            ax.axvline(tau_rot_s, color='blue', linestyle='-', linewidth=3, alpha=0.7,
                       label=f'τ_rot = {tau_rot_s:.1f} s')

        if tau_trans_s:
            ax.axvline(tau_trans_s, color='red', linestyle='-', linewidth=3, alpha=0.7,
                       label=f'τ_sat = {tau_trans_s:.1f} s')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('MSD_trans (mm²)', fontsize=12)
        ax.set_title(f'Translational MSD', fontsize=13)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

    ax = axes[1]
    if len(msd_rot) > 0:
        t_rot = msd_rot[:, 0]
        msd_r = msd_rot[:, 1]
        valid = (t_rot > 0) & (msd_r > 0)
        t_sec = t_rot[valid] * 60
        msd_r_v = msd_r[valid]

        ax.loglog(t_sec, msd_r_v, 'b-', linewidth=2.5, label='MSD_rot', zorder=10)

        fit_params = results.get('fit_params')
        D_r = results.get('D_r')
        if fit_params and D_r:
            t_fit_line = np.logspace(np.log10(t_sec.min()), np.log10(t_sec.max()), 100)
            fit_line = 2 * D_r * (t_fit_line / 60)
            ax.loglog(t_fit_line, fit_line, 'k--', linewidth=2, alpha=0.7, label='Fit: MSD = 2D_r·t')

            t_start = fit_params['t_start_sec']
            t_end = fit_params['t_end_sec']
            ax.axvspan(t_start, t_end, alpha=0.15, color='gray', label=f'Fit range: {t_start:.0f}-{t_end:.0f} s')

        if tau_rot_s:
            ax.axvline(tau_rot_s, color='green', linestyle='-', linewidth=3, alpha=0.8)
            msd_at_tau = np.interp(tau_rot_s, t_sec, msd_r_v)
            ax.plot(tau_rot_s, msd_at_tau, 'go', markersize=12, zorder=15)
            ax.text(tau_rot_s * 1.2, msd_at_tau, f'τ_rot = {tau_rot_s:.1f} s',
                    fontsize=11, color='green', fontweight='bold', va='center')

        if D_r:
            D_r_s = D_r / 60
            r_sq = fit_params.get('r_squared', 0) if fit_params else 0
            ax.text(0.02, 0.98, f'D_r = {D_r_s:.4f} rad²/s\nτ_rot = 1/D_r = {tau_rot_s:.1f} s\nR² = {r_sq:.4f}',
                    transform=ax.transAxes, fontsize=11, fontweight='bold',
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('MSD_rot (rad²)', fontsize=12)
        ax.set_title(f'Rotational MSD', fontsize=13)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)

    if N_rot is not None and tau_rot_s and tau_trans_s:
        title = f"T = {temperature}°C  |  τ_rot = {tau_rot_s:.1f} s  |  τ_sat = {tau_trans_s:.1f} s  |  N_rot = {N_rot:.2f}"
    elif tau_trans_s:
        title = f"T = {temperature}°C  |  τ_sat = {tau_trans_s:.1f} s  |  N_rot = N/A (no diffusive regime)"
    else:
        title = f"T = {temperature}°C"
    fig.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    fig_dir = "FIGURES/ROTATIONAL"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"msd_analysis_EXP_T{int(temperature)}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_msd_all_temperatures(results_by_temp, tracking_point='com', show=True):
    import matplotlib.pyplot as plt

    SMOOTH_WINDOW = 15

    temp_colors = {
        10: '#1f77b4',
        20: '#ff7f0e',
        30: '#d62728',
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex=True)

    temps = sorted(results_by_temp.keys())

    ax = axes[0, 0]
    for temp in temps:
        data = results_by_temp[temp]
        msd_rot = data['msd_rot']
        results = data['results']
        if len(msd_rot) == 0:
            continue
        t = msd_rot[:, 0]
        msd = msd_rot[:, 1]
        std_err = msd_rot[:, 2] if msd_rot.shape[1] > 2 else None
        valid = (t > 0) & (msd > 0)
        color = temp_colors.get(temp, 'gray')
        t_sec = t[valid] * 60

        if std_err is not None:
            ax.fill_between(t_sec, msd[valid] - std_err[valid], msd[valid] + std_err[valid],
                            alpha=0.2, color=color, linewidth=0)

        ax.loglog(t_sec, msd[valid], '-', color=color, linewidth=2.5, label=f'T = {temp}°C')

        fit_params = results.get('fit_params')
        if fit_params:
            log_2Dr = fit_params['log_2Dr']
            t_min = t_sec / 60
            fit_line = np.exp(log_2Dr) * t_min
            ax.loglog(t_sec, fit_line, '--', color=color, linewidth=1.5, alpha=0.7)

    ax.set_ylabel(r'MSD$_{\rm rot}$ (rad²)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    annotations = []
    for temp in temps:
        results = results_by_temp[temp]['results']
        tau_rot = results.get('tau_rot')
        fit_params = results.get('fit_params')
        r_sq = fit_params.get('r_squared', 0) if fit_params else 0
        if tau_rot:
            tau_rot_s = tau_rot * 60
            annotations.append(f"T={temp}°C: τ_rot = {tau_rot_s:.1f} s (R²={r_sq:.3f})")
    if annotations:
        textstr = '\n'.join(annotations)
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax = axes[0, 1]
    for temp in temps:
        data = results_by_temp[temp]
        msd_trans = data['msd_trans']
        results = data['results']
        if len(msd_trans) == 0:
            continue
        t = msd_trans[:, 0]
        msd = msd_trans[:, 1]
        std_err = msd_trans[:, 2] if msd_trans.shape[1] > 2 else None
        valid = (t > 0) & (msd > 0)
        color = temp_colors.get(temp, 'gray')
        t_sec = t[valid] * 60

        if std_err is not None:
            ax.fill_between(t_sec, msd[valid] - std_err[valid], msd[valid] + std_err[valid],
                            alpha=0.2, color=color, linewidth=0)

        ax.loglog(t_sec, msd[valid], '-', color=color, linewidth=2.5, label=f'T = {temp}°C')

        plateau_mean = results.get('plateau_mean')
        plateau_std = results.get('plateau_std')
        if plateau_mean is not None and plateau_std is not None:
            ax.axhspan(plateau_mean - plateau_std, plateau_mean + plateau_std,
                       alpha=0.1, color=color, zorder=0)
            ax.axhline(plateau_mean, color=color, linestyle=':', linewidth=1.5, alpha=0.5)

        tau_trans = results.get('tau_trans')
        if tau_trans:
            tau_trans_s = tau_trans * 60
            linestyle = '-'
            if temp == 20:
                linestyle = '--'
            ax.axvline(tau_trans_s, color=color, linestyle=linestyle, linewidth=2.5, alpha=0.8)

    ax.set_ylabel(r'MSD$_{\rm trans}$ (mm²)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    annotations = []
    for temp in temps:
        results = results_by_temp[temp]['results']
        tau_trans = results.get('tau_trans')
        N_rot = results.get('N_rot')
        if tau_trans:
            tau_trans_s = tau_trans * 60
            N_rot_str = f"{N_rot:.2f}" if N_rot is not None else "N/A"
            annotations.append(f"T={temp}°C: τ_sat = {tau_trans_s:.1f} s, N_rot = {N_rot_str}")
    if annotations:
        textstr = '\n'.join(annotations)
        ax.text(0.03, 0.97, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax = axes[1, 0]
    ax.axhline(1, color='k', linestyle='--', alpha=0.5, linewidth=1, label='slope = 1')
    for temp in temps:
        data = results_by_temp[temp]
        msd_rot = data['msd_rot']
        if len(msd_rot) == 0:
            continue
        t = msd_rot[:, 0]
        msd = msd_rot[:, 1]
        valid = (t > 0) & (msd > 0)
        color = temp_colors.get(temp, 'gray')
        t_sec = t[valid] * 60

        log_t = np.log10(t[valid])
        log_msd = np.log10(msd[valid])
        slope = np.gradient(log_msd, log_t)
        slope_smooth = uniform_filter1d(slope, size=SMOOTH_WINDOW, mode='nearest')

        ax.semilogx(t_sec, slope_smooth, '-', color=color, linewidth=2.5, label=f'T = {temp}°C')

    ax.set_ylabel(r'd(log MSD$_{\rm rot}$)/d(log t)')
    ax.set_ylim(-0.5, 2.0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.axhline(1, color='k', linestyle='--', alpha=0.5, linewidth=1, label='slope = 1')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='slope = 0')
    for temp in temps:
        data = results_by_temp[temp]
        msd_trans = data['msd_trans']
        if len(msd_trans) == 0:
            continue
        t = msd_trans[:, 0]
        msd = msd_trans[:, 1]
        valid = (t > 0) & (msd > 0)
        color = temp_colors.get(temp, 'gray')
        t_sec = t[valid] * 60

        log_t = np.log10(t[valid])
        log_msd = np.log10(msd[valid])
        slope = np.gradient(log_msd, log_t)
        slope_smooth = uniform_filter1d(slope, size=SMOOTH_WINDOW, mode='nearest')

        ax.semilogx(t_sec, slope_smooth, '-', color=color, linewidth=2.5, label=f'T = {temp}°C')

    ax.set_ylabel(r'd(log MSD$_{\rm trans}$)/d(log t)')
    ax.set_ylim(-0.5, 2.0)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    for temp in temps:
        data = results_by_temp[temp]
        msd_rot = data['msd_rot']
        results = data['results']
        if len(msd_rot) == 0:
            continue
        t = msd_rot[:, 0]
        msd = msd_rot[:, 1]
        valid = (t > 0) & (msd > 0)
        color = temp_colors.get(temp, 'gray')
        t_sec = t[valid] * 60
        ax.loglog(t_sec, msd[valid] / t_sec, '-', color=color, linewidth=2.5, label=f'T = {temp}°C')

        D_r = results.get('D_r')
        if D_r:
            ax.axhline(2 * D_r / 60, color=color, linestyle=':', alpha=0.6, linewidth=1.5)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'MSD$_{\rm rot}$ / t (rad²/s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for temp in temps:
        data = results_by_temp[temp]
        msd_trans = data['msd_trans']
        if len(msd_trans) == 0:
            continue
        t = msd_trans[:, 0]
        msd = msd_trans[:, 1]
        valid = (t > 0) & (msd > 0)
        color = temp_colors.get(temp, 'gray')
        t_sec = t[valid] * 60
        ax.loglog(t_sec, msd[valid] / t_sec, '-', color=color, linewidth=2.5, label=f'T = {temp}°C')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'MSD$_{\rm trans}$ / t (mm²/s)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    tp_label = {'com': 'Center of Mass', 'end1': 'End 1', 'end2': 'End 2'}.get(tracking_point, tracking_point)
    n_rot_str = "  |  ".join([
        f"T={temp}°C: N_rot={results_by_temp[temp]['results'].get('N_rot', 0):.3f}"
        for temp in temps
        if results_by_temp[temp]['results'].get('N_rot') is not None
    ])
    fig.suptitle(f"Experimental MSD Analysis ({tp_label})\n{n_rot_str}", fontsize=12, fontweight='bold')

    plt.tight_layout()

    fig_dir = "FIGURES/ROTATIONAL"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"msd_analysis_EXP_all_{tracking_point}.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_summary_figure(results_by_temp, tracking_point='end1', show=True):
    import matplotlib.pyplot as plt

    temp_colors = {
        10: '#1f77b4',
        20: '#7f7f7f',
        30: '#d62728',
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    temps = sorted(results_by_temp.keys())

    ax = axes[0, 0]
    for temp in temps:
        data = results_by_temp[temp]
        msd_rot = data['msd_rot']
        results = data['results']
        if len(msd_rot) == 0:
            continue
        t = msd_rot[:, 0]
        msd = msd_rot[:, 1]
        valid = (t > 0) & (msd > 0)
        color = temp_colors.get(temp, 'gray')
        t_sec = t[valid] * 60

        D_r = results.get('D_r')
        if D_r is not None:
            D_r_s = D_r / 60
            label = f'T = {temp}°C, $D_r$ = {D_r_s:.2f} rad²/s'
        else:
            label = f'T = {temp}°C (no diffusive regime)'

        ax.loglog(t_sec, msd[valid], '-', color=color, linewidth=2.5, label=label)

        fit_params = results.get('fit_params')
        if fit_params and D_r is not None:
            log_2Dr = fit_params['log_2Dr']
            t_min = t_sec / 60
            fit_line = np.exp(log_2Dr) * t_min
            ax.loglog(t_sec, fit_line, '--', color=color, linewidth=1.5, alpha=0.7)

    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$\langle(\theta(t) - \theta_0)^2\rangle$ [rad²]')
    ax.legend(loc='lower right', fontsize=9)
    ax.text(0.02, 0.98, '(a)', transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for temp in temps:
        data = results_by_temp[temp]
        msd_trans = data['msd_trans']
        results = data['results']
        if len(msd_trans) == 0:
            continue
        t = msd_trans[:, 0]
        msd = msd_trans[:, 1]
        valid = (t > 0) & (msd > 0)
        color = temp_colors.get(temp, 'gray')
        t_sec = t[valid] * 60

        ax.loglog(t_sec, msd[valid], '-', color=color, linewidth=2.5, label=f'T = {temp}°C')

        plateau_mean = results.get('plateau_mean')
        plateau_std = results.get('plateau_std')
        if plateau_mean is not None and plateau_std is not None:
            ax.axhspan(plateau_mean - plateau_std, plateau_mean + plateau_std,
                       alpha=0.1, color=color, zorder=0)
            ax.axhline(plateau_mean, color=color, linestyle=':', linewidth=1.5, alpha=0.5)

        tau_trans = results.get('tau_trans')
        if tau_trans:
            tau_trans_s = tau_trans * 60
            ax.axvline(tau_trans_s, color=color, linestyle='-', linewidth=2, alpha=0.7)

    ax.set_xlabel(r'$t$ [s]')
    ax.set_ylabel(r'$\langle(r(t) - r_0)^2\rangle$ [mm²]')
    ax.legend(loc='lower right', fontsize=9)
    ax.text(0.02, 0.98, '(b)', transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]

    T_arr = np.array(temps)
    tau_rot_arr = np.array([(results_by_temp[t]['results'].get('tau_rot') or np.nan) * 60 for t in temps])
    tau_trans_arr = np.array([(results_by_temp[t]['results'].get('tau_trans') or np.nan) * 60 for t in temps])

    rate_rot_arr = 1.0 / tau_rot_arr
    rate_trans_arr = 1.0 / tau_trans_arr

    valid_rot = ~np.isnan(rate_rot_arr)
    valid_trans = ~np.isnan(rate_trans_arr)
    if np.any(valid_rot):
        ax.plot(T_arr[valid_rot], rate_rot_arr[valid_rot], 's-', color='#2ca02c', markersize=12, linewidth=2,
                label=r'$1/\tau_{\rm rot} = D_r$')
    if np.any(valid_trans):
        ax.plot(T_arr[valid_trans], rate_trans_arr[valid_trans], 'o-', color='#1f77b4', markersize=12, linewidth=2,
                label=r'$1/\tau_{\rm sat}$')

    ax.set_xlabel(r'$T$ [°C]')
    ax.set_ylabel(r'$1/\tau$ [s$^{-1}$]')
    ax.legend(loc='upper left', fontsize=10)
    ax.text(0.02, 0.98, '(c)', transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 35)

    ax = axes[1, 1]

    N_rot_arr = np.array([(results_by_temp[t]['results'].get('N_rot') or np.nan) for t in temps])

    valid_nrot = ~np.isnan(N_rot_arr)
    if np.any(valid_nrot):
        ax.plot(T_arr[valid_nrot], N_rot_arr[valid_nrot], 's', color='#2ca02c', markersize=14, linewidth=0)
    else:
        ax.text(0.5, 0.5, 'No diffusive regime\nN_rot = N/A', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')

    ax.set_xlabel(r'$T$ [°C]')
    ax.set_ylabel(r'$N_{\rm rot} = \tau_{\rm rot} / \tau_{\rm trans}$')
    ax.text(0.02, 0.98, '(d)', transform=ax.transAxes, fontsize=14, fontweight='bold',
            va='top', ha='left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(5, 35)

    plt.tight_layout()

    fig_dir = "FIGURES"
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"summary_N_rot_{tracking_point}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    fig_path_pdf = os.path.join(fig_dir, f"summary_N_rot_{tracking_point}.pdf")
    plt.savefig(fig_path_pdf, bbox_inches='tight')
    print(f"  Saved: {fig_path_pdf}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute N_rot for experimental worm data'
    )
    parser.add_argument('--all', action='store_true',
                        help='Process all temperatures')
    parser.add_argument('--temperature', '-t', type=float,
                        help='Process specific temperature (e.g., 20)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    parser.add_argument('--include-short', action='store_true',
                        help='Include short_dynamics trajectories')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save data to npz files')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively')
    parser.add_argument('--debug-segments', action='store_true',
                        help='Generate debug figures for all segments')
    parser.add_argument('--debug-full', action='store_true',
                        help='Generate full trajectory debug plots (no segmentation)')
    parser.add_argument('--all-points', action='store_true',
                        help='Generate 3-panel figures with all points (end1, end2, com)')
    parser.add_argument('--align-points', action='store_true',
                        help='Generate before/after alignment figures (end1, end2, com)')
    parser.add_argument('--summary', action='store_true',
                        help='Generate 4-panel summary figure for publication')
    parser.add_argument('--events', action='store_true',
                        help='Compute trapping/translocation events (detection: end1)')
    parser.add_argument('--events-com', action='store_true',
                        help='Compute trapping/translocation events (detection: CoM, alignment: end1)')
    parser.add_argument('--e2e-autocorr', action='store_true',
                        help='Compute e2e vector autocorrelation (aligned methodology)')
    parser.add_argument('--workers', type=int, default=24,
                        help='Number of parallel workers for --debug-* options (default: 24)')

    args = parser.parse_args()

    print("Loading experimental trajectories...")
    trajs_by_temp = load_trajectories_by_temperature(include_short=args.include_short)

    if not trajs_by_temp:
        print("No trajectories found!")
        return

    if args.debug_segments:
        from .debug_plots import generate_all_debug_figures
        generate_all_debug_figures(trajs_by_temp, n_workers=args.workers)
        return

    if args.debug_full:
        from .debug_plots import generate_full_trajectory_plots
        generate_full_trajectory_plots(trajs_by_temp, tracking_point='end1', n_workers=args.workers)
        return

    if args.all_points:
        from .debug_plots import plot_all_points_by_temperature
        for tp in ['end1', 'end2', 'com']:
            output_path = f"FIGURES/DEBUG_FULL/all_points_{tp}.png"
            plot_all_points_by_temperature(trajs_by_temp, tracking_point=tp,
                                           save_path=output_path)
        return

    if args.align_points:
        from .debug_plots import plot_aligned_points_by_temperature
        for tp in ['end1', 'end2']:
            output_path = f"FIGURES/DEBUG_FULL/aligned_points_{tp}.png"
            plot_aligned_points_by_temperature(trajs_by_temp, tracking_point=tp,
                                               save_path=output_path)
        return

    if args.events or args.events_com:
        from .events import process_all_temperatures, save_events, print_summary
        detection_point = 'com' if args.events_com else 'end1'
        print(f"\nComputing trapping/translocation events (alignment: end1, detection: {detection_point})...")
        results = process_all_temperatures(trajs_by_temp, detection_point=detection_point)
        save_events(results, detection_point=detection_point)
        print_summary(results, detection_point=detection_point)
        return

    if args.e2e_autocorr:
        os.makedirs("DATA_E2E_AUTOCORR_EXP", exist_ok=True)

        print("\n" + "="*60)
        print("E2E VECTOR AUTOCORRELATION (aligned methodology)")
        print("with SHORT DYNAMICS (25 fps) + LONG DYNAMICS (0.5 fps)")
        print("="*60)

        print("\nLoading short dynamics data...")
        short_trajs_by_temp = load_short_dynamics_by_temperature()

        all_tau_decorr = {}
        all_autocorr_data = {}
        all_autocorr_short = {}
        all_autocorr_merged = {}

        for temp, trajs in trajs_by_temp.items():
            print(f"\n{'='*50}")
            print(f"Temperature: {temp}°C")
            print(f"{'='*50}")

            print(f"\n[LONG DYNAMICS] {len(trajs)} trajectories (dt=2s)")
            long_results = calculate_e2e_autocorr_aligned(trajs, verbose=True)

            short_trajs = short_trajs_by_temp.get(temp, [])
            short_results = None

            if short_trajs:
                print(f"\n[SHORT DYNAMICS] {len(short_trajs)} trajectories (dt=0.04s)")
                short_results = calculate_e2e_autocorr_short(short_trajs, verbose=True)
            else:
                print(f"\n[SHORT DYNAMICS] No data available for T={temp}°C")

            import pandas as pd

            for obs_type in ('full', 'unit'):
                obs_suffix = "" if obs_type == "unit" else "_full_vec"

                long_autocorr, tau_decorr_long = long_results[obs_type]
                if long_autocorr is not None:
                    if obs_type == 'unit':
                        all_tau_decorr[temp] = tau_decorr_long
                    all_autocorr_data.setdefault(temp, {})[obs_type] = long_autocorr

                    df = pd.DataFrame({
                        'time_min': long_autocorr[:, 0],
                        'autocorr_mean': long_autocorr[:, 1],
                        'std_error': long_autocorr[:, 2]
                    })
                    csv_path = f"DATA_E2E_AUTOCORR_EXP/e2e_autocorr{obs_suffix}_T{int(temp)}C.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"  Saved: {csv_path}")

                if short_results is not None:
                    short_autocorr, _ = short_results[obs_type]
                    if short_autocorr is not None:
                        all_autocorr_short.setdefault(temp, {})[obs_type] = short_autocorr

                        df = pd.DataFrame({
                            'time_min': short_autocorr[:, 0],
                            'autocorr_mean': short_autocorr[:, 1],
                            'std_error': short_autocorr[:, 2]
                        })
                        csv_path = f"DATA_E2E_AUTOCORR_EXP/e2e_autocorr_short{obs_suffix}_T{int(temp)}C.csv"
                        df.to_csv(csv_path, index=False)
                        print(f"  Saved: {csv_path}")

                        if long_autocorr is not None:
                            print(f"\n[MERGING {obs_type.upper()}] Short + Long at crossover = {CROSSOVER_TIME_SEC}s")
                            merged_autocorr, _ = merge_autocorrelations(
                                short_autocorr, long_autocorr, verbose=True)
                            all_autocorr_merged.setdefault(temp, {})[obs_type] = merged_autocorr

                            df = pd.DataFrame({
                                'time_min': merged_autocorr[:, 0],
                                'autocorr_mean': merged_autocorr[:, 1],
                                'std_error': merged_autocorr[:, 2]
                            })
                            csv_path = f"DATA_E2E_AUTOCORR_EXP/e2e_autocorr_merged{obs_suffix}_T{int(temp)}C.csv"
                            df.to_csv(csv_path, index=False)
                            print(f"  Saved: {csv_path}")

        print("\n" + "="*60)
        print("SUMMARY: τ_decorr (e2e vector autocorrelation)")
        print("="*60)
        for temp in sorted(all_tau_decorr.keys()):
            tau = all_tau_decorr[temp]
            print(f"  T = {temp:2.0f}°C: τ_decorr = {tau:.4f} min = {tau*60:.2f} s")

        import pandas as pd
        summary_df = pd.DataFrame([
            {'temperature': temp, 'tau_decorr_min': tau, 'tau_decorr_s': tau*60}
            for temp, tau in sorted(all_tau_decorr.items())
        ])
        summary_path = "DATA_E2E_AUTOCORR_EXP/tau_decorr_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")

        import matplotlib.pyplot as plt

        temp_colors = {10: 'blue', 20: 'green', 30: 'red'}

        obs_styles = {'full': '-', 'unit': ':'}
        obs_labels = {'full': 'full vec', 'unit': 'unit vec'}

        fig, ax = plt.subplots(figsize=(8, 6))

        for temp in sorted(all_autocorr_data.keys()):
            color = temp_colors.get(temp, 'black')
            tau = all_tau_decorr.get(temp, np.nan)
            for obs_type in ('full', 'unit'):
                if obs_type not in all_autocorr_data[temp]:
                    continue
                autocorr_result = all_autocorr_data[temp][obs_type]
                time_sec = autocorr_result[:, 0] * 60
                mean_ac = autocorr_result[:, 1]
                std_err = autocorr_result[:, 2]
                label = f'{int(temp)}°C {obs_labels[obs_type]}'
                if obs_type == 'unit' and np.isfinite(tau):
                    label += f' (τ = {tau*60:.1f} s)'
                ax.plot(time_sec, mean_ac, ls=obs_styles[obs_type],
                        color=color, linewidth=2, label=label)
                ax.fill_between(time_sec, mean_ac - std_err, mean_ac + std_err,
                               color=color, alpha=0.15)

        ax.axhline(y=1/np.e, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel(r'$C(\tau)$', fontsize=12)
        ax.set_title('E2E Autocorrelation (Exp, Cavity) — Full & Unit vector', fontsize=14)
        ax.set_xscale('log')
        ax.set_ylim(-0.2, 1.05)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        fig_path = "DATA_E2E_AUTOCORR_EXP/e2e_autocorr_all_temps.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure: {fig_path}")
        plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        if all_autocorr_merged:
            fig, ax = plt.subplots(figsize=(8, 6))

            for temp in sorted(all_autocorr_merged.keys()):
                for obs_type in ('full', 'unit'):
                    if obs_type not in all_autocorr_merged[temp]:
                        continue
                    merged = all_autocorr_merged[temp][obs_type]
                    time_sec = merged[:, 0] * 60
                    mean_ac = merged[:, 1]
                    std_err = merged[:, 2]

                    color = temp_colors.get(temp, 'black')
                    label = f'{int(temp)}°C {obs_labels[obs_type]}'
                    ax.plot(time_sec, mean_ac, ls=obs_styles[obs_type],
                            color=color, linewidth=2, label=label)
                    ax.fill_between(time_sec, mean_ac - std_err, mean_ac + std_err,
                                   color=color, alpha=0.15)

            ax.axvline(x=CROSSOVER_TIME_SEC, color='gray', linestyle=':', linewidth=1, alpha=0.7)
            ax.axhline(y=1/np.e, color='gray', linestyle='--', linewidth=1, alpha=0.7)

            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel(r'$C(\tau)$', fontsize=12)
            ax.set_title('E2E Autocorrelation (Merged: Short + Long)', fontsize=14)
            ax.set_xscale('log')
            ax.set_ylim(-0.2, 1.05)
            ax.set_xlim(0.03, None)
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)

            fig_path = "DATA_E2E_AUTOCORR_EXP/e2e_autocorr_merged_all_temps.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure: {fig_path}")
            plt.savefig(fig_path.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close()

        if all_autocorr_short:
            for temp in sorted(all_autocorr_short.keys()):
                if temp not in all_autocorr_data:
                    continue

                fig, ax = plt.subplots(figsize=(8, 6))

                if 'unit' in all_autocorr_short.get(temp, {}):
                    short = all_autocorr_short[temp]['unit']
                    time_sec_short = short[:, 0] * 60
                    ax.plot(time_sec_short, short[:, 1], '--',
                            color=temp_colors.get(temp, 'black'),
                            linewidth=1.5, label='Short dynamics (dt=0.04s)')

                if 'unit' in all_autocorr_data.get(temp, {}):
                    long = all_autocorr_data[temp]['unit']
                    time_sec_long = long[:, 0] * 60
                    ax.plot(time_sec_long, long[:, 1], '-',
                            color=temp_colors.get(temp, 'black'),
                            linewidth=2, label='Long dynamics (dt=2s)')

                if temp in all_autocorr_merged and 'unit' in all_autocorr_merged[temp]:
                    merged = all_autocorr_merged[temp]['unit']
                    time_sec_merged = merged[:, 0] * 60
                    ax.plot(time_sec_merged, merged[:, 1], '-',
                            color='black', linewidth=3, alpha=0.3,
                            label='Merged')

                ax.axvline(x=CROSSOVER_TIME_SEC, color='gray', linestyle=':', linewidth=1, alpha=0.7)
                ax.axhline(y=1/np.e, color='gray', linestyle='--', linewidth=1, alpha=0.7)

                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel(r'$C(\tau)$', fontsize=12)
                ax.set_title(f'E2E Autocorrelation Comparison - T = {int(temp)}°C', fontsize=14)
                ax.set_xscale('log')
                ax.set_ylim(-0.2, 1.05)
                ax.set_xlim(0.03, None)
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)

                fig_path = f"DATA_E2E_AUTOCORR_EXP/e2e_autocorr_comparison_T{int(temp)}C.png"
                plt.savefig(fig_path, dpi=150, bbox_inches='tight')
                print(f"Saved figure: {fig_path}")
                plt.close()

        if args.show:
            plt.show()

        return

    all_results = []
    msd_data_all = {}

    if args.all:
        for temp, trajs in trajs_by_temp.items():
            results, msd_data_by_tp = process_temperature(
                trajs, temp,
                plot=False,
                save_data=not args.no_save,
                show=False
            )
            all_results.append(results)
            msd_data_all[temp] = {
                'msd_data_by_tp': msd_data_by_tp,
                'results': results,
            }

        if args.plot:
            tracking_points = ['end1', 'end2']
            for tp in tracking_points:
                print(f"\nGenerating combined MSD figure for {tp}...")
                data_for_plot = {}
                for temp, data in msd_data_all.items():
                    tp_data = data['msd_data_by_tp'][tp]
                    data_for_plot[temp] = {
                        'msd_rot': tp_data['msd_rot'],
                        'msd_trans': tp_data['msd_trans'],
                        'results': tp_data['results'],
                    }
                plot_msd_all_temperatures(data_for_plot, tracking_point=tp, show=args.show)

            if args.summary:
                print(f"\nGenerating summary figure for end1...")
                data_for_summary = {}
                for temp, data in msd_data_all.items():
                    tp_data = data['msd_data_by_tp']['end1']
                    data_for_summary[temp] = {
                        'msd_rot': tp_data['msd_rot'],
                        'msd_trans': tp_data['msd_trans'],
                        'results': tp_data['results'],
                    }
                plot_summary_figure(data_for_summary, tracking_point='end1', show=args.show)

    elif args.temperature is not None:
        temp = args.temperature
        if temp not in trajs_by_temp:
            available = list(trajs_by_temp.keys())
            print(f"Temperature {temp}°C not found. Available: {available}")
            return

        results, msd_data_by_tp = process_temperature(
            trajs_by_temp[temp], temp,
            plot=args.plot,
            save_data=not args.no_save,
            show=args.show or args.plot
        )
        all_results.append(results)

    else:
        parser.print_help()
        return

    if all_results:
        print("\n" + "="*80)
        print("SUMMARY: Experimental N_rot (end1, plateau-based τ_sat)")
        print("="*80)
        print(f"{'T (°C)':<10} {'τ_rot (s)':<12} {'τ_sat (s)':<14} {'N_rot':<10} {'D_r (rad²/s)':<14} {'plateau':<10}")
        print("-"*80)
        for r in all_results:
            tau_rot_s = (r.get('tau_rot') or 0) * 60
            tau_trans_s = (r.get('tau_trans') or 0) * 60
            N_rot = r.get('N_rot')
            D_r_s = (r.get('D_r') or 0) / 60
            plateau_m = r.get('plateau_mean', 0) or 0
            N_rot_s = f"{N_rot:.2f}" if N_rot is not None else "N/A"
            tau_trans_str = f"{tau_trans_s:.1f}" if tau_trans_s > 0 else "N/A"
            D_r_str = f"{D_r_s:.4f}" if D_r_s > 0 else "N/A"
            print(f"{r['temperature']:<10.0f} {tau_rot_s:<12.1f} {tau_trans_str:<14} {N_rot_s:<10} {D_r_str:<14} {plateau_m:<10.1f}")
        print("="*80)

        if not args.no_save:
            summary_file = "DATA_EXP/rotational_summary_exp.txt"
            os.makedirs("DATA_EXP", exist_ok=True)
            with open(summary_file, 'w') as f:
                f.write("# Experimental N_rot Summary (end1, plateau-based τ_sat)\n")
                f.write("# Methodology: mirror + alignment to right cavity\n")
                f.write("# N_rot = τ_rot / τ_sat where τ_sat = first crossing of MSD plateau mean\n")
                f.write("# D_r only computed if power-law exponent α ∈ [0.8, 1.3]\n")
                f.write("# All times in SECONDS, D_r in rad²/s\n")
                f.write("#\n")
                f.write(f"# {'T(C)':<8} {'τ_rot(s)':<12} {'τ_sat(s)':<14} {'N_rot':<10} {'D_r(rad²/s)':<14} {'plateau':<10}\n")
                for r in all_results:
                    tau_rot_s = (r.get('tau_rot') or 0) * 60
                    tau_trans_s = (r.get('tau_trans') or 0) * 60
                    N_rot = r.get('N_rot')
                    D_r_s = (r.get('D_r') or 0) / 60
                    plateau_m = r.get('plateau_mean', 0) or 0
                    N_rot_s = f"{N_rot:.2f}" if N_rot is not None else "N/A"
                    tau_trans_str = f"{tau_trans_s:.1f}" if tau_trans_s > 0 else "N/A"
                    D_r_str = f"{D_r_s:.4f}" if D_r_s > 0 else "N/A"
                    f.write(f"  {r['temperature']:<8.0f} {tau_rot_s:<12.1f} {tau_trans_str:<14} {N_rot_s:<10} {D_r_str:<14} {plateau_m:<10.1f}\n")
            print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()
