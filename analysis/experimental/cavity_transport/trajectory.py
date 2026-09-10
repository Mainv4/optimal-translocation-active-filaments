
import os
import numpy as np
from tqdm import tqdm

from .config import (
    LEFT_CAVITY, RIGHT_CAVITY, CAVITY_Y,
    DUMP_FREQ, DT_SIM, TAU_LJ, MIN_SEGMENT_LENGTH
)


def is_escaped(x, y):
    tolerance = 1e-1
    outside_x = (x < LEFT_CAVITY[0] - tolerance) | (x > RIGHT_CAVITY[1] + tolerance)
    outside_y = (y < CAVITY_Y[0] - tolerance) | (y > CAVITY_Y[1] + tolerance)
    return outside_x | outside_y


def mask_escaped_trajectories(time, head_x, head_y, N_polymers):
    print("Checking for escaped worms...")

    head_x_masked = head_x.copy()
    head_y_masked = head_y.copy()
    total_masked = 0

    for n in range(N_polymers):
        if head_x.ndim > 1:
            x = head_x[:, n]
            y = head_y[:, n] if head_y.ndim > 1 else np.zeros_like(time)
        else:
            x = head_x
            y = head_y if head_y.size > 0 else np.zeros_like(time)

        escaped_points = is_escaped(x, y)

        if np.any(escaped_points):
            first_escape = np.argmax(escaped_points)
            if head_x.ndim > 1:
                head_x_masked[first_escape:, n] = np.nan
                if head_y.ndim > 1:
                    head_y_masked[first_escape:, n] = np.nan
            else:
                head_x_masked[first_escape:] = np.nan
                head_y_masked[first_escape:] = np.nan
            total_masked += len(time) - first_escape

    if total_masked > 0:
        print(f"  Masked {total_masked} points from escaped worms")
    else:
        print("  No escaped worms detected")

    return head_x_masked, head_y_masked


def load_head_positions(path):
    print(f"Loading head positions from {path}...")

    try:
        from analyse_events.utilities import read_mass_center_head_tail_x
        result = read_mass_center_head_tail_x(path)
        _, head_bundle, _ = result
        time_raw, head_x_raw, head_y_raw, N_polymers = head_bundle

        print(f"  Loaded {N_polymers} polymers with {len(time_raw)} time points")

        time_scaled = time_raw * DUMP_FREQ * DT_SIM * TAU_LJ / 60
        head_x_scaled = head_x_raw / 2 + 14.2 / 2
        head_y_scaled = head_y_raw / 2

        head_x_masked, head_y_masked = mask_escaped_trajectories(
            time_scaled, head_x_scaled, head_y_scaled, N_polymers
        )

        return time_scaled, head_x_masked, head_y_masked, N_polymers

    except (ImportError, FileNotFoundError, ValueError, TypeError) as e:
        print(f"  Could not use analyse_events: {e}")
        print("  Attempting direct file loading...")

        sim_name = os.path.basename(path.rstrip('/'))
        data_file = f"DATA_Head/{sim_name}__HeadOverTime.dat"

        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        print(f"  Loading from {data_file}")

        data = np.loadtxt(data_file, skiprows=1)

        time_raw = data[:, 0]
        n_cols = data.shape[1]
        N_polymers = (n_cols - 1) // 2

        print(f"  Found {N_polymers} polymers with {len(time_raw)} time points")

        head_x_raw = data[:, 1::2]
        head_y_raw = data[:, 2::2]

        time_scaled = time_raw * DUMP_FREQ * DT_SIM * TAU_LJ / 60
        head_x_scaled = head_x_raw / 2 + 14.2 / 2
        head_y_scaled = head_y_raw / 2

        head_x_masked, head_y_masked = mask_escaped_trajectories(
            time_scaled, head_x_scaled, head_y_scaled, N_polymers
        )

        return time_scaled, head_x_masked, head_y_masked, N_polymers


def _extract_contiguous_segments(time, x, y, mask, cavity_id):
    segments = []

    if not np.any(mask):
        return segments

    padded_mask = np.concatenate(([False], mask, [False]))
    diff_mask = np.diff(padded_mask.astype(int))

    start_indices = np.where(diff_mask == 1)[0]
    end_indices = np.where(diff_mask == -1)[0]

    for start, end in zip(start_indices, end_indices):
        if end > start:
            segment_time = time[start:end]
            segment_x = x[start:end]
            segment_y = y[start:end]

            cavity_id_array = np.full_like(segment_time, cavity_id)
            segment = np.column_stack(
                (segment_time, segment_x, segment_y, cavity_id_array)
            )
            segments.append(segment)

    return segments


def extract_cavity_segments(time, head_x, head_y, N_polymers):
    print("Extracting cavity segments...")
    cavity_segments = []

    for n in tqdm(range(N_polymers), desc="Processing polymers"):
        if head_x.ndim > 1:
            x = head_x[:, n]
            y = head_y[:, n] if head_y.ndim > 1 else np.zeros_like(time)
        else:
            x = head_x
            y = head_y if head_y.size > 0 else np.zeros_like(time)

        left_mask = (
            (x >= LEFT_CAVITY[0]) & (x <= LEFT_CAVITY[1]) &
            (y >= CAVITY_Y[0]) & (y <= CAVITY_Y[1]) &
            ~np.isnan(x) & ~np.isnan(y)
        )
        left_segments = _extract_contiguous_segments(time, x, y, left_mask, cavity_id=-1)

        right_mask = (
            (x >= RIGHT_CAVITY[0]) & (x <= RIGHT_CAVITY[1]) &
            (y >= CAVITY_Y[0]) & (y <= CAVITY_Y[1]) &
            ~np.isnan(x) & ~np.isnan(y)
        )
        right_segments = _extract_contiguous_segments(time, x, y, right_mask, cavity_id=1)

        cavity_segments.extend(left_segments)
        cavity_segments.extend(right_segments)

        print(f"  Polymer {n+1}: {len(left_segments)} left + {len(right_segments)} right segments")

    cavity_segments = [seg for seg in cavity_segments if len(seg) >= MIN_SEGMENT_LENGTH]

    if not cavity_segments:
        raise ValueError("No valid cavity segments found in the data")

    print(f"Total cavity segments: {len(cavity_segments)} (min length = {MIN_SEGMENT_LENGTH})")
    return cavity_segments
