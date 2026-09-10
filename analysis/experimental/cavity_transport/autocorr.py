
import numpy as np
from tqdm import tqdm

from .config import (
    LEFT_CAVITY, RIGHT_CAVITY, CAVITY_Y,
    DUMP_FREQ, DT_SIM, TAU_LJ, MIN_SEGMENT_LENGTH, TIME_STEP_DURATION
)


def load_e2e_positions(path):
    import os

    print(f"Loading head and tail positions from {path}...")

    try:
        from analyse_events.utilities import read_mass_center_head_tail_x
        _, head_bundle, tail_bundle = read_mass_center_head_tail_x(path)
        time_raw, head_x_raw, head_y_raw, N_polymers = head_bundle
        _, tail_x_raw, tail_y_raw, _ = tail_bundle

        print(f"  Loaded {N_polymers} polymers with {len(time_raw)} time points")

    except (ImportError, FileNotFoundError, ValueError, TypeError) as e:
        print(f"  Could not use analyse_events: {e}")
        print("  Attempting direct file loading...")

        sim_name = os.path.basename(path.rstrip('/'))

        head_file = f"DATA_Head/{sim_name}__HeadOverTime.dat"
        if not os.path.exists(head_file):
            raise FileNotFoundError(f"Head data file not found: {head_file}")
        head_data = np.loadtxt(head_file, skiprows=1)
        time_raw = head_data[:, 0]
        head_x_raw = head_data[:, 1::2]
        head_y_raw = head_data[:, 2::2]
        N_polymers = head_x_raw.shape[1]

        tail_file = f"DATA_Tail/{sim_name}__TailOverTime.dat"
        if not os.path.exists(tail_file):
            raise FileNotFoundError(f"Tail data file not found: {tail_file}")
        tail_data = np.loadtxt(tail_file, skiprows=1)
        tail_x_raw = tail_data[:, 1::2]
        tail_y_raw = tail_data[:, 2::2]

        print(f"  Loaded {N_polymers} polymers with {len(time_raw)} time points")

    time = time_raw * DUMP_FREQ * DT_SIM * TAU_LJ / 60
    head_x = head_x_raw / 2 + 14.2 / 2
    head_y = head_y_raw / 2
    tail_x = tail_x_raw / 2 + 14.2 / 2
    tail_y = tail_y_raw / 2

    e2e_x = tail_x - head_x
    e2e_y = tail_y - head_y

    return time, e2e_x, e2e_y, head_x, head_y, N_polymers


def _extract_contiguous_segments_e2e(time, e2e_x, e2e_y, head_x, mask, cavity_id):
    segments = []

    if not np.any(mask):
        return segments

    padded_mask = np.concatenate(([False], mask, [False]))
    diff_mask = np.diff(padded_mask.astype(int))

    start_indices = np.where(diff_mask == 1)[0]
    end_indices = np.where(diff_mask == -1)[0]

    for start, end in zip(start_indices, end_indices):
        if end > start:
            segment = np.column_stack((
                time[start:end],
                e2e_x[start:end],
                e2e_y[start:end],
                np.full(end - start, cavity_id)
            ))
            segments.append(segment)

    return segments


def extract_e2e_cavity_segments(time, e2e_x, e2e_y, head_x, head_y, N_polymers):
    print("Extracting e2e cavity segments...")
    cavity_segments = []

    for n in tqdm(range(N_polymers), desc="Processing polymers"):
        if head_x.ndim > 1:
            hx = head_x[:, n]
            hy = head_y[:, n] if head_y.ndim > 1 else np.zeros_like(time)
            ex = e2e_x[:, n]
            ey = e2e_y[:, n]
        else:
            hx = head_x
            hy = head_y if head_y.size > 0 else np.zeros_like(time)
            ex = e2e_x
            ey = e2e_y

        left_mask = (
            (hx >= LEFT_CAVITY[0]) & (hx <= LEFT_CAVITY[1]) &
            (hy >= CAVITY_Y[0]) & (hy <= CAVITY_Y[1]) &
            ~np.isnan(hx) & ~np.isnan(hy) &
            ~np.isnan(ex) & ~np.isnan(ey)
        )
        left_segments = _extract_contiguous_segments_e2e(time, ex, ey, hx, left_mask, cavity_id=-1)

        right_mask = (
            (hx >= RIGHT_CAVITY[0]) & (hx <= RIGHT_CAVITY[1]) &
            (hy >= CAVITY_Y[0]) & (hy <= CAVITY_Y[1]) &
            ~np.isnan(hx) & ~np.isnan(hy) &
            ~np.isnan(ex) & ~np.isnan(ey)
        )
        right_segments = _extract_contiguous_segments_e2e(time, ex, ey, hx, right_mask, cavity_id=1)

        cavity_segments.extend(left_segments)
        cavity_segments.extend(right_segments)

    cavity_segments = [seg for seg in cavity_segments if len(seg) >= MIN_SEGMENT_LENGTH]

    if not cavity_segments:
        raise ValueError("No valid e2e cavity segments found in the data")

    print(f"Total e2e cavity segments: {len(cavity_segments)} (min length = {MIN_SEGMENT_LENGTH})")
    return cavity_segments


def calculate_e2e_autocorr_for_one_segment(segment, max_lag_steps):
    e2e_x = segment[:, 1]
    e2e_y = segment[:, 2]

    norms = np.sqrt(e2e_x**2 + e2e_y**2)
    valid = norms > 1e-10

    if np.sum(valid) < 10:
        nan_arr = np.full(max_lag_steps, np.nan)
        return {'full': nan_arr, 'unit': nan_arr.copy()}

    ux = np.zeros_like(e2e_x)
    uy = np.zeros_like(e2e_y)
    ux[valid] = e2e_x[valid] / norms[valid]
    uy[valid] = e2e_y[valid] / norms[valid]

    mean_norm_sq = np.mean(norms[valid]**2)
    c0_unit = np.mean((ux * ux + uy * uy)[valid]) if np.sum(valid) > 0 else 1.0

    segment_len = len(e2e_x)
    autocorr_full = np.full(max_lag_steps, np.nan)
    autocorr_unit = np.full(max_lag_steps, np.nan)

    for lag in range(1, min(segment_len, max_lag_steps + 1)):
        valid_pairs = valid[:segment_len-lag] & valid[lag:]
        if np.sum(valid_pairs) > 0:
            dot_unit = ux[:segment_len-lag] * ux[lag:] + uy[:segment_len-lag] * uy[lag:]
            autocorr_unit[lag - 1] = np.mean(dot_unit[valid_pairs])
            dot_full = e2e_x[:segment_len-lag] * e2e_x[lag:] + e2e_y[:segment_len-lag] * e2e_y[lag:]
            autocorr_full[lag - 1] = np.mean(dot_full[valid_pairs])

    if c0_unit > 1e-10:
        autocorr_unit = autocorr_unit / c0_unit
    if mean_norm_sq > 1e-10:
        autocorr_full = autocorr_full / mean_norm_sq

    return {'full': autocorr_full, 'unit': autocorr_unit}


def _find_tau_decorr_autocorr(time_lags, mean_autocorr):
    threshold = 1.0 / np.e
    below = mean_autocorr < threshold
    if np.any(below):
        idx = np.where(below)[0][0]
        if idx > 0:
            t0, t1 = time_lags[idx-1], time_lags[idx]
            c0, c1 = mean_autocorr[idx-1], mean_autocorr[idx]
            if c0 != c1:
                return t0 + (t1 - t0) * (c0 - threshold) / (c0 - c1)
            return t0
        return time_lags[0]
    return None


def calculate_e2e_autocorr(cavity_segments):
    print("Calculating e2e autocorrelation (both observables)...")

    if not cavity_segments:
        print("  No segments found")
        return {'full': (np.array([]), None), 'unit': (np.array([]), None)}

    max_lag_steps = np.max([len(seg) for seg in cavity_segments]) - 1
    all_full = []
    all_unit = []

    for segment in tqdm(cavity_segments, desc="  Processing segments"):
        result = calculate_e2e_autocorr_for_one_segment(segment, max_lag_steps)
        all_full.append(result['full'])
        all_unit.append(result['unit'])

    time_lags = np.arange(1, max_lag_steps + 1) * TIME_STEP_DURATION

    output = {}
    for obs_type, all_data in [('full', all_full), ('unit', all_unit)]:
        arr = np.array(all_data)
        mean_ac = np.nanmean(arr, axis=0)
        std_ac = np.nanstd(arr, axis=0)
        count = np.sum(~np.isnan(arr), axis=0)
        std_err = std_ac / np.sqrt(np.maximum(count, 1))

        valid = ~np.isnan(mean_ac)
        if not np.any(valid):
            print(f"  Warning: {obs_type} autocorrelation is all NaN")
            output[obs_type] = (np.array([]), None)
            continue

        last_valid = np.where(valid)[0][-1] + 1
        result = np.column_stack((
            time_lags[:last_valid],
            mean_ac[:last_valid],
            std_err[:last_valid]
        ))

        tau = _find_tau_decorr_autocorr(result[:, 0], result[:, 1])

        print(f"  {obs_type}: {len(result)} pts", end="")
        if tau is not None:
            print(f", τ_decorr = {tau:.4f} min = {tau*60:.2f} s")
        else:
            print(", τ_decorr NOT REACHED")

        output[obs_type] = (result, tau)

    return output
