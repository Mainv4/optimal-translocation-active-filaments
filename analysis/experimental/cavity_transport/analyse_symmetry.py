
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

LEFT_CAVITY = (0, 8)
RIGHT_CAVITY = (88, 96)
CAVITY_Y = (-6, 10)
DUMP_FREQ = 100000
DT_SIM = 0.001
TAU_LJ = 1e-2
TIME_STEP_DURATION = DUMP_FREQ * DT_SIM * TAU_LJ / 60
MIN_SEGMENT_LENGTH = 10


def load_e2e_positions(path):
    print(f"Loading head and tail positions from {path}...")

    sim_name = os.path.basename(str(path).rstrip('/'))
    base_dir = Path(path).parent

    head_file = base_dir / f"DATA_Head/{sim_name}__HeadOverTime.dat"
    if not head_file.exists():
        raise FileNotFoundError(f"Head data file not found: {head_file}")

    head_data = np.loadtxt(head_file, skiprows=1)
    time_raw = head_data[:, 0]
    head_x_raw = head_data[:, 1::2]
    head_y_raw = head_data[:, 2::2]
    N_polymers = head_x_raw.shape[1]

    tail_file = base_dir / f"DATA_Tail/{sim_name}__TailOverTime.dat"
    if not tail_file.exists():
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


def extract_cavity_segments(time, e2e_x, e2e_y, head_x, head_y, N_polymers):
    print("Extracting cavity segments...")
    left_segments = []
    right_segments = []

    for n in tqdm(range(N_polymers), desc="Processing polymers"):
        if head_x.ndim > 1:
            hx = head_x[:, n]
            hy = head_y[:, n]
            ex = e2e_x[:, n]
            ey = e2e_y[:, n]
        else:
            hx, hy, ex, ey = head_x, head_y, e2e_x, e2e_y

        left_mask = (
            (hx >= LEFT_CAVITY[0]) & (hx <= LEFT_CAVITY[1]) &
            (hy >= CAVITY_Y[0]) & (hy <= CAVITY_Y[1]) &
            ~np.isnan(hx) & ~np.isnan(hy) &
            ~np.isnan(ex) & ~np.isnan(ey)
        )

        right_mask = (
            (hx >= RIGHT_CAVITY[0]) & (hx <= RIGHT_CAVITY[1]) &
            (hy >= CAVITY_Y[0]) & (hy <= CAVITY_Y[1]) &
            ~np.isnan(hx) & ~np.isnan(hy) &
            ~np.isnan(ex) & ~np.isnan(ey)
        )

        for mask, segments_list, cavity_id in [(left_mask, left_segments, -1),
                                                (right_mask, right_segments, 1)]:
            if not np.any(mask):
                continue

            padded_mask = np.concatenate(([False], mask, [False]))
            diff_mask = np.diff(padded_mask.astype(int))
            start_indices = np.where(diff_mask == 1)[0]
            end_indices = np.where(diff_mask == -1)[0]

            for start, end in zip(start_indices, end_indices):
                if end - start >= MIN_SEGMENT_LENGTH:
                    segment = np.column_stack((
                        time[start:end],
                        ex[start:end],
                        ey[start:end],
                        np.full(end - start, cavity_id)
                    ))
                    segments_list.append(segment)

    print(f"  Left cavity segments:  {len(left_segments)}")
    print(f"  Right cavity segments: {len(right_segments)}")

    return left_segments, right_segments


def calculate_tau_decorr(segments):
    if not segments:
        return None, 0

    max_lag_steps = np.max([len(seg) for seg in segments]) - 1
    all_autocorrs = []

    for segment in segments:
        e2e_x = segment[:, 1]
        e2e_y = segment[:, 2]

        norms = np.sqrt(e2e_x**2 + e2e_y**2)
        valid = norms > 1e-10

        if np.sum(valid) < 10:
            continue

        ux = np.zeros_like(e2e_x)
        uy = np.zeros_like(e2e_y)
        ux[valid] = e2e_x[valid] / norms[valid]
        uy[valid] = e2e_y[valid] / norms[valid]

        segment_len = len(e2e_x)
        autocorr = np.full(max_lag_steps, np.nan)
        c0 = np.mean((ux * ux + uy * uy)[valid]) if np.sum(valid) > 0 else 1.0

        for lag in range(1, min(segment_len, max_lag_steps + 1)):
            dot_products = ux[:segment_len-lag] * ux[lag:] + uy[:segment_len-lag] * uy[lag:]
            valid_pairs = valid[:segment_len-lag] & valid[lag:]
            if np.sum(valid_pairs) > 0:
                autocorr[lag - 1] = np.mean(dot_products[valid_pairs])

        if c0 > 1e-10:
            autocorr = autocorr / c0

        all_autocorrs.append(autocorr)

    if not all_autocorrs:
        return None, 0

    all_autocorrs_array = np.array(all_autocorrs)
    average_autocorr = np.nanmean(all_autocorrs_array, axis=0)
    time_lags = np.arange(1, max_lag_steps + 1) * TIME_STEP_DURATION

    valid_indices = ~np.isnan(average_autocorr)
    if not np.any(valid_indices):
        return None, len(all_autocorrs)

    threshold = 1.0 / np.e
    below_threshold = average_autocorr < threshold

    if not np.any(below_threshold):
        return None, len(all_autocorrs)

    idx = np.where(below_threshold)[0][0]
    if idx > 0:
        t0, t1 = time_lags[idx-1], time_lags[idx]
        c0, c1 = average_autocorr[idx-1], average_autocorr[idx]
        if c0 != c1:
            tau_decorr = t0 + (t1 - t0) * (c0 - threshold) / (c0 - c1)
        else:
            tau_decorr = t0
    else:
        tau_decorr = time_lags[0]

    return tau_decorr, len(all_autocorrs)


def analyse_symmetry(sim_path):
    print(f"\n{'='*60}")
    print(f"Analysing: {sim_path}")
    print('='*60)

    time, e2e_x, e2e_y, head_x, head_y, N_polymers = load_e2e_positions(sim_path)

    left_segments, right_segments = extract_cavity_segments(
        time, e2e_x, e2e_y, head_x, head_y, N_polymers
    )

    print("\nCalculating tau_decorr for LEFT cavity...")
    tau_left, n_left = calculate_tau_decorr(left_segments)

    print("Calculating tau_decorr for RIGHT cavity...")
    tau_right, n_right = calculate_tau_decorr(right_segments)

    all_segments = left_segments + right_segments
    print("Calculating tau_decorr for COMBINED...")
    tau_combined, n_combined = calculate_tau_decorr(all_segments)

    print(f"\n{'─'*40}")
    print("Results:")
    print(f"  tau_decorr (left):     {tau_left:.4f} min ({n_left} segments)" if tau_left else f"  tau_decorr (left):     N/A ({n_left} segments)")
    print(f"  tau_decorr (right):    {tau_right:.4f} min ({n_right} segments)" if tau_right else f"  tau_decorr (right):    N/A ({n_right} segments)")
    print(f"  tau_decorr (combined): {tau_combined:.4f} min ({n_combined} segments)" if tau_combined else f"  tau_decorr (combined): N/A ({n_combined} segments)")

    relative_diff = None
    if tau_left and tau_right:
        relative_diff = abs(tau_left - tau_right) / ((tau_left + tau_right) / 2) * 100
        print(f"\n  Relative difference: {relative_diff:.1f}%")

        if relative_diff < 10:
            print("  → Symmetry OK (< 10%)")
        elif relative_diff < 20:
            print("  → Moderate asymmetry (10-20%)")
        else:
            print("  → Significant asymmetry (> 20%)")

    return {
        'path': str(sim_path),
        'tau_left': tau_left,
        'tau_right': tau_right,
        'tau_combined': tau_combined,
        'n_left': len(left_segments),
        'n_right': len(right_segments),
        'relative_diff': relative_diff
    }


def main():
    parser = argparse.ArgumentParser(description='Analyse cavity symmetry for tau_decorr')
    parser.add_argument('path', nargs='?', help='Simulation directory (e.g., Pe_0.3_T_0.1_k_1.0)')
    parser.add_argument('--all', action='store_true', help='Run on all simulations')
    parser.add_argument('--sample', type=int, default=10, help='Number of random simulations to sample (with --all)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent

    if args.all:
        sim_dirs = sorted(base_dir.glob('Pe_*_T_*_k_*'))
        if not sim_dirs:
            print("No simulation directories found!")
            sys.exit(1)

        if len(sim_dirs) > args.sample:
            np.random.seed(42)
            indices = np.random.choice(len(sim_dirs), args.sample, replace=False)
            sim_dirs = [sim_dirs[i] for i in indices]
            print(f"Sampling {args.sample} simulations")

        results = []
        for sim_dir in sim_dirs:
            try:
                result = analyse_symmetry(sim_dir)
                results.append(result)
            except Exception as e:
                print(f"Error processing {sim_dir}: {e}")

        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)

        valid_results = [r for r in results if r['relative_diff'] is not None]
        if valid_results:
            diffs = [r['relative_diff'] for r in valid_results]
            print(f"\nRelative differences (left vs right):")
            print(f"  Mean:   {np.mean(diffs):.1f}%")
            print(f"  Median: {np.median(diffs):.1f}%")
            print(f"  Max:    {np.max(diffs):.1f}%")
            print(f"  Min:    {np.min(diffs):.1f}%")

            symmetric = sum(1 for d in diffs if d < 10)
            moderate = sum(1 for d in diffs if 10 <= d < 20)
            asymmetric = sum(1 for d in diffs if d >= 20)

            print(f"\nClassification:")
            print(f"  Symmetric (< 10%):   {symmetric}/{len(diffs)}")
            print(f"  Moderate (10-20%):   {moderate}/{len(diffs)}")
            print(f"  Asymmetric (> 20%):  {asymmetric}/{len(diffs)}")

            if np.mean(diffs) < 15:
                print("\n  Overall: Symmetry assumption is VALID")
            else:
                print("\n  Overall: Symmetry assumption may be QUESTIONABLE")

    elif args.path:
        sim_path = Path(args.path)
        if not sim_path.exists():
            sim_path = base_dir / args.path

        if not sim_path.exists():
            print(f"Directory not found: {args.path}")
            sys.exit(1)

        analyse_symmetry(sim_path)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
