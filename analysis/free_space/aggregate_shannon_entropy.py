
import os
import glob
import re
import argparse

import numpy as np
import pandas as pd


def compute_shannon_entropy(h):
    h_flat = h.flatten()
    h_nonzero = h_flat[h_flat > 0]
    h_nonzero = h_nonzero / h_nonzero.sum()
    return -np.sum(h_nonzero * np.log(h_nonzero))


def extract_params(filename):
    basename = os.path.basename(filename)
    basename = basename.replace('.npy', '').replace('_metrics.csv', '')
    pattern = r'Pe_([0-9]*\.?[0-9]+)_T_([0-9]*\.?[0-9]+)_k_([0-9]*\.?[0-9]+)'
    match = re.match(pattern, basename)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None


def compute_entropy_from_npy(npy_file, n_bins=50):
    data = np.load(npy_file)
    Rg = data[:, :, 0].flatten() / 39
    Re = data[:, :, 1].flatten() / 39

    mask = np.isfinite(Rg) & np.isfinite(Re) & (Rg > 0)
    Rg = Rg[mask]
    Re = Re[mask]

    if len(Rg) < 100:
        return np.nan

    h, _, _ = np.histogram2d(Rg, Re, bins=n_bins, range=[[0, 1], [0, 1]])
    return compute_shannon_entropy(h)


def main():
    parser = argparse.ArgumentParser(description="Aggregate Shannon entropy into CSV")
    parser.add_argument('--generate-missing', action='store_true',
                        help='Compute entropy for .npy files without metrics CSV')
    args = parser.parse_args()

    metrics_dir = "DATA_Corr_Rg_Re/metrics"
    npy_dir = "DATA_Corr_Rg_Re"
    output_file = "shannon_entropy_free_space.csv"

    rows = []

    metrics_files = glob.glob(os.path.join(metrics_dir, "*_metrics.csv"))
    existing_params = set()

    for f in metrics_files:
        try:
            df = pd.read_csv(f)
            if 'shannon_entropy' in df.columns:
                params = extract_params(f)
                if params:
                    rows.append({'Pe': params[0], 'T': params[1], 'k': params[2],
                                 'shannon_entropy': df['shannon_entropy'].iloc[0]})
                    existing_params.add(params)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    print(f"Loaded {len(rows)} entries from existing metrics CSVs")

    if args.generate_missing:
        npy_files = glob.glob(os.path.join(npy_dir, "Pe_*.npy"))
        computed = 0
        for npy_file in npy_files:
            params = extract_params(npy_file)
            if params and params not in existing_params:
                entropy = compute_entropy_from_npy(npy_file)
                if not np.isnan(entropy):
                    rows.append({'Pe': params[0], 'T': params[1], 'k': params[2],
                                 'shannon_entropy': entropy})
                    computed += 1

                    os.makedirs(metrics_dir, exist_ok=True)
                    base = os.path.basename(npy_file).replace('.npy', '')
                    metric_df = pd.DataFrame([rows[-1]])
                    metric_df.to_csv(os.path.join(metrics_dir, f"{base}_metrics.csv"), index=False)

        print(f"Computed {computed} new entropy values from .npy files")

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(['Pe', 'T', 'k']).reset_index(drop=True)
    result_df.to_csv(output_file, index=False)
    print(f"Saved {len(result_df)} entries to {output_file}")


if __name__ == "__main__":
    main()
