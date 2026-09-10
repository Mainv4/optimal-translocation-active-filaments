
import pandas as pd
import numpy as np
from pathlib import Path

ROTATIONAL_DATA = Path("DATA_ROTATIONAL/summary.txt")
CSV_PATH = Path(__file__).parent / "data_confinement.csv"
CSV_BACKUP = Path(__file__).parent / "data_confinement_backup.csv"

def parse_rotational_summary(filepath):
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split()
            if len(parts) >= 11:
                try:
                    tau_eff = float(parts[5])
                    tau_star = float(parts[6])
                    tau_geom = float(parts[7])
                    tau_0 = float(parts[9])
                    tau_rot = float(parts[10])

                    row = {
                        'Pe': float(parts[1]),
                        'T': float(parts[2]),
                        'kappa': float(parts[3]),
                        'N_rot_eff': tau_rot / tau_eff if tau_eff > 0 else np.nan,
                        'N_rot_star': tau_rot / tau_star if tau_star > 0 else np.nan,
                        'N_rot_geom': tau_rot / tau_geom if tau_geom > 0 else np.nan,
                        'N_rot_0': tau_rot / tau_0 if tau_0 > 0 else np.nan,
                        'tau_rot': tau_rot,
                    }
                    data.append(row)
                except (ValueError, IndexError) as e:
                    print(f"Skipping line: {line[:50]}... ({e})")

    df = pd.DataFrame(data)
    print(f"Parsed {len(df)} rows from rotational summary")
    return df


def main():
    print(f"Loading {CSV_PATH}")
    df_csv = pd.read_csv(CSV_PATH, comment='#')
    print(f"  Existing CSV: {len(df_csv)} rows, {len(df_csv.columns)} columns")

    print(f"Creating backup at {CSV_BACKUP}")
    df_csv.to_csv(CSV_BACKUP, index=False)

    old_rot_cols = ['N_rot', 'N_rot_eff', 'N_rot_star', 'N_rot_geom', 'N_rot_0',
                    'tau_trans', 'tau_eff', 'tau_star', 'tau_geom', 'tau_0',
                    'tau_rot', 'D_r', 'mobility']
    cols_to_drop = [c for c in old_rot_cols if c in df_csv.columns]
    if cols_to_drop:
        print(f"  Dropping old columns: {cols_to_drop}")
        df_csv = df_csv.drop(columns=cols_to_drop)

    print(f"\nParsing {ROTATIONAL_DATA}")
    df_rot = parse_rotational_summary(ROTATIONAL_DATA)

    df_csv['Pe'] = df_csv['Pe'].round(2)
    df_csv['T'] = df_csv['T'].round(2)
    df_csv['kappa'] = df_csv['kappa'].round(2)

    df_rot['Pe'] = df_rot['Pe'].round(2)
    df_rot['T'] = df_rot['T'].round(2)
    df_rot['kappa'] = df_rot['kappa'].round(2)

    print("\nMerging on (Pe, T, kappa)...")
    df_merged = pd.merge(
        df_csv,
        df_rot,
        on=['Pe', 'T', 'kappa'],
        how='left'
    )

    n_matched = df_merged['N_rot_eff'].notna().sum()
    print(f"  Matched: {n_matched} / {len(df_merged)} rows")

    rot_keys = set(zip(df_rot['Pe'], df_rot['T'], df_rot['kappa']))
    csv_keys = set(zip(df_csv['Pe'], df_csv['T'], df_csv['kappa']))
    unmatched = rot_keys - csv_keys
    if unmatched:
        print(f"\n  Warning: {len(unmatched)} rotational entries not in CSV:")
        for pe, t, k in sorted(unmatched)[:5]:
            print(f"    Pe={pe}, T={t}, k={k}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    header = """# Confinement Reference Dataset - Active Polymer N=40
#
# Reference base: H_conf (Shannon entropy in confinement, cavity only)
# All other observables fuzzy-matched with tolerance=0.05 on (Pe, T, kappa)
#
# Rotational number data (N_rot = τ_rot / τ_trans):
# - N_rot_eff: using τ_eff (MSD reaches R²=9)
# - N_rot_star: using τ_star (MSD reaches R²=14.06)
# - N_rot_geom: using τ_geom (MSD reaches R²=16)
# - N_rot_0: using τ_0 (first MSD maximum)
# - tau_rot: rotational timescale = 1/D_r (min)
#
"""

    print(f"\nSaving to {CSV_PATH}")
    with open(CSV_PATH, 'w') as f:
        f.write(header)
        df_merged.to_csv(f, index=False)

    new_cols = ['N_rot_eff', 'N_rot_star', 'N_rot_geom', 'N_rot_0', 'tau_rot']
    print(f"Done! New columns: {', '.join(new_cols)}")

    print("\nSummary of new columns:")
    for col in new_cols:
        if col in df_merged.columns:
            valid = df_merged[col].notna().sum()
            if valid > 0:
                print(f"  {col}: {valid} values, mean={df_merged[col].mean():.4f}, range=[{df_merged[col].min():.4f}, {df_merged[col].max():.4f}]")


if __name__ == "__main__":
    main()
