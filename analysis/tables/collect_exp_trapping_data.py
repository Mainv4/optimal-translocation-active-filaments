import sqlite3
from pathlib import Path

import pandas as pd

CONFINEMENT_PATH = Path(__file__).parent.parent / "CONFINEMENT" / "N_40"
OUTPUT_DB = Path(__file__).parent / "active-polymer-worms-viz" / "exp_trapping_times.db"

MAX_TRAP_TIME = 10.0


def main():
    print("=" * 60)
    print("Collecting experimental trapping data (NEW ALIGNED METHOD)")
    print("=" * 60)

    print("\nStep 1: Loading worm lengths...")
    worm_lengths = {}

    data_dir = CONFINEMENT_PATH / "EXP_DATA_Cavity"
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        return

    exp_folders = [f for f in data_dir.iterdir() if f.is_dir() and not f.name.startswith('.')]
    print(f"Found {len(exp_folders)} experiment folders")

    for exp_folder in exp_folders:
        csv_files = list(exp_folder.glob("worm_*.csv"))
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, index_col=0)
                if len(df) == 0:
                    continue
                worm_key = f"{exp_folder.name}_{csv_file.stem}"
                worm_lengths[worm_key] = df['length_ref'].iloc[0]
            except Exception as e:
                print(f"  Warning: Could not read {csv_file}: {e}")

    print(f"Loaded {len(worm_lengths)} worm lengths")

    print("\nStep 2: Loading trapping times (NEW aligned method)...")
    data = []
    temps = [10, 20, 30]

    for T in temps:
        filepath = CONFINEMENT_PATH / "DATA_EXP" / f"trapping_events_aligned_T{T}.txt"
        if not filepath.exists():
            print(f"  Warning: File not found: {filepath}")
            continue

        count = 0
        matched = 0
        with open(filepath) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                count += 1
                parts = line.strip().split(',')
                if len(parts) < 5:
                    continue

                try:
                    duration_s = float(parts[2].strip())
                    worm_id = parts[4].strip()

                    time_min = duration_s / 60.0

                    if worm_id in worm_lengths and 0 < time_min <= MAX_TRAP_TIME:
                        data.append((T, worm_lengths[worm_id], time_min))
                        matched += 1
                except (ValueError, IndexError):
                    continue

        print(f"  T={T}C: {count} events, {matched} matched with lengths")

    print(f"\nTotal events with valid lengths: {len(data)}")

    print(f"\nStep 3: Updating database at {OUTPUT_DB}...")

    conn = sqlite3.connect(OUTPUT_DB)

    conn.execute("DELETE FROM exp_trapping_events")

    conn.executemany(
        "INSERT INTO exp_trapping_events (T_exp, length_mm, time_min) VALUES (?, ?, ?)",
        data
    )
    conn.commit()

    cursor = conn.cursor()
    cursor.execute("SELECT T_exp, COUNT(*), AVG(time_min), AVG(length_mm) FROM exp_trapping_events GROUP BY T_exp")
    print("\nVerification:")
    print(f"{'T_exp':>6} | {'Count':>6} | {'Mean time (min)':>15} | {'Mean length':>12}")
    print("-" * 55)
    for row in cursor.fetchall():
        print(f"{row[0]:>6} | {row[1]:>6} | {row[2]:>15.4f} | {row[3]:>12.2f}")

    conn.close()
    print(f"\nDone! Updated {OUTPUT_DB}")


if __name__ == "__main__":
    main()
