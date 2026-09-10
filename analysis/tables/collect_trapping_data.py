
import re
import sqlite3
from pathlib import Path

import numpy as np
from tqdm import tqdm

BASE_PATH = Path(__file__).parent.parent

DATA_SOURCES = [
    (40, BASE_PATH / "CONFINEMENT" / "N_40" / "DATA_trapping_time"),
    (50, BASE_PATH / "CONFINEMENT" / "N_50" / "DATA_trapping_time"),
    (60, BASE_PATH / "CONFINEMENT" / "N_60" / "DATA_trapping_time"),
]

OUTPUT_DB = Path(__file__).parent / "active-polymer-worms-viz" / "trapping_times.db"

PARAM_PATTERN = re.compile(r"Pe_([\d.]+)_T_([\d.]+)_k_([\d.]+)")


def parse_trapping_file(filepath: Path) -> list[float]:
    times = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) >= 1:
                try:
                    times.append(float(parts[0]))
                except ValueError:
                    continue
    return times


def extract_parameters(filename: str) -> tuple[float, float, float] | None:
    match = PARAM_PATTERN.search(filename)
    if match:
        return float(match.group(1)), float(match.group(2)), float(match.group(3))
    return None


def create_database(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE trapping_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            N INTEGER NOT NULL,
            Pe REAL NOT NULL,
            T REAL NOT NULL,
            kappa REAL NOT NULL,
            time_min REAL NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE parameter_sets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            N INTEGER NOT NULL,
            Pe REAL NOT NULL,
            T REAL NOT NULL,
            kappa REAL NOT NULL,
            n_events INTEGER NOT NULL,
            mean_time REAL,
            median_time REAL,
            max_time REAL,
            UNIQUE(N, Pe, T, kappa)
        )
    """)

    cursor.execute("CREATE INDEX idx_events_params ON trapping_events(N, Pe, T, kappa)")
    cursor.execute("CREATE INDEX idx_sets_N ON parameter_sets(N)")

    conn.commit()
    return conn


def main():
    print("=" * 60)
    print("Collecting trapping time data into SQLite database")
    print("=" * 60)

    for N, path in DATA_SOURCES:
        if not path.exists():
            print(f"WARNING: Path for N={N} does not exist: {path}")
        else:
            n_files = len(list(path.glob("*trapping_times.txt")))
            print(f"N={N}: {n_files} files in {path}")

    print(f"\nCreating database: {OUTPUT_DB}")
    conn = create_database(OUTPUT_DB)
    cursor = conn.cursor()

    total_events = 0
    total_files = 0
    parameter_data = {}

    for N, data_path in DATA_SOURCES:
        if not data_path.exists():
            continue

        files = list(data_path.glob("*trapping_times.txt"))
        print(f"\nProcessing N={N}: {len(files)} files")

        for filepath in tqdm(files, desc=f"N={N}"):
            params = extract_parameters(filepath.name)
            if params is None:
                print(f"  WARNING: Could not extract parameters from {filepath.name}")
                continue

            Pe, T, kappa = params
            times = parse_trapping_file(filepath)

            if len(times) == 0:
                continue

            for t in times:
                cursor.execute(
                    "INSERT INTO trapping_events (N, Pe, T, kappa, time_min) VALUES (?, ?, ?, ?, ?)",
                    (N, Pe, T, kappa, t)
                )

            key = (N, Pe, T, kappa)
            if key not in parameter_data:
                parameter_data[key] = []
            parameter_data[key].extend(times)

            total_events += len(times)
            total_files += 1

    print(f"\nCalculating aggregates for {len(parameter_data)} parameter sets...")
    for (N, Pe, T, kappa), times in parameter_data.items():
        times_arr = np.array(times)
        cursor.execute(
            """INSERT INTO parameter_sets
               (N, Pe, T, kappa, n_events, mean_time, median_time, max_time)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (N, Pe, T, kappa, len(times),
             float(np.mean(times_arr)),
             float(np.median(times_arr)),
             float(np.max(times_arr)))
        )

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {total_files}")
    print(f"Total trapping events: {total_events}")
    print(f"Parameter sets: {len(parameter_data)}")
    print(f"Database saved to: {OUTPUT_DB}")
    print(f"Database size: {OUTPUT_DB.stat().st_size / 1024:.1f} KB")

    print("\nPer-N breakdown:")
    for N in [40, 50, 60]:
        n_sets = sum(1 for k in parameter_data if k[0] == N)
        n_events = sum(len(v) for k, v in parameter_data.items() if k[0] == N)
        print(f"  N={N}: {n_sets} parameter sets, {n_events} events")


if __name__ == "__main__":
    main()
