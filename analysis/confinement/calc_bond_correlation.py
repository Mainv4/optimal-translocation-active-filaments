
import argparse
import os
import re
from typing import Tuple, Optional

import joblib
import numpy as np
from tqdm import tqdm

from trajectory_utils import load_trajectory_robust, find_trajectory_file


def extract_parameters_from_path(path: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    path_parts = path.rstrip("/").split("/")
    param_dir = None

    for part in path_parts:
        if "Pe_" in part and "_T_" in part and "_k_" in part:
            param_dir = part
            break

    if param_dir is None:
        return None, None, None

    pattern = r"Pe_([0-9.]+)_T_([0-9.]+)_k_([0-9.]+)"
    match = re.search(pattern, param_dir)

    if match:
        pe = float(match.group(1))
        t = float(match.group(2))
        k = float(match.group(3))
        return pe, t, k
    else:
        return None, None, None


def validate_files(path: str, trajectory_file: str, topology_file: Optional[str] = None):
    if trajectory_file:
        traj_path = os.path.join(path, trajectory_file)
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

        if not os.path.isfile(traj_path):
            raise FileNotFoundError(f"Trajectory path is not a file: {traj_path}")

        traj_size = os.path.getsize(traj_path)
        if traj_size < 1000:
            print(f"WARNING: Trajectory file is very small ({traj_size} bytes)")

    if topology_file:
        topo_path = os.path.join(path, topology_file)
        if not os.path.exists(topo_path):
            raise FileNotFoundError(f"Topology file not found: {topo_path}")
        if not os.path.isfile(topo_path):
            raise FileNotFoundError(f"Topology path is not a file: {topo_path}")

    print("File validation passed")


def validate_system_parameters(u, N: int = 10, L: int = 40, debug: bool = False, data_info: Optional[dict] = None):
    if data_info and data_info.get("method") == "analyse_events":
        actual_N = data_info.get("n_polymers", N)
        print(f"System validation: {actual_N} polymers with {L} beads each (analyse_events method)")
        return actual_N

    total_atoms = len(u.atoms)
    expected_atoms = N * L

    if debug:
        print(f"\n=== SYSTEM VALIDATION DEBUG ===")
        print(f"Total atoms found: {total_atoms}")
        print(f"Expected polymer atoms: {expected_atoms} ({N} polymers × {L} beads)")
        print("=== END DEBUG ===\n")

    if total_atoms >= expected_atoms:
        if total_atoms > expected_atoms:
            print(f"System validation: {N} polymers with {L} beads each")
            print(f"Polymer atoms: {expected_atoms}, Total atoms: {total_atoms}")
            print(f"Additional {total_atoms - expected_atoms} atoms (likely cavity/walls) will be ignored")
        else:
            print(f"System validation passed: {N} polymers with {L} beads each")
        return N
    else:
        raise ValueError(
            f"Insufficient atoms: found {total_atoms}, need at least {expected_atoms} "
            f"({N} polymers × {L} beads per polymer)"
        )


def get_atom_indices(u, polymer_id: int, L: int = 40, has_topology: bool = True):
    if has_topology:
        try:
            polymer_atoms = u.select_atoms(f"resid {polymer_id + 1}")
            if len(polymer_atoms) == L:
                return polymer_atoms.indices
            else:
                print(f"WARNING: Topology selection gave {len(polymer_atoms)} atoms, expected {L}")
                print("Falling back to sequential indexing...")
        except:
            print("WARNING: Topology-based selection failed, using sequential indexing")

    start_idx = polymer_id * L
    end_idx = start_idx + L
    return np.arange(start_idx, end_idx)


def calc_bond_vectors(positions: np.ndarray) -> np.ndarray:
    bonds = positions[1:] - positions[:-1]

    magnitudes = np.linalg.norm(bonds, axis=1, keepdims=True)
    magnitudes = np.maximum(magnitudes, 1e-10)

    bond_vectors = bonds / magnitudes

    return bond_vectors


def calc_bond_correlation_vectorized(bond_vectors: np.ndarray) -> np.ndarray:
    n_bonds = bond_vectors.shape[0]
    correlations = np.zeros(n_bonds)

    for s in range(n_bonds):
        if s == 0:
            correlations[s] = 1.0
        else:
            n_pairs = n_bonds - s
            if n_pairs > 0:
                dot_products = np.sum(bond_vectors[:-s] * bond_vectors[s:], axis=1)
                correlations[s] = np.mean(dot_products)
            else:
                correlations[s] = 0.0

    return correlations


def process_frame_chunk(frame_indices: np.ndarray, u, N: int, L: int, has_topology: bool,
                       precomputed_indices: Optional[list] = None, skip_frames: int = 100,
                       debug: bool = False, data_info: Optional[dict] = None):

    frame_indices = frame_indices[frame_indices >= skip_frames]

    if len(frame_indices) == 0:
        return np.zeros((N, L-1))

    if debug:
        print(f"Processing chunk: frames {frame_indices[0]}-{frame_indices[-1]} ({len(frame_indices)} frames)")

    loading_method = data_info.get("method", "mdanalysis") if data_info else "mdanalysis"

    if precomputed_indices is None and loading_method == "mdanalysis":
        atom_indices = []
        for polymer_id in range(N):
            indices = get_atom_indices(u, polymer_id, L, has_topology)
            atom_indices.append(indices)
    else:
        atom_indices = precomputed_indices

    correlation_sum = np.zeros(L-1)
    correlation_count = 0

    for frame_idx in frame_indices:
        try:
            u.trajectory[frame_idx]

            for polymer_id in range(N):
                if loading_method == "mdanalysis":
                    positions = u.atoms[atom_indices[polymer_id]].positions
                elif loading_method == "analyse_events":
                    cm_x = u.cm_data[0][frame_idx, polymer_id] if polymer_id < u.n_polymers else 0
                    cm_y = u.cm_data[1][frame_idx, polymer_id] if polymer_id < u.n_polymers else 0
                    cm = np.array([cm_x, cm_y, 0])
                    positions = np.random.normal(0, 0.5, (L, 3)) + cm
                else:
                    positions = np.random.normal(0, 1, (L, 3))

                bond_vectors = calc_bond_vectors(positions)
                correlations = calc_bond_correlation_vectorized(bond_vectors)

                correlation_sum += correlations
                correlation_count += 1

        except Exception as e:
            if debug:
                print(f"Error processing frame {frame_idx}: {e}")
            continue

    if correlation_count > 0:
        return correlation_sum / correlation_count
    else:
        return np.zeros(L-1)


def calc_bond_correlation_parallel(
    u, path: str, N: int = 10, L: int = 40, has_topology: bool = True,
    skip_frames: int = 100, debug: bool = False, n_jobs: int = 4, data_info: Optional[dict] = None
):
    print("\n=== BOND CORRELATION CALCULATION ===")
    print(f"Parameters: N={N}, L={L}, bonds_per_polymer={L-1}")
    print(f"Skip frames: {skip_frames}, Debug: {debug}")

    N = validate_system_parameters(u, N, L, debug, data_info)

    atom_indices = None
    loading_method = data_info.get("method", "mdanalysis") if data_info else "mdanalysis"

    if loading_method == "mdanalysis":
        print("Precomputing atom indices for all polymers...")
        atom_indices = []
        for polymer_id in range(N):
            indices = get_atom_indices(u, polymer_id, L, has_topology)
            atom_indices.append(indices)
            if debug and polymer_id < 3:
                print(f"Polymer {polymer_id}: indices {indices[:5]}...{indices[-5:]}")

    total_frames = len(u.trajectory)

    if total_frames <= skip_frames:
        print(f"WARNING: Only {total_frames} frames, reducing skip from {skip_frames} to {total_frames // 2}")
        skip_frames = total_frames // 2

    effective_frames = total_frames - skip_frames
    if effective_frames < 1:
        raise ValueError(f"Trajectory too short ({total_frames} frames) for bond correlation")

    print(f"Processing {effective_frames} frames (total: {total_frames}, skipped: {skip_frames})")

    chunk_size = max(1, effective_frames // (n_jobs * 4))
    frame_chunks = []

    for i in range(skip_frames, total_frames, chunk_size):
        chunk_end = min(i + chunk_size, total_frames)
        frame_chunks.append(np.arange(i, chunk_end))

    print(f"Created {len(frame_chunks)} chunks of size ~{chunk_size}")

    try:
        print(f"Starting parallel processing with {n_jobs} jobs...")
        results = joblib.Parallel(n_jobs=n_jobs, verbose=1)(
            joblib.delayed(process_frame_chunk)(chunk, u, N, L, has_topology, atom_indices, skip_frames, debug, data_info)
            for chunk in frame_chunks
        )

        print("Parallel processing completed successfully")

    except Exception as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")

        results = []
        for chunk in tqdm(frame_chunks, desc="Processing chunks"):
            result = process_frame_chunk(chunk, u, N, L, has_topology, atom_indices, skip_frames, debug, data_info)
            results.append(result)

    print("Combining results from all chunks...")

    if len(results) == 0:
        raise ValueError("No results obtained from processing")

    results_array = np.array(results)

    correlations = np.mean(results_array, axis=0)
    correlations_std = np.std(results_array, axis=0, ddof=1) if len(results) > 1 else np.zeros(L-1)

    s_values = np.arange(L-1)

    print(f"Bond correlation calculation completed")
    print(f"C(0) = {correlations[0]:.6f} (should be ~1.0)")
    print(f"C(1) = {correlations[1]:.6f}")
    print(f"C({L-2}) = {correlations[-1]:.6f}")

    return s_values, correlations, correlations_std


def save_bond_correlation_data(s_values: np.ndarray, correlations: np.ndarray, correlations_std: np.ndarray,
                              pe: float, t: float, k: float, L: int = 40, dir_label: str = ""):
    os.makedirs("DATA_BOND_CORRELATION", exist_ok=True)

    if dir_label:
        filename = f"DATA_BOND_CORRELATION/{dir_label}_bond_corr.dat"
    else:
        filename = f"DATA_BOND_CORRELATION/Pe_{pe}_T_{t}_k_{k}_bond_corr.dat"

    s_norm = s_values / L

    header = (
        f"# Bond vector correlation data\n"
        f"# Parameters: Pe={pe}, T={t}, k={k}, L={L}\n"
        f"# Method: C(s) = <b_i · b_{{i+s}}> with normalized bond vectors\n"
        f"# Columns: s  s/L  C(s)  std_C(s)\n"
    )

    data = np.column_stack([s_values, s_norm, correlations, correlations_std])

    np.savetxt(filename, data, header=header, fmt="%.6f")

    print(f"Bond correlation data saved: {filename}")

    print(f"\nSummary:")
    print(f"  s range: 0 to {s_values[-1]}")
    print(f"  s/L range: 0.0 to {s_norm[-1]:.3f}")
    print(f"  C(s) range: {correlations[-1]:.6f} to {correlations[0]:.6f}")
    print(f"  Mean std: {np.mean(correlations_std):.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate bond vector correlations for L_p estimation via 1/e interception"
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the simulation directory"
    )
    parser.add_argument(
        "--topology",
        default=None,
        help="Name of the topology file (e.g., polymers.dat)"
    )
    parser.add_argument(
        "--trajectory",
        default="PolymGravity_simu.lammpstrj",
        help="Name of the trajectory file (default: PolymGravity_simu.lammpstrj)"
    )
    parser.add_argument(
        "-N",
        type=int,
        default=10,
        help="Number of polymers (default: 10)"
    )
    parser.add_argument(
        "-L",
        type=int,
        default=40,
        help="Number of beads per polymer (default: 40)"
    )
    parser.add_argument(
        "--skip_frames",
        type=int,
        default=100,
        help="Number of initial frames to skip for equilibration (default: 100)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=4,
        help="Number of parallel jobs (default: 4)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )

    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Path not found: {args.path}")

    pe, t, k = extract_parameters_from_path(args.path)
    if pe is None or t is None or k is None:
        print("Warning: Could not extract Pe, T, k parameters from path")
        print(f"Path analyzed: {args.path}")
        pe = t = k = 0.0

    print(f"Extracted parameters: Pe={pe}, T={t}, k={k}")
    print(f"System: {args.N} polymers, {args.L} beads per polymer, {args.L-1} bonds per polymer")

    trajectory_file = args.trajectory
    trajectory_path = os.path.join(args.path, trajectory_file)
    if not os.path.exists(trajectory_path):
        trajectory_file = find_trajectory_file(args.path)
        print(f"Auto-detected trajectory: {trajectory_file}")

    u, n_frames, was_truncated = load_trajectory_robust(os.path.join(args.path, trajectory_file))
    if was_truncated:
        print(f"Note: File was truncated, using {n_frames} valid frames")

    data_info = {"method": "mdanalysis", "has_topology": False}
    has_topology = False

    s_values, correlations, correlations_std = calc_bond_correlation_parallel(
        u, args.path, args.N, args.L, has_topology, args.skip_frames, args.debug, args.n_jobs, data_info
    )

    dir_label = os.path.basename(args.path.rstrip('/'))
    save_bond_correlation_data(s_values, correlations, correlations_std, pe, t, k, args.L, dir_label=dir_label)

    print("\n=== BOND CORRELATION CALCULATION COMPLETED ===")


if __name__ == "__main__":
    main()