

import argparse
import os
import sys

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d


def determine_and_validate_N(u, N_given=None, chain_length=40):
    total_atoms = len(u.atoms)
    
    if total_atoms % chain_length != 0:
        raise ValueError(f"Total number of atoms ({total_atoms}) is not divisible by chain length ({chain_length})")
    
    N_calculated = total_atoms // chain_length
    
    if N_given is not None:
        if N_given != N_calculated:
            raise ValueError(f"Given number of polymers ({N_given}) does not match calculated value ({N_calculated}) "
                           f"based on total atoms ({total_atoms}) and chain length ({chain_length})")
        print(f"Validated: {N_given} polymers with {chain_length} atoms each (total: {total_atoms} atoms)")
        return N_given
    else:
        print(f"Automatically determined: {N_calculated} polymers with {chain_length} atoms each (total: {total_atoms} atoms)")
        return N_calculated


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


def find_one_over_e_interception(s_norm: np.ndarray, correlations: np.ndarray) -> float:
    one_over_e = 1.0 / np.e

    if correlations[-1] > one_over_e or correlations[0] < one_over_e:
        return np.nan

    try:
        crossing_indices = np.where(correlations <= one_over_e)[0]
        if len(crossing_indices) == 0:
            return np.nan

        cross_idx = crossing_indices[0]

        if cross_idx == 0:
            return s_norm[0]

        idx_before = cross_idx - 1
        idx_after = cross_idx

        s1, s2 = s_norm[idx_before], s_norm[idx_after]
        c1, c2 = correlations[idx_before], correlations[idx_after]

        if c2 != c1:
            s_intercept = s1 + (s2 - s1) * (one_over_e - c1) / (c2 - c1)
        else:
            s_intercept = s1

        return s_intercept

    except Exception:
        return np.nan


def calculate_instantaneous_lp(positions: np.ndarray, chain_length: int = 40) -> float:
    try:
        bond_vectors = calc_bond_vectors(positions)
        correlations = calc_bond_correlation_vectorized(bond_vectors)

        n_bonds = len(correlations)
        s_values = np.arange(n_bonds)
        s_norm = s_values / chain_length

        s_intercept = find_one_over_e_interception(s_norm, correlations)

        if not np.isnan(s_intercept):
            return s_intercept
        else:
            return np.nan

    except Exception:
        return np.nan


def read_trajectory(path, name_of_trajectory_file, name_of_topology_file=None):
    print("Reading the trajectory...")
    if name_of_topology_file:
        if '/' in name_of_topology_file or name_of_topology_file.startswith('/'):
            topology_path = name_of_topology_file
        else:
            topology_path = path + "/" + name_of_topology_file

        u = mda.Universe(
            topology_path,
            path + "/" + name_of_trajectory_file,
            topology_format="DATA",
            format="LAMMPSDUMP",
        )
    else:
        u = mda.Universe(
            path + "/" + name_of_trajectory_file,
            format="LAMMPSDUMP",
        )
    print("Trajectory read")
    return u


def compute_data(u, N, path, has_topology=True, chain_length=40):
    print("Computing the radius of gyration, end-to-end distance, CM speed, and instantaneous persistence length...")
    data = np.zeros((u.trajectory.n_frames, N, 4))
    dt = u.trajectory.dt
    prev_cm_pos = np.zeros((N, 3))

    first_frame = True
    for ts in tqdm(u.trajectory):
        current_cm_pos = np.zeros((N, 3))
        for i in range(N):
            if has_topology:
                polymer = u.select_atoms("resid " + str(i + 1))
            else:
                start_idx = i * chain_length
                end_idx = (i + 1) * chain_length
                polymer = u.atoms[start_idx:end_idx]
            
            pos = polymer.positions
            
            if has_topology:
                cm = polymer.center_of_mass()
            else:
                cm = np.mean(pos, axis=0)
            
            current_cm_pos[i] = cm

            pos_xy = pos[:, :2]
            data[ts.frame - 1, i, 0] = np.sqrt(
                np.sum(np.mean((pos_xy - np.mean(pos_xy, axis=0)) ** 2, axis=0))
            )
            data[ts.frame - 1, i, 1] = np.linalg.norm(pos_xy[-1] - pos_xy[0])

            if not first_frame:
                displacement = cm - prev_cm_pos[i]
                speed = np.linalg.norm(displacement) / dt if dt > 1e-9 else 0.0
                data[ts.frame - 1, i, 2] = speed
            else:
                data[ts.frame - 1, i, 2] = 0.0

            lp_over_l = calculate_instantaneous_lp(pos, chain_length)
            data[ts.frame - 1, i, 3] = lp_over_l

        prev_cm_pos = current_cm_pos
        if first_frame:
            first_frame = False

    print("Radius of gyration, end-to-end distance, CM speed, and instantaneous persistence length computed")
    print("Saving the data...")
    output_dir = "DATA_Corr_Rg_Re"
    os.makedirs(output_dir, exist_ok=True)
    filename_path = path.strip('/').replace("/", "_")
    
    if filename_path.startswith("FREE_SPACE_"):
        filename_path = filename_path[len("FREE_SPACE_"):]
    
    if filename_path.startswith("ACTIVE_") or filename_path.startswith("ACTIVE"):
        parts = filename_path.split("_")
        if parts[0] == "ACTIVE":
            filename_path = "_".join(parts[1:])
        elif parts[0] == "ACTIVE" and len(parts) > 1 and parts[1].startswith("T"):
            filename_path = "_".join(parts[2:])
    
    np.save(os.path.join(output_dir, f"{filename_path}.npy"), data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the mean square displacement of the CM, the Head, and the Tail polymers."
    )
    parser.add_argument(
        "--directory",
        type=str,
        help="The path to the directory containing the trajectory.",
    )
    parser.add_argument("--topology", type=str, help="The name of the topology file.", required=False)
    parser.add_argument(
        "--trajectory", type=str, help="The name of the trajectory file."
    )
    parser.add_argument("-N", type=int, help="The number of polymers.", required=False)
    parser.add_argument("--force", action="store_true", help="Force recalculation even if output exists")
    args = parser.parse_args()

    path = args.directory
    topology = args.topology if args.topology else None
    trajectory = args.trajectory.split("/")[-1]

    output_dir = "DATA_Corr_Rg_Re"
    os.makedirs(output_dir, exist_ok=True)

    filename_path = path.strip('/').replace("/", "_")
    if filename_path.startswith("FREE_SPACE_"):
        filename_path = filename_path[len("FREE_SPACE_"):]
    if filename_path.startswith("ACTIVE_") or filename_path.startswith("ACTIVE"):
        parts = filename_path.split("_")
        if parts[0] == "ACTIVE":
            filename_path = "_".join(parts[1:])
        elif parts[0] == "ACTIVE" and len(parts) > 1 and parts[1].startswith("T"):
            filename_path = "_".join(parts[2:])
    output_file = os.path.join(output_dir, f"{filename_path}.npy")

    if os.path.exists(output_file) and not args.force:
        print(f"Output exists, skipping: {output_file}")
        sys.exit(0)

    u = read_trajectory(path, trajectory, topology)

    N = determine_and_validate_N(u, args.N)

    compute_data(u, N, path, has_topology=(topology is not None))
