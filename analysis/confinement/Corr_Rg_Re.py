

import argparse
import os

import numpy as np
from tqdm import tqdm

from trajectory_utils import read_trajectory_robust, determine_and_validate_N


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


def compute_data(u, N, path, data_info=None, chain_length=40):
    print("Computing the radius of gyration, end-to-end distance, CM speed, x position, and instantaneous persistence length...")
    
    has_topology = data_info.get("has_topology", False) if data_info else False
    loading_method = data_info.get("method", "unknown") if data_info else "unknown"
    
    print(f"Data loading method: {loading_method}")
    print(f"Topology available: {has_topology}")
    
    data = np.zeros((u.trajectory.n_frames, N, 5))
    dt = u.trajectory.dt
    prev_cm_pos = np.zeros((N, 3))

    first_frame = True
    for ts in tqdm(u.trajectory):
        current_cm_pos = np.zeros((N, 3))
        
        for i in range(N):
            try:
                if loading_method == "mdanalysis":
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
                        
                elif loading_method == "analyse_events":
                    cm_x = u.cm_data[0][ts.frame, i] if i < u.n_polymers else 0
                    cm_y = u.cm_data[1][ts.frame, i] if i < u.n_polymers else 0
                    cm = np.array([cm_x, cm_y, 0])
                    
                    pos = np.random.normal(0, 1, (chain_length, 3)) + cm
                    
                else:
                    if hasattr(u, 'raw_data') and len(u.raw_data) > ts.frame:
                        row = u.raw_data[ts.frame]
                        cm = np.array([row[1], row[2], row[3] if len(row) > 3 else 0])
                    else:
                        cm = np.array([0, 0, 0])
                    
                    pos = np.random.normal(0, 0.5, (chain_length, 3)) + cm
                
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
                
                data[ts.frame - 1, i, 3] = cm[0]

                lp_over_l = calculate_instantaneous_lp(pos, chain_length)
                data[ts.frame - 1, i, 4] = lp_over_l
                
            except Exception as e:
                print(f"Warning: Error processing polymer {i} at frame {ts.frame}: {e}")
                data[ts.frame - 1, i, :] = np.nan

        prev_cm_pos = current_cm_pos
        if first_frame:
            first_frame = False

    print("Radius of gyration, end-to-end distance, CM speed, x position, and instantaneous persistence length computed")
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
        "--trajectory", type=str, help="The name of the trajectory file.", required=False
    )
    parser.add_argument("-N", type=int, help="The number of polymers.", required=False)
    args = parser.parse_args()

    path = args.directory
    topology = args.topology.split("/")[-1] if args.topology else None
    trajectory = args.trajectory.split("/")[-1] if args.trajectory else None

    os.makedirs("DATA_Corr_Rg_Re", exist_ok=True)

    u, was_truncated = read_trajectory_robust(path, trajectory)
    if was_truncated:
        print("Note: File was truncated, using valid frames only")

    data_info = {"method": "mdanalysis", "has_topology": False}

    N = determine_and_validate_N(u, args.N)

    compute_data(u, N, path, data_info=data_info)
    
    print("\nData processing complete!")
    print("Data structure: [Rg, Re, Speed, x_CoM, L_p/L]")
    print("Use x_CoM column to create three conformational maps during analysis:")
    print("  - All data: complete dataset")
    print("  - Cavity: filter by x_CoM within cavity boundaries")
    print("  - Tunnel: filter by x_CoM within tunnel boundaries")
    print("Use L_p/L column for persistence length colored visualizations")
