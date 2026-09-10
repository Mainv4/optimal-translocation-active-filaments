

import argparse
import os

import numpy as np
from tqdm import tqdm

from trajectory_utils import read_trajectory_robust, determine_and_validate_N


def calculate_center_of_mass(u, path, L, N):
    path_clean = path.rstrip('/')
    output_file_name = path_clean.replace('/', '__') + '__CenterOfMassOverTime.dat'
    print(f"Output file name: {output_file_name}")

    os.makedirs('DATA_CenterOfMass', exist_ok=True)

    with open(os.path.join('DATA_CenterOfMass', output_file_name), 'w') as output_file:
        print(f"Number of polymers: {N}")
        print(f"Chain length: {L}")

        xy_cm_array = []
        for ts in tqdm(u.trajectory, desc='Computing center of mass positions'):
            for i in range(N):
                start_idx = i * L
                end_idx = (i + 1) * L
                polymer = u.atoms[start_idx:end_idx]

                com = polymer.positions.mean(axis=0)
                x = com[0]
                y = com[1]
                xy_cm_array.append([x, y])

        xy_cm_array = np.reshape(xy_cm_array, (len(u.trajectory), N, 2))

        output_file.write('Time')
        for i in range(N):
            output_file.write(f' x_cm_{i+1} y_cm_{i+1}')
        output_file.write('\n')

        for i in range(len(u.trajectory)):
            output_file.write(str(i))
            for j in range(N):
                output_file.write(f' {xy_cm_array[i][j][0]:.4f} {xy_cm_array[i][j][1]:.4f}')
            output_file.write('\n')

    print(f"Center of mass positions saved to DATA_CenterOfMass/{output_file_name}")


def calculate_cavity_coordinates(u, path, L, N):
    total_atoms = len(u.atoms)
    polymer_atoms = N * L
    cavity_atom_count = total_atoms - polymer_atoms

    if cavity_atom_count <= 0:
        print("No cavity atoms detected (polymers only)")
        return

    print(f"Calculating cavity coordinates ({cavity_atom_count} cavity atoms)...")

    os.makedirs('DATA_Cavity', exist_ok=True)

    path_clean = path.rstrip('/')
    output_file_name = path_clean.replace('/', '__') + '__CavityCoordinates.dat'

    with open(os.path.join('DATA_Cavity', output_file_name), 'w') as output_file:
        for ts in u.trajectory[:1]:
            cavity_positions = u.atoms[polymer_atoms:].positions[:, :2]
            cavity_unique = np.unique(cavity_positions, axis=0)
            cavity_unique = cavity_unique[cavity_unique[:, 0].argsort()]

            output_file.write('#x\ty coordinates of the cavity\n')
            for pos in cavity_unique:
                output_file.write(f'{pos[0]}\t{pos[1]}\n')

    print(f"Cavity coordinates saved to DATA_Cavity/{output_file_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate center of mass positions of polymers over time (modernized version with auto-detection)."
    )
    parser.add_argument("--path", help="Path to the simulation directory", required=True)
    parser.add_argument("--trajectory", help="Name of the trajectory file (auto-detected if not specified)")
    parser.add_argument("-L", type=int, default=40, help="Length of each polymer (default: 40)")
    parser.add_argument("-N", type=int, help="Number of polymers (auto-detected if not specified)")
    args = parser.parse_args()

    u, has_topology = read_trajectory_robust(args.path, args.trajectory)

    N = determine_and_validate_N(u, args.N, args.L)

    calculate_center_of_mass(u, args.path, args.L, N)

    calculate_cavity_coordinates(u, args.path, args.L, N)


if __name__ == "__main__":
    main()
