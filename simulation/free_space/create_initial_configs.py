

import argparse
import os
import numpy as np
import MDAnalysis as mda
from tqdm import tqdm
import logging

def read_trajectory(path: str, name_of_trajectory_file: str, name_of_topology_file: str) -> mda.core.universe.Universe:
    logging.info("Reading the trajectory...")
    u = mda.Universe(os.path.join(path, name_of_topology_file), os.path.join(path, name_of_trajectory_file), topology_format='DATA', format='LAMMPSDUMP')
    logging.info("Trajectory read")
    return u

def extract_pol(u: mda.core.universe.Universe, Rg_max: float, N_max: int, N: int) -> np.ndarray:
    pol_index = []
    time_index = []
    n = 0
    N_array = np.arange(1, N+1)
    positions_list = []
    for ts in tqdm(u.trajectory, desc='Computing radius of gyration'):
        if ts.frame < len(u.trajectory) / 2:
            continue
        for i in range(N):
            selection = u.select_atoms(f'resid {N_array[i]}')
            positions = selection.positions
            center_of_mass = np.mean(positions, axis=0)
            squared_distances = np.sum((positions - center_of_mass) ** 2, axis=1).mean()
            if squared_distances < Rg_max**2:
                positions -= center_of_mass
                positions_list.append(positions)
                n += 1
                if n == N_max:
                    break
        if n == N_max:
            break
    logging.info(f"pol_index: {pol_index}")
    logging.info(f"time_index: {time_index}")
    positions_list = np.array(positions_list)
    return positions_list

def get_positions(u: mda.core.universe.Universe, pol_index: list, time_index: list) -> np.ndarray:
    positions = []
    for i in tqdm(range(len(pol_index)), desc='Extracting positions'):
        ts = u.trajectory[time_index[i]]
        selection = u.select_atoms(f'resid {pol_index[i]}')
        pos = selection.positions
        pos = np.array(pos)
        logging.debug(f"pos.shape: {pos.shape}")
        positions.append(pos)
    return np.array(positions)


def generate_lammps_data_bonds(num_polymers, polymer_size):
    lammps_data = f"\nBonds\n\n"
    num_bonds = 0
    for polymer in range(num_polymers):
        
        for i in range(1, polymer_size):
            num_bonds += 1
            bond_index = i + (polymer * polymer_size)
            lammps_data += f"{num_bonds} 1 {bond_index} {bond_index+1}\n"
    
    return lammps_data

def generate_lammps_data_angles(num_polymers, polymer_size, angle_proportion):
    lammps_data = f"\nAngles\n\n"
    
    num_angle = 0
    for polymer in range(num_polymers):
        num_angles = int(polymer_size * angle_proportion)
        start_index = polymer_size * polymer + (polymer_size - num_angles)
        
        for i in range(start_index, start_index + num_angles):
            atom_index = i
            num_angle += 1
            lammps_data += f"{num_angle} 1 {atom_index - 1} {atom_index} {atom_index+1}\n"
    for polymer in range(num_polymers):
        num_angles = int(polymer_size * angle_proportion)
        start_index = polymer_size * polymer + (polymer_size - num_angles)
        
        for i in range(start_index, start_index + num_angles):
            atom_index = i
            num_angle += 1
            lammps_data += f"{num_angle} 2 {atom_index - 1} {atom_index} {atom_index+1}\n"
        
    return lammps_data, num_angle


def generate_lammps_data_positions(box_size, num_polymers, polymer_size, positions):
    a = box_size[0]
    L = box_size[1]
    lammps_data = f"\nAtoms\n\n"
    for polymer in range(len(positions)):
        
        for i in range(1, polymer_size + 1):
            x = positions[polymer, i-1, 0]
            y = positions[polymer, i-1, 1]
            z = positions[polymer, i-1, 2] + 2.0
            lammps_data += f"{i + (polymer * polymer_size)} {polymer + 1} 1 {x:.5f} {y:.5f} {z:.5f} 0 0 0\n"

    return lammps_data


def write_lammps_data_header(num_atoms, num_bonds, num_angles, xlo, xhi, ylo, yhi, zlo, zhi):
    header = f"LAMMPS data file\n\n{num_atoms} atoms\n{num_bonds} bonds\n{num_angles} angles\n0 dihedrals\n0 impropers\n\n"
    header += "2 atom types\n1 bond types\n2 angle types\n0 dihedral types\n0 improper types\n\n"
    header += f"{xlo} {xhi} xlo xhi\n{ylo} {yhi} ylo yhi\n{zlo} {zhi} zlo zhi\n\n"
    header += "Masses\n\n1 1\n2 1\n"
    
    return header

def compute_and_print_distances(positions: np.ndarray) -> None:
    for idx, polymer in enumerate(positions):
        distances = np.linalg.norm(polymer[1:] - polymer[:-1], axis=1)
        logging.info(f"Polymer {idx + 1} distances between consecutive beads: {distances}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract initial configurations from a trajectory file and save them in a new lammps topology file.")
    parser.add_argument("--path", help="Path to the directory containing the trajectory files", required=True)
    parser.add_argument("--topology", help="Name of the topology file", required=True)
    parser.add_argument("--trajectory", help="Name of the trajectory file", required=True)
    parser.add_argument("-N", help="Number of polymers", required=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    trajectory_path = args.path
    topology_file = args.topology
    name_of_trajectory_file = args.trajectory
    N = int(args.N)
    if trajectory_path in topology_file:
        topology_file = topology_file.split('/')[-1]
    if trajectory_path in name_of_trajectory_file:
        name_of_trajectory_file = name_of_trajectory_file.split('/')[-1]

    u = read_trajectory(trajectory_path, name_of_trajectory_file, topology_file)

    Rg_max = 0.001
    N_max = 10

    positions = extract_pol(u, Rg_max, N_max, N)
    if len(positions) == 0:
        logging.info("No initial configurations extracted")
        logging.info(f"Rg_max: {Rg_max}")
        while len(positions) < N_max:
            logging.info("Increasing Rg_max...")
            Rg_max += 0.1
            positions = extract_pol(u, Rg_max, N_max, N)
            logging.info(f"Rg_max: {Rg_max}")
            logging.info(f"len(positions): {len(positions)}")
            if Rg_max > 12.5:
                logging.info("No initial configurations extracted")
                break
    if len(positions) != N_max:
        while len(positions) < 10:
            positions = np.concatenate((positions, positions), axis=0)
            logging.info(f"positions.shape: {positions.shape}")
        positions = positions[:N_max]

    logging.info(f"positions.shape: {positions.shape}")
    
    compute_and_print_distances(positions)
    

    box_size = u.dimensions[:3]
    num_polymers = len(positions)
    polymer_size = positions.shape[1]
    lammps_data_positions = generate_lammps_data_positions(box_size, num_polymers, polymer_size, positions)

    lammps_data_bonds = generate_lammps_data_bonds(num_polymers, polymer_size)

    angle_proportion = 0.97
    lammps_data_angles, num_angles = generate_lammps_data_angles(num_polymers, polymer_size, angle_proportion)

    num_atoms = num_polymers * polymer_size
    num_bonds = num_polymers * (polymer_size - 1)

    xlo = 0
    xhi = 200
    ylo = 0
    yhi = 18
    zlo = 0
    zhi = box_size[2]

    header = write_lammps_data_header(num_atoms, num_bonds, num_angles, xlo, xhi, ylo, yhi, zlo, zhi)

    lammps_data = header + lammps_data_positions + lammps_data_bonds + lammps_data_angles
    os.makedirs("Initial_configs", exist_ok=True)
    with open("Initial_configs/initial_config_" + trajectory_path.replace('/', '__') + ".data", "w") as f:
        f.write(lammps_data)
