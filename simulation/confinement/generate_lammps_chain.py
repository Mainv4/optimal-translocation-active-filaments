import random
import argparse


def generate_lammps_data_positions(box_size, num_polymers, polymer_size):
    a = box_size[0]
    L = box_size[1]
    lammps_data = f"\nAtoms\n\n"
    for polymer in range(num_polymers):
        x = 0.0
        y = a/2
        z = a/2
        
        for i in range(1, polymer_size + 1):
            lammps_data += f"{i + (polymer * polymer_size)} {polymer + 1} 1 {x:.5f} {y:.5f} {z:.5f} 0 0 0\n"
            
            x += 1.0
        
    return lammps_data

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
        
    
    return lammps_data

def write_lammps_data_header(num_atoms, num_bonds, num_angles, xlo, xhi, ylo, yhi, zlo, zhi):
    header = f"LAMMPS data file\n\n{num_atoms} atoms\n{num_bonds} bonds\n{num_angles} angles\n0 dihedrals\n0 impropers\n\n"
    header += "2 atom types\n1 bond types\n2 angle types\n0 dihedral types\n0 improper types\n\n"
    header += f"{xlo} {xhi} xlo xhi\n{ylo} {yhi} ylo yhi\n{zlo} {zhi} zlo zhi\n\n"
    header += "Masses\n\n1 1\n2 1\n"
    
    return header

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LAMMPS data files for polymer chains.")
    parser.add_argument("--box_size", nargs=2, type=float, default=[20, 150], help="Size of the box. Two numbers separated by a space: a and L such that the box is a**2 * L")
    parser.add_argument("--num_polymers", type=int, default=1, help="Number of polymers to generate")
    parser.add_argument("--polymer_size", type=int, default=100, help="Size of each polymer")
    parser.add_argument("--angle_proportion", type=float, default=0.1, help="Proportion of angles per polymer")
    parser.add_argument("--output_file", type=str, default="lammps_data.data", help="Name of the output file")
    args = parser.parse_args()

    lammps_data_position = generate_lammps_data_positions(args.box_size, args.num_polymers, args.polymer_size)
    lammps_data_bonds = generate_lammps_data_bonds(args.num_polymers, args.polymer_size)
    lammps_data_angles = generate_lammps_data_angles(args.num_polymers, args.polymer_size, args.angle_proportion)


    num_atoms = args.num_polymers * args.polymer_size
    num_bonds = args.num_polymers * (args.polymer_size - 1)
    num_angles = int(num_atoms * args.angle_proportion)
    xlo = 0 
    xhi = args.box_size[1]
    ylo = 0
    yhi = args.box_size[0]
    zlo = 0
    zhi = args.box_size[0]
    lammps_data_header = write_lammps_data_header(num_atoms, num_bonds, num_angles, xlo, xhi, ylo, yhi, zlo, zhi)

    lammps_data = lammps_data_header + lammps_data_position + lammps_data_bonds + lammps_data_angles

    with open(args.output_file, "w") as f:
        f.write(lammps_data)

