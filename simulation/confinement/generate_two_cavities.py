import math
import numpy as np
import argparse

def generate_cylindrical_wall(width, length, atom_spacing, diameter, a, L):
    circumference = 2 * math.pi * radius
    x_global_translate = 0

    def num_atoms_circle_r(r, atom_spacing):
        circumference = 2 * math.pi * r
        return int(circumference / atom_spacing) + 1

    lammps_data = f"LAMMPS data file # Cylindrical wall\n\n"
    num_atoms = 0
    lammps_data += f"{num_atoms} atoms\n\n"
    
    lammps_data += "Atoms\n\n"
    k = 1
    for i in range(10):
        for j in range(int(length / atom_spacing)):
            z = i * atom_spacing
            x = j * atom_spacing + x_global_translate
            y = 0
            lammps_data += f"{k} 1 2 {x:.5f} {y:.5f} {z:.5f} 0 0 0\n"
            k += 1
            y = width
            lammps_data += f"{k} 1 2 {x:.5f} {y:.5f} {z:.5f} 0 0 0\n"
            k += 1
    def circle(x_translate, k):
        tempo_lammps_data = ""
        for i in range(10):
            for j in range(2 * int(diameter / atom_spacing)):
                r = diameter / 2
                angle = j * (math.pi / int(diameter / atom_spacing))
                z = i * atom_spacing
                x = (r) * math.cos(angle) + x_translate + x_global_translate
                y = (r) * math.sin(angle) + width/2
                if (y>0 and y<width):
                    if i == 0:
                        pass
                    if x < -diameter/2:
                        tempo_lammps_data += f"{k} 1 2 {x:.5f} {y:.5f} {z:.5f} 0 0 0\n"
                        k += 1
                    if x > length + diameter/2:
                        tempo_lammps_data += f"{k} 1 2 {x:.5f} {y:.5f} {z:.5f} 0 0 0\n"
                        k += 1
                else:
                    tempo_lammps_data += f"{k} 1 2 {x:.5f} {y:.5f} {z:.5f} 0 0 0\n"
                    k += 1
        return tempo_lammps_data, k

    lammps_data_tempo, k = circle(length + diameter/2 - 1/2 * atom_spacing, k)
    lammps_data += lammps_data_tempo

    lammps_data_tempo, k = circle(-diameter/2 + 1/2 * atom_spacing, k)
    lammps_data += lammps_data_tempo

    lammps_data = lammps_data.replace(f"{num_atoms} atoms", f"{k-1} atoms")
    return lammps_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LAMMPS data file for a cylindrical wall, given the radius, height, and atom spacing")
    parser.add_argument("-r", "--radius", type=float, required=True, help="Radius of the cylindrical wall")
    parser.add_argument("-l", "--length", type=float, required=True, help="Length of the cylindrical wall")
    parser.add_argument("-as", "--atom_spacing", type=float, required=True, help="Spacing between atoms")
    parser.add_argument("-b", "--size_of_the_box", nargs=2, type=float, required=True, help="Size of the box containing the cylindrical wall. Two numbers separated by a space: a and L such that the box is a**2 * L")
    parser.add_argument("-d", "--diameter", type=float, required=True, help="Diameter of the spherical cavities at the extremities")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Name of the output file")
    args = parser.parse_args()

    radius = args.radius
    length = args.length
    atom_spacing = args.atom_spacing
    diameter = args.diameter
    size_of_the_box = args.size_of_the_box
    a = size_of_the_box[0]
    L = size_of_the_box[1]
    if diameter < 2 * radius:
        print("Diameter: ", diameter)
        print("Radius: ", radius)
        raise ValueError("Diameter of the cavities cannot be smaller than 2 * radius")
    if a < 2 * radius:
        print("a: ", a)
        print("Radius: ", radius)
        raise ValueError("Size of the box cannot be smaller than 2 * radius")
    if L < length + 2 * diameter:
        print("L: ", L)
        print("Length: ", length)
        print("Diameter: ", diameter)
        raise ValueError("Size of the box cannot be smaller than the length of the cylindrical wall + 2 * diameter of the cavities")

    lammps_data = generate_cylindrical_wall(radius, length, atom_spacing, diameter, a, L)

    with open(args.output_file, "w") as f:
        f.write(lammps_data)
