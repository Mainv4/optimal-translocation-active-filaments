

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from tqdm import tqdm

sys.path.append(".")

from analyse_events.plot_distribution import plot_distribution
from analyse_events.plot_distribution_time_spent import \
    plot_distribution_time_spent
from analyse_events.plot_mean_number_of_times_crossing_wall import \
    plot_mean_number_of_times_crossing_wall
from analyse_events.plot_where_polymers import plot_where_polymers
from analyse_events.plot_x_coordinates_vs_time import \
    plot_x_coordinates_vs_time
from analyse_events.translocation_time import \
    plot_translocation_time_distribution
from analyse_events.trapping_time import plot_trapping_time_distribution
from analyse_events.utilities import (read_data_x,
                                      read_mass_center_head_tail_x,
                                      set_plot_style)


def read_data_cavity(path):
    path_array = path.split("/")
    start_of_name = path_array[1].split("__")[0]
    list_of_files = os.listdir("DATA_Cavity")
    for file in list_of_files:
        if file.startswith(start_of_name):
            cavity_file_name = file
            break
    data = np.loadtxt("DATA_Cavity/" + cavity_file_name, skiprows=1)
    return data


def main():
    os.makedirs("FIGURES/CrossingEvents", exist_ok=True)
    os.makedirs("FIGURES/x_vs_time", exist_ok=True)
    os.makedirs("FIGURES/PenetrationDistance", exist_ok=True)

    argparser = argparse.ArgumentParser(
        "Analyse the crossings of the polymers through the tunnel."
    )
    argparser.add_argument("--path", type=str, help="Path to the trajectory file.")
    argparser.add_argument(
        "-x1", type=float, help="Position of the beginning of the tunnel."
    )
    argparser.add_argument("-x2", type=float, help="Position of the end of the tunnel.")
    argparser.add_argument(
        "--dt_min",
        type=float,
        help="Minimum time the polymer must stay in the new box for the box index to change.",
    )
    args = argparser.parse_args()

    print(f"Path to the trajectory file: {args.path}")
    print(f"Position of the beginning of the tunnel: {args.x1}")
    print(f"Position of the end of the tunnel: {args.x2}")
    print(
        f"Minimum time the polymer must stay in the new box for the box index to change: {args.dt_min}"
    )

    plot_functions = [
        (
            plot_distribution,
            "Plotting the distribution of the x coordinates of the polymers...",
            False,
        ),
        (
            plot_distribution_time_spent,
            "Plotting the distribution of the time spent in the tunnel...",
            False,
        ),
        (
            plot_trapping_time_distribution,
            "Plotting the distribution of trapping times in cavities...",
            False,
        ),
        (
            plot_translocation_time_distribution,
            "Plotting the distribution of translocation times...",
            False,
        ),
    ]

    for plot_func, message, requires_dt_min in plot_functions:
        print(message)
        if requires_dt_min:
            plot_func(args.path, args.x1, args.x2, args.dt_min)
        else:
            plot_func(args.path, args.x1, args.x2)


if __name__ == "__main__":
    main()
