import matplotlib.pyplot as plt
import numpy as np
import os
from analyse_events.utilities import set_plot_style, read_mass_center_head_tail_x

def plot_distribution_time_spent(path, x1, x2):
    if x1 is None or x2 is None:
        print(f"Skipping plot_distribution_time_spent: x1 and x2 required")
        return

    set_plot_style()

    massCenter_data, head_data, tail_data = read_mass_center_head_tail_x(path)
    time, x_cm, y_cm, N = massCenter_data
    x_head = head_data[1]
    x_tail = tail_data[1]

    N = int(N)
    nTimeSteps = len(time)

    dump_freq = 10000
    dt = 0.001
    tau = 1e-2
    time = time * dump_freq * dt * tau
    
    os.makedirs('FIGURES/Distribution_occupancy/' + path + '/', exist_ok=True)

    time_spent_in_left_cavity = []
    time_spent_in_right_cavity = []
    time_spent_in_tunnel = []
    for i in range(N):
        j = 1
        counter = 0
        while j < nTimeSteps:
            if x_cm[j, i] < x1 and x_cm[j-1, i] >= x1:
                try:
                    while x_cm[j, i] < x1:
                        counter += 1
                        j += 1
                    time_spent_in_left_cavity.append(counter)
                except IndexError:
                    pass
            j += 1
            counter = 0
    for i in range(N):
        j = 1
        counter = 0
        while j < nTimeSteps:
            if x_cm[j, i] > x2 and x_cm[j-1, i] <= x2:
                try:
                    while x_cm[j, i] > x2:
                        counter += 1
                        j += 1
                    time_spent_in_right_cavity.append(counter)
                except IndexError:
                    pass
            j += 1
            counter = 0
    for i in range(N):
        j = 1
        counter = 0
        while j < nTimeSteps:
            if (x_cm[j, i] > x1 and x_cm[j-1, i] <= x1) or (x_cm[j, i] < x2 and x_cm[j-1, i] >= x2):
                try:
                    while (x_cm[j, i] > x1 and x_cm[j, i] < x2):
                        counter += 1
                        j += 1
                    time_spent_in_tunnel.append(counter)
                except IndexError:
                    pass
            j += 1
            counter = 0


    plt.hist(time_spent_in_left_cavity, bins=20, label=r'Left cavity', alpha=0.5, density=True)
    plt.hist(time_spent_in_right_cavity, bins=20, label=r'Right cavity', alpha=0.5, density=True)
    plt.hist(time_spent_in_tunnel, bins=20, label=r'Tunnel', alpha=0.5, density=True)
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\langle N_{occ} \rangle$')
    plt.legend()
    plt.tight_layout()
    plt.yscale('log')
    plt.savefig('FIGURES/Distribution_occupancy/' + path + '/' + path.replace('/', '__') + '__DistributionTimeSpent.png')
    plt.close()


