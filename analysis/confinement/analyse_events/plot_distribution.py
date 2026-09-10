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
    x_cm = massCenter_data[0]
    time = massCenter_data[1]
    N = massCenter_data[2]
    nTimeSteps = massCenter_data[3]
    x_head = head_data[0]
    x_tail = tail_data[0]

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

def plot_distribution(path, x1, x2):
    if x1 is None or x2 is None:
        print(f"Skipping plot_distribution: x1 and x2 required")
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
    occupancy = np.zeros((3, 3))
    for i in range(nTimeSteps):
        for j in range(N):
            if x_cm[i, j] < x1:
                occupancy[0, 0] += 1
            elif x_cm[i, j] > x2:
                occupancy[0, 2] += 1
            else:
                occupancy[0, 1] += 1
            if x_head[i, j] < x1:
                occupancy[1, 0] += 1
            elif x_head[i, j] > x2:
                occupancy[1, 2] += 1
            else:
                occupancy[1, 1] += 1
            if x_tail[i, j] < x1:
                occupancy[2, 0] += 1
            elif x_tail[i, j] > x2:
                occupancy[2, 2] += 1
            else:
                occupancy[2, 1] += 1
    occupancy = occupancy / N / nTimeSteps
    set_plot_style()
    plt.bar([0, 3, 6], occupancy[0], label=r'C.M.')
    plt.bar([1, 4, 7], occupancy[1], label=r'Head')
    plt.bar([2, 5, 8], occupancy[2], label=r'Tail')
    plt.xticks([1, 4, 7], [r'cav.1', r'tun.', r'cav.2'])
    plt.ylabel(r'$\langle N_{occ} \rangle$')
    plt.xlabel(r'$x$')
    plt.legend()
    plt.tight_layout()
    plt.savefig('FIGURES/Distribution_occupancy/' + path + '/' + path.replace('/', '__') + '__MeanDistribution.png')
    plt.close()


