import numpy as np
import matplotlib.pyplot as plt
import os
from analyse_events.utilities import set_plot_style, read_mass_center_head_tail_x

def plot_x_coordinates_vs_time(path, x_1, x_2, dt_min):
    set_plot_style()

    massCenter_data, head_data, tail_data = read_mass_center_head_tail_x(path)
    time, x_cm, y_cm, N = massCenter_data
    x_head = head_data[1]
    x_tail = tail_data[1]

    min_ht = np.min([np.min(x_head), np.min(x_tail)])
    max_ht = np.max([np.max(x_head), np.max(x_tail)])


    x_cm /= 2


    N = int(N)
    nTimeSteps = len(time)

    dump_freq = 100000
    dt = 0.001
    tau = 1e-2
    time = time * dump_freq * dt * tau / 60

    

    
    os.makedirs('FIGURES/x_vs_time/' + path + '/', exist_ok=True)
    for i in range(N):
        plt.figure(figsize=(18, 6))
        plt.plot(time, x_cm[:, i]+14.2/2, label='Center of mass')
        
        min_ht = np.min([np.min(x_head[:, i]), np.min(x_tail[:, i])])
        max_ht = np.max([np.max(x_head[:, i]), np.max(x_tail[:, i])])
        
        x_1 = 16/2
        x_2 = 176/2
        x_4 = 0
        x_5 = 96

        plt.hlines(x_1, 0, time[-1], colors='k', linestyles='dashed')
        plt.hlines(x_2, 0, time[-1], colors='k', linestyles='dashed')
        plt.hlines(x_4, 0, time[-1], colors='k', linestyles='dashed', lw=0.5)
        plt.hlines(x_5, 0, time[-1], colors='k', linestyles='dashed', lw=0.5)
        plt.grid(False)
        plt.xlabel(r'$t$ (min)', fontsize=18)
        plt.ylabel(r'$x$ (mm)', fontsize=18)
        plt.ylim(-4, 100)
        if time[-1] < 60:
            plt.xticks(np.arange(0, time[-1], 5))
        elif time[-1] < 120:
            plt.xticks(np.arange(0, time[-1], 10))
        else:
            plt.xticks(np.arange(0, time[-1], 30))
        plt.tick_params(axis='both', which='major', labelsize=18)


        plt.tight_layout()
        plt.savefig('FIGURES/x_vs_time/' + path + '/' + path.replace('/', '__') + '__x_vs_time_' + str(i + 1) + '.png')
        plt.close()

def plot_distribution(path, x1, x2):
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
    time = time * dump_freq * dt * tau / 60
    
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


