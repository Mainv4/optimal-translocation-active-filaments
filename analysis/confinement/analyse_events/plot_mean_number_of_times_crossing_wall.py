import numpy as np
import matplotlib.pyplot as plt
import os
from analyse_events.utilities import set_plot_style, read_mass_center_head_tail_x

def plot_mean_number_of_times_crossing_wall(path, x_1, x_2, dt_min):
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
    
    N_times_crossed_wall_cm_left = np.zeros((N, nTimeSteps))
    N_times_crossed_wall_head_left = np.zeros((N, nTimeSteps))
    N_times_crossed_wall_tail_left = np.zeros((N, nTimeSteps))
    N_times_crossed_wall_cm_right = np.zeros((N, nTimeSteps))
    N_times_crossed_wall_head_right = np.zeros((N, nTimeSteps))
    N_times_crossed_wall_tail_right = np.zeros((N, nTimeSteps))
    
    frame_diff = 1
    for i in range(N):
        for j in range(frame_diff, nTimeSteps):
            for data, left, right in zip(
                [x_cm, x_head, x_tail],
                [N_times_crossed_wall_cm_left, N_times_crossed_wall_head_left, N_times_crossed_wall_tail_left],
                [N_times_crossed_wall_cm_right, N_times_crossed_wall_head_right, N_times_crossed_wall_tail_right]
            ):
                if data[j, i] < x_1 and data[j-frame_diff, i] >= x_1:
                    left[i, j] = left[i, j-frame_diff] + 1
                else:
                    left[i, j] = left[i, j-frame_diff]
                if data[j, i] > x_2 and data[j-frame_diff, i] <= x_2:
                    left[i, j] = left[i, j-frame_diff] + 1
                else:
                    left[i, j] = left[i, j-frame_diff]
                if data[j, i] > x_1 and data[j-frame_diff, i] <= x_1:
                    right[i, j] = right[i, j-frame_diff] + 1
                else:
                    right[i, j] = right[i, j-frame_diff]
                if data[j, i] < x_2 and data[j-frame_diff, i] >= x_2:
                    right[i, j] = right[i, j-frame_diff] + 1
                else:
                    right[i, j] = right[i, j-frame_diff]

    N_times_crossed_wall_cm = N_times_crossed_wall_cm_left + N_times_crossed_wall_cm_right
    N_times_crossed_wall_head = N_times_crossed_wall_head_left + N_times_crossed_wall_head_right
    N_times_crossed_wall_tail = N_times_crossed_wall_tail_left + N_times_crossed_wall_tail_right
    mean_N_times_crossed_wall_cm = np.mean(N_times_crossed_wall_cm, axis=0)
    mean_N_times_crossed_wall_head = np.mean(N_times_crossed_wall_head, axis=0)
    mean_N_times_crossed_wall_tail = np.mean(N_times_crossed_wall_tail, axis=0)
    
    plt.plot(time, mean_N_times_crossed_wall_cm, label='Center of mass')
    plt.plot(time, mean_N_times_crossed_wall_head, label='Head')
    plt.plot(time, mean_N_times_crossed_wall_tail, label='Tail')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$\langle N_{crossed} \rangle$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'FIGURES/CrossingEvents/{path.replace("/", "__")}__MeanNumberOfTimesCrossedWall.png')
    plt.close()
    
    os.makedirs(f'FIGURES/CrossingEvents/{path}/', exist_ok=True)
    for i in range(N):
        plt.plot(time, N_times_crossed_wall_cm[i], label='Center of mass')
        plt.plot(time, N_times_crossed_wall_head[i], label='Head')
        plt.plot(time, N_times_crossed_wall_tail[i], label=f'Tail pol. {i + 1}')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$N_{crossed}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'FIGURES/CrossingEvents/{path}/{path.replace("/", "__")}__NumberOfTimesCrossedWall_{i + 1}.png')
        plt.close()


