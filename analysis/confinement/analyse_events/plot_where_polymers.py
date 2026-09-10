import numpy as np
import matplotlib.pyplot as plt
import os
from analyse_events.utilities import set_plot_style, read_mass_center_head_tail_x
from scipy.optimize import curve_fit

def plot_where_polymers(path, x_1, x_2, dt_min):
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

    where_cm = np.full((N, nTimeSteps), -1)
    where_head = np.full((N, nTimeSteps), -1)
    where_tail = np.full((N, nTimeSteps), -1)

    for i in range(N):
        for j in range(nTimeSteps):
            for data, where in zip([x_cm, x_head, x_tail], [where_cm, where_head, where_tail]):
                if data[j, i] <= x_1:
                    where[i, j] = 0
                elif data[j, i] >= x_2:
                    where[i, j] = 2
                else:
                    where[i, j] = 1

    os.makedirs(f'FIGURES/WherePolymers/{path}/', exist_ok=True)
    for i in range(N):
        plt.plot(time, where_cm[i], label='Center of mass')
        plt.plot(time, where_head[i], label='Head')
        plt.plot(time, where_tail[i], label=f'Tail pol. {i + 1}')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'FIGURES/WherePolymers/{path}/{path.replace("/", "__")}__WherePolymers_{i + 1}.png')
        plt.close()

    mean_where_cm = np.mean(where_cm, axis=0)
    mean_where_head = np.mean(where_head, axis=0)
    mean_where_tail = np.mean(where_tail, axis=0)
    plt.plot(time, mean_where_cm, label='Center of mass')
    plt.plot(time, mean_where_head, label='Head')
    plt.plot(time, mean_where_tail, label='Tail')
    plt.xlabel(r'$t$')
    plt.ylabel(r'$x$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'FIGURES/WherePolymers/{path.replace("/", "__")}__MeanWherePolymers.png')
    plt.close()

    N_times_crossed_wall_cm = np.zeros((N, nTimeSteps))
    N_times_crossed_wall_head = np.zeros((N, nTimeSteps))
    N_times_crossed_wall_tail = np.zeros((N, nTimeSteps))

    for i in range(N):
        for j in range(1, nTimeSteps):
            for where, N_times in zip(
                [where_cm, where_head, where_tail],
                [N_times_crossed_wall_cm, N_times_crossed_wall_head, N_times_crossed_wall_tail]
            ):
                if where[i, j] != where[i, j-1]:
                    N_times[i, j] = N_times[i, j-1] + 1
                else:
                    N_times[i, j] = N_times[i, j-1]

    os.makedirs(f'FIGURES/WherePolymers_crossed_wall/{path}/', exist_ok=True)
    for i in range(N):
        plt.plot(time, N_times_crossed_wall_cm[i], label='Center of mass')
        plt.plot(time, N_times_crossed_wall_head[i], label='Head')
        plt.plot(time, N_times_crossed_wall_tail[i], label=f'Tail pol. {i + 1}')
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'FIGURES/WherePolymers_crossed_wall/{path}/{path.replace("/", "__")}__WherePolymers_crossed_wall_{i + 1}.png')
        plt.close()

    mean_N_times_crossed_wall_cm = np.mean(N_times_crossed_wall_cm, axis=0)
    mean_N_times_crossed_wall_head = np.mean(N_times_crossed_wall_head, axis=0)
    mean_N_times_crossed_wall_tail = np.mean(N_times_crossed_wall_tail, axis=0)
    plt.plot(time, mean_N_times_crossed_wall_cm, label='Center of mass')
    plt.plot(time, mean_N_times_crossed_wall_head, label='Head')
    plt.plot(time, mean_N_times_crossed_wall_tail, label='Tail')
    
    func = lambda x, a, b: a * x + b
    popt_cm, _ = curve_fit(func, time, mean_N_times_crossed_wall_cm)
    popt_head, _ = curve_fit(func, time, mean_N_times_crossed_wall_head)
    popt_tail, _ = curve_fit(func, time, mean_N_times_crossed_wall_tail)
    plt.plot(time, func(time, *popt_cm), color='C0', linestyle='dashed')
    plt.plot(time, func(time, *popt_head), color='C1', linestyle='dashed')
    plt.plot(time, func(time, *popt_tail), color='C2', linestyle='dashed')
    acm = popt_cm[0] * 1e3
    ahead = popt_head[0] * 1e3
    atail = popt_tail[0] * 1e3
    plt.text(0.05, 0.61, r'$a_{{\rm cm}}= {:.2f} \times 10^{{-3}} \mathrm{{s}}^{{-1}}$'.format(acm), transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.55, r'$a_{{\rm head}}= {:.2f} \times 10^{{-3}} \mathrm{{s}}^{{-1}}$'.format(ahead), transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.49, r'$a_{{\rm tail}}= {:.2f} \times 10^{{-3}} \mathrm{{s}}^{{-1}}$'.format(atail), transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.40, r'$a_{{\rm head}}/a_{{\rm cm}}= {:.2f}$'.format(ahead/acm), transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.34, r'$a_{{\rm tail}}/a_{{\rm cm}}= {:.2f}$'.format(atail/acm), transform=plt.gca().transAxes, fontsize=12)
    plt.text(0.05, 0.28, r'$a_{{\rm head}}/a_{{\rm tail}}= {:.2f}$'.format(ahead/atail), transform=plt.gca().transAxes, fontsize=12)
    plt.xlabel(r'$t$')
    plt.ylim(-2, 60)
    plt.ylabel(r'$\langle N_{\rm crossed} \rangle$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'FIGURES/WherePolymers_crossed_wall/{path.replace("/", "__")}__MeanWherePolymers_crossed_wall.png')
    plt.close()

    def compute_time_spent(where_data):
        time_spent_in_left_cavity = []
        time_spent_in_right_cavity = []
        time_spent_in_tunnel = []
        for i in range(N):
            j = 1
            counter = 0
            while j < nTimeSteps:
                if where_data[i, j] == 0 and where_data[i, j-1] != 0:
                    try:
                        while where_data[i, j] == 0:
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
                if where_data[i, j] == 2 and where_data[i, j-1] != 2:
                    try:
                        while where_data[i, j] == 2:
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
                if where_data[i, j] == 1 and where_data[i, j-1] != 1:
                    try:
                        while where_data[i, j] == 1:
                            counter += 1
                            j += 1
                        time_spent_in_tunnel.append(counter)
                    except IndexError:
                        pass
                j += 1
                counter = 0
        return time_spent_in_left_cavity, time_spent_in_right_cavity, time_spent_in_tunnel

    def plot_time_spent_distribution(time_spent, label, path_suffix):
        time_spent = [x for x in time_spent if x < 300]
        plt.hist(time_spent, bins=50, label=label, alpha=0.5, density=True)
        try:
            popt, _ = curve_fit(func, np.arange(300), np.histogram(time_spent, bins=300, density=True)[0])
            if 1/popt[1] > 1:
                plt.plot(np.arange(300), func(np.arange(300), *popt), color='k', linestyle='dashed', lw=1)
                plt.text(0.05, 0.85, r'$\tau_{{\rm {}}}= {:.2f} \mathrm{{s}}$'.format(label.lower(), 1 / popt[1]), transform=plt.gca().transAxes, fontsize=12)
        except:
            pass

    for where_data, label in zip([where_cm, where_head, where_tail], ['CM', 'Head', 'Tail']):
        time_spent_in_left_cavity, time_spent_in_right_cavity, time_spent_in_tunnel = compute_time_spent(where_data)
        plt.figure()
        plot_time_spent_distribution(time_spent_in_left_cavity, 'Left cavity', label)
        plot_time_spent_distribution(time_spent_in_right_cavity, 'Right cavity', label)
        plot_time_spent_distribution(time_spent_in_tunnel, 'Tunnel', label)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\langle N_{occ} \rangle$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'FIGURES/WherePolymers_crossed_wall/{path.replace("/", "__")}__DistributionTimeSpent_{label}.png')
        plt.yscale('log')
        plt.ylim(1e-3, 1e0)
        plt.savefig(f'FIGURES/WherePolymers_crossed_wall/{path.replace("/", "__")}__DistributionTimeSpent_{label}_log.png')
        plt.close()

