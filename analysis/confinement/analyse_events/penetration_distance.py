import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from analyse_events.utilities import set_plot_style, read_mass_center_head_tail_x

__all__ = ['plot_penetration_distance']

def plot_penetration_distance(path, x1, x2):
    set_plot_style()
    
    massCenter_data, _, _ = read_mass_center_head_tail_x(path)
    time, x_cm, y_cm, N = massCenter_data
    N = int(N)
    nTimeSteps = len(time)

    penetration_distances = []
    times_to_max_penetration = []
    channel_width = x2 - x1

    penetration_distances_per_polymer = {i: [] for i in range(N)}
    times_to_max_penetration_per_polymer = {i: [] for i in range(N)}

    for i in range(N):
        j = 1
        while j < nTimeSteps-1:
            if (x_cm[j, i] >= x1 and x_cm[j-1, i] < x1) or (x_cm[j, i] <= x2 and x_cm[j-1, i] > x2):
                max_distance = 0
                entry_point = x1 if x_cm[j-1, i] < x1 else x2
                entry_time = time[j]
                try:
                    while x1 <= x_cm[j, i] <= x2:
                        current_distance = abs(x_cm[j, i] - entry_point)
                        if current_distance > max_distance:
                            max_distance = current_distance
                            max_time = time[j]
                        j += 1
                    if max_distance > 0:
                        normalized_distance = max_distance / channel_width
                        penetration_distances.append(normalized_distance)
                        time_to_max = max_time - entry_time
                        times_to_max_penetration.append(time_to_max)
                        penetration_distances_per_polymer[i].append(normalized_distance)
                        times_to_max_penetration_per_polymer[i].append(time_to_max)
                except IndexError:
                    pass
            j += 1

    os.makedirs('FIGURES/PenetrationDistance/', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.hist(penetration_distances, bins=20, density=True, 
             color='blue', alpha=0.7, label='Combined penetration')
    plt.xlabel(r'$\delta_{\rm p.}/L_{\rm channel}$')
    plt.ylabel(r'$P(\delta_{\rm p.})$')
    plt.xlim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('FIGURES/PenetrationDistance/' + path.replace('/', '_') + '.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    cmap = cm.get_cmap('tab20', N)
    for i in range(N):
        if penetration_distances_per_polymer[i]:
            plt.scatter(times_to_max_penetration_per_polymer[i], 
                        penetration_distances_per_polymer[i], 
                        alpha=0.6, 
                        edgecolors='w', 
                        s=50, 
                        color=cmap(i),
                        label=f'Polymer {i}' if i < 20 else "")
    plt.xlabel(r'$\Delta t_{\rm max}$')
    plt.ylabel(r'$\delta_{\rm p.}/L_{\rm channel}$')
    plt.ylim(0.001, 1)

    plt.yscale('log')
    plt.xscale('log')
    plt.title('Penetration Distance vs. Time to Max Penetration')
    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Polymer ID', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
    plt.tight_layout()
    plt.savefig('FIGURES/PenetrationDistance/' + path.replace('/', '_') + '__Penetration_vs_Time.png')
    plt.close()

    plt.figure(figsize=(12, 8))
    for i in range(N):
        if penetration_distances_per_polymer[i]:
            plt.hist(penetration_distances_per_polymer[i], bins=10, alpha=0.5, label=f'Polymer {i}' if i < 10 else "")
    plt.xlabel('Normalized penetration distance ($\delta_{\rm p.}/L_{\rm channel}$)')
    plt.ylabel('Frequency')
    plt.title('Individual Polymer Penetration Distance Distributions')
    plt.xlim(0, 1)
    plt.legend(title='Polymer ID', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
    plt.tight_layout()
    plt.savefig('FIGURES/PenetrationDistance/' + path.replace('/', '_') + '__IndividualPenetrations_Histogram.png')
    plt.close()

    mean_penetration = np.mean(penetration_distances)
    std_penetration = np.std(penetration_distances)
    print(f"Mean normalized penetration distance: {mean_penetration:.3f} ± {std_penetration:.3f}")
    
    return penetration_distances, mean_penetration, std_penetration
