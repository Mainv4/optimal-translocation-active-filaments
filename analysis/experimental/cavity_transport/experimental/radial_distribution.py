
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LogNorm

from .loader import load_trajectories_by_temperature
from .compute import (
    RIGHT_CAVITY_CENTER,
    EXP_CAVITY_RADIUS_GEOM,
    ALIGNED_CAVITY_THRESHOLD,
)

ALIGN_X_TARGET = 48.0


def collect_cavity_positions(trajectories_by_T, tracking_point='end1'):
    positions_by_T = {}

    for T, trajs in sorted(trajectories_by_T.items()):
        all_x = []
        all_y = []
        n_aligned = 0

        for traj in trajs:
            if tracking_point == 'com':
                x_raw = traj['x']
                y_raw = traj['y']
            elif tracking_point == 'end1':
                if 'end1_x' not in traj or traj['end1_x'] is None:
                    continue
                x_raw = traj['end1_x']
                y_raw = traj['end1_y']
            elif tracking_point == 'end2':
                if 'end2_x' not in traj or traj['end2_x'] is None:
                    continue
                x_raw = traj['end2_x']
                y_raw = traj['end2_y']
            else:
                raise ValueError(f"Unknown tracking_point: {tracking_point}")

            valid = ~np.isnan(x_raw) & ~np.isnan(y_raw)
            x = x_raw[valid]
            y = y_raw[valid]

            if len(x) == 0:
                continue

            x_max_A = np.max(x)
            dx_A = ALIGN_X_TARGET - x_max_A
            x_aligned_A = x + dx_A

            x_mirror = -x
            x_max_B = np.max(x_mirror)
            dx_B = ALIGN_X_TARGET - x_max_B
            x_aligned_B = x_mirror + dx_B

            n_cavity_A = np.sum(x_aligned_A > ALIGNED_CAVITY_THRESHOLD)
            n_cavity_B = np.sum(x_aligned_B > ALIGNED_CAVITY_THRESHOLD)

            if n_cavity_A >= n_cavity_B:
                x_aligned = x_aligned_A
            else:
                x_aligned = x_aligned_B

            in_cavity = x_aligned > ALIGNED_CAVITY_THRESHOLD
            if not np.any(in_cavity):
                continue

            n_aligned += 1

            x_centered = x_aligned[in_cavity] - RIGHT_CAVITY_CENTER[0]
            y_centered = y[in_cavity] - RIGHT_CAVITY_CENTER[1]

            all_x.extend(x_centered)
            all_y.extend(y_centered)

        all_x = np.array(all_x)
        all_y = np.array(all_y)
        r = np.sqrt(all_x**2 + all_y**2)

        positions_by_T[T] = {
            'x': all_x,
            'y': all_y,
            'r': r,
            'n_worms': len(trajs),
            'n_aligned': n_aligned,
        }

        print(f"T={T}°C: {len(trajs)} worms, {n_aligned} aligned, {len(all_x):,} cavity points")

    return positions_by_T


def plot_radial_distribution(positions_by_T, save_path=None):
    temperatures = sorted(positions_by_T.keys())
    n_temps = len(temperatures)

    fig, axes = plt.subplots(n_temps, 3, figsize=(14, 4 * n_temps))
    if n_temps == 1:
        axes = axes.reshape(1, 3)

    R = EXP_CAVITY_RADIUS_GEOM

    hist_bins = 60
    hist_range = [[-6, 6], [-6, 6]]
    histograms = {}
    global_vmax = 1
    for T in temperatures:
        x = positions_by_T[T]['x']
        y = positions_by_T[T]['y']
        H, _, _ = np.histogram2d(x, y, bins=hist_bins, range=hist_range)
        histograms[T] = H
        if H.max() > global_vmax:
            global_vmax = H.max()

    for row, T in enumerate(temperatures):
        data = positions_by_T[T]
        x = data['x']
        y = data['y']
        r = data['r']
        n_worms = data['n_worms']

        ax = axes[row, 0]
        bins = np.linspace(0, 6, 51)
        ax.hist(r, bins=bins, density=True, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axvline(R, color='red', linestyle='--', linewidth=2, label=f'R = {R} mm')
        ax.set_xlabel('r (mm)')
        ax.set_ylabel('Density')
        ax.set_title(f'T = {int(T)}°C')
        ax.legend(loc='upper right')
        ax.set_xlim(0, 6)
        ax.grid(True, alpha=0.3)

        r_mean = np.mean(r)
        r_std = np.std(r)
        inside_frac = np.mean(r <= R) * 100
        hist_counts, hist_edges = np.histogram(r, bins=50, range=(0, 6))
        bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        r_mode = bin_centers[np.argmax(hist_counts)]
        ax.text(0.05, 0.95, f'<r> = {r_mean:.2f} mm\nmode = {r_mode:.2f} mm\nσ = {r_std:.2f} mm\n{inside_frac:.1f}% inside',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax = axes[row, 1]
        ax.scatter(x, y, s=0.3, alpha=0.2, c='steelblue', rasterized=True)

        circle = Circle((0, 0), R, fill=False, color='red', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        ax.plot(0, 0, 'r+', markersize=12, markeredgewidth=2)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.grid(True, alpha=0.3)

        ax.text(0.02, 0.98, f'{n_worms} worms\n{len(x):,} pts',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax = axes[row, 2]
        h = ax.hist2d(x, y, bins=hist_bins, range=hist_range,
                      norm=LogNorm(vmin=1, vmax=global_vmax), cmap='viridis', rasterized=True)
        plt.colorbar(h[3], ax=ax, label='Count')

        circle = Circle((0, 0), R, fill=False, color='red', linestyle='--', linewidth=2)
        ax.add_patch(circle)
        ax.plot(0, 0, 'r+', markersize=12, markeredgewidth=2)

        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_aspect('equal')

    axes[0, 0].set_title(f'Radial distribution\nT = {int(temperatures[0])}°C')
    axes[0, 1].set_title(f'2D scatter\nT = {int(temperatures[0])}°C')
    axes[0, 2].set_title(f'2D density\nT = {int(temperatures[0])}°C')

    fig.suptitle(f'Head position distribution in cavity (R = {R} mm)', fontsize=14, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
        return None

    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize head position distributions in cavity')
    parser.add_argument('--tracking-point', default='end1', choices=['end1', 'end2', 'com'],
                        help='Which point to track (default: end1)')
    parser.add_argument('--plot', action='store_true', help='Generate and save figure')
    parser.add_argument('--output', default='FIGURES/EXP_radial_distribution.png',
                        help='Output path for figure')
    args = parser.parse_args()

    print("Loading experimental trajectories...")
    trajs = load_trajectories_by_temperature()

    print(f"\nCollecting cavity positions (tracking: {args.tracking_point})...")
    positions = collect_cavity_positions(trajs, tracking_point=args.tracking_point)

    if args.plot:
        print("\nGenerating figure...")
        plot_radial_distribution(positions, save_path=args.output)
    else:
        print("\nUse --plot to generate figure")


if __name__ == '__main__':
    main()
