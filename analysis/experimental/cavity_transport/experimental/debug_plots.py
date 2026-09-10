
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .compute import (
    LEFT_CAVITY_CENTER, RIGHT_CAVITY_CENTER,
    EXP_CAVITY_RADIUS_GEOM,
    ALIGNED_CAVITY_CENTER, ALIGNED_CAVITY_THRESHOLD,
)


def compute_segment_msd(x, y, theta, max_lag_fraction=0.5):
    n = len(x)
    max_lag = max(int(n * max_lag_fraction), 1)

    msd_rot = np.full(max_lag, np.nan)
    msd_trans = np.full(max_lag, np.nan)

    for lag in range(1, max_lag + 1):
        dtheta = theta[lag:] - theta[:-lag]
        if len(dtheta) > 0:
            msd_rot[lag - 1] = np.mean(dtheta**2)

        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        if len(dx) > 0:
            msd_trans[lag - 1] = np.mean(dx**2 + dy**2)

    lag_indices = np.arange(1, max_lag + 1)
    return lag_indices, msd_rot, msd_trans


def plot_segment_debug(segment, tracking_point='com', segment_info=None, save_path=None):
    col_map = {
        'com': (1, 2),
        'end1': (4, 5),
        'end2': (6, 7),
    }
    if tracking_point not in col_map:
        raise ValueError(f"tracking_point must be one of {list(col_map.keys())}")
    x_col, y_col = col_map[tracking_point]

    time = segment[:, 0]
    x = segment[:, x_col]
    y = segment[:, y_col]
    cavity_id = segment[0, 3]

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return None

    if cavity_id < 0:
        center_x, center_y = LEFT_CAVITY_CENTER
        cavity_name = "Left"
    else:
        center_x, center_y = RIGHT_CAVITY_CENTER
        cavity_name = "Right"

    x_centered = x - center_x
    y_centered = y - center_y

    if cavity_id > 0:
        x_centered = -x_centered

    theta_raw = np.arctan2(y_centered, x_centered)
    theta = np.unwrap(theta_raw)

    dt = np.median(np.diff(time))
    lag_indices, msd_rot, msd_trans = compute_segment_msd(x, y, theta)
    lag_time = lag_indices * dt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    time_sec = (time - time[0]) * 60
    scatter = ax.scatter(x, y, c=time_sec, cmap='viridis', s=5, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)')

    cavity_circle = Circle((center_x, center_y), EXP_CAVITY_RADIUS_GEOM,
                           fill=False, color='red', linestyle='--', linewidth=1.5)
    ax.add_patch(cavity_circle)

    ax.plot(center_x, center_y, 'r+', markersize=10, markeredgewidth=2)

    ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
    ax.plot(x[-1], y[-1], 'rs', markersize=8, label='End')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Trajectory ({tracking_point})')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time_sec, theta, 'b-', linewidth=1)

    theta_min, theta_max = theta.min(), theta.max()
    for k in range(-10, 11):
        y_line = k * np.pi
        if theta_min - np.pi < y_line < theta_max + np.pi:
            ax.axhline(y_line, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
            if k != 0:
                ax.text(time_sec[-1] * 1.02, y_line, f'{k}π', fontsize=8, va='center')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (rad)')
    ax.set_title('Unwrapped angle θ(t)')
    ax.grid(True, alpha=0.3)

    total_rotation = (theta[-1] - theta[0]) / np.pi
    ax.text(0.05, 0.95, f'Δθ = {total_rotation:.2f}π rad', transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[2]
    lag_time_sec = lag_time * 60

    valid_rot = ~np.isnan(msd_rot) & (msd_rot > 0)
    valid_trans = ~np.isnan(msd_trans) & (msd_trans > 0)

    if np.any(valid_rot):
        ax.loglog(lag_time_sec[valid_rot], msd_rot[valid_rot], 'b-',
                  linewidth=1.5, label='Rotational (rad²)')

    if np.any(valid_trans):
        ax.loglog(lag_time_sec[valid_trans], msd_trans[valid_trans], 'k-',
                  linewidth=1.5, label='Translational (mm²)')

    if np.any(valid_rot) or np.any(valid_trans):
        t_ref = lag_time_sec[lag_time_sec > 0]
        if len(t_ref) > 1:
            y_ref = t_ref / t_ref[0]
            if np.any(valid_rot):
                y_ref_scaled = y_ref * msd_rot[valid_rot][0]
                ax.loglog(t_ref, y_ref_scaled, 'b--', alpha=0.4, linewidth=1, label='slope=1 (rot)')
            if np.any(valid_trans):
                y_ref_scaled = y_ref * msd_trans[valid_trans][0]
                ax.loglog(t_ref, y_ref_scaled, 'k--', alpha=0.4, linewidth=1, label='slope=1 (trans)')

    ax.set_xlabel('Lag time (s)')
    ax.set_ylabel('MSD')
    ax.set_title('MSD vs lag time')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    if segment_info:
        title = (f"T={segment_info.get('temperature', '?')}°C | "
                 f"Worm: {segment_info.get('worm_id', '?')} | "
                 f"Segment {segment_info.get('segment_idx', '?')} | "
                 f"{cavity_name} cavity | "
                 f"N={len(time)} pts")
    else:
        title = f"{cavity_name} cavity | N={len(time)} points"
    fig.suptitle(title, fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=72, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def plot_segment_debug_aligned(segment, tracking_point='end1', segment_info=None, save_path=None):
    time = segment[:, 0]
    x = segment[:, 1]
    y = segment[:, 2]

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return None

    center_x, center_y = ALIGNED_CAVITY_CENTER

    x_centered = x - center_x
    y_centered = y - center_y

    theta_raw = np.arctan2(y_centered, x_centered)
    theta = np.unwrap(theta_raw)

    dt = np.median(np.diff(time))
    lag_indices, msd_rot, msd_trans = compute_segment_msd(x, y, theta)
    lag_time = lag_indices * dt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    time_sec = (time - time[0]) * 60
    scatter = ax.scatter(x, y, c=time_sec, cmap='viridis', s=5, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)')

    cavity_circle = Circle((center_x, center_y), EXP_CAVITY_RADIUS_GEOM,
                           fill=False, color='red', linestyle='--', linewidth=1.5)
    ax.add_patch(cavity_circle)

    ax.plot(center_x, center_y, 'r+', markersize=10, markeredgewidth=2)

    ax.axvline(ALIGNED_CAVITY_THRESHOLD, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.plot(x[0], y[0], 'go', markersize=8, label='Start')
    ax.plot(x[-1], y[-1], 'rs', markersize=8, label='End')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title(f'Aligned trajectory ({tracking_point})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(38, 52)
    ax.set_ylim(-6, 6)

    ax = axes[1]
    ax.plot(time_sec, theta, 'b-', linewidth=1)

    theta_min, theta_max = theta.min(), theta.max()
    for k in range(-10, 11):
        y_line = k * np.pi
        if theta_min - np.pi < y_line < theta_max + np.pi:
            ax.axhline(y_line, color='gray', linestyle=':', alpha=0.5, linewidth=0.5)
            if k != 0:
                ax.text(time_sec[-1] * 1.02, y_line, f'{k}π', fontsize=8, va='center')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('θ (rad)')
    ax.set_title('Unwrapped angle θ(t)')
    ax.grid(True, alpha=0.3)

    total_rotation = (theta[-1] - theta[0]) / np.pi
    ax.text(0.05, 0.95, f'Δθ = {total_rotation:.2f}π rad', transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax = axes[2]
    lag_time_sec = lag_time * 60

    valid_rot = ~np.isnan(msd_rot) & (msd_rot > 0)
    valid_trans = ~np.isnan(msd_trans) & (msd_trans > 0)

    if np.any(valid_rot):
        ax.loglog(lag_time_sec[valid_rot], msd_rot[valid_rot], 'b-',
                  linewidth=1.5, label='Rotational (rad²)')

    if np.any(valid_trans):
        ax.loglog(lag_time_sec[valid_trans], msd_trans[valid_trans], 'k-',
                  linewidth=1.5, label='Translational (mm²)')

    if np.any(valid_rot) or np.any(valid_trans):
        t_ref = lag_time_sec[lag_time_sec > 0]
        if len(t_ref) > 1:
            y_ref = t_ref / t_ref[0]
            if np.any(valid_rot):
                y_ref_scaled = y_ref * msd_rot[valid_rot][0]
                ax.loglog(t_ref, y_ref_scaled, 'b--', alpha=0.4, linewidth=1, label='slope=1 (rot)')
            if np.any(valid_trans):
                y_ref_scaled = y_ref * msd_trans[valid_trans][0]
                ax.loglog(t_ref, y_ref_scaled, 'k--', alpha=0.4, linewidth=1, label='slope=1 (trans)')

    ax.set_xlabel('Lag time (s)')
    ax.set_ylabel('MSD')
    ax.set_title('MSD vs lag time')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    if segment_info:
        version = segment_info.get('version', '?')
        title = (f"T={segment_info.get('temperature', '?')}°C | "
                 f"Worm: {segment_info.get('worm_id', '?')} | "
                 f"Segment {segment_info.get('segment_idx', '?')} | "
                 f"Version {version} | "
                 f"Right cavity (aligned) | "
                 f"N={len(time)} pts")
    else:
        title = f"Right cavity (aligned) | N={len(time)} points"
    fig.suptitle(title, fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=72, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def _plot_aligned_task(args):
    segment, tracking_point, segment_info, save_path = args
    try:
        plot_segment_debug_aligned(segment, tracking_point=tracking_point,
                                   segment_info=segment_info, save_path=save_path)
        return True
    except Exception:
        return False


def _plot_task(args):
    segment, tracking_point, segment_info, save_path = args
    try:
        plot_segment_debug(segment, tracking_point=tracking_point,
                          segment_info=segment_info, save_path=save_path)
        return True
    except Exception:
        return False


def generate_all_debug_figures(trajectories_by_T, output_dir='FIGURES/DEBUG_EXP', n_workers=24):
    from .compute import extract_aligned_segments

    tracking_points = ['end1', 'end2']

    all_tasks = []

    for T, trajs in sorted(trajectories_by_T.items()):
        print(f"\n--- Temperature: {T}°C ({len(trajs)} worms) ---")

        for tp in tracking_points:
            print(f"  Extracting aligned segments for {tp}...")
            segments, metadata = extract_aligned_segments(trajs, tracking_point=tp,
                                                          verbose=True, return_metadata=True)

            if not segments:
                print(f"    No segments found for {tp}")
                continue

            segments_by_worm = {}
            for i, (seg, meta) in enumerate(zip(segments, metadata)):
                worm_id = meta.get('worm_id', f'worm_{i:03d}')
                if worm_id not in segments_by_worm:
                    segments_by_worm[worm_id] = []
                segments_by_worm[worm_id].append((i, seg, meta))

            tp_dir = os.path.join(output_dir, f"T{int(T)}", tp)

            for worm_id, worm_segments in segments_by_worm.items():
                worm_dir = os.path.join(tp_dir, worm_id)
                os.makedirs(worm_dir, exist_ok=True)

                for local_idx, (_, seg, meta) in enumerate(worm_segments):
                    segment_info = {
                        'temperature': T,
                        'worm_id': worm_id,
                        'segment_idx': local_idx,
                        'version': meta.get('version', '?'),
                    }
                    save_path = os.path.join(worm_dir, f"segment_{local_idx:03d}.png")
                    all_tasks.append((seg, tp, segment_info, save_path))

    print(f"\nGenerating {len(all_tasks)} debug figures with {n_workers} workers...")

    success_count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_plot_aligned_task, task) for task in all_tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Plotting"):
            if future.result():
                success_count += 1

    print(f"\nTotal: {success_count}/{len(all_tasks)} debug figures saved to {output_dir}/")


def plot_full_trajectory(traj, tracking_point='end1', save_path=None):
    time = traj['time']

    if tracking_point == 'com':
        x = traj['x']
        y = traj['y']
    elif tracking_point == 'end1':
        if 'end1_x' not in traj or traj['end1_x'] is None:
            return None
        x = traj['end1_x']
        y = traj['end1_y']
        if 'time_endpoints' in traj:
            time = traj['time_endpoints']
    elif tracking_point == 'end2':
        if 'end2_x' not in traj or traj['end2_x'] is None:
            return None
        x = traj['end2_x']
        y = traj['end2_y']
        if 'time_endpoints' in traj:
            time = traj['time_endpoints']
    else:
        raise ValueError(f"tracking_point must be 'com', 'end1', or 'end2', got '{tracking_point}'")

    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return None

    fig, ax = plt.subplots(figsize=(14, 5))

    time_sec = (time - time[0]) * 60

    scatter = ax.scatter(x, y, c=time_sec, cmap='viridis', s=3, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Time (s)')


    left_circle = Circle(LEFT_CAVITY_CENTER, EXP_CAVITY_RADIUS_GEOM,
                         fill=False, color='red', linestyle='--', linewidth=1.5,
                         label=f'Cavity (r={EXP_CAVITY_RADIUS_GEOM} mm)')
    ax.add_patch(left_circle)
    ax.plot(*LEFT_CAVITY_CENTER, 'r+', markersize=10, markeredgewidth=2)

    right_circle = Circle(RIGHT_CAVITY_CENTER, EXP_CAVITY_RADIUS_GEOM,
                          fill=False, color='red', linestyle='--', linewidth=1.5)
    ax.add_patch(right_circle)
    ax.plot(*RIGHT_CAVITY_CENTER, 'r+', markersize=10, markeredgewidth=2)

    channel_y_half = 1.0
    channel_x_left = LEFT_CAVITY_CENTER[0] + EXP_CAVITY_RADIUS_GEOM
    channel_x_right = RIGHT_CAVITY_CENTER[0] - EXP_CAVITY_RADIUS_GEOM
    channel_rect = Rectangle(
        (channel_x_left, -channel_y_half),
        channel_x_right - channel_x_left,
        2 * channel_y_half,
        fill=False, color='blue', linestyle=':', linewidth=1.0,
        label='Channel (2 mm × 80 mm)'
    )
    ax.add_patch(channel_rect)

    ax.plot(x[0], y[0], 'go', markersize=10, markeredgecolor='black', markeredgewidth=1, label='Start')
    ax.plot(x[-1], y[-1], 'rs', markersize=10, markeredgecolor='black', markeredgewidth=1, label='End')

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    x_min, x_max = -50, 50
    y_min, y_max = -10, 10
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    worm_id = traj.get('worm_id', 'unknown')
    duration_min = (time[-1] - time[0])
    n_points = len(time)
    title = f"Worm: {worm_id} | {tracking_point} | Duration: {duration_min:.1f} min | N={n_points} pts"
    ax.set_title(title, fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        return None

    return fig


def _plot_full_task(args):
    traj, tracking_point, save_path = args
    try:
        result = plot_full_trajectory(traj, tracking_point=tracking_point, save_path=save_path)
        return result is None
    except Exception:
        return False


def generate_full_trajectory_plots(trajectories_by_T, tracking_point='end1',
                                   output_dir='FIGURES/DEBUG_FULL', n_workers=24):
    all_tasks = []

    for T, trajs in sorted(trajectories_by_T.items()):
        print(f"\n--- Temperature: {T}°C ({len(trajs)} worms) ---")
        temp_dir = os.path.join(output_dir, f"T{int(T)}")
        os.makedirs(temp_dir, exist_ok=True)

        for traj in trajs:
            worm_id = traj.get('worm_id', 'unknown')
            worm_id_clean = worm_id.replace('/', '_').replace(' ', '_')
            save_path = os.path.join(temp_dir, f"{worm_id_clean}.png")
            all_tasks.append((traj, tracking_point, save_path))

    print(f"\nGenerating {len(all_tasks)} full trajectory plots with {n_workers} workers...")

    success_count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_plot_full_task, task) for task in all_tasks]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Plotting"):
            if future.result():
                success_count += 1

    print(f"\nTotal: {success_count}/{len(all_tasks)} full trajectory plots saved to {output_dir}/")


def plot_all_points_by_temperature(trajectories_by_T, tracking_point='com', save_path=None):
    temperatures = sorted(trajectories_by_T.keys())
    n_temps = len(temperatures)

    fig, axes = plt.subplots(n_temps, 1, figsize=(14, 4*n_temps), sharex=True)
    if n_temps == 1:
        axes = [axes]

    for ax, T in zip(axes, temperatures):
        trajs = trajectories_by_T[T]

        all_x = []
        all_y = []

        for traj in trajs:
            if tracking_point == 'com':
                x = traj['x']
                y = traj['y']
            elif tracking_point == 'end1':
                if 'end1_x' not in traj or traj['end1_x'] is None:
                    continue
                x = traj['end1_x']
                y = traj['end1_y']
            elif tracking_point == 'end2':
                if 'end2_x' not in traj or traj['end2_x'] is None:
                    continue
                x = traj['end2_x']
                y = traj['end2_y']
            else:
                raise ValueError(f"Unknown tracking_point: {tracking_point}")

            valid = ~np.isnan(x) & ~np.isnan(y)
            all_x.extend(x[valid])
            all_y.extend(y[valid])

        all_x = np.array(all_x)
        all_y = np.array(all_y)

        print(f"T={T}°C: {len(trajs)} worms, {len(all_x)} points")

        ax.scatter(all_x, all_y, s=0.5, alpha=0.3, c='blue', rasterized=True)

        left_circle = Circle(LEFT_CAVITY_CENTER, EXP_CAVITY_RADIUS_GEOM,
                            fill=False, color='red', linestyle='-', linewidth=2)
        right_circle = Circle(RIGHT_CAVITY_CENTER, EXP_CAVITY_RADIUS_GEOM,
                             fill=False, color='red', linestyle='-', linewidth=2)
        ax.add_patch(left_circle)
        ax.add_patch(right_circle)

        ax.plot(*LEFT_CAVITY_CENTER, 'r+', markersize=12, markeredgewidth=2)
        ax.plot(*RIGHT_CAVITY_CENTER, 'r+', markersize=12, markeredgewidth=2)

        channel_y_half = 1.0
        channel_x_left = LEFT_CAVITY_CENTER[0] + EXP_CAVITY_RADIUS_GEOM
        channel_x_right = RIGHT_CAVITY_CENTER[0] - EXP_CAVITY_RADIUS_GEOM
        channel_rect = Rectangle(
            (channel_x_left, -channel_y_half),
            channel_x_right - channel_x_left,
            2 * channel_y_half,
            fill=False, color='red', linestyle='-', linewidth=2
        )
        ax.add_patch(channel_rect)

        if ax == axes[-1]:
            ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'T = {int(T)}°C  ({len(trajs)} worms, {len(all_x):,} points)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-52, 52)
        ax.set_ylim(-10, 10)

    fig.suptitle(f'All experimental points ({tracking_point})\n'
                 f'Cavity centers: x = ±{abs(LEFT_CAVITY_CENTER[0])} mm, '
                 f'radius = {EXP_CAVITY_RADIUS_GEOM} mm',
                 fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig


def align_to_right_cavity(x, y):
    valid = ~np.isnan(x) & ~np.isnan(y)
    x_v = x[valid]
    y_v = y[valid]

    if len(x_v) < 10:
        return None

    x_max = np.max(x_v)

    if x_max < 43.0:
        return None

    dx = 48.0 - x_max
    return (x_v + dx, y_v)


def _add_geometry(ax, right_only=False):
    right_circle = Circle(RIGHT_CAVITY_CENTER, EXP_CAVITY_RADIUS_GEOM,
                         fill=False, color='red', linestyle='-', linewidth=2)
    ax.add_patch(right_circle)
    ax.plot(*RIGHT_CAVITY_CENTER, 'r+', markersize=12, markeredgewidth=2)

    if not right_only:
        left_circle = Circle(LEFT_CAVITY_CENTER, EXP_CAVITY_RADIUS_GEOM,
                            fill=False, color='red', linestyle='-', linewidth=2)
        ax.add_patch(left_circle)
        ax.plot(*LEFT_CAVITY_CENTER, 'r+', markersize=12, markeredgewidth=2)

        channel_y_half = 1.0
        channel_x_left = LEFT_CAVITY_CENTER[0] + EXP_CAVITY_RADIUS_GEOM
        channel_x_right = RIGHT_CAVITY_CENTER[0] - EXP_CAVITY_RADIUS_GEOM
        channel_rect = Rectangle(
            (channel_x_left, -channel_y_half),
            channel_x_right - channel_x_left,
            2 * channel_y_half,
            fill=False, color='red', linestyle='-', linewidth=2
        )
        ax.add_patch(channel_rect)
        ax.set_xlim(-52, 52)
    else:
        ax.set_xlim(38, 52)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-10, 10)


def plot_aligned_points_by_temperature(trajectories_by_T, tracking_point='end1', save_path=None):
    temperatures = sorted(trajectories_by_T.keys())
    n_temps = len(temperatures)

    fig, axes = plt.subplots(n_temps, 2, figsize=(14, 4*n_temps))
    if n_temps == 1:
        axes = axes.reshape(1, 2)

    for row, T in enumerate(temperatures):
        trajs = trajectories_by_T[T]

        raw_x, raw_y = [], []
        aligned_x, aligned_y = [], []
        n_aligned_A = 0
        n_aligned_B = 0

        for traj in trajs:
            if tracking_point == 'end1':
                if 'end1_x' not in traj or traj['end1_x'] is None:
                    continue
                x = traj['end1_x'].copy()
                y = traj['end1_y'].copy()
            elif tracking_point == 'end2':
                if 'end2_x' not in traj or traj['end2_x'] is None:
                    continue
                x = traj['end2_x'].copy()
                y = traj['end2_y'].copy()
            else:
                continue

            valid = ~np.isnan(x) & ~np.isnan(y)
            x_valid = x[valid]
            y_valid = y[valid]

            raw_x.extend(x_valid)
            raw_y.extend(y_valid)

            result_A = align_to_right_cavity(x_valid, y_valid)
            if result_A is not None:
                aligned_x.extend(result_A[0])
                aligned_y.extend(result_A[1])
                n_aligned_A += 1

            x_mirror = -x_valid
            result_B = align_to_right_cavity(x_mirror, y_valid)
            if result_B is not None:
                aligned_x.extend(result_B[0])
                aligned_y.extend(result_B[1])
                n_aligned_B += 1

        raw_x = np.array(raw_x)
        raw_y = np.array(raw_y)
        aligned_x = np.array(aligned_x)
        aligned_y = np.array(aligned_y)

        print(f"T={T}°C: {len(trajs)} worms, {n_aligned_A} orig + {n_aligned_B} mirror aligned, {len(aligned_x):,} pts")

        ax = axes[row, 0]
        ax.scatter(raw_x, raw_y, s=0.5, alpha=0.3, c='blue', rasterized=True)
        _add_geometry(ax, right_only=False)
        ax.set_ylabel('y (mm)')
        if row == 0:
            ax.set_title('Raw data')
        ax.text(0.02, 0.98, f'T={int(T)}°C\n{len(trajs)} worms\n{len(raw_x):,} pts',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax = axes[row, 1]
        ax.scatter(aligned_x, aligned_y, s=0.5, alpha=0.3, c='green', rasterized=True)
        _add_geometry(ax, right_only=True)
        if row == 0:
            ax.set_title('Aligned to right cavity (A + mirror B)')
        ax.text(0.98, 0.98, f'T={int(T)}°C\n{n_aligned_A}+{n_aligned_B} aligned\n{len(aligned_x):,} pts',
                transform=ax.transAxes, fontsize=9, va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    axes[-1, 0].set_xlabel('x (mm)')
    axes[-1, 1].set_xlabel('x (mm)')

    fig.suptitle(f'Trajectory alignment to right cavity ({tracking_point})', fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close(fig)
    else:
        plt.show()

    return fig
