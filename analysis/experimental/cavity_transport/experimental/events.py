
import numpy as np
import os
from typing import Dict, List

CAVITY_X_MIN = 40.0
CAVITY_X_MAX = 48.0
CHANNEL_X_MAX = 40.0

ALIGNMENT_THRESHOLD = 43.0

MERGE_THRESHOLD_S = 0.0


def align_trajectory(trajectory: Dict) -> Dict:
    end1_x = trajectory.get('end1_x')
    time_endpoints = trajectory.get('time_endpoints')

    if end1_x is None or len(end1_x) == 0:
        return None

    com_x = trajectory.get('x')
    com_y = trajectory.get('y')
    time_com = trajectory.get('time')

    if com_x is None or len(com_x) == 0:
        return None

    x_max_end1 = np.nanmax(end1_x)
    x_min_end1 = np.nanmin(end1_x)

    reaches_right = x_max_end1 > ALIGNMENT_THRESHOLD
    reaches_left = x_min_end1 < -ALIGNMENT_THRESHOLD

    if not reaches_right and not reaches_left:
        return None

    need_mirror = reaches_left and (not reaches_right or abs(x_min_end1) > x_max_end1)

    if need_mirror:
        end1_x_proc = -end1_x.copy()
        com_x_proc = -com_x.copy()
    else:
        end1_x_proc = end1_x.copy()
        com_x_proc = com_x.copy()

    x_max_aligned = np.nanmax(end1_x_proc)
    dx = CAVITY_X_MAX - x_max_aligned

    end1_x_aligned = end1_x_proc + dx
    com_x_aligned = com_x_proc + dx

    aligned = {
        'worm_id': trajectory.get('worm_id', 'unknown'),
        'temperature': trajectory.get('temperature'),
        'end1_x_aligned': end1_x_aligned,
        'time_endpoints': time_endpoints,
        'com_x_aligned': com_x_aligned,
        'time_com': time_com,
        'end1_y': trajectory.get('end1_y'),
        'com_y': com_y,
        'mirrored': need_mirror,
        'dx_shift': dx,
    }

    return aligned


def detect_trapping_events(aligned_traj: Dict, detection_point: str = 'end1') -> List[Dict]:
    if detection_point == 'end1':
        x = aligned_traj.get('end1_x_aligned')
        time = aligned_traj.get('time_endpoints')
    else:
        x = aligned_traj.get('com_x_aligned')
        time = aligned_traj.get('time_com')

    if x is None or len(x) == 0 or time is None:
        return []

    worm_id = aligned_traj.get('worm_id', 'unknown')

    time_s = time * 60

    in_cavity = x > CAVITY_X_MIN

    events = []
    in_event = False
    t_start = None
    x_values = []

    for i in range(len(time)):
        if in_cavity[i] and not in_event:
            in_event = True
            t_start = time_s[i]
            x_values = [x[i]]
        elif in_cavity[i] and in_event:
            x_values.append(x[i])
        elif not in_cavity[i] and in_event:
            in_event = False
            t_end = time_s[i]
            duration = t_end - t_start
            events.append({
                't_start': t_start,
                't_end': t_end,
                'duration': duration,
                'x_mean': np.mean(x_values),
                'worm_id': worm_id,
            })

    if in_event and len(x_values) > 0:
        t_end = time_s[-1]
        duration = t_end - t_start
        events.append({
            't_start': t_start,
            't_end': t_end,
            'duration': duration,
            'x_mean': np.mean(x_values),
            'worm_id': worm_id,
        })

    if MERGE_THRESHOLD_S > 0 and len(events) > 1:
        events = merge_nearby_events(events)

    return events


def merge_nearby_events(events: List[Dict]) -> List[Dict]:
    if len(events) <= 1:
        return events

    merged = []
    current = events[0].copy()

    for i in range(1, len(events)):
        gap = events[i]['t_start'] - current['t_end']
        if gap < MERGE_THRESHOLD_S:
            current['t_end'] = events[i]['t_end']
            current['duration'] = current['t_end'] - current['t_start']
            current['x_mean'] = (current['x_mean'] + events[i]['x_mean']) / 2
        else:
            merged.append(current)
            current = events[i].copy()

    merged.append(current)

    return merged


def detect_translocation_events(aligned_traj: Dict, detection_point: str = 'end1') -> List[Dict]:
    if detection_point == 'end1':
        x = aligned_traj.get('end1_x_aligned')
        time = aligned_traj.get('time_endpoints')
    else:
        x = aligned_traj.get('com_x_aligned')
        time = aligned_traj.get('time_com')

    if x is None or len(x) == 0 or time is None:
        return []

    worm_id = aligned_traj.get('worm_id', 'unknown')
    time_s = time * 60

    CHANNEL_CENTER = 44.0
    in_left = x < CHANNEL_CENTER - 4
    in_right = x > CHANNEL_CENTER + 4

    events = []
    current_region = None
    t_leave = None
    start_region = None

    for i in range(len(time)):
        if in_left[i]:
            region = 'left'
        elif in_right[i]:
            region = 'right'
        else:
            region = 'channel'

        if current_region is None:
            current_region = region
            if region in ['left', 'right']:
                start_region = region
            continue

        if current_region in ['left', 'right'] and region == 'channel':
            t_leave = time_s[i]
            start_region = current_region

        elif current_region == 'channel' and region in ['left', 'right'] and t_leave is not None:
            t_enter = time_s[i]
            duration = t_enter - t_leave

            if start_region == 'left' and region == 'right':
                direction = 'L2R'
                event_type = 'success'
            elif start_region == 'right' and region == 'left':
                direction = 'R2L'
                event_type = 'success'
            elif start_region == region and start_region is not None:
                direction = f'{start_region[0].upper()}2{start_region[0].upper()}'
                event_type = 'attempt'
            else:
                direction = 'unknown'
                event_type = 'unknown'

            events.append({
                't_start': t_leave,
                't_end': t_enter,
                'duration': duration,
                'direction': direction,
                'type': event_type,
                'worm_id': worm_id,
            })

            t_leave = None

        current_region = region

    return events


def compute_trapping_statistics(events: List[Dict]) -> Dict:
    if len(events) == 0:
        return {
            'n_events': 0,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
        }

    durations = np.array([e['duration'] for e in events])

    return {
        'n_events': len(events),
        'mean': np.mean(durations),
        'median': np.median(durations),
        'std': np.std(durations),
        'min': np.min(durations),
        'max': np.max(durations),
    }


def compute_translocation_statistics(events: List[Dict]) -> Dict:
    success_events = [e for e in events if e['type'] == 'success']
    attempt_events = [e for e in events if e['type'] == 'attempt']

    if len(success_events) == 0:
        success_stats = {'mean': np.nan, 'median': np.nan, 'std': np.nan}
    else:
        durations = np.array([e['duration'] for e in success_events])
        success_stats = {
            'mean': np.mean(durations),
            'median': np.median(durations),
            'std': np.std(durations),
        }

    return {
        'n_success': len(success_events),
        'n_attempt': len(attempt_events),
        'success_mean': success_stats['mean'],
        'success_median': success_stats['median'],
        'success_std': success_stats.get('std', np.nan),
    }


def process_all_temperatures(trajs_by_temp: Dict, detection_point: str = 'end1') -> Dict:
    results = {}

    for temp, trajs in sorted(trajs_by_temp.items()):
        print(f"\n{'='*60}")
        print(f"Processing T = {temp}°C ({len(trajs)} worms)")
        print(f"{'='*60}")

        all_trapping = []
        all_translocation = []
        n_aligned = 0

        for traj in trajs:
            aligned = align_trajectory(traj)
            if aligned is None:
                continue

            n_aligned += 1

            trap_events = detect_trapping_events(aligned, detection_point)
            all_trapping.extend(trap_events)

            trans_events = detect_translocation_events(aligned, detection_point)
            all_translocation.extend(trans_events)

        trap_stats = compute_trapping_statistics(all_trapping)
        trans_stats = compute_translocation_statistics(all_translocation)

        results[temp] = {
            'trapping_events': all_trapping,
            'trapping_stats': trap_stats,
            'translocation_events': all_translocation,
            'translocation_stats': trans_stats,
            'n_worms': len(trajs),
            'n_aligned': n_aligned,
        }

        print(f"  Aligned: {n_aligned}/{len(trajs)} worms")
        print(f"  Trapping: {trap_stats['n_events']} events, "
              f"mean = {trap_stats['mean']:.1f} s, median = {trap_stats['median']:.1f} s")
        print(f"  Translocation: {trans_stats['n_success']} success, {trans_stats['n_attempt']} attempts")
        if trans_stats['n_success'] > 0:
            print(f"    Success: mean = {trans_stats['success_mean']:.1f} s, "
                  f"median = {trans_stats['success_median']:.1f} s")

    return results


def save_events(results: Dict, output_dir: str = 'DATA_EXP', detection_point: str = 'end1'):
    os.makedirs(output_dir, exist_ok=True)

    for temp, data in results.items():
        trap_file = os.path.join(output_dir, f'trapping_events_aligned_T{int(temp)}.txt')
        with open(trap_file, 'w') as f:
            f.write(f"# Trapping events (aligned with end1, detected with {detection_point})\n")
            f.write("# t_start(s), t_end(s), duration(s), x_mean(mm), worm_id\n")
            for e in data['trapping_events']:
                f.write(f"{e['t_start']:.1f}, {e['t_end']:.1f}, {e['duration']:.1f}, "
                       f"{e['x_mean']:.2f}, {e['worm_id']}\n")
        print(f"  Saved: {trap_file}")

        trans_file = os.path.join(output_dir, f'translocation_events_aligned_T{int(temp)}.txt')
        with open(trans_file, 'w') as f:
            f.write(f"# Translocation events (aligned with end1, detected with {detection_point})\n")
            f.write("# t_start(s), t_end(s), duration(s), direction, type, worm_id\n")
            for e in data['translocation_events']:
                f.write(f"{e['t_start']:.1f}, {e['t_end']:.1f}, {e['duration']:.1f}, "
                       f"{e['direction']}, {e['type']}, {e['worm_id']}\n")
        print(f"  Saved: {trans_file}")

    summary_file = os.path.join(output_dir, 'events_summary_aligned.txt')
    with open(summary_file, 'w') as f:
        f.write(f"# Events Summary (aligned with end1, detected with {detection_point})\n")
        f.write("# Cavity detection: x > 40 mm (after alignment)\n")
        f.write("#\n")
        f.write("# TRAPPING STATISTICS\n")
        f.write("# T(C)  N_events  mean(s)  median(s)  std(s)  min(s)  max(s)\n")
        for temp, data in sorted(results.items()):
            s = data['trapping_stats']
            f.write(f"  {int(temp):3d}     {s['n_events']:4d}     {s['mean']:5.1f}     "
                   f"{s['median']:5.1f}   {s['std']:5.1f}    {s['min']:5.1f}   {s['max']:6.1f}\n")
        f.write("#\n")
        f.write("# TRANSLOCATION STATISTICS\n")
        f.write("# T(C)  N_success  N_attempt  mean(s)  median(s)\n")
        for temp, data in sorted(results.items()):
            s = data['translocation_stats']
            f.write(f"  {int(temp):3d}     {s['n_success']:4d}        {s['n_attempt']:4d}        "
                   f"{s['success_mean']:5.1f}     {s['success_median']:5.1f}\n")

    print(f"  Saved: {summary_file}")


def print_summary(results: Dict, detection_point: str = 'end1'):
    print("\n" + "=" * 80)
    print(f"EVENTS SUMMARY (aligned with end1, detected with {detection_point})")
    print("=" * 80)

    print("\nTRAPPING (time in cavity)")
    print("-" * 80)
    print(f"{'T (°C)':<10} {'N_events':<10} {'τ_trap_mean (s)':<18} {'τ_trap_median (s)':<18}")
    print("-" * 80)

    for temp, data in sorted(results.items()):
        s = data['trapping_stats']
        print(f"{int(temp):<10} {s['n_events']:<10} {s['mean']:<18.1f} {s['median']:<18.1f}")

    print("\nTRANSLOCATION (cavity crossing)")
    print("-" * 80)
    print(f"{'T (°C)':<10} {'N_success':<12} {'N_attempt':<12} "
          f"{'τ_transloc_mean (s)':<20} {'τ_transloc_median (s)':<20}")
    print("-" * 80)

    for temp, data in sorted(results.items()):
        s = data['translocation_stats']
        print(f"{int(temp):<10} {s['n_success']:<12} {s['n_attempt']:<12} "
              f"{s['success_mean']:<20.1f} {s['success_median']:<20.1f}")

    print("=" * 80)
