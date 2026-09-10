import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from analyse_events.utilities import (read_mass_center_head_tail_x,
                                      set_plot_style)


def calculate_max_penetration(x_positions, direction, time, left_cavity, right_cavity):
    print("x_positions:", x_positions)
    print("left_cavity:", left_cavity[1])
    print("right_cavity:", right_cavity[0])
    if direction == "L2L":
        penetration = x_positions - left_cavity[1]
        print("left_cavity:", left_cavity)
        print("penetration:", penetration)
    elif direction == "R2R":
        penetration = x_positions - right_cavity[0]
        print("right_cavity:", right_cavity)
        print("penetration:", penetration)
    else:
        return None, None, None

    penetration = np.abs(penetration)

    max_index = np.argmax(penetration)
    max_penetration = penetration[max_index]

    if isinstance(time, np.ndarray):
        return max_penetration, time[0], time[max_index]
    else:
        return max_penetration, time, time


def calculate_translocation_times(x_positions, time, left_cavity, right_cavity):
    if len(x_positions) == 0 or len(time) == 0:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    trans_times = []
    start_times = []
    end_times = []
    trans_directions = []
    attempt_times = []
    attempt_directions = []
    attempt_penetrations = []

    current_cavity = None
    last_cavity = None
    initial_cavity = None
    transition_start = None
    in_transit = False

    for i in range(len(time)):
        x = x_positions[i]
        is_in_left = left_cavity[0] <= x <= left_cavity[1]
        is_in_right = right_cavity[0] <= x <= right_cavity[1]

        if is_in_left:
            current_cavity = "left"
        elif is_in_right:
            current_cavity = "right"
        else:
            if not in_transit and last_cavity is not None:
                in_transit = True
                transition_start = time[i]
                initial_cavity = last_cavity
            continue

        if in_transit and current_cavity is not None:
            duration = time[i] - transition_start

            if current_cavity != initial_cavity:
                movement = f"{initial_cavity[0].upper()}2{current_cavity[0].upper()}"
                trans_times.append(duration)
                start_times.append(transition_start)
                end_times.append(time[i])
                trans_directions.append(movement)
            else:
                movement = f"{initial_cavity[0].upper()}2{initial_cavity[0].upper()}"

                start_index = np.argmin(np.abs(time - transition_start))
                x_positions_attempt = x_positions[start_index : i + 1]
                time_attempt = time[start_index : i + 1]

                max_penetration, _, _ = calculate_max_penetration(
                    x_positions_attempt,
                    movement,
                    time_attempt,
                    left_cavity,
                    right_cavity,
                )

                attempt_times.append(duration)
                attempt_directions.append(movement)
                attempt_penetrations.append(max_penetration)

            in_transit = False
            initial_cavity = current_cavity
            transition_start = None

        last_cavity = current_cavity

    return (
        np.array(trans_times),
        np.array(start_times),
        np.array(end_times),
        np.array(trans_directions),
        np.array(attempt_times),
        np.array(attempt_directions),
        np.array(attempt_penetrations),
    )


def save_translocation_times(
    all_translocation_times, all_attempt_times, polymer_events, path
):
    data_dir = "DATA_translocation_time/"
    os.makedirs(data_dir, exist_ok=True)

    filename = path.replace("/", "__") + "__translocation_times.txt"
    filepath = os.path.join(data_dir, filename)

    total_time = max(
        max(polymer["ends"]) for polymer in polymer_events if len(polymer["ends"]) > 0
    )
    if total_time == 0:
        total_time = 1

    with open(filepath, "w") as f:
        f.write("# Translocation and attempt times (minutes)\n")
        f.write("# Format: time,polymer_id,direction,type,penetration_distance\n")

        for polymer in polymer_events:
            polymer_id = polymer["polymer_id"]
            for time, direction in zip(polymer["times"], polymer["directions"]):
                f.write(f"{time:.6f},{polymer_id},{direction},translocation,N/A\n")

        for polymer in polymer_events:
            polymer_id = polymer["polymer_id"]
            for time, direction, penetration_distance in zip(
                polymer["attempt_times"],
                polymer["attempt_directions"],
                polymer["attempt_penetration_distances"],
            ):
                f.write(
                    f"{time:.6f},{polymer_id},{direction},attempt,{penetration_distance:.6f}\n"
                )

        f.write("\n# Statistics\n")
        f.write(f"# Total observation time: {total_time:.6f} min\n")

        if len(all_translocation_times) > 0:
            l2r_count = sum(1 for d in polymer_events[0]["directions"] if d == "L2R")
            r2l_count = sum(1 for d in polymer_events[0]["directions"] if d == "R2L")
            f.write(f"\n# Successful Translocations:\n")
            f.write(f"# Total events: {len(all_translocation_times)}\n")
            f.write(f"# L2R events: {l2r_count}\n")
            f.write(f"# R2L events: {r2l_count}\n")
            f.write(f"# Mean time: {np.mean(all_translocation_times):.6f} min\n")
            f.write(f"# Median time: {np.median(all_translocation_times):.6f} min\n")
            f.write(f"# Max time: {np.max(all_translocation_times):.6f} min\n")

        f.write(f"\n# Failed Attempts:\n")
        f.write(f"# Total attempts: {len(all_attempt_times)}\n")
        if len(all_attempt_times) > 0:
            l2l_count = sum(
                1 for d in polymer_events[0]["attempt_directions"] if d == "L2L"
            )
            r2r_count = sum(
                1 for d in polymer_events[0]["attempt_directions"] if d == "R2R"
            )
            f.write(f"# L2L attempts: {l2l_count}\n")
            f.write(f"# R2R attempts: {r2r_count}\n")
            f.write(f"# Mean attempt time: {np.mean(all_attempt_times):.6f} min\n")
            f.write(f"# Median attempt time: {np.median(all_attempt_times):.6f} min\n")
            f.write(f"# Max attempt time: {np.max(all_attempt_times):.6f} min\n")
            f.write(
                f"# Mean penetration distance: {np.mean(polymer_events[0]['attempt_penetration_distances']):.6f} min\n"
            )
            f.write(
                f"# Median penetration distance: {np.median(polymer_events[0]['attempt_penetration_distances']):.6f} min\n"
            )


def plot_translocation_time_distribution(path, x1=None, x2=None):
    massCenter_data, _, _ = read_mass_center_head_tail_x(path)
    time, x_cm, _, N = massCenter_data

    dump_freq = 100000
    dt = 0.001
    tau = 1e-2
    time = time * dump_freq * dt * tau / 60

    x_cm = x_cm / 2 + 14.2 / 2

    left_cavity = (0, 8)
    right_cavity = (88, 96)

    print(f"\nDebug information:")
    print(f"X position range: {np.min(x_cm)} to {np.max(x_cm)} mm")
    print(f"Time range: {np.min(time)} to {np.max(time)} minutes")
    print(f"Number of polymers: {N}")
    print(f"Cavity bounds: left={left_cavity}, right={right_cavity}")

    all_translocation_times = []
    all_attempt_times = []
    all_directions = []
    polymer_events = []

    for i in range(N):
        (
            times,
            starts,
            ends,
            directions,
            attempt_times,
            attempt_directions,
            attempt_penetration_distances,
        ) = calculate_translocation_times(x_cm[:, i], time, left_cavity, right_cavity)

        polymer_event_data = {
            "polymer_id": i + 1,
            "times": times,
            "starts": starts,
            "ends": ends,
            "directions": directions,
            "attempt_times": attempt_times,
            "attempt_directions": attempt_directions,
            "x_positions": x_cm[:, i],
            "time": time,
            "attempt_penetration_distances": attempt_penetration_distances,
        }
        polymer_events.append(polymer_event_data)

        all_translocation_times.extend(times)
        all_attempt_times.extend(attempt_times)
        all_directions.extend(directions)

        print(f"\n=== Translocation events for Polymer {i + 1} ===")
        if len(times) > 0:
            for j, (duration, start, end, direction) in enumerate(
                zip(times, starts, ends, directions), 1
            ):
                print(
                    f"Event {j}: {direction}, Duration = {duration:.2f} min "
                    f"(t = {start:.2f} to {end:.2f} min)"
                )
        else:
            print("No translocation events")

    if not all_translocation_times:
        print("\nWarning: No translocation events detected!")
        print("This might indicate an issue with the detection parameters or data.")
        print(
            f"Please check the cavity bounds: left={left_cavity}, right={right_cavity}"
        )

        base_path = "FIGURES/translocation_times/" + path + "/"
        os.makedirs(base_path, exist_ok=True)

        for scale in ["linear", "log"]:
            fig = plt.figure(figsize=(8, 6))
            plt.xlabel(r"$\tau_t$ (min)", fontsize=14)
            plt.ylabel(r"$P(\tau_t)$", fontsize=14)
            if scale == "log":
                plt.yscale("log")
                plt.ylim(1e-5, 1.2)
            plt.grid(True, alpha=0.3)
            plt.xlim(-0.5, 15)

            plt.text(
                0.5,
                0.5,
                "No translocation events detected",
                transform=plt.gca().transAxes,
                verticalalignment="center",
                horizontalalignment="center",
                bbox=dict(facecolor="white", alpha=0.8),
                fontsize=12,
            )

            scale_suffix = "_log" if scale == "log" else "_linear"
            fig.savefig(f"{base_path}translocation_time_distribution{scale_suffix}.png")
            plt.close(fig)

        return

    save_translocation_times(
        all_translocation_times, all_attempt_times, polymer_events, path
    )


    base_path = "FIGURES/translocation_times/" + path + "/"
    os.makedirs(base_path, exist_ok=True)

    set_plot_style()

    figs = []
    for scale in ["linear", "log"]:
        fig = plt.figure(figsize=(8, 6))
        figs.append(fig)

        plt.hist(all_translocation_times, bins=30, density=True, alpha=0.7)
        plt.xlabel(r"$\tau_t$ (min)", fontsize=14)
        plt.ylabel(r"$P(\tau_t)$", fontsize=14)
        if scale == "log":
            plt.yscale("log")
            plt.ylim(1e-5, 1.2)

        plt.grid(True, alpha=0.3)
        plt.xlim(-0.5, 15)

        stats_text = f"N events: {len(all_translocation_times)}\n"
        stats_text += f"Mean: {np.mean(all_translocation_times):.2f} min\n"
        stats_text += f"Median: {np.median(all_translocation_times):.2f} min\n"
        stats_text += f"Max: {np.max(all_translocation_times):.2f} min\n"
        stats_text += f'L2R: {sum(d=="L2R" for d in all_directions)}\n'
        stats_text += f'R2L: {sum(d=="R2L" for d in all_directions)}'

        plt.text(
            0.95,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.8),
            fontsize=12,
        )

        scale_suffix = "_log" if scale == "log" else "_linear"
        fig.savefig(f"{base_path}translocation_time_distribution{scale_suffix}.png")

    plt.close("all")

    print(f"\nOverall statistics:")
    print(f"Total number of translocation events: {len(all_translocation_times)}")
    print(f"Mean translocation time: {np.mean(all_translocation_times):.2f} min")
    print(f"Median translocation time: {np.median(all_translocation_times):.2f} min")
    print(f"Maximum translocation time: {np.max(all_translocation_times):.2f} min")
    print(f"Left to Right events: {sum(d=='L2R' for d in all_directions)}")
    print(f"Right to Left events: {sum(d=='R2L' for d in all_directions)}")
