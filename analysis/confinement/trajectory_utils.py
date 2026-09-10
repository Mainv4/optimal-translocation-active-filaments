
import glob
import os
import tempfile

import MDAnalysis as mda


def find_trajectory_file(path):
    patterns = [
        "PolymGravity_simu.lammpstrj",
        "PolymGravity.*.lammpstrj",
        "*.lammpstrj",
    ]

    for pattern in patterns:
        matches = glob.glob(os.path.join(path, pattern))
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return os.path.basename(matches[0])

    raise FileNotFoundError(f"No trajectory file found in {path}")


def find_valid_trajectory_end(filepath):
    with open(filepath, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()

        actual_end = file_size
        search_size = 100000

        while search_size < file_size:
            f.seek(max(0, file_size - search_size))
            end_chunk = f.read()

            found_data = False
            for i in range(len(end_chunk) - 1, -1, -1):
                if end_chunk[i] != 0:
                    actual_end = file_size - len(end_chunk) + i + 1
                    found_data = True
                    break

            if found_data:
                break

            search_size *= 10

        has_null_padding = actual_end < file_size

        f.seek(0)
        marker = b'ITEM: TIMESTEP'
        frame_positions = []

        chunk_size = 50 * 1024 * 1024
        pos = 0
        overlap = b''

        while pos < actual_end:
            f.seek(pos)
            chunk = overlap + f.read(min(chunk_size, actual_end - pos))

            idx = 0
            while True:
                marker_pos = chunk.find(marker, idx)
                if marker_pos == -1:
                    break
                abs_pos = pos - len(overlap) + marker_pos
                if abs_pos < actual_end:
                    frame_positions.append(abs_pos)
                idx = marker_pos + 1

            overlap = chunk[-1000:] if len(chunk) > 1000 else chunk
            pos += chunk_size

        n_total_frames = len(frame_positions)

        if n_total_frames == 0:
            return file_size, 0, False

        last_frame_start = frame_positions[-1]
        last_frame_size = actual_end - last_frame_start
        min_expected_size = 10000

        if last_frame_size < min_expected_size or has_null_padding:
            if n_total_frames >= 2:
                return frame_positions[-1], n_total_frames - 1, True
            else:
                return actual_end, 0, True

        return file_size, n_total_frames, False


def load_trajectory_robust(filepath, format="LAMMPSDUMP"):
    try:
        u = mda.Universe(filepath, format=format)
        n_frames = u.trajectory.n_frames
        return u, n_frames, False
    except (UnicodeDecodeError, KeyError, ValueError) as e:
        print(f"  Normal loading failed: {type(e).__name__}")

    cut_position, n_complete_frames, is_truncated = find_valid_trajectory_end(filepath)

    if not is_truncated and n_complete_frames > 1:
        print(f"  File appears valid but MDAnalysis failed - trying to skip last frame")
        with open(filepath, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            chunk_size = 5 * 1024 * 1024
            f.seek(max(0, file_size - chunk_size))
            chunk = f.read()
            marker = b'ITEM: TIMESTEP'
            positions = []
            idx = 0
            while True:
                pos = chunk.find(marker, idx)
                if pos == -1:
                    break
                positions.append(file_size - len(chunk) + pos)
                idx = pos + 1
            if len(positions) >= 2:
                cut_position = positions[-1]
                n_complete_frames = n_complete_frames - 1
                is_truncated = True

    if n_complete_frames == 0:
        raise RuntimeError(f"No complete frames found in file {filepath}")

    print(f"  Truncated file detected: {n_complete_frames} complete frames")
    print(f"  Using {cut_position:,} of {os.path.getsize(filepath):,} bytes")

    base_dir = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    temp_file = os.path.join(base_dir, base_name + '.tmp_fixed.lammpstrj')

    if os.path.exists(temp_file) and os.path.getsize(temp_file) == cut_position:
        print(f"  Using existing fixed trajectory: {temp_file}")
    else:
        print(f"  Creating fixed trajectory: {temp_file}")
        with open(filepath, 'rb') as src:
            with open(temp_file, 'wb') as dst:
                remaining = cut_position
                while remaining > 0:
                    chunk_size = min(10 * 1024 * 1024, remaining)
                    dst.write(src.read(chunk_size))
                    remaining -= chunk_size

    u = mda.Universe(temp_file, format=format)
    actual_frames = u.trajectory.n_frames
    print(f"  Successfully loaded {actual_frames} frames from truncated file")
    return u, actual_frames, True


def determine_and_validate_N(u, N_given=None, chain_length=40):
    total_atoms = len(u.atoms)

    if N_given is not None:
        expected_polymer_atoms = N_given * chain_length
        if expected_polymer_atoms <= total_atoms:
            print(f"Using given N={N_given} polymers with {chain_length} atoms each")
            if expected_polymer_atoms < total_atoms:
                print(f"Note: {total_atoms - expected_polymer_atoms} additional atoms (likely cavity/walls) will be ignored")
            return N_given
        else:
            raise ValueError(f"Given N={N_given} would require {expected_polymer_atoms} atoms, but only {total_atoms} available")

    N_calculated = total_atoms // chain_length
    if N_calculated > 0:
        polymer_atoms = N_calculated * chain_length
        extra_atoms = total_atoms - polymer_atoms
        print(f"Auto-detected: {N_calculated} polymers with {chain_length} atoms each")
        if extra_atoms > 0:
            print(f"Note: {extra_atoms} additional atoms (likely cavity/walls) will be ignored")
        return N_calculated
    else:
        raise ValueError(f"Total atoms ({total_atoms}) less than chain length ({chain_length})")


def read_trajectory_robust(path, name_of_trajectory_file=None):
    print("Reading trajectory data...")

    if name_of_trajectory_file is None:
        name_of_trajectory_file = find_trajectory_file(path)
        print(f"Auto-detected trajectory: {name_of_trajectory_file}")

    traj_path = os.path.join(path, name_of_trajectory_file)

    u, n_frames, was_truncated = load_trajectory_robust(traj_path)
    print(f"Loaded trajectory with {len(u.atoms)} atoms over {n_frames} frames")

    return u, was_truncated
