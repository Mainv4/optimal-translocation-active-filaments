
import numpy as np
from tqdm import tqdm

from .config import LEFT_CAVITY, RIGHT_CAVITY, CAVITY_Y, TIME_STEP_DURATION


def smooth_msd(msd_array, window=5):
    if len(msd_array) == 0 or window < 2:
        return msd_array

    time = msd_array[:, 0]
    msd = msd_array[:, 1]
    std_err = msd_array[:, 2] if msd_array.shape[1] > 2 else np.zeros_like(msd)

    kernel = np.ones(window) / window
    msd_smooth = np.convolve(msd, kernel, mode='same')

    half_w = window // 2
    msd_smooth[:half_w] = msd[:half_w]
    msd_smooth[-half_w:] = msd[-half_w:]

    return np.column_stack((time, msd_smooth, std_err))


def calculate_rotational_msd_for_one_segment(segment, max_lag_steps):
    x = segment[:, 1]
    y = segment[:, 2]
    cavity_id = segment[0, 3]

    cavity_center_x = (
        LEFT_CAVITY[0] + (LEFT_CAVITY[1] - LEFT_CAVITY[0]) / 2
        if cavity_id < 0
        else RIGHT_CAVITY[0] + (RIGHT_CAVITY[1] - RIGHT_CAVITY[0]) / 2
    )
    cavity_center_y = (CAVITY_Y[0] + CAVITY_Y[1]) / 2

    x_centered = x - cavity_center_x
    y_centered = y - cavity_center_y

    if cavity_id > 0:
        x_centered = -x_centered

    theta_0 = np.arctan2(y_centered, x_centered)
    theta = np.unwrap(theta_0)

    segment_len = len(theta)
    msd_segment = np.full(max_lag_steps, np.nan)

    for lag in range(1, min(segment_len, max_lag_steps + 1)):
        displacements = (theta[lag:] - theta[:-lag]) ** 2
        if len(displacements) > 0:
            msd_segment[lag - 1] = np.mean(displacements)

    return msd_segment


def calculate_translational_msd_for_one_segment(segment, max_lag_steps):
    x = segment[:, 1]
    y = segment[:, 2]

    segment_len = len(x)
    msd_segment = np.full(max_lag_steps, np.nan)

    for lag in range(1, min(segment_len, max_lag_steps + 1)):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        displacements = dx**2 + dy**2
        if len(displacements) > 0:
            msd_segment[lag - 1] = np.mean(displacements)

    return msd_segment


def calculate_rotational_msd(cavity_segments):
    print("Calculating rotational MSD...")

    if not cavity_segments:
        print("  No segments found")
        return np.array([]), []

    max_lag_steps = np.max([len(seg) for seg in cavity_segments]) - 1
    all_individual_msds = []

    for segment in tqdm(cavity_segments, desc="  Processing segments"):
        msd_one_seg = calculate_rotational_msd_for_one_segment(segment, max_lag_steps)
        all_individual_msds.append(msd_one_seg)

    all_individual_msds_array = np.array(all_individual_msds)
    average_msd = np.nanmean(all_individual_msds_array, axis=0)
    std_msd = np.nanstd(all_individual_msds_array, axis=0)
    count = np.sum(~np.isnan(all_individual_msds_array), axis=0)
    std_error = std_msd / np.sqrt(np.maximum(count, 1))

    time_lags = np.arange(1, max_lag_steps + 1) * TIME_STEP_DURATION

    valid_indices = ~np.isnan(average_msd)
    if not np.any(valid_indices):
        print("  Warning: Average rotational MSD is all NaN")
        return np.array([]), all_individual_msds

    last_valid = np.where(valid_indices)[0][-1] + 1
    msd_result = np.column_stack((
        time_lags[:last_valid],
        average_msd[:last_valid],
        std_error[:last_valid]
    ))

    print(f"  Rotational MSD computed: {len(msd_result)} time points")
    return msd_result, all_individual_msds


def calculate_translational_msd(cavity_segments, max_lag_steps=2000):
    print("Calculating translational MSD...")

    if not cavity_segments:
        print("  No segments found")
        return np.array([]), []

    all_individual_msds = []

    for segment in tqdm(cavity_segments, desc="  Processing segments"):
        msd_one_seg = calculate_translational_msd_for_one_segment(segment, max_lag_steps)
        all_individual_msds.append(msd_one_seg)

    all_individual_msds_array = np.array(all_individual_msds)
    average_msd = np.nanmean(all_individual_msds_array, axis=0)
    std_msd = np.nanstd(all_individual_msds_array, axis=0)
    count = np.sum(~np.isnan(all_individual_msds_array), axis=0)
    std_error = std_msd / np.sqrt(np.maximum(count, 1))

    time_lags = np.arange(1, max_lag_steps + 1) * TIME_STEP_DURATION

    valid_indices = ~np.isnan(average_msd)
    if not np.any(valid_indices):
        print("  Warning: Average translational MSD is all NaN")
        return np.array([]), all_individual_msds

    last_valid = np.where(valid_indices)[0][-1] + 1
    msd_result = np.column_stack((
        time_lags[:last_valid],
        average_msd[:last_valid],
        std_error[:last_valid]
    ))

    print(f"  Translational MSD computed: {len(msd_result)} time points")
    return msd_result, all_individual_msds
