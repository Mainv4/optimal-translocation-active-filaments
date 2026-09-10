
import sys
import pandas as pd
import numpy as np
import os
import glob
import re
from pathlib import Path
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).parent / "exp_analyse_scripts"))

TOLERANCE = 0.05


def extract_parameters_from_filename(filename):
    import re
    match = re.search(r'Pe_([\d.]+)_T_([\d.]+)_k_([\d.]+)', filename)
    if match:
        pe = match.group(1).rstrip('.')
        t = match.group(2).rstrip('.')
        k = match.group(3).rstrip('.')
        return float(pe), float(t), float(k)
    return None, None, None


def find_closest_match(target_params, available_params_dict, tolerance=TOLERANCE):
    target_pe, target_t, target_k = target_params

    best_data = None
    best_dist = float('inf')

    for (pe, t, k), data in available_params_dict.items():
        if (abs(pe - target_pe) <= tolerance and
            abs(t - target_t) <= tolerance and
            abs(k - target_k) <= tolerance):
            dist = abs(pe - target_pe) + abs(t - target_t) + abs(k - target_k)
            if dist < best_dist:
                best_dist = dist
                best_data = data

    return best_data


def load_free_space_entropy():
    csv_path = "shannon_entropy_free_space.csv"

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return {}

    df = pd.read_csv(csv_path)

    entropy_dict = {}
    for _, row in df.iterrows():
        key = (row['Pe'], row['T'], row['k'])
        entropy_dict[key] = row['shannon_entropy']

    print(f"Loaded {len(entropy_dict)} free space entropy values")
    return entropy_dict


def calculate_lp_r2_from_bond_corr(bond_corr_file):
    try:
        data = np.loadtxt(bond_corr_file, comments='#')

        if len(data) < 3:
            return np.nan

        s = data[:, 0]
        C_s = data[:, 2]

        positive = C_s > 0
        if not np.any(positive):
            return np.nan

        zero_crossings = np.where(np.diff(positive.astype(int)) < 0)[0]
        if len(zero_crossings) > 0:
            idx_max = zero_crossings[0] + 1
        else:
            idx_max = len(s)

        s_fit = s[:idx_max]
        C_fit = C_s[:idx_max]

        if len(s_fit) < 3:
            return np.nan

        def exp_decay(s, A, lp):
            return A * np.exp(-s / lp)

        p0 = [1.0, 5.0]

        popt, _ = curve_fit(exp_decay, s_fit, C_fit, p0=p0, maxfev=5000)

        C_pred = exp_decay(s_fit, *popt)
        ss_res = np.sum((C_fit - C_pred) ** 2)
        ss_tot = np.sum((C_fit - np.mean(C_fit)) ** 2)

        if ss_tot == 0:
            return np.nan

        r2 = 1 - (ss_res / ss_tot)

        return r2

    except Exception as e:
        return np.nan


def load_free_space_lp():
    csv_path = "DATA_LP_COMPLETE/complete_lp_data.csv"
    bond_corr_dir = "DATA_BOND_CORRELATION"

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return {}

    df = pd.read_csv(csv_path)

    lp_dict = {}
    for _, row in df.iterrows():
        key = (row['Pe'], row['T'], row['k'])
        pe, t, k = key

        bond_corr_file = os.path.join(bond_corr_dir, f"Pe_{pe}_T_{t}_k_{k}_bond_corr.dat")
        if os.path.exists(bond_corr_file):
            r2 = calculate_lp_r2_from_bond_corr(bond_corr_file)
        else:
            r2 = np.nan

        lp_dict[key] = (row['lp_over_l'], row['intercept_error'], r2)

    print(f"Loaded {len(lp_dict)} free space persistence length values with errors and R²")
    return lp_dict


def _extract_tau_decorr_from_file(filepath):
    time_factor = 1.0
    with open(filepath, 'r') as f:
        for line in f:
            if 'Time factor:' in line:
                time_factor = float(line.split(':')[1].strip().split()[0])
            elif not line.startswith('#'):
                break

    autocorr = np.loadtxt(filepath)
    if autocorr.size == 0:
        return None, None

    times = np.arange(len(autocorr)) * time_factor
    threshold = 1.0 / np.e

    above_threshold = autocorr > threshold
    if np.any(above_threshold) and not np.all(above_threshold):
        crossing_indices = np.where(np.diff(above_threshold.astype(int)) == -1)[0]
        if len(crossing_indices) > 0:
            idx = crossing_indices[0]
            t1, t2 = times[idx], times[idx + 1]
            c1, c2 = autocorr[idx], autocorr[idx + 1]
            tau_decorr = t1 + (threshold - c1) * (t2 - t1) / (c2 - c1)
            tau_decorr_error = (t2 - t1) / 2.0
            return tau_decorr, tau_decorr_error

    return None, None


def load_decorrelation_times():
    data_dir = "DATA_E2E_AUTOCORR"

    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found")
        return {}

    full_files = glob.glob(os.path.join(data_dir, "Pe_*_T_*_k_*_E2E_AUTOCORR.dat"))
    unit_files = glob.glob(os.path.join(data_dir, "Pe_*_T_*_k_*_E2E_UNIT_VEC_AUTOCORR.dat"))

    full_by_params = {}
    for fp in full_files:
        pe, t, k = extract_parameters_from_filename(os.path.basename(fp))
        if pe is not None:
            full_by_params[(pe, t, k)] = fp

    unit_by_params = {}
    for fp in unit_files:
        pe, t, k = extract_parameters_from_filename(os.path.basename(fp))
        if pe is not None:
            unit_by_params[(pe, t, k)] = fp

    all_params = set(full_by_params.keys()) | set(unit_by_params.keys())
    tau_decorr_dict = {}

    for params in all_params:
        tau_full, tau_full_err = np.nan, np.nan
        tau_unit, tau_unit_err = np.nan, np.nan

        if params in full_by_params:
            try:
                tf, tfe = _extract_tau_decorr_from_file(full_by_params[params])
                if tf is not None:
                    tau_full, tau_full_err = tf, tfe
            except Exception as e:
                print(f"Error loading full {os.path.basename(full_by_params[params])}: {e}")

        if params in unit_by_params:
            try:
                tu, tue = _extract_tau_decorr_from_file(unit_by_params[params])
                if tu is not None:
                    tau_unit, tau_unit_err = tu, tue
            except Exception as e:
                print(f"Error loading unit {os.path.basename(unit_by_params[params])}: {e}")

        if not (np.isnan(tau_full) and np.isnan(tau_unit)):
            tau_decorr_dict[params] = (tau_full, tau_full_err, tau_unit, tau_unit_err)

    n_full = sum(1 for v in tau_decorr_dict.values() if not np.isnan(v[0]))
    n_unit = sum(1 for v in tau_decorr_dict.values() if not np.isnan(v[2]))
    print(f"Loaded {len(tau_decorr_dict)} decorrelation time entries (full: {n_full}, unit: {n_unit})")
    return tau_decorr_dict


def load_confinement_entropy():
    csv_path = "../CONFINEMENT/N_40/conformational_metrics_H_values_all_classifications.csv"

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return {}

    df = pd.read_csv(csv_path)

    df_cavity = df[df['classification'] == 'cavity']

    entropy_dict = {}
    for _, row in df_cavity.iterrows():
        key = (row['Pe'], row['T'], row['k'])
        entropy_dict[key] = row['H']

    print(f"Loaded {len(entropy_dict)} confinement entropy values (cavity)")
    return entropy_dict


def load_confinement_lp():
    csv_path = "../CONFINEMENT/N_40/DATA_LP_COMPLETE/complete_lp_data.csv"
    bond_corr_dir = "../CONFINEMENT/N_40/DATA_BOND_CORRELATION"

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return {}

    df = pd.read_csv(csv_path)

    lp_dict = {}
    for _, row in df.iterrows():
        key = (row['Pe'], row['T'], row['k'])
        pe, t, k = key

        bond_corr_file = os.path.join(bond_corr_dir, f"Pe_{pe}_T_{t}_k_{k}_bond_corr.dat")
        if os.path.exists(bond_corr_file):
            r2 = calculate_lp_r2_from_bond_corr(bond_corr_file)
        else:
            r2 = np.nan

        lp_dict[key] = (row['lp_over_l'], row['intercept_error'], r2)

    print(f"Loaded {len(lp_dict)} confinement persistence length values with errors and R²")
    return lp_dict


def load_trapping_times():
    data_dir = "../CONFINEMENT/N_40/DATA_trapping_time"

    if not os.path.exists(data_dir):
        print(f"Warning: {data_dir} not found")
        return {}

    trapping_files = glob.glob(os.path.join(data_dir, "*trapping_times.txt"))

    ttrap_dict = {}

    for filepath in trapping_files:
        filename = os.path.basename(filepath)
        pe, t, k = extract_parameters_from_filename(filename)

        if pe is None:
            continue

        try:
            times = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith('#') or 'Mean time:' in line or 'Median' in line or 'Max' in line or 'Total events' in line:
                        continue
                    parts = line.strip().split(',')
                    if len(parts) >= 1:
                        try:
                            time = float(parts[0])
                            times.append(time)
                        except ValueError:
                            continue

            MIN_TRAPPING_EVENTS = 5
            if len(times) >= MIN_TRAPPING_EVENTS:
                mean_time = np.mean(times)
                std_time = np.std(times, ddof=1) if len(times) > 1 else 0.0
                ttrap_dict[(pe, t, k)] = (mean_time, std_time)
            elif len(times) > 0:
                print(f"  Skipping {filename}: only {len(times)} events (min={MIN_TRAPPING_EVENTS})")

        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue

    print(f"Loaded {len(ttrap_dict)} trapping time values with std")
    return ttrap_dict


def load_tau_decorr_cavity():
    tau_decorr_cavity_dict = {}

    for npz_dir in ["../CONFINEMENT/N_40/DATA_AUTOCORR", "../CONFINEMENT/N_40/DATA_ROTATIONAL"]:
        if not os.path.exists(npz_dir):
            continue
        autocorr_files = glob.glob(os.path.join(npz_dir, "Pe_*.npz"))
        autocorr_files += glob.glob(os.path.join(npz_dir, "TEST_*.npz"))

        for filepath in autocorr_files:
            pe, t, k = extract_parameters_from_filename(os.path.basename(filepath))
            if pe is None:
                continue
            params = (pe, t, k)
            if params in tau_decorr_cavity_dict:
                continue
            try:
                data = np.load(filepath)
                tau_unit_val = data.get('tau_decorr', None)
                tau_full_val = data.get('tau_decorr_full', None)

                tau_unit = float(tau_unit_val) if tau_unit_val is not None and not np.isnan(tau_unit_val) else np.nan
                tau_full = float(tau_full_val) if tau_full_val is not None and not np.isnan(tau_full_val) else np.nan

                if not np.isnan(tau_unit) or not np.isnan(tau_full):
                    tau_decorr_cavity_dict[params] = (tau_full, tau_unit)
            except Exception as e:
                print(f"Error loading {os.path.basename(filepath)}: {e}")

    n_full = sum(1 for v in tau_decorr_cavity_dict.values() if not np.isnan(v[0]))
    n_unit = sum(1 for v in tau_decorr_cavity_dict.values() if not np.isnan(v[1]))
    print(f"Loaded {len(tau_decorr_cavity_dict)} tau_decorr_cavity entries (full: {n_full}, unit: {n_unit})")
    return tau_decorr_cavity_dict


def load_rotational_data():
    summary_path = "../CONFINEMENT/N_40/DATA_ROTATIONAL/summary.txt"

    if not os.path.exists(summary_path):
        print(f"Warning: {summary_path} not found")
        return {}

    PE_PREFERRED = {
        (0.1, 0.1, 0.4),
        (0.5, 0.1, 0.4),
    }

    rotational_dict = {}

    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split()
            if len(parts) >= 8:
                try:
                    directory = parts[0]
                    pe = float(parts[1])
                    t = float(parts[2])
                    k = float(parts[3])
                    n_rot_raw = parts[4]
                    n_rot = float(n_rot_raw) if n_rot_raw != 'N/A' else float('nan')
                    tau_trans = float(parts[5])
                    plateau = float(parts[6])
                    tau_rot = float(parts[7])

                    key = (pe, t, k)
                    is_test = directory.startswith('TEST_')

                    if key in PE_PREFERRED:
                        if is_test and key in rotational_dict:
                            continue

                    rotational_dict[key] = (n_rot, tau_trans, plateau, tau_rot)
                except (ValueError, IndexError) as e:
                    print(f"Skipping line: {line[:50]}... ({e})")

    print(f"Loaded {len(rotational_dict)} rotational number values (N_rot = τ_rot/τ_trans)")
    return rotational_dict


def load_translocation_data():
    csv_path = "../CONFINEMENT/N_40/DATA_translocation_time/translocation_rates_summary_sim.csv"

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return {}

    df = pd.read_csv(csv_path, skipinitialspace=True)

    df.columns = df.columns.str.strip()

    transloc_dict = {}
    for _, row in df.iterrows():
        key = (row['Pe'], row['T'], row['kappa'])
        transloc_dict[key] = (
            row['Rate_ev_per_hour'],
            row['Success_rate_percent'],
            row['N_success'],
            row['N_attempts'],
            row['Total']
        )

    print(f"Loaded {len(transloc_dict)} translocation statistics")
    return transloc_dict


def load_diffusion_coefficients():
    csv_path = "DATA_DIFFUSION/diffusion_coefficients.csv"

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return {}

    df = pd.read_csv(csv_path)

    dlong_dict = {}
    for _, row in df.iterrows():
        key = (row['Pe'], row['T'], row['k'])
        dlong_dict[key] = (
            row['D_long'],
            row['D_error'],
            row['R_squared'],
            row['n_points']
        )

    print(f"Loaded {len(dlong_dict)} diffusion coefficient values with error bars")
    return dlong_dict


def load_free_space_lp_individual():
    npy_dir = "DATA_Corr_Rg_Re"

    if not os.path.exists(npy_dir):
        print(f"Warning: {npy_dir} not found")
        return {}

    print(f"Loading individual L_p data from {npy_dir}/...")

    npy_files = glob.glob(os.path.join(npy_dir, "Pe_*.npy"))

    if not npy_files:
        print(f"Warning: No .npy files found in {npy_dir}")
        return {}

    lp_individual_dict = {}
    processed_count = 0
    failed_count = 0

    for npy_file in npy_files:
        try:
            params = extract_parameters_from_filename(npy_file.replace('.npy', '_bond_corr.dat'))
            if params is None:
                failed_count += 1
                continue

            pe, t, k = params

            data = np.load(npy_file)
            lp_vals = data[:, :, 3]

            lp_clean = lp_vals[~np.isnan(lp_vals)]

            if len(lp_clean) > 0:
                lp_mean = np.mean(lp_clean)
                lp_std = np.std(lp_clean)
                lp_individual_dict[(pe, t, k)] = (lp_mean, lp_std)
                processed_count += 1
            else:
                failed_count += 1

        except Exception as e:
            failed_count += 1
            continue

    print(f"Loaded {len(lp_individual_dict)} free space individual lp values with std ({processed_count} processed, {failed_count} failed)")
    return lp_individual_dict


def load_confinement_lp_individual():
    npy_dir = Path(__file__).parent.parent / "CONFINEMENT" / "N_40" / "DATA_Corr_Rg_Re"
    cavity_file = Path(__file__).parent.parent / "CONFINEMENT" / "N_40" / "DATA_Cavity" / "Pe_0.01_T_0.01_k_0.02____CavityCoordinates.dat"

    if not npy_dir.exists():
        print(f"Warning: {npy_dir} not found")
        return {}

    try:
        data_cavity = np.loadtxt(cavity_file, skiprows=1)
        x_min_cavity = np.min(data_cavity[:, 0])
        x_max_cavity = np.max(data_cavity[:, 0])
    except Exception as e:
        print(f"Warning: Could not load cavity boundaries: {e}")
        return {}

    npy_files = list(npy_dir.glob("*Pe_*.npy"))
    if not npy_files:
        print(f"Warning: No .npy files found in {npy_dir}")
        return {}

    print(f"Loading confinement lp_individual from {npy_dir}/...")

    lp_individual_dict = {}
    processed_count = 0
    failed_count = 0

    for npy_file in npy_files:
        try:
            params = extract_parameters_from_filename(str(npy_file).replace('.npy', '_bond_corr.dat'))
            if params is None:
                failed_count += 1
                continue

            data = np.load(npy_file)

            if data.shape[2] < 5:
                failed_count += 1
                continue

            x_cm = data[:, :, 3].flatten()
            lp_vals = data[:, :, 4].flatten()

            x_shift = x_cm + abs(x_min_cavity)
            x_max_c = x_max_cavity + abs(x_min_cavity)
            x_norm = x_shift / x_max_c
            x_norm[x_norm > 0.5] = 1 - x_norm[x_norm > 0.5]

            cavity_mask = (x_norm <= 0.083) & ~np.isnan(lp_vals) & ~np.isnan(x_cm)
            lp_cavity = lp_vals[cavity_mask]

            if len(lp_cavity) > 0:
                lp_mean = np.mean(lp_cavity)
                lp_std = np.std(lp_cavity)
                lp_individual_dict[params] = (lp_mean, lp_std)
                processed_count += 1
            else:
                failed_count += 1

        except Exception as e:
            failed_count += 1
            continue

    print(f"Loaded {len(lp_individual_dict)} confinement lp_individual values - cavity only ({processed_count} processed, {failed_count} failed)")
    return lp_individual_dict


def load_msd_rmse():
    import msd_matching as msd_mod

    sim_files = sorted(msd_mod.SIM_DATA_DIR.glob("*_MSD_CM.dat"))
    print(f"  MSD RMSE: found {len(sim_files)} simulation MSD files")

    exp_data = {}
    for temp in ["10C", "20C", "30C"]:
        exp = msd_mod.load_experimental_msd(temp)
        if exp is not None:
            exp_data[temp] = exp

    rmse_dict = {}
    for f in sim_files:
        params = msd_mod.parse_filename(f.name)
        if params is None:
            continue

        times_s, msd_s = msd_mod.load_simulation_msd(f)
        if times_s is None:
            continue

        key = (params["Pe"], params["T"], params["k"])
        rmse_10, rmse_20, rmse_30 = np.nan, np.nan, np.nan

        for temp in ["10C", "20C", "30C"]:
            if temp in exp_data:
                rmse = msd_mod.compute_rmse_log(
                    exp_data[temp]["times"], exp_data[temp]["msd"],
                    times_s, msd_s, *msd_mod.MATCH_INTERVAL)
                if temp == "10C":
                    rmse_10 = rmse
                elif temp == "20C":
                    rmse_20 = rmse
                else:
                    rmse_30 = rmse

        rmse_dict[key] = (rmse_10, rmse_20, rmse_30)

    print(f"  MSD RMSE: computed for {len(rmse_dict)} parameter sets")
    return rmse_dict


def compile_freespace_data():
    print("Loading all data sources...\n")

    h_free_dict = load_free_space_entropy()
    lp_free_dict = load_free_space_lp()
    lp_free_individual_dict = load_free_space_lp_individual()
    tau_decorr_dict = load_decorrelation_times()
    dlong_dict = load_diffusion_coefficients()
    h_conf_dict = load_confinement_entropy()
    lp_conf_dict = load_confinement_lp()
    lp_conf_individual_dict = load_confinement_lp_individual()
    ttrap_dict = load_trapping_times()
    tau_decorr_cavity_dict = load_tau_decorr_cavity()
    transloc_dict = load_translocation_data()
    msd_rmse_dict = load_msd_rmse()

    base_params = sorted(h_free_dict.keys())

    print(f"\n[FREE SPACE REFERENCE]")
    print(f"Using {len(base_params)} H_free points as reference base")
    print(f"Fuzzy matching all other observables (tolerance={TOLERANCE})")

    data_rows = []

    match_stats = {
        'lp_free': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'lp_free_individual': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'tau_decorr_free': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'D_long': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'H_conf': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'lp_conf': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'lp_conf_individual': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'ttrap': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'tau_decorr_cav': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'transloc': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'msd_rmse': {'exact': 0, 'fuzzy': 0, 'none': 0}
    }

    for params in base_params:
        pe, t, k = params

        h_free_val = h_free_dict[params]

        lp_free_val = find_closest_match(params, lp_free_dict, TOLERANCE)
        lp_free_individual_val = find_closest_match(params, lp_free_individual_dict, TOLERANCE)
        tau_decorr_val = find_closest_match(params, tau_decorr_dict, TOLERANCE)
        dlong_val = find_closest_match(params, dlong_dict, TOLERANCE)

        h_conf_val = find_closest_match(params, h_conf_dict, TOLERANCE)
        lp_conf_val = find_closest_match(params, lp_conf_dict, TOLERANCE)
        lp_conf_individual_val = find_closest_match(params, lp_conf_individual_dict, TOLERANCE)
        ttrap_val = find_closest_match(params, ttrap_dict, TOLERANCE)
        tau_decorr_cavity_val = find_closest_match(params, tau_decorr_cavity_dict, TOLERANCE)
        transloc_val = find_closest_match(params, transloc_dict, TOLERANCE)
        msd_rmse_val = find_closest_match(params, msd_rmse_dict, TOLERANCE)

        for name, val, source_dict in [
            ('lp_free', lp_free_val, lp_free_dict),
            ('lp_free_individual', lp_free_individual_val, lp_free_individual_dict),
            ('tau_decorr_free', tau_decorr_val, tau_decorr_dict),
            ('D_long', dlong_val, dlong_dict),
            ('H_conf', h_conf_val, h_conf_dict),
            ('lp_conf', lp_conf_val, lp_conf_dict),
            ('lp_conf_individual', lp_conf_individual_val, lp_conf_individual_dict),
            ('ttrap', ttrap_val, ttrap_dict),
            ('tau_decorr_cav', tau_decorr_cavity_val, tau_decorr_cavity_dict),
            ('transloc', transloc_val, transloc_dict),
            ('msd_rmse', msd_rmse_val, msd_rmse_dict)
        ]:
            if val is not None:
                if params in source_dict:
                    match_stats[name]['exact'] += 1
                else:
                    match_stats[name]['fuzzy'] += 1
            else:
                match_stats[name]['none'] += 1

        if dlong_val is not None:
            d_long, d_error, d_r2, d_n = dlong_val
        else:
            d_long, d_error, d_r2, d_n = np.nan, np.nan, np.nan, np.nan

        if lp_free_val is not None:
            lp_free, lp_free_err, lp_free_r2 = lp_free_val
        else:
            lp_free, lp_free_err, lp_free_r2 = np.nan, np.nan, np.nan

        if lp_conf_val is not None:
            lp_conf, lp_conf_err, lp_conf_r2 = lp_conf_val
        else:
            lp_conf, lp_conf_err, lp_conf_r2 = np.nan, np.nan, np.nan

        if lp_free_individual_val is not None:
            lp_free_indiv, lp_free_indiv_std = lp_free_individual_val
        else:
            lp_free_indiv, lp_free_indiv_std = np.nan, np.nan

        if tau_decorr_val is not None:
            tau_free_full, tau_free_full_err, tau_free_unit, tau_free_unit_err = tau_decorr_val
        else:
            tau_free_full, tau_free_full_err = np.nan, np.nan
            tau_free_unit, tau_free_unit_err = np.nan, np.nan

        if tau_decorr_cavity_val is not None:
            tau_cav_full, tau_cav_unit = tau_decorr_cavity_val
        else:
            tau_cav_full, tau_cav_unit = np.nan, np.nan

        if ttrap_val is not None:
            ttrap, ttrap_std = ttrap_val
        else:
            ttrap, ttrap_std = np.nan, np.nan

        if transloc_val is not None:
            transloc_rate, transloc_success_rate, transloc_n_success, transloc_n_attempts, transloc_total = transloc_val
        else:
            transloc_rate, transloc_success_rate, transloc_n_success, transloc_n_attempts, transloc_total = np.nan, np.nan, np.nan, np.nan, np.nan

        if msd_rmse_val is not None:
            rmse_d10, rmse_d20, rmse_d30 = msd_rmse_val
        else:
            rmse_d10, rmse_d20, rmse_d30 = np.nan, np.nan, np.nan

        row = {
            'Pe': pe,
            'T': t,
            'kappa': k,
            'H_free': h_free_val,
            'lp_free': lp_free,
            'lp_free_error': lp_free_err,
            'lp_free_R2': lp_free_r2,
            'lp_free_individual': lp_free_indiv,
            'lp_free_individual_std': lp_free_indiv_std,
            'tau_decorr_free_full': tau_free_full,
            'tau_decorr_free_full_error': tau_free_full_err,
            'tau_decorr_free_unit': tau_free_unit,
            'tau_decorr_free_unit_error': tau_free_unit_err,
            'D_long': d_long,
            'D_long_error': d_error,
            'D_long_R2': d_r2,
            'D_long_n_points': d_n,
            'H_conf': h_conf_val if h_conf_val is not None else np.nan,
            'lp_conf': lp_conf,
            'lp_conf_error': lp_conf_err,
            'lp_conf_R2': lp_conf_r2,
            'lp_conf_individual': lp_conf_individual_val if lp_conf_individual_val is not None else np.nan,
            'ttrap': ttrap,
            'ttrap_std': ttrap_std,
            'tau_decorr_cavity_full': tau_cav_full,
            'tau_decorr_cavity_unit': tau_cav_unit,
            'transloc_rate_per_hour': transloc_rate,
            'transloc_success_rate': transloc_success_rate,
            'transloc_n_success': transloc_n_success,
            'transloc_n_attempts': transloc_n_attempts,
            'transloc_total_events': transloc_total,
            'RMSE_D10': rmse_d10,
            'RMSE_D20': rmse_d20,
            'RMSE_D30': rmse_d30
        }

        data_rows.append(row)

    print(f"\nFuzzy Matching Statistics (tolerance={TOLERANCE}):")
    print("="*70)
    print(f"H_free      : {len(base_params):3d} (reference base, 100%)")
    for observable, stats in match_stats.items():
        total = stats['exact'] + stats['fuzzy']
        pct = 100 * total / len(base_params) if len(base_params) > 0 else 0
        print(f"{observable:12s}: {stats['exact']:3d} exact, {stats['fuzzy']:3d} fuzzy, {stats['none']:3d} no match ({pct:.1f}%)")
    print("="*70)

    df = pd.DataFrame(data_rows)

    df = df.sort_values(['Pe', 'T', 'kappa']).reset_index(drop=True)

    return df


def compile_confinement_data():
    print("Loading all data sources...\n")

    h_free_dict = load_free_space_entropy()
    lp_free_dict = load_free_space_lp()
    lp_free_individual_dict = load_free_space_lp_individual()
    tau_decorr_dict = load_decorrelation_times()
    dlong_dict = load_diffusion_coefficients()
    h_conf_dict = load_confinement_entropy()
    lp_conf_dict = load_confinement_lp()
    lp_conf_individual_dict = load_confinement_lp_individual()
    ttrap_dict = load_trapping_times()
    tau_decorr_cavity_dict = load_tau_decorr_cavity()
    transloc_dict = load_translocation_data()
    rotational_dict = load_rotational_data()
    msd_rmse_dict = load_msd_rmse()

    base_params = sorted(h_conf_dict.keys())

    print(f"\n[CONFINEMENT REFERENCE]")
    print(f"Using {len(base_params)} H_conf points as reference base")
    print(f"Fuzzy matching all other observables (tolerance={TOLERANCE})")

    data_rows = []

    match_stats = {
        'H_free': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'lp_free': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'lp_free_individual': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'tau_decorr_free': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'D_long': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'lp_conf': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'lp_conf_individual': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'ttrap': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'tau_decorr_cav': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'transloc': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'rotational': {'exact': 0, 'fuzzy': 0, 'none': 0},
        'msd_rmse': {'exact': 0, 'fuzzy': 0, 'none': 0}
    }

    for params in base_params:
        pe, t, k = params

        h_conf_val = h_conf_dict[params]

        h_free_val = find_closest_match(params, h_free_dict, TOLERANCE)
        lp_free_val = find_closest_match(params, lp_free_dict, TOLERANCE)
        lp_free_individual_val = find_closest_match(params, lp_free_individual_dict, TOLERANCE)
        tau_decorr_val = find_closest_match(params, tau_decorr_dict, TOLERANCE)
        dlong_val = find_closest_match(params, dlong_dict, TOLERANCE)

        lp_conf_val = find_closest_match(params, lp_conf_dict, TOLERANCE)
        lp_conf_individual_val = find_closest_match(params, lp_conf_individual_dict, TOLERANCE)
        ttrap_val = find_closest_match(params, ttrap_dict, TOLERANCE)
        tau_decorr_cavity_val = find_closest_match(params, tau_decorr_cavity_dict, TOLERANCE)
        transloc_val = find_closest_match(params, transloc_dict, TOLERANCE)
        rotational_val = find_closest_match(params, rotational_dict, TOLERANCE)
        msd_rmse_val = find_closest_match(params, msd_rmse_dict, TOLERANCE)

        for name, val, source_dict in [
            ('H_free', h_free_val, h_free_dict),
            ('lp_free', lp_free_val, lp_free_dict),
            ('lp_free_individual', lp_free_individual_val, lp_free_individual_dict),
            ('tau_decorr_free', tau_decorr_val, tau_decorr_dict),
            ('D_long', dlong_val, dlong_dict),
            ('lp_conf', lp_conf_val, lp_conf_dict),
            ('lp_conf_individual', lp_conf_individual_val, lp_conf_individual_dict),
            ('ttrap', ttrap_val, ttrap_dict),
            ('tau_decorr_cav', tau_decorr_cavity_val, tau_decorr_cavity_dict),
            ('transloc', transloc_val, transloc_dict),
            ('rotational', rotational_val, rotational_dict),
            ('msd_rmse', msd_rmse_val, msd_rmse_dict)
        ]:
            if val is not None:
                if params in source_dict:
                    match_stats[name]['exact'] += 1
                else:
                    match_stats[name]['fuzzy'] += 1
            else:
                match_stats[name]['none'] += 1

        if dlong_val is not None:
            d_long, d_error, d_r2, d_n = dlong_val
        else:
            d_long, d_error, d_r2, d_n = np.nan, np.nan, np.nan, np.nan

        if lp_free_val is not None:
            lp_free, lp_free_err, lp_free_r2 = lp_free_val
        else:
            lp_free, lp_free_err, lp_free_r2 = np.nan, np.nan, np.nan

        if lp_conf_val is not None:
            lp_conf, lp_conf_err, lp_conf_r2 = lp_conf_val
        else:
            lp_conf, lp_conf_err, lp_conf_r2 = np.nan, np.nan, np.nan

        if lp_free_individual_val is not None:
            lp_free_indiv, lp_free_indiv_std = lp_free_individual_val
        else:
            lp_free_indiv, lp_free_indiv_std = np.nan, np.nan

        if tau_decorr_val is not None:
            tau_free_full, tau_free_full_err, tau_free_unit, tau_free_unit_err = tau_decorr_val
        else:
            tau_free_full, tau_free_full_err = np.nan, np.nan
            tau_free_unit, tau_free_unit_err = np.nan, np.nan

        if tau_decorr_cavity_val is not None:
            tau_cav_full, tau_cav_unit = tau_decorr_cavity_val
        else:
            tau_cav_full, tau_cav_unit = np.nan, np.nan

        if ttrap_val is not None:
            ttrap, ttrap_std = ttrap_val
        else:
            ttrap, ttrap_std = np.nan, np.nan

        if transloc_val is not None:
            transloc_rate, transloc_success_rate, transloc_n_success, transloc_n_attempts, transloc_total = transloc_val
        else:
            transloc_rate, transloc_success_rate, transloc_n_success, transloc_n_attempts, transloc_total = np.nan, np.nan, np.nan, np.nan, np.nan

        if rotational_val is not None:
            n_rot, tau_trans, plateau_mean, tau_rot = rotational_val
        else:
            n_rot, tau_trans, plateau_mean, tau_rot = np.nan, np.nan, np.nan, np.nan

        if msd_rmse_val is not None:
            rmse_d10, rmse_d20, rmse_d30 = msd_rmse_val
        else:
            rmse_d10, rmse_d20, rmse_d30 = np.nan, np.nan, np.nan

        row = {
            'Pe': pe,
            'T': t,
            'kappa': k,
            'H_free': h_free_val if h_free_val is not None else np.nan,
            'lp_free': lp_free,
            'lp_free_error': lp_free_err,
            'lp_free_R2': lp_free_r2,
            'lp_free_individual': lp_free_indiv,
            'lp_free_individual_std': lp_free_indiv_std,
            'tau_decorr_free_full': tau_free_full,
            'tau_decorr_free_full_error': tau_free_full_err,
            'tau_decorr_free_unit': tau_free_unit,
            'tau_decorr_free_unit_error': tau_free_unit_err,
            'D_long': d_long,
            'D_long_error': d_error,
            'D_long_R2': d_r2,
            'D_long_n_points': d_n,
            'H_conf': h_conf_val,
            'lp_conf': lp_conf,
            'lp_conf_error': lp_conf_err,
            'lp_conf_R2': lp_conf_r2,
            'lp_conf_individual': lp_conf_individual_val if lp_conf_individual_val is not None else np.nan,
            'ttrap': ttrap,
            'ttrap_std': ttrap_std,
            'tau_decorr_cavity_full': tau_cav_full,
            'tau_decorr_cavity_unit': tau_cav_unit,
            'transloc_rate_per_hour': transloc_rate,
            'transloc_success_rate': transloc_success_rate,
            'transloc_n_success': transloc_n_success,
            'transloc_n_attempts': transloc_n_attempts,
            'transloc_total_events': transloc_total,
            'N_rot': n_rot,
            'tau_trans': tau_trans,
            'plateau_mean': plateau_mean,
            'tau_rot': tau_rot,
            'RMSE_D10': rmse_d10,
            'RMSE_D20': rmse_d20,
            'RMSE_D30': rmse_d30
        }

        data_rows.append(row)

    print(f"\nFuzzy Matching Statistics (tolerance={TOLERANCE}):")
    print("="*70)
    print(f"H_conf      : {len(base_params):3d} (reference base, 100%)")
    for observable, stats in match_stats.items():
        total = stats['exact'] + stats['fuzzy']
        pct = 100 * total / len(base_params) if len(base_params) > 0 else 0
        print(f"{observable:12s}: {stats['exact']:3d} exact, {stats['fuzzy']:3d} fuzzy, {stats['none']:3d} no match ({pct:.1f}%)")
    print("="*70)

    df = pd.DataFrame(data_rows)

    df = df.sort_values(['Pe', 'T', 'kappa']).reset_index(drop=True)

    return df


def main():
    print("="*70)
    print("Compiling All Simulation Observables")
    print("="*70)

    print("\n" + "="*70)
    print("1. FREE SPACE REFERENCE DATASET")
    print("="*70)

    df_freespace = compile_freespace_data()

    print("\n" + "="*70)
    print("Free Space Data Summary")
    print("="*70)
    print(f"Total parameter combinations: {len(df_freespace)}")
    print(f"\nData availability:")
    for col in ['H_free', 'lp_free', 'lp_free_individual', 'tau_decorr_free_full', 'tau_decorr_free_unit', 'D_long', 'H_conf', 'lp_conf', 'lp_conf_individual', 'ttrap', 'tau_decorr_cavity_full', 'tau_decorr_cavity_unit', 'RMSE_D10', 'RMSE_D20', 'RMSE_D30']:
        n_available = df_freespace[col].notna().sum()
        pct = 100 * n_available / len(df_freespace)
        print(f"  {col:25s}: {n_available:3d}/{len(df_freespace)} ({pct:.1f}%)")

    output_path_freespace = "active-polymer-worms-viz/data_freespace.csv"

    header_lines_freespace = [
        "# Free Space Reference Dataset - Active Polymer N=40",
        "#",
        "# Reference base: H_free (Shannon entropy in free space)",
        "# All other observables fuzzy-matched with tolerance=0.05 on (Pe, T, kappa)",
        "#",
        "# Use this dataset when correlating free space observables or mixed observables",
        "# with emphasis on free space data completeness.",
        "#",
        "# Columns:",
        "# - Pe, T, kappa: simulation parameters (dimensionless)",
        "# - H_free: Shannon entropy in free space (REFERENCE BASE)",
        "# - lp_free: Persistence length from averaged correlation (normalized)",
        "# - lp_free_individual: Mean of individual polymer persistence lengths (normalized)",
        "# - tau_decorr_free_full: E2E full-vector decorrelation time in free space (seconds)",
        "# - tau_decorr_free_unit: E2E unit-vector decorrelation time in free space (seconds)",
        "# - D_long: Long-time diffusion coefficient in free space (mm²/s)",
        "# - H_conf: Shannon entropy in confinement, CAVITY ONLY",
        "# - lp_conf: Persistence length from averaged correlation in confinement (normalized)",
        "# - lp_conf_individual: Mean of individual polymer persistence lengths in confinement (normalized)",
        "# - ttrap: Mean trapping time in cavity (minutes)",
        "# - tau_decorr_cavity_full: E2E full-vector decorrelation time in cavity ONLY (minutes)",
        "# - tau_decorr_cavity_unit: E2E unit-vector decorrelation time in cavity ONLY (minutes)",
        "# - RMSE_D10/D20/D30: MSD matching RMSE (log-space) vs experimental 10/20/30C over [10-50s]",
        "#"
    ]

    with open(output_path_freespace, 'w') as f:
        for line in header_lines_freespace:
            f.write(line + '\n')
        df_freespace.to_csv(f, index=False, float_format='%.6f')

    print(f"\n✓ Data exported to: {output_path_freespace}")

    print("\n\n" + "="*70)
    print("2. CONFINEMENT REFERENCE DATASET")
    print("="*70)

    df_confinement = compile_confinement_data()

    print("\n" + "="*70)
    print("Confinement Data Summary")
    print("="*70)
    print(f"Total parameter combinations: {len(df_confinement)}")
    print(f"\nData availability:")
    for col in ['H_free', 'lp_free', 'lp_free_individual',
                'tau_decorr_free_full', 'tau_decorr_free_unit', 'D_long',
                'H_conf', 'lp_conf', 'lp_conf_individual', 'ttrap',
                'tau_decorr_cavity_full', 'tau_decorr_cavity_unit',
                'N_rot', 'tau_trans', 'plateau_mean', 'tau_rot',
                'RMSE_D10', 'RMSE_D20', 'RMSE_D30']:
        n_available = df_confinement[col].notna().sum()
        pct = 100 * n_available / len(df_confinement)
        print(f"  {col:25s}: {n_available:3d}/{len(df_confinement)} ({pct:.1f}%)")

    output_path_confinement = "active-polymer-worms-viz/data_confinement.csv"

    header_lines_confinement = [
        "# Confinement Reference Dataset - Active Polymer N=40",
        "#",
        "# Reference base: H_conf (Shannon entropy in confinement, cavity only)",
        "# All other observables fuzzy-matched with tolerance=0.05 on (Pe, T, kappa)",
        "#",
        "# Use this dataset when correlating confinement observables or mixed observables",
        "# with emphasis on confinement data completeness.",
        "#",
        "# Columns:",
        "# - Pe, T, kappa: simulation parameters (dimensionless)",
        "# - H_free: Shannon entropy in free space",
        "# - lp_free: Persistence length from averaged correlation (normalized)",
        "# - lp_free_individual: Mean of individual polymer persistence lengths (normalized)",
        "# - tau_decorr_free_full: E2E full-vector decorrelation time in free space (seconds)",
        "# - tau_decorr_free_unit: E2E unit-vector decorrelation time in free space (seconds)",
        "# - D_long: Long-time diffusion coefficient in free space (mm²/s)",
        "# - H_conf: Shannon entropy in confinement, CAVITY ONLY (REFERENCE BASE)",
        "# - lp_conf: Persistence length from averaged correlation in confinement (normalized)",
        "# - lp_conf_individual: Mean of individual polymer persistence lengths in confinement (normalized)",
        "# - ttrap: Mean trapping time in cavity (minutes)",
        "# - tau_decorr_cavity_full: E2E full-vector decorrelation time in cavity ONLY (minutes)",
        "# - tau_decorr_cavity_unit: E2E unit-vector decorrelation time in cavity ONLY (minutes)",
        "#",
        "# Rotational number analysis (plateau-based τ_trans):",
        "# - N_rot: Rotational number = τ_rot / τ_trans",
        "# - tau_trans: Saturation time = first crossing of MSD plateau mean (minutes)",
        "# - plateau_mean: MSD plateau mean value (excludes last 20% of signal)",
        "# - tau_rot: Rotational timescale = 1/D_r (minutes, only if α ∈ [0.8, 1.3])",
        "#",
        "# MSD matching RMSE (log-space, over [10-50s] interval):",
        "# - RMSE_D10: RMSE vs experimental 10C MSD",
        "# - RMSE_D20: RMSE vs experimental 20C MSD",
        "# - RMSE_D30: RMSE vs experimental 30C MSD",
        "#"
    ]

    with open(output_path_confinement, 'w') as f:
        for line in header_lines_confinement:
            f.write(line + '\n')
        df_confinement.to_csv(f, index=False, float_format='%.6f')

    print(f"\n✓ Data exported to: {output_path_confinement}")

    print("\n\n" + "="*70)
    print("COMPILATION COMPLETE")
    print("="*70)
    print(f"Generated 2 datasets:")
    print(f"  1. {output_path_freespace} ({len(df_freespace)} points, H_free base)")
    print(f"  2. {output_path_confinement} ({len(df_confinement)} points, H_conf base)")
    print("="*70)


if __name__ == "__main__":
    main()
