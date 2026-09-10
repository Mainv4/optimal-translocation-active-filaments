
import argparse
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

from .config import CAVITY_RADIUS_15
from .trajectory import load_head_positions, extract_cavity_segments
from .msd import calculate_rotational_msd, calculate_translational_msd
from .diffusion import fit_rotational_diffusion, fit_power_law, find_saturation_time_plateau, calculate_mobility
from .rotational_number import calculate_rotational_number, summarize_results
from .plotting import plot_combined_msd, plot_e2e_autocorr
from .autocorr import load_e2e_positions, extract_e2e_cavity_segments, calculate_e2e_autocorr


def process_single_directory(path, plot=False, save_data=True, show=True, msd_only=False):
    print(f"\n{'='*60}")
    print(f"Processing: {path}")
    if msd_only:
        print("  (MSD only - skipping E2E autocorrelation)")
    print(f"{'='*60}")

    time, head_x, head_y, N_polymers = load_head_positions(path)

    cavity_segments = extract_cavity_segments(time, head_x, head_y, N_polymers)

    msd_rot, _ = calculate_rotational_msd(cavity_segments)
    msd_trans, _ = calculate_translational_msd(cavity_segments)

    A_power, alpha_power, A_error, alpha_error = fit_power_law(msd_rot)

    if alpha_power is not None and 0.8 <= alpha_power <= 1.3:
        D_r, D_r_error, tau_rot, fit_start_idx, fit_end_idx = fit_rotational_diffusion(msd_rot)
    else:
        alpha_str = f"{alpha_power:.2f}" if alpha_power is not None else "N/A"
        print(f"  No diffusive regime (α = {alpha_str}, not in [0.8, 1.3]) → D_r = None")
        D_r, D_r_error, tau_rot = None, None, None
        fit_start_idx, fit_end_idx = None, None

    print("\n  Finding saturation time from plateau mean:")
    tau_trans, plateau_mean, plateau_std = find_saturation_time_plateau(msd_trans)

    mobility = calculate_mobility(msd_trans, tau_trans)

    N_rot = calculate_rotational_number(tau_trans, tau_rot)

    autocorr_result_unit = np.array([])
    autocorr_result_full = np.array([])
    tau_decorr_unit = None
    tau_decorr_full = None
    if not msd_only:
        try:
            time_e2e, e2e_x, e2e_y, head_x_e2e, head_y_e2e, N_polymers_e2e = load_e2e_positions(path)
            e2e_segments = extract_e2e_cavity_segments(time_e2e, e2e_x, e2e_y, head_x_e2e, head_y_e2e, N_polymers_e2e)
            autocorr_results = calculate_e2e_autocorr(e2e_segments)
            autocorr_result_unit = autocorr_results['unit'][0]
            tau_decorr_unit = autocorr_results['unit'][1]
            autocorr_result_full = autocorr_results['full'][0]
            tau_decorr_full = autocorr_results['full'][1]
        except Exception as e:
            print(f"  Warning: Could not calculate e2e autocorrelation: {e}")

    results = summarize_results(D_r, D_r_error, tau_rot, tau_trans, N_rot, mobility)
    results['tau_trans'] = tau_trans
    results['plateau_mean'] = plateau_mean
    results['plateau_std'] = plateau_std
    results['tau_decorr'] = tau_decorr_unit
    results['tau_decorr_unit'] = tau_decorr_unit
    results['tau_decorr_full'] = tau_decorr_full
    results['fit_start_idx'] = fit_start_idx
    results['fit_end_idx'] = fit_end_idx
    results['A_power'] = A_power
    results['alpha_power'] = alpha_power

    if save_data and tau_trans is not None:
        output_dir = "DATA_ROTATIONAL"
        os.makedirs(output_dir, exist_ok=True)
        traj_name = os.path.basename(path.rstrip('/'))
        output_file = os.path.join(output_dir, f"{traj_name}.npz")

        np.savez(
            output_file,
            msd_rot=msd_rot,
            msd_trans=msd_trans,
            D_r=D_r if D_r is not None else np.nan,
            D_r_error=D_r_error if D_r_error is not None else np.nan,
            tau_rot=tau_rot if tau_rot is not None else np.nan,
            tau_trans=tau_trans if tau_trans is not None else np.nan,
            plateau_mean=plateau_mean if plateau_mean is not None else np.nan,
            plateau_std=plateau_std if plateau_std is not None else np.nan,
            N_rot=N_rot if N_rot is not None else np.nan,
            mobility_trans=mobility if mobility is not None else np.nan,
            autocorr=autocorr_result_unit,
            autocorr_full=autocorr_result_full,
            tau_decorr=tau_decorr_unit if tau_decorr_unit is not None else np.nan,
            tau_decorr_full=tau_decorr_full if tau_decorr_full is not None else np.nan,
        )
        print(f"  Saved results to: {output_file}")

    if plot:
        fig_dir = "FIGURES/ROTATIONAL"
        os.makedirs(fig_dir, exist_ok=True)
        traj_name = os.path.basename(path.rstrip('/'))

        fig_path = os.path.join(fig_dir, f"msd_analysis_{traj_name}.png")
        print(f"  Generating MSD plot: {fig_path}")
        plot_combined_msd(msd_rot, msd_trans, results, save_path=fig_path, show=show)

        if not msd_only and len(autocorr_result_unit) > 0:
            fig_dir_e2e = "FIGURES/E2E_AUTOCORR_CAVITY"
            os.makedirs(fig_dir_e2e, exist_ok=True)
            fig_path_ac = os.path.join(fig_dir_e2e, f"e2e_autocorr_{traj_name}.png")
            print(f"  Generating autocorr plot: {fig_path_ac}")
            plot_e2e_autocorr(autocorr_result_unit, tau_decorr=tau_decorr_unit,
                              autocorr_result_full=autocorr_result_full,
                              tau_decorr_full=tau_decorr_full,
                              title=f"E2E Autocorrelation (cavity) - {traj_name}",
                              save_path=fig_path_ac, show=show)
    else:
        print("  Skipping plots (--plot not specified)")

    return results


def find_simulation_directories():
    cwd = os.getcwd()
    dirs = []
    for entry in os.listdir(cwd):
        if (entry.startswith('Pe_') or entry.startswith('TEST_')) and os.path.isdir(entry):
            dirs.append(entry)
    return sorted(dirs)


def parse_parameters(dirname):
    name = dirname
    if name.startswith('TEST_'):
        name = name[5:]
    match = re.match(r'Pe_([\d.]+)_T_([\d.]+)_k_([\d.]+)', name)
    if match:
        return {
            'Pe': float(match.group(1)),
            'T': float(match.group(2)),
            'k': float(match.group(3)),
        }
    return None


def process_directory_wrapper(args_tuple):
    path, plot, save_data, msd_only = args_tuple
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = process_single_directory(path, plot=plot, save_data=save_data, show=False, msd_only=msd_only)
        return path, result, None
    except Exception as e:
        return path, None, str(e)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def process_autocorr_only(path, plot=False, save_data=True, show=True):
    print(f"\n{'='*60}")
    print(f"E2E Autocorrelation: {path}")
    print(f"{'='*60}")

    time_e2e, e2e_x, e2e_y, head_x, head_y, N_polymers = load_e2e_positions(path)

    e2e_segments = extract_e2e_cavity_segments(time_e2e, e2e_x, e2e_y, head_x, head_y, N_polymers)

    autocorr_results = calculate_e2e_autocorr(e2e_segments)
    autocorr_result_unit = autocorr_results['unit'][0]
    tau_decorr_unit = autocorr_results['unit'][1]
    autocorr_result_full = autocorr_results['full'][0]
    tau_decorr_full = autocorr_results['full'][1]

    results = {
        'tau_decorr': tau_decorr_unit,
        'tau_decorr_unit': tau_decorr_unit,
        'tau_decorr_full': tau_decorr_full,
    }

    if save_data:
        output_dir = "DATA_AUTOCORR"
        os.makedirs(output_dir, exist_ok=True)
        traj_name = os.path.basename(path.rstrip('/'))
        output_file = os.path.join(output_dir, f"{traj_name}.npz")

        np.savez(
            output_file,
            autocorr=autocorr_result_unit,
            autocorr_full=autocorr_result_full,
            tau_decorr=tau_decorr_unit if tau_decorr_unit is not None else np.nan,
            tau_decorr_full=tau_decorr_full if tau_decorr_full is not None else np.nan,
        )
        print(f"  Saved results to: {output_file}")

    if plot and len(autocorr_result_unit) > 0:
        fig_dir = "FIGURES/E2E_AUTOCORR_CAVITY"
        os.makedirs(fig_dir, exist_ok=True)
        traj_name = os.path.basename(path.rstrip('/'))
        fig_path = os.path.join(fig_dir, f"e2e_autocorr_{traj_name}.png")
        print(f"  Generating plot: {fig_path}")
        plot_e2e_autocorr(autocorr_result_unit, tau_decorr=tau_decorr_unit,
                          autocorr_result_full=autocorr_result_full,
                          tau_decorr_full=tau_decorr_full,
                          title=f"E2E Autocorrelation (cavity) - {traj_name}",
                          save_path=fig_path, show=show)

    return results


def process_autocorr_wrapper(args_tuple):
    path, plot, save_data = args_tuple
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        result = process_autocorr_only(path, plot=plot, save_data=save_data, show=False)
        return path, result, None
    except Exception as e:
        return path, None, str(e)
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def main():
    parser = argparse.ArgumentParser(
        description='Calculate Rotational Number N_rot = τ_rot / τ_trans'
    )
    parser.add_argument('--path', type=str, help='Path to simulation directory')
    parser.add_argument('--all', action='store_true',
                        help='Process all Pe_* and TEST_* directories')
    parser.add_argument('--autocorr-only', action='store_true',
                        help='Compute ONLY e2e autocorrelation (faster, skips MSD/N_rot)')
    parser.add_argument('--msd-only', action='store_true',
                        help='Compute ONLY MSD analysis (skips e2e autocorrelation)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots (saved to FIGURES/)')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to npz files')
    parser.add_argument('--batch-size', type=int, default=12,
                        help='Number of parallel workers (default: 12)')

    args = parser.parse_args()

    if args.autocorr_only:
        if args.all:
            dirs = find_simulation_directories()
            if not dirs:
                print("No Pe_* or TEST_* directories found in current directory")
                return

            print(f"Found {len(dirs)} simulation directories")
            print(f"Computing E2E AUTOCORRELATION ONLY in parallel with {args.batch_size} workers...")

            task_args = [(d, args.plot, not args.no_save) for d in dirs]

            all_results = []
            errors = []
            completed = 0

            with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
                futures = {executor.submit(process_autocorr_wrapper, arg): arg[0] for arg in task_args}

                for future in as_completed(futures):
                    dirname = futures[future]
                    completed += 1
                    _, result, error = future.result()

                    if error:
                        errors.append((dirname, error))
                        print(f"  [{completed}/{len(dirs)}] {dirname}: ERROR - {error}")
                    elif result and result.get('tau_decorr') is not None:
                        params = parse_parameters(dirname)
                        if params:
                            all_results.append({
                                'directory': dirname,
                                **params,
                                **result
                            })
                        print(f"  [{completed}/{len(dirs)}] {dirname}: τ_decorr = {result['tau_decorr']:.4f} min")
                    else:
                        print(f"  [{completed}/{len(dirs)}] {dirname}: No valid result")

            if errors:
                print(f"\n{len(errors)} errors occurred:")
                for dirname, err in errors:
                    print(f"  - {dirname}: {err}")

            if all_results:
                print("\n" + "="*80)
                print("SUMMARY: E2E Autocorrelation (τ_decorr)")
                print("="*80)
                print(f"{'Directory':<35} {'Pe':>6} {'T':>6} {'k':>6} {'τ_decorr (min)':>15}")
                print("-"*80)
                for r in sorted(all_results, key=lambda x: (x['Pe'], x['T'], x['k'])):
                    tau_decorr = r.get('tau_decorr')
                    tau_decorr_str = f"{tau_decorr:.4f}" if tau_decorr is not None else "N/A"
                    print(f"{r['directory']:<35} {r['Pe']:>6.2f} {r['T']:>6.2f} {r['k']:>6.2f} {tau_decorr_str:>15}")
                print("="*80)

                summary_file = "DATA_AUTOCORR/summary.txt"
                os.makedirs("DATA_AUTOCORR", exist_ok=True)
                with open(summary_file, 'w') as f:
                    f.write("# E2E Autocorrelation Summary\n")
                    f.write("#\n")
                    f.write("# τ_decorr = decorrelation time where C(τ) = 1/e\n")
                    f.write("# C(τ) = <û(t)·û(t+τ)> with û = unit e2e vector\n")
                    f.write("#\n")
                    f.write(f"# {'Directory':<33} {'Pe':>6} {'T':>6} {'k':>6} {'tau_decorr':>12}\n")
                    for r in sorted(all_results, key=lambda x: (x['Pe'], x['T'], x['k'])):
                        tau_decorr = r.get('tau_decorr')
                        tau_decorr_str = f"{tau_decorr:.4f}" if tau_decorr is not None else "N/A"
                        f.write(f"  {r['directory']:<33} {r['Pe']:>6.2f} {r['T']:>6.2f} {r['k']:>6.2f} {tau_decorr_str:>12}\n")
                print(f"Summary saved to: {summary_file}")

        elif args.path:
            process_autocorr_only(args.path, plot=args.plot, save_data=not args.no_save, show=True)
        else:
            parser.print_help()
        return

    if args.all:
        dirs = find_simulation_directories()
        if not dirs:
            print("No Pe_* or TEST_* directories found in current directory")
            return

        print(f"Found {len(dirs)} simulation directories")
        mode_str = " (MSD only)" if args.msd_only else ""
        print(f"Processing in parallel with {args.batch_size} workers...{mode_str}")

        task_args = [(d, args.plot, not args.no_save, args.msd_only) for d in dirs]

        all_results = []
        errors = []
        completed = 0

        with ProcessPoolExecutor(max_workers=args.batch_size) as executor:
            futures = {executor.submit(process_directory_wrapper, arg): arg[0] for arg in task_args}

            for future in as_completed(futures):
                dirname = futures[future]
                completed += 1
                _, result, error = future.result()

                if error:
                    errors.append((dirname, error))
                    print(f"  [{completed}/{len(dirs)}] {dirname}: ERROR - {error}")
                elif result and result.get('tau_trans') is not None:
                    params = parse_parameters(dirname)
                    if params:
                        all_results.append({
                            'directory': dirname,
                            **params,
                            **result
                        })
                    N_rot_str = f"{result['N_rot']:.4f}" if result.get('N_rot') is not None else "N/A"
                    print(f"  [{completed}/{len(dirs)}] {dirname}: N_rot = {N_rot_str}")
                else:
                    print(f"  [{completed}/{len(dirs)}] {dirname}: No valid result")

        if errors:
            print(f"\n{len(errors)} errors occurred:")
            for dirname, err in errors:
                print(f"  - {dirname}: {err}")

        if all_results:
            print("\n" + "="*120)
            print("SUMMARY: Rotational Numbers — τ_trans from plateau mean")
            print("="*120)
            print(f"{'Directory':<30} {'Pe':>6} {'T':>6} {'k':>6} {'N_rot':>8} {'τ_trans':>8} {'plateau':>10} {'τ_rot':>8}")
            print("-"*120)
            for r in all_results:
                tau_trans = r.get('tau_trans', 0) or 0
                plat_mean = r.get('plateau_mean', 0) or 0
                tau_rot = r.get('tau_rot', 0) or 0
                N_rot_val = r.get('N_rot')
                N_rot_str = f"{N_rot_val:>8.3f}" if N_rot_val is not None else "     N/A"
                print(f"{r['directory']:<30} {r['Pe']:>6.2f} {r['T']:>6.2f} {r['k']:>6.2f} "
                      f"{N_rot_str} {tau_trans:>8.2f} {plat_mean:>10.2f} {tau_rot:>8.2f}")
            print("="*120)

            summary_file = "DATA_ROTATIONAL/summary.txt"
            os.makedirs("DATA_ROTATIONAL", exist_ok=True)
            with open(summary_file, 'w') as f:
                f.write("# Rotational Number Summary\n")
                f.write("#\n")
                f.write("# N_rot = τ_rot / τ_trans\n")
                f.write("#   τ_trans = first time MSD reaches plateau mean\n")
                f.write("#   τ_rot = 1/D_r (only if α ∈ [0.8, 1.3])\n")
                f.write("#   plateau = mean MSD on plateau (last 20% excluded)\n")
                f.write("#\n")
                f.write(f"# {'Directory':<28} {'Pe':>6} {'T':>6} {'k':>6} {'N_rot':>8} {'tau_trans':>10} {'plateau':>10} {'tau_rot':>8}\n")
                for r in all_results:
                    tau_trans = r.get('tau_trans', 0) or 0
                    plat_mean = r.get('plateau_mean', 0) or 0
                    tau_rot = r.get('tau_rot', 0) or 0
                    N_rot_val = r.get('N_rot')
                    N_rot_str = f"{N_rot_val:>8.3f}" if N_rot_val is not None else "     N/A"
                    f.write(f"  {r['directory']:<28} {r['Pe']:>6.2f} {r['T']:>6.2f} {r['k']:>6.2f} "
                           f"{N_rot_str} {tau_trans:>10.2f} {plat_mean:>10.2f} {tau_rot:>8.2f}\n")
            print(f"Summary saved to: {summary_file}")

    elif args.path:
        process_single_directory(args.path, plot=args.plot, save_data=not args.no_save, show=True, msd_only=args.msd_only)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
