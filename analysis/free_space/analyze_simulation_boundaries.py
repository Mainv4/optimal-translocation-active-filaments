

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from collections import defaultdict

CHAIN_LENGTH = 39

def set_plot_style():
    plt.style.use('ggplot')
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

def parse_filename(filename):
    basename = os.path.basename(filename).replace('.npy', '')
    parts = basename.split('_')
    
    params = {}
    for i in range(0, len(parts), 2):
        if i+1 < len(parts):
            key = parts[i]
            value = float(parts[i+1])
            params[key] = value
    
    return params

def load_simulation_data(data_dir="DATA_Corr_Rg_Re"):
    print("Loading simulation conformational data...")
    
    npy_files = glob.glob(os.path.join(data_dir, "Pe_*.npy"))
    print(f"Found {len(npy_files)} simulation files")
    
    simulation_data = {}
    
    for filepath in npy_files:
        try:
            params = parse_filename(filepath)
            
            data = np.load(filepath)
            print(f"Loaded {filepath}: shape {data.shape}")
            
            rg_raw = data[:, :, 0].flatten()
            re_raw = data[:, :, 1].flatten()
            
            rg_norm = rg_raw / CHAIN_LENGTH
            re_norm = re_raw / CHAIN_LENGTH
            
            valid_mask = np.isfinite(rg_norm) & np.isfinite(re_norm) & (rg_norm > 0) & (re_norm > 0)
            rg_clean = rg_norm[valid_mask]
            re_clean = re_norm[valid_mask]
            
            simulation_data[filepath] = {
                'params': params,
                'Rg': rg_clean,
                'Re': re_clean,
                'n_points': len(rg_clean)
            }
            
            print(f"  Parameters: Pe={params.get('Pe', 'N/A')}, T={params.get('T', 'N/A')}, k={params.get('k', 'N/A')}")
            print(f"  Valid points: {len(rg_clean)}")
            print(f"  Rg range: {np.min(rg_clean):.3f} - {np.max(rg_clean):.3f}")
            print(f"  Re range: {np.min(re_clean):.3f} - {np.max(re_clean):.3f}")
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    print(f"Successfully loaded {len(simulation_data)} simulation datasets")
    return simulation_data

def extract_empirical_minimum_boundary(simulation_data):
    print("Extracting empirical minimum boundary using absolute minimum...")
    print("Method: Find minimum Rg for each unique Re value across all simulation datasets")
    
    all_rg = []
    all_re = []
    
    for filepath, data in simulation_data.items():
        all_rg.extend(data['Rg'])
        all_re.extend(data['Re'])
    
    all_rg = np.array(all_rg)
    all_re = np.array(all_re)
    
    print(f"Combined dataset: {len(all_rg)} total data points")
    print(f"Rg range: {np.min(all_rg):.3f} - {np.max(all_rg):.3f}")
    print(f"Re range: {np.min(all_re):.3f} - {np.max(all_re):.3f}")
    
    from collections import defaultdict
    re_to_min_rg = defaultdict(lambda: float('inf'))
    
    for rg, re in zip(all_rg, all_re):
        re_rounded = round(re, 4)
        if rg < re_to_min_rg[re_rounded]:
            re_to_min_rg[re_rounded] = rg
    
    re_values_raw = np.array(sorted(re_to_min_rg.keys()))
    rg_min_boundary_raw = np.array([re_to_min_rg[re] for re in re_values_raw])
    
    print(f"Extracted raw boundary with {len(re_values_raw)} unique Re values")
    print(f"Raw boundary Re range: {np.min(re_values_raw):.3f} - {np.max(re_values_raw):.3f}")
    print(f"Raw boundary Rg range: {np.min(rg_min_boundary_raw):.3f} - {np.max(rg_min_boundary_raw):.3f}")
    
    from scipy.interpolate import interp1d
    from scipy.ndimage import uniform_filter1d
    
    rg_smoothed = uniform_filter1d(rg_min_boundary_raw, size=10)
    
    interp_func = interp1d(re_values_raw, rg_smoothed, kind='cubic', bounds_error=False, fill_value='extrapolate')
    
    re_values = np.linspace(np.min(re_values_raw), np.max(re_values_raw), 200)
    rg_min_boundary = interp_func(re_values)
    
    rg_min_boundary = uniform_filter1d(rg_min_boundary, size=3)
    
    print(f"Created smooth boundary with {len(re_values)} interpolated points")
    print(f"Smooth boundary Re range: {np.min(re_values):.3f} - {np.max(re_values):.3f}")
    print(f"Smooth boundary Rg range: {np.min(rg_min_boundary):.3f} - {np.max(rg_min_boundary):.3f}")
    
    return re_values, rg_min_boundary

def plot_empirical_boundary(simulation_data, re_values, rg_min_boundary, output_dir="FIGURES"):
    print("Creating boundary visualization plots...")
    
    set_plot_style()
    
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(simulation_data)))
    
    for i, (filepath, data) in enumerate(simulation_data.items()):
        params = data['params']
        label = f"Pe={params.get('Pe', '?')}, T={params.get('T', '?')}, k={params.get('k', '?')}"
        
        n_plot = min(1000, len(data['Re']))
        indices = np.random.choice(len(data['Re']), n_plot, replace=False)
        
        plt.scatter(data['Rg'][indices], data['Re'][indices], 
                   alpha=0.3, s=1, color=colors[i], label=label)
    
    plt.plot(rg_min_boundary, re_values, 'b-', linewidth=3, 
             label='Empirical Min Boundary (absolute minimum)')
    
    if os.path.exists("Rg_Re_Circle.txt"):
        circle_data = np.loadtxt("Rg_Re_Circle.txt")
        plt.plot(circle_data[:, 0], circle_data[:, 1], 'r-', linewidth=3, 
                 label='Circle Reference (Max Boundary)')
    
    plt.xlabel(r'$R_g/L$')
    plt.ylabel(r'$R_e/L$')
    plt.title('Simulation Data with Empirical Minimum Boundary')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'empirical_min_boundary_overview.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(rg_min_boundary, re_values, 'bo-', linewidth=2, markersize=4,
             label='Empirical Min Boundary')
    plt.xlabel(r'$R_g/L$ (minimum)')
    plt.ylabel(r'$R_e/L$')
    plt.title('Empirical Minimum Rg Boundary vs Re')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'empirical_min_boundary_detail.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Boundary plots saved to {output_dir}/")

def save_boundary_data(re_values, rg_min_boundary, output_file="empirical_min_boundary.txt"):
    print(f"Saving empirical boundary data to {output_file}...")
    
    boundary_data = np.column_stack([rg_min_boundary, re_values])
    
    header = "# Empirical minimum Rg boundary extracted from simulation data\n"
    header += "# Column 1: Rg/L (minimum)\n"
    header += "# Column 2: Re/L\n"
    header += f"# Generated from {len(re_values)} unique Re values\n"
    header += "# Method: Find minimum Rg for each unique Re value across all simulation datasets"
    
    np.savetxt(output_file, boundary_data, fmt='%.6f', header=header)
    print(f"Boundary data saved: {len(boundary_data)} points")

def main():
    print("=" * 60)
    print("SIMULATION BOUNDARY ANALYSIS")
    print("=" * 60)
    
    simulation_data = load_simulation_data()
    
    if not simulation_data:
        print("No simulation data found. Exiting.")
        return
    
    re_values, rg_min_boundary = extract_empirical_minimum_boundary(simulation_data)
    
    plot_empirical_boundary(simulation_data, re_values, rg_min_boundary)
    
    save_boundary_data(re_values, rg_min_boundary)
    
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total simulation files analyzed: {len(simulation_data)}")
    print(f"Total data points: {sum(data['n_points'] for data in simulation_data.values())}")
    print(f"Empirical boundary points: {len(re_values)}")
    print(f"Re range covered: {np.min(re_values):.3f} - {np.max(re_values):.3f}")
    print(f"Min Rg range: {np.min(rg_min_boundary):.3f} - {np.max(rg_min_boundary):.3f}")
    print("\nEmpirical boundary data ready! Generated empirical_min_boundary.txt for plotting scripts.")

if __name__ == "__main__":
    main()