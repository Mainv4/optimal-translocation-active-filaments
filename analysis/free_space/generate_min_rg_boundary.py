

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import os

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

def progressive_unfolding_model(re_norm, alpha1, beta1, alpha2, alpha3, alpha4):
    rg_min = np.zeros_like(re_norm)
    
    re_1 = 0.2
    re_2 = 0.4
    re_3 = 0.6
    re_4 = 0.8
    
    mask_1 = re_norm < re_1
    rg_min[mask_1] = alpha1 * np.sqrt(re_norm[mask_1]) + beta1
    
    mask_2 = (re_norm >= re_1) & (re_norm < re_2)
    rg_spiral = alpha1 * np.sqrt(re_norm[mask_2]) + beta1
    f_1 = (re_norm[mask_2] - re_1) / (re_2 - re_1)
    rg_leg = alpha2 * re_norm[mask_2]
    rg_min[mask_2] = np.sqrt(rg_spiral**2 + f_1 * rg_leg**2)
    
    mask_3 = (re_norm >= re_2) & (re_norm < re_3)
    rg_spiral = alpha1 * np.sqrt(re_norm[mask_3]) + beta1
    f_2 = (re_norm[mask_3] - re_2) / (re_3 - re_2)
    rg_legs = alpha3 * re_norm[mask_3]
    rg_min[mask_3] = np.sqrt(rg_spiral**2 + f_2 * rg_legs**2)
    
    mask_4 = (re_norm >= re_3) & (re_norm < re_4)
    theta_opt = np.pi * (1 - re_norm[mask_4])
    rg_min[mask_4] = alpha4 * re_norm[mask_4] * np.sin(theta_opt/2)
    
    mask_5 = re_norm >= re_4
    rod_limit = 1.0 / (2 * np.sqrt(3))
    rg_min[mask_5] = rod_limit * np.sqrt(1 - (1 - re_norm[mask_5])**2)
    
    return rg_min

def load_empirical_data(filename="empirical_min_boundary.txt"):
    print(f"Loading empirical boundary data from {filename}...")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Empirical boundary file not found: {filename}")
    
    data = np.loadtxt(filename)
    rg_empirical = data[:, 0]
    re_empirical = data[:, 1]
    
    print(f"Loaded {len(re_empirical)} empirical boundary points")
    print(f"Re range: {np.min(re_empirical):.3f} - {np.max(re_empirical):.3f}")
    print(f"Rg range: {np.min(rg_empirical):.3f} - {np.max(rg_empirical):.3f}")
    
    return re_empirical, rg_empirical

def fit_model_to_data(re_empirical, rg_empirical):
    print("Fitting progressive unfolding model to empirical data...")
    
    initial_guess = [0.3, 0.06, 0.25, 0.2, 0.35]
    
    lower_bounds = [0.1, 0.04, 0.1, 0.1, 0.2]
    upper_bounds = [0.6, 0.1, 0.4, 0.4, 0.5]
    bounds = (lower_bounds, upper_bounds)
    
    try:
        fitted_params, param_covariance = curve_fit(
            progressive_unfolding_model, 
            re_empirical, 
            rg_empirical,
            p0=initial_guess,
            bounds=bounds,
            maxfev=10000
        )
        
        param_errors = np.sqrt(np.diag(param_covariance))
        
        rg_predicted = progressive_unfolding_model(re_empirical, *fitted_params)
        ss_res = np.sum((rg_empirical - rg_predicted) ** 2)
        ss_tot = np.sum((rg_empirical - np.mean(rg_empirical)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print("Model fitting completed successfully!")
        print(f"R-squared: {r_squared:.4f}")
        print("Fitted parameters:")
        param_names = ['α₁ (spiral)', 'β₁ (spiral)', 'α₂ (1-leg)', 'α₃ (2-legs)', 'α₄ (bent)']
        for i, (name, value, error) in enumerate(zip(param_names, fitted_params, param_errors)):
            print(f"  {name}: {value:.4f} ± {error:.4f}")
        
        return fitted_params, param_errors, r_squared
        
    except Exception as e:
        print(f"Model fitting failed: {e}")
        print("Using default parameters based on physical estimates...")
        fitted_params = np.array([0.35, 0.065, 0.25, 0.25, 0.35])
        param_errors = np.zeros_like(fitted_params)
        r_squared = 0.0
        return fitted_params, param_errors, r_squared

def generate_boundary_data(fitted_params, re_range=None, n_points=100):
    print("Generating minimum boundary data using fitted model...")
    
    if re_range is None:
        re_range = (0.01, 0.95)
    
    re_boundary = np.concatenate([
        np.linspace(re_range[0], 0.3, n_points//3),
        np.linspace(0.3, 0.7, n_points//3),
        np.linspace(0.7, re_range[1], n_points//3)
    ])
    
    re_boundary = np.unique(re_boundary)
    
    rg_boundary = progressive_unfolding_model(re_boundary, *fitted_params)
    
    print(f"Generated {len(re_boundary)} boundary points")
    print(f"Re range: {np.min(re_boundary):.3f} - {np.max(re_boundary):.3f}")
    print(f"Rg range: {np.min(rg_boundary):.3f} - {np.max(rg_boundary):.3f}")
    
    return re_boundary, rg_boundary

def save_boundary_reference(re_boundary, rg_boundary, fitted_params, r_squared, 
                          output_file="Rg_Re_MinBoundary.txt"):
    print(f"Saving minimum boundary reference to {output_file}...")
    
    boundary_data = np.column_stack([rg_boundary, re_boundary])
    
    header = "# Minimum Rg boundary based on progressive unfolding model\n"
    header += "# Generated from empirical simulation data analysis\n"
    header += "# Column 1: Rg/L (minimum radius of gyration, normalized)\n"
    header += "# Column 2: Re/L (end-to-end distance, normalized)\n"
    header += "# \n"
    header += "# Progressive unfolding model parameters:\n"
    param_names = ['alpha1_spiral', 'beta1_spiral', 'alpha2_1leg', 'alpha3_2legs', 'alpha4_bent']
    for name, value in zip(param_names, fitted_params):
        header += f"# {name}: {value:.6f}\n"
    header += f"# Model R-squared: {r_squared:.4f}\n"
    header += f"# Chain length L: {CHAIN_LENGTH} bond lengths\n"
    header += f"# Number of points: {len(boundary_data)}"
    
    np.savetxt(output_file, boundary_data, fmt='%.8f', header=header)
    
    print(f"Boundary reference saved: {len(boundary_data)} points")

def plot_model_validation(re_empirical, rg_empirical, re_boundary, rg_boundary, 
                         fitted_params, r_squared, output_dir="FIGURES"):
    print("Creating model validation plots...")
    
    set_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(rg_empirical, re_empirical, color='red', s=50, alpha=0.7,
               label='Empirical Data (5th percentile)')

    plt.plot(rg_boundary, re_boundary, 'b-', linewidth=3,
             label=f'Progressive Unfolding Model (R² = {r_squared:.3f})')

    if os.path.exists("Rg_Re_Circle.txt"):
        circle_data = np.loadtxt("Rg_Re_Circle.txt")
        plt.plot(circle_data[:, 0], circle_data[:, 1], 'g-', linewidth=2, alpha=0.7,
                 label='Circle Reference (Max Boundary)')

    regime_boundaries = [0.2, 0.4, 0.6, 0.8]
    regime_labels = ['Spiral→1-leg', '1-leg→2-legs', '2-legs→Bent', 'Bent→Straight']
    colors = ['orange', 'purple', 'brown', 'pink']

    for boundary, label, color in zip(regime_boundaries, regime_labels, colors):
        plt.axhline(y=boundary, color=color, linestyle='--', alpha=0.6, linewidth=1)
        plt.text(plt.xlim()[0] + 0.005, boundary + 0.01, label,
                color=color, fontsize=8, alpha=0.8)

    plt.xlabel(r'$R_g/L$')
    plt.ylabel(r'$R_e/L$')
    plt.title('Progressive Unfolding Model: Fit to Empirical Simulation Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'min_boundary_model_fit.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    rg_predicted = progressive_unfolding_model(re_empirical, *fitted_params)
    residuals = rg_empirical - rg_predicted
    
    plt.subplot(1, 2, 1)
    plt.scatter(re_empirical, residuals, alpha=0.7, color='blue')
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
    plt.xlabel(r'$R_e/L$')
    plt.ylabel('Residuals (Empirical - Model)')
    plt.title('Residuals vs Re')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=15, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='-', alpha=0.5)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'min_boundary_residuals.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    
    re_fine = np.linspace(0.01, 0.95, 1000)
    rg_fine = progressive_unfolding_model(re_fine, *fitted_params)
    
    regime_colors = ['red', 'orange', 'green', 'blue', 'purple']
    regime_names = ['Pure Spiral', 'Spiral+1leg', 'Spiral+2legs', 'Bent Linear', 'Nearly Straight']
    regime_bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for i in range(5):
        mask = (re_fine >= regime_bounds[i]) & (re_fine < regime_bounds[i+1])
        plt.plot(rg_fine[mask], re_fine[mask], color=regime_colors[i],
                linewidth=3, label=f'Regime {i+1}: {regime_names[i]}')

    plt.scatter(rg_empirical, re_empirical, color='black', s=20, alpha=0.5,
               label='Empirical Data')

    plt.xlabel(r'$R_g/L$')
    plt.ylabel(r'$R_e/L$')
    plt.title('Progressive Unfolding Regimes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'min_boundary_regimes.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Validation plots saved to {output_dir}/")

def main():
    print("=" * 60)
    print("MINIMUM RG BOUNDARY GENERATION")
    print("=" * 60)
    
    try:
        re_empirical, rg_empirical = load_empirical_data()
        
        fitted_params, param_errors, r_squared = fit_model_to_data(re_empirical, rg_empirical)
        
        re_boundary, rg_boundary = generate_boundary_data(fitted_params)
        
        save_boundary_reference(re_boundary, rg_boundary, fitted_params, r_squared)
        
        plot_model_validation(re_empirical, rg_empirical, re_boundary, rg_boundary,
                            fitted_params, r_squared)
        
        print("\n" + "=" * 60)
        print("BOUNDARY GENERATION COMPLETE")
        print("=" * 60)
        print(f"Generated Rg_Re_MinBoundary.txt with {len(re_boundary)} points")
        print(f"Model fit quality (R²): {r_squared:.4f}")
        print("Files created:")
        print("  - Rg_Re_MinBoundary.txt (reference file)")
        print("  - FIGURES/min_boundary_*.png (validation plots)")
        print("\nMinimum boundary ready for conformational analysis!")
        
    except Exception as e:
        print(f"Error in boundary generation: {e}")
        print("Check that empirical_min_boundary.txt exists and is valid.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())