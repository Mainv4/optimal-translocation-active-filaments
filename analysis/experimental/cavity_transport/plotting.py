
import os
import numpy as np
import matplotlib.pyplot as plt

from .config import CAVITY_RADIUS
from .msd import smooth_msd


def calculate_log_derivative(time, msd, window=5):
    valid = (time > 0) & (msd > 0)
    log_t = np.log10(time[valid])
    log_msd = np.log10(msd[valid])

    slope = np.gradient(log_msd, log_t)

    if window > 1 and len(slope) > window:
        kernel = np.ones(window) / window
        slope_smooth = np.convolve(slope, kernel, mode='valid')
        offset = window // 2
        time_out = time[valid][offset:offset + len(slope_smooth)]
        return time_out, slope_smooth

    return time[valid], slope


def plot_msd_analysis(time, msd, title, R=None, tau_trans=None, D=None,
                      save_path=None, show=True):
    if R is None:
        R = CAVITY_RADIUS

    _fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    valid = (time > 0) & (msd > 0)
    t_valid = time[valid]
    msd_valid = msd[valid]

    ax1 = axes[0]
    ax1.loglog(t_valid, msd_valid, 'b-', linewidth=1.5, label='MSD')

    ax1.axhline(R**2, color='r', linestyle='--', alpha=0.7, label=f'R² = {R**2:.1f}')

    if tau_trans is not None:
        ax1.axvline(tau_trans, color='g', linestyle=':', alpha=0.7, label=f'τ_trans = {tau_trans:.2f}')

    if D is not None:
        t_fit = np.array([t_valid[0], t_valid[-1]])
        msd_fit = 2 * D * t_fit
        ax1.loglog(t_fit, msd_fit, 'k--', alpha=0.5, label=f'2D·t')

    ax1.set_ylabel('MSD')
    ax1.set_title(title)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    time_deriv, slope = calculate_log_derivative(t_valid, msd_valid)
    ax2.semilogx(time_deriv, slope, 'b-', linewidth=1.5)
    ax2.axhline(1, color='k', linestyle='--', alpha=0.5, label='slope = 1 (diffusive)')
    ax2.axhline(2, color='r', linestyle='--', alpha=0.5, label='slope = 2 (ballistic)')
    ax2.axhline(0, color='g', linestyle='--', alpha=0.5, label='slope = 0 (confined)')

    if tau_trans is not None:
        ax2.axvline(tau_trans, color='g', linestyle=':', alpha=0.7)

    ax2.set_ylabel('d(log MSD)/d(log t)')
    ax2.set_ylim(-0.5, 2.5)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    msd_over_t = msd_valid / t_valid
    ax3.loglog(t_valid, msd_over_t, 'b-', linewidth=1.5)

    if D is not None:
        ax3.axhline(2 * D, color='k', linestyle='--', alpha=0.5, label=f'2D = {2*D:.4f}')

    if tau_trans is not None:
        ax3.axvline(tau_trans, color='g', linestyle=':', alpha=0.7)

    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('MSD/t')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure: {save_path}")

    plt.close()


def plot_rotational_msd(msd_rot, D_r=None, tau_rot=None, save_path=None, show=True):
    if len(msd_rot) == 0:
        print("  Cannot plot empty MSD")
        return

    time = msd_rot[:, 0]
    msd = msd_rot[:, 1]

    plot_msd_analysis(
        time, msd,
        title='Rotational MSD Analysis',
        R=np.pi,
        tau_trans=tau_rot,
        D=D_r,
        save_path=save_path,
        show=show
    )


def plot_translational_msd(msd_trans, tau_trans=None, R=None, save_path=None, show=True):
    if R is None:
        R = CAVITY_RADIUS

    if len(msd_trans) == 0:
        print("  Cannot plot empty MSD")
        return

    time = msd_trans[:, 0]
    msd = msd_trans[:, 1]

    plot_msd_analysis(
        time, msd,
        title='Translational MSD Analysis',
        R=R,
        tau_trans=tau_trans,
        D=None,
        save_path=save_path,
        show=show
    )


def plot_combined_msd(msd_rot, msd_trans, results, save_path=None, show=True):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12), sharex='col')

    if len(msd_rot) > 0:
        t_rot = msd_rot[:, 0]
        msd_r = msd_rot[:, 1]
        valid_rot = (t_rot > 0) & (msd_r > 0)
        t_rot_v = t_rot[valid_rot]
        msd_r_v = msd_r[valid_rot]

        msd_rot_valid = np.column_stack((t_rot_v, msd_r_v, np.zeros_like(msd_r_v)))
        msd_rot_smooth = smooth_msd(msd_rot_valid, window=11)
        t_rot_smooth = msd_rot_smooth[:, 0]
        msd_r_smooth = msd_rot_smooth[:, 1]

        fit_start_idx = results.get('fit_start_idx', 0)
        fit_end_idx = results.get('fit_end_idx', len(t_rot_v))
        if fit_start_idx is not None and fit_end_idx is not None and fit_end_idx > fit_start_idx:
            t_fit_start = t_rot_v[fit_start_idx] if fit_start_idx < len(t_rot_v) else t_rot_v[0]
            t_fit_end = t_rot_v[min(fit_end_idx - 1, len(t_rot_v) - 1)]
        else:
            t_fit_start, t_fit_end = None, None

        ax = axes[0, 0]
        ax.loglog(t_rot_smooth, msd_r_smooth, 'b-', linewidth=1.5)

        if t_fit_start and t_fit_end:
            ax.axvspan(t_fit_start, t_fit_end, alpha=0.2, color='green', label='fit range')

        if results.get('D_r'):
            ax.loglog(t_rot_v, 2 * results['D_r'] * t_rot_v, 'k-', alpha=0.7, linewidth=2,
                      label=f"2D_r·t (τ_rot={results.get('tau_rot', 0):.2f})")

        A_power = results.get('A_power')
        alpha_power = results.get('alpha_power')
        if A_power and alpha_power:
            ax.loglog(t_rot_v, A_power * t_rot_v**alpha_power, 'r--', alpha=0.7, linewidth=1.5,
                      label=f"A·t^α (α={alpha_power:.2f})")

        ax.set_ylabel('MSD_rot (rad²)')
        ax.set_title('Rotational MSD')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        time_deriv, slope = calculate_log_derivative(t_rot_smooth, msd_r_smooth, window=3)
        ax.semilogx(time_deriv, slope, 'b-', linewidth=1.5)
        ax.axhline(1, color='k', linestyle='--', alpha=0.5, label='slope = 1 (diffusive)')
        ax.axhline(1.5, color='orange', linestyle='--', alpha=0.5, label='slope = 1.5 (threshold)')
        ax.axhline(2, color='r', linestyle='--', alpha=0.5, label='slope = 2 (ballistic)')

        if t_fit_start and t_fit_end:
            ax.axvspan(t_fit_start, t_fit_end, alpha=0.2, color='green')

        ax.set_ylabel('d(log MSD)/d(log t)')
        ax.set_ylim(-0.5, 2.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 0]
        msd_over_t_rot = msd_r_smooth / t_rot_smooth
        ax.loglog(t_rot_smooth, msd_over_t_rot, 'b-', linewidth=1.5)

        if t_fit_start and t_fit_end:
            ax.axvspan(t_fit_start, t_fit_end, alpha=0.2, color='green')

        if results.get('D_r'):
            ax.axhline(2 * results['D_r'], color='k', linestyle='-', linewidth=2, alpha=0.7,
                       label=f"2D_r = {2*results['D_r']:.4f}")

        ax.set_xlabel('Time (min)')
        ax.set_ylabel('MSD_rot / t')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    if len(msd_trans) > 0:
        t_trans = msd_trans[:, 0]
        msd_t = msd_trans[:, 1]
        valid_trans = (t_trans > 0) & (msd_t > 0)
        t_trans_v = t_trans[valid_trans]
        msd_t_v = msd_t[valid_trans]

        msd_trans_valid = np.column_stack((t_trans_v, msd_t_v, np.zeros_like(msd_t_v)))
        msd_trans_smooth = smooth_msd(msd_trans_valid, window=11)
        t_trans_smooth = msd_trans_smooth[:, 0]
        msd_t_smooth = msd_trans_smooth[:, 1]

        tau_trans = results.get('tau_trans')
        plateau_mean = results.get('plateau_mean')
        plateau_std = results.get('plateau_std')

        ax = axes[0, 1]
        ax.loglog(t_trans_smooth, msd_t_smooth, 'k-', linewidth=1.5, label='MSD')

        if plateau_mean is not None:
            ax.axhline(plateau_mean, color='r', linestyle='--', alpha=0.7,
                       label=f'plateau = {plateau_mean:.1f}')
            if plateau_std is not None and plateau_std > 0:
                ax.axhspan(plateau_mean - plateau_std, plateau_mean + plateau_std,
                           alpha=0.15, color='red')

        if tau_trans:
            ax.axvline(tau_trans, color='r', linestyle='-', linewidth=2, alpha=0.8,
                       label=f'τ_sat = {tau_trans:.2f}')

        ax.set_ylabel('MSD_trans')
        ax.set_title('Translational MSD')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        time_deriv, slope = calculate_log_derivative(t_trans_smooth, msd_t_smooth, window=3)
        ax.semilogx(time_deriv, slope, 'k-', linewidth=1.5)
        ax.axhline(1, color='k', linestyle='--', alpha=0.5, label='slope = 1 (diffusive)')
        ax.axhline(0, color='g', linestyle='--', alpha=0.5, label='slope = 0 (confined)')
        if tau_trans:
            ax.axvline(tau_trans, color='r', linestyle='-', linewidth=2, alpha=0.8)
        ax.set_ylabel('d(log MSD)/d(log t)')
        ax.set_ylim(-0.5, 2.5)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2, 1]
        msd_over_t_trans = msd_t_smooth / t_trans_smooth
        ax.loglog(t_trans_smooth, msd_over_t_trans, 'k-', linewidth=1.5)
        if tau_trans:
            ax.axvline(tau_trans, color='r', linestyle='-', linewidth=2, alpha=0.8)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('MSD_trans / t')
        ax.grid(True, alpha=0.3)

    N_rot = results.get('N_rot')
    if N_rot is not None:
        fig.suptitle(f"N_rot = τ_rot/τ_trans = {N_rot:.3f}", fontsize=14, fontweight='bold')
    else:
        alpha = results.get('alpha_power')
        alpha_str = f"α={alpha:.2f}" if alpha is not None else "α=N/A"
        fig.suptitle(f"N_rot = N/A (no diffusive regime, {alpha_str})", fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure: {save_path}")

    plt.close()


def plot_e2e_autocorr(autocorr_result, tau_decorr=None,
                      autocorr_result_full=None, tau_decorr_full=None,
                      title=None, save_path=None, show=True):
    if len(autocorr_result) == 0:
        print("  No autocorrelation data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if autocorr_result_full is not None and len(autocorr_result_full) > 0:
        time_f = autocorr_result_full[:, 0]
        ac_f = autocorr_result_full[:, 1]
        se_f = autocorr_result_full[:, 2]
        ax.plot(time_f, ac_f, 'k-', linewidth=1.5, label=r'Full: $\langle\mathbf{R}\cdot\mathbf{R}\rangle / \langle|\mathbf{R}|^2\rangle$')
        ax.fill_between(time_f, ac_f - se_f, ac_f + se_f, alpha=0.15, color='black')
        if tau_decorr_full is not None:
            ax.axvline(tau_decorr_full, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                       label=f'τ_full = {tau_decorr_full:.3f} min')

    time = autocorr_result[:, 0]
    autocorr = autocorr_result[:, 1]
    std_error = autocorr_result[:, 2]
    ax.plot(time, autocorr, color='tab:orange', linewidth=1.5, linestyle=':', label=r'Unit: $\langle\hat{u}\cdot\hat{u}\rangle$')
    ax.fill_between(time, autocorr - std_error, autocorr + std_error,
                    alpha=0.2, color='tab:orange')
    if tau_decorr is not None:
        ax.axvline(tau_decorr, color='tab:orange', linestyle='--', linewidth=1.5, alpha=0.7,
                   label=f'τ_unit = {tau_decorr:.3f} min')

    threshold = 1.0 / np.e
    ax.axhline(threshold, color='gray', linestyle='--', alpha=0.5,
               label=f'1/e ≈ {threshold:.3f}')

    ax.set_xlabel('Time lag τ (min)')
    ax.set_ylabel('C(τ)')
    ax.set_xscale('log')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved figure: {save_path}")

    plt.close()
