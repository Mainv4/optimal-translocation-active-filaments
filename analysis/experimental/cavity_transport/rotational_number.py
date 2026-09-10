
import numpy as np


def calculate_rotational_number(tau_trans, tau_rot):
    if tau_trans is None or tau_rot is None:
        print("  Cannot calculate N_rot: missing time scale")
        return None

    if tau_trans <= 0 or np.isinf(tau_trans):
        print("  Cannot calculate N_rot: invalid τ_trans")
        return None

    N_rot = tau_rot / tau_trans

    print(f"\n{'='*50}")
    print(f"ROTATIONAL NUMBER CALCULATION")
    print(f"{'='*50}")
    print(f"  τ_trans = {tau_trans:.4f} min (translational saturation)")
    print(f"  τ_rot   = {tau_rot:.4f} min (1/D_r)")
    print(f"  N_rot   = τ_rot / τ_trans = {N_rot:.4f}")
    print(f"{'='*50}\n")

    if N_rot > 10:
        print("  Interpretation: N_rot >> 1 → Slow rotation, polymer explores before rotating")
    elif N_rot < 0.1:
        print("  Interpretation: N_rot << 1 → Fast rotation, many rotations during exploration")
    else:
        print("  Interpretation: N_rot ~ 1 → Comparable rotation and translation time scales")

    return N_rot


def summarize_results(D_r, D_r_error, tau_rot, tau_trans, N_rot, mobility_trans=None):
    return {
        'D_r': D_r,
        'D_r_error': D_r_error,
        'tau_rot': tau_rot,
        'tau_trans': tau_trans,
        'N_rot': N_rot,
        'mobility_trans': mobility_trans,
    }
