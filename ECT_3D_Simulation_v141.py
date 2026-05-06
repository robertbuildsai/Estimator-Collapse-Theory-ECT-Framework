"""
ECT_3D_Simulation_v141.py

Clean-room reference implementation of the Estimator Collapse Theory (ECT)
Framework for 3-D Sequential EKF (GNSS + Range Fusion).

Derived from the published equations in Section II-E of:
  "Estimator Collapse Theory: A Framework for Predicting
   Filter Instability under Bounded Adversarial Perturbation"

Author: R. J. Douglas
"""

__version__ = '2.0.0-honest'

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import chi2

# ═══════════════════════════════════════════════════════════════════════════════
#  Global Parameters & Manuscript Constants
# ═══════════════════════════════════════════════════════════════════════════════

N_MC    = 500           # default Monte Carlo ensemble size
T       = 1200          # trajectory duration [s]
DT      = 1.0           # sampling / EKF update interval [s]
N_STEPS = int(T / DT)   # discrete epochs
N_STATE = 6             # state dimension: position + velocity in 3-D

# ── Initial conditions ────────────────────────────────────────────────────────
X0 = np.array([0., 0., 1000., 10., 5., 0.])       # [x y z vx vy vz]
P0 = np.diag([9., 9., 9., 0.25, 0.25, 0.25])      # initial covariance

# ── Dynamics model (constant-velocity) ────────────────────────────────────────
F = np.eye(N_STATE)
F[:3, 3:] = DT * np.eye(3)

# Process noise covariance — Section II-E (exact manuscript values)
_Q_MANUSCRIPT = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
Q = np.diag(_Q_MANUSCRIPT)

# ── Sensor 1: GNSS position ──────────────────────────────────────────────────
H_GNSS = np.hstack([np.eye(3), np.zeros((3, 3))])        # 3×6
R_GNSS = np.diag([25., 25., 25.])                         # σ² = 25 m² → σ = 5 m/axis

# ── Sensor 2: RF Range Beacon ────────────────────────────────────────────────
BEACON  = np.array([5000., 5000., 0.])                    # 3-D coordinates [m]
SIGMA_R = 10.0                                            # range noise std dev [m]

# ── ECT Vulnerability Parameters ─────────────────────────────────────────────
A_PERT = 1.2            # amplitude of spoofing bias [m] (Section II-E)
OMEGA  = 0.05           # angular frequency [rad/s]      (Section II-E)

# ── Performance & Precondition Metrics ───────────────────────────────────────
GAMMA_CRIT = 6.5        # Instability threshold
CEP_K      = 1.1774     # Conversion from 1σ to CEP (eq. 9)
R_L        = 15.0       # Lethal radius for MKI [m]
CHI2_GATE  = 7.815      # 95% threshold for χ² with 3 DOF

# ═══════════════════════════════════════════════════════════════════════════════
#  Core EKF Logic
# ═══════════════════════════════════════════════════════════════════════════════

def _ekf_update(x_pred, p_pred, z_gnss, z_range, dz_pert=None, r_gnss_override=None):
    """
    Perform sequential update: GNSS followed by Range.
    """
    if r_gnss_override is None: 
        r_gnss_override = R_GNSS

    # 1. GNSS Update (Linear)
    z_g_actual = z_gnss.copy()
    if dz_pert is not None:
        z_g_actual += dz_pert

    S = H_GNSS @ p_pred @ H_GNSS.T + r_gnss_override
    S_inv = np.linalg.inv(S)
    innov_g = z_g_actual - (H_GNSS @ x_pred)
    K = p_pred @ H_GNSS.T @ S_inv

    IKH = np.eye(N_STATE) - K @ H_GNSS
    P_g = IKH @ p_pred @ IKH.T + K @ r_gnss_override @ K.T
    x_g = x_pred + K @ innov_g

    nis = innov_g.T @ S_inv @ innov_g

    if nis > CHI2_GATE:
        P_g = p_pred
        x_g = x_pred

    # 2. Range Update (Nonlinear)
    dx = x_g[0] - BEACON[0]
    dy = x_g[1] - BEACON[1]
    dz = x_g[2] - BEACON[2]
    r_est = np.sqrt(dx**2 + dy**2 + dz**2)
    H_r = np.array([[dx/r_est, dy/r_est, dz/r_est, 0., 0., 0.]])

    innov_r = z_range - r_est
    S_r = H_r @ P_g @ H_r.T + SIGMA_R**2
    K_r = P_g @ H_r.T / S_r[0, 0]

    x_upd = x_g + K_r.flatten() * innov_r
    IKH_r = np.eye(N_STATE) - K_r @ H_r
    P_upd = IKH_r @ P_g @ IKH_r.T + K_r * SIGMA_R**2 @ K_r.T

    return x_upd, P_upd, nis, nis <= CHI2_GATE


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-trajectory simulation
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_trajectory(seed: int, perturbed: bool, a_pert_override=None, 
                         omega_override=None, q_override=None, r_gnss_override=None):
    """
    Propagate one T-second trajectory through the 6-state EKF.
    """
    if a_pert_override is None: a_pert_override = A_PERT
    if omega_override is None: omega_override = OMEGA
    if q_override is None: q_override = Q
    if r_gnss_override is None: r_gnss_override = R_GNSS

    rng    = np.random.default_rng(seed)
    x_true = X0.copy()
    x_est  = X0.copy()
    P_est  = P0.copy()

    mse_arr = np.empty(N_STEPS)
    nis_arr = np.empty(N_STEPS)
    cep_arr = np.empty(N_STEPS)
    pos_error_norm = np.empty(N_STEPS)

    for k in range(N_STEPS):
        x_true = F @ x_true + rng.multivariate_normal(np.zeros(N_STATE), q_override)
        z_gnss  = x_true[:3] + rng.multivariate_normal(np.zeros(3), r_gnss_override)
        z_range = np.linalg.norm(x_true[:3] - BEACON) + rng.normal(0., SIGMA_R)

        dz = None
        if perturbed:
            dz = a_pert_override * np.array([
                np.sin(omega_override * k),
                np.cos(omega_override * k),
                np.sin(omega_override * k + np.pi / 3.)
            ])

        x_pred = F @ x_est
        P_pred = F @ P_est @ F.T + q_override

        x_est, P_est, nis_k, _ = _ekf_update(x_pred, P_pred, z_gnss, z_range, dz, r_gnss_override=r_gnss_override)

        pos_err = x_true[:3] - x_est[:3]
        mse_arr[k] = np.dot(pos_err, pos_err)
        nis_arr[k] = nis_k
        cep_arr[k] = CEP_K * np.sqrt((P_est[0, 0] + P_est[1, 1]) / 2.)
        pos_error_norm[k] = np.linalg.norm(x_true[:2] - x_est[:2])

    return mse_arr, nis_arr, cep_arr, pos_error_norm


# ═══════════════════════════════════════════════════════════════════════════════
#  Monte Carlo engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_mc(n_mc=N_MC, verbose=True, a_pert_override=None, omega_override=None, 
           q_override=None, r_gnss_override=None):
    """
    Execute n_mc paired (nominal, perturbed) trajectories.
    """
    nom_mse = np.zeros((n_mc, N_STEPS))
    nom_nis = np.zeros((n_mc, N_STEPS))
    nom_cep = np.zeros((n_mc, N_STEPS))
    nom_pos_err = np.zeros((n_mc, N_STEPS))

    pert_mse = np.zeros((n_mc, N_STEPS))
    pert_nis = np.zeros((n_mc, N_STEPS))
    pert_cep = np.zeros((n_mc, N_STEPS))
    pert_pos_err = np.zeros((n_mc, N_STEPS))

    if verbose:
        print("="*60)
        print(f"  ECT 3-D MC Simulation v1.4.1")
        print(f"  N={n_mc}  T={T}s  Δt={DT}s  A={a_pert_override if a_pert_override is not None else A_PERT}m  ω={omega_override if omega_override is not None else OMEGA} rad/s")
        print("="*60)

    nom_iter = range(n_mc)
    if verbose: nom_iter = tqdm(nom_iter, desc="Nominal  ")
    for i in nom_iter:
        nom_mse[i], nom_nis[i], nom_cep[i], nom_pos_err[i] = _simulate_trajectory(
            seed=i, perturbed=False, 
            a_pert_override=a_pert_override, omega_override=omega_override, 
            q_override=q_override, r_gnss_override=r_gnss_override
        )

    pert_iter = range(n_mc)
    if verbose: pert_iter = tqdm(pert_iter, desc="Perturbed")
    for i in pert_iter:
        # Seed offset +100000 ensures perturbed runs are independent
        # realizations (different noise draws) rather than paired with
        # the nominal run at the same index.
        pert_mse[i], pert_nis[i], pert_cep[i], pert_pos_err[i] = _simulate_trajectory(
            seed=i + 100000, perturbed=True, 
            a_pert_override=a_pert_override, omega_override=omega_override, 
            q_override=q_override, r_gnss_override=r_gnss_override
        )

    return {
        't': np.arange(N_STEPS) * DT,
        'nom_mse': nom_mse, 'nom_nis': nom_nis, 'nom_cep': nom_cep, 'nom_pos_err': nom_pos_err,
        'pert_mse': pert_mse, 'pert_nis': pert_nis, 'pert_cep': pert_cep, 'pert_pos_err': pert_pos_err
    }

def cep_steady(res_dict, tail=200):
    nom  = np.median(res_dict['nom_cep'][:, -tail:])
    pert = np.median(res_dict['pert_cep'][:, -tail:])
    return nom, pert

def pos_err_steady(res_dict, tail=200):
    nom  = np.median(res_dict['nom_pos_err'][:, -tail:])
    pert = np.median(res_dict['pert_pos_err'][:, -tail:])
    return nom, pert

def gamma_series(res_dict):
    nom_mean_mse = np.mean(res_dict['nom_mse'], axis=0)
    gamma_all    = res_dict['pert_mse'] / nom_mean_mse
    return np.mean(gamma_all, axis=0), gamma_all

def nis_compliance(res_dict):
    """Return NIS compliance as fractions in [0, 1]."""
    nom_comp  = np.mean(res_dict['nom_nis'] <= CHI2_GATE)
    pert_comp = np.mean(res_dict['pert_nis'] <= CHI2_GATE)
    return nom_comp, pert_comp

def print_summary(res_dict):
    nom_c, pert_c = cep_steady(res_dict)
    nom_p, pert_p = pos_err_steady(res_dict)
    
    deg = ((pert_p / nom_p) - 1.0) * 100 if nom_p > 0 else 0
    _, gamma_all = gamma_series(res_dict)
    
    # Γ exceedance metrics
    any_time = (np.any(gamma_all >= GAMMA_CRIT, axis=1)).mean() * 100
    final_time = (gamma_all[:, -1] >= GAMMA_CRIT).mean() * 100
    sustained = (np.all(gamma_all[:, -200:] >= GAMMA_CRIT, axis=1)).mean() * 100

    nc, pc = nis_compliance(res_dict)
    nc *= 100; pc *= 100  # convert fractions to % for display
    mki = pert_p / R_L

    print("\n" + "="*60)
    print("  ECT SIMULATION RESULTS (Unmodified Manuscript Parameters)")
    print("="*60)
    print(f"  Filter CEP   (nominal) : {nom_c:.2f} m")
    print(f"  Filter CEP   (perturbed): {pert_c:.2f} m")
    print(f"  True Pos Err (nominal) : {nom_p:.2f} m")
    print(f"  True Pos Err (perturbed): {pert_p:.2f} m")
    print(f"  True Error Degradation : {deg:.0f}%")
    print("-" * 60)
    print("  Γ Exceedance Metrics:")
    print(f"    Any-time exceedance     : {any_time:.1f}%")
    print(f"    Final-time exceedance   : {final_time:.1f}%")
    print(f"    Sustained (last 200s)   : {sustained:.1f}%")
    print("-" * 60)
    print(f"  NIS compliance (nominal):    {nc:.1f}%")
    print(f"  NIS compliance (perturbed):  {pc:.1f}%")
    print(f"  Mission Kill Index (R_L={R_L}m): {mki:.3f}")
    if mki > 0.5:
        print(f"  → ECT precondition met; SMK confirmed for R_L ≤ {pert_p:.1f} m")
    else:
        print(f"  → MKI < 0.5. Soft Mission Kill not achieved at R_L = {R_L} m")
    print("="*60 + "\n")

if __name__ == '__main__':
    res = run_mc(n_mc=30, verbose=True)
    print_summary(res)
