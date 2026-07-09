"""
ECT_3D_Simulation_v141.py

Clean-room reference implementation of the Estimator Collapse Theory (ECT)
Framework for 3-D Sequential EKF (GNSS + Range Fusion).

Derived from the published equations in Section II-E of:
  "Estimator Collapse Theory: A Framework for Predicting
   Filter Instability under Bounded Adversarial Perturbation"

Author: R. J. Douglas
"""

__version__ = '2.0.1-honest'

import sys

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

# Process noise covariance — Section II-E (verified parameter set, v2.0.0)
_Q_MANUSCRIPT = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
Q = np.diag(_Q_MANUSCRIPT)

# ── Sensor 1: GNSS position ──────────────────────────────────────────────────
H_GNSS = np.hstack([np.eye(3), np.zeros((3, 3))])        # 3×6
R_GNSS = np.diag([25., 25., 25.])                         # σ² = 25 m² → σ = 5 m/axis

# ── Sensor 2: RF Range Beacon ────────────────────────────────────────────────
BEACON  = np.array([5000., 5000., 0.])                    # 3-D coordinates [m]
SIGMA_R = 10.0                                            # range noise std dev [m]

# ── ECT Vulnerability Parameters ─────────────────────────────────────────────
A_PERT = 2.5            # amplitude of spoofing bias [m] (Section II-E, verified v2.0.0)
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
        print(f"  ECT 3-D MC Simulation v{__version__}")
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
        # Paired seeding: same seed as nominal run i, isolating perturbation
        # effect from MC variance in the Γ ratio (verified v2.0.0).
        pert_mse[i], pert_nis[i], pert_cep[i], pert_pos_err[i] = _simulate_trajectory(
            seed=i, perturbed=True,
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

def gamma_nominal_false_positive(res_dict):
    """
    False-positive rate of the Γ any-time exceedance statistic: fraction of
    UNPERTURBED runs that cross Γ_crit at least once when scored with the same
    statistic (split-half, so no run is compared against a mean that includes
    itself). Single-run MSE is heavy-tailed, so any-time exceedance fires on
    healthy runs too — report this baseline next to the perturbed rate.
    """
    nom_mse = res_dict['nom_mse']
    half = nom_mse.shape[0] // 2
    if half < 1:
        return float('nan')
    g_a = nom_mse[:half] / np.mean(nom_mse[half:], axis=0)
    g_b = nom_mse[half:] / np.mean(nom_mse[:half], axis=0)
    g = np.vstack([g_a, g_b])
    return float((g >= GAMMA_CRIT).any(axis=1).mean())

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
    fp = gamma_nominal_false_positive(res_dict) * 100

    print("\n" + "="*60)
    print(f"  ECT SIMULATION RESULTS (v{__version__} verified parameters)")
    print("="*60)
    print(f"  Filter CEP   (nominal) : {nom_c:.2f} m")
    print(f"  Filter CEP   (perturbed): {pert_c:.2f} m")
    print(f"  True Pos Err (nominal) : {nom_p:.2f} m")
    print(f"  True Pos Err (perturbed): {pert_p:.2f} m")
    print(f"  True Error Degradation : {deg:.0f}%")
    print("-" * 60)
    print("  Γ Exceedance Metrics:")
    print(f"    Any-time exceedance     : {any_time:.1f}%")
    print(f"    Nominal false-positive  : {fp:.1f}%  (same statistic, no attack)")
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

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure generation (Figures 2–5)
# ═══════════════════════════════════════════════════════════════════════════════

def make_figures(res_dict, outdir='Figures', dpi=300):
    """
    Regenerate Figures 2–5 from a run_mc result so the archived figures always
    match the code that claims to produce them. Figure 1 (EKF loop diagram) is
    a hand-drawn schematic and is not regenerated.
    """
    import os
    os.makedirs(outdir, exist_ok=True)
    t = res_dict['t']
    n_runs = res_dict['nom_mse'].shape[0]
    gamma_t, gamma_all = gamma_series(res_dict)
    fp = gamma_nominal_false_positive(res_dict) * 100

    # ── Figure 2: Γ(t) temporal evolution ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    p5, p95 = np.percentile(gamma_all, [5, 95], axis=0)
    ax.fill_between(t, p5, p95, alpha=0.18, color='#2a78d6', label='5th–95th percentile')
    ax.plot(t, gamma_t, color='#2a78d6', lw=1.8, label='Mean Γ(t) — perturbed')
    ax.axhline(GAMMA_CRIT, color='#d03b3b', ls='--', lw=1.4, label=f'Γ_crit = {GAMMA_CRIT}')
    ax.axhline(1.0, color='#1baf7a', ls='--', lw=1.2, label='Nominal (Γ = 1)')
    ax.set(xlabel='Time [s]', ylabel='Γ(t)', xlim=(0, T),
           title=f'Fig 2. Estimator Instability Number Γ(t), N = {n_runs} (v{__version__})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{outdir}/Figure_2.png', dpi=dpi); plt.close(fig)

    # ── Figure 3: per-run max-Γ distribution vs nominal false-positive ──────
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    nom_mse = res_dict['nom_mse']
    half = n_runs // 2
    g_fp = np.vstack([nom_mse[:half] / np.mean(nom_mse[half:], axis=0),
                      nom_mse[half:] / np.mean(nom_mse[:half], axis=0)])
    bins = np.linspace(0, max(gamma_all.max(axis=1).max(), g_fp.max(axis=1).max()) * 1.05, 40)
    ax.hist(g_fp.max(axis=1), bins=bins, color='#1baf7a', alpha=0.65,
            label='Nominal runs (false-positive baseline)')
    ax.hist(gamma_all.max(axis=1), bins=bins, color='#2a78d6', alpha=0.65,
            label='Perturbed runs')
    ax.axvline(GAMMA_CRIT, color='#d03b3b', ls='--', lw=1.4, label=f'Γ_crit = {GAMMA_CRIT}')
    ax.set(xlabel='max Γ over run', ylabel='Runs',
           title=f'Fig 3. Any-time max Γ per run: perturbed vs unperturbed baseline '
                 f'(FP rate {fp:.0f}%), N = {n_runs}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{outdir}/Figure_3.png', dpi=dpi); plt.close(fig)

    # ── Figure 4: Confidently Wrong — filter CEP invariant, true error grows ─
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(t, np.median(res_dict['nom_pos_err'], axis=0), color='#1baf7a', lw=1.6,
            label='True horizontal error — nominal')
    ax.plot(t, np.median(res_dict['pert_pos_err'], axis=0), color='#2a78d6', lw=1.6,
            label='True horizontal error — perturbed')
    ax.plot(t, np.median(res_dict['nom_cep'], axis=0), color='#eda100', lw=1.6, ls='--',
            label='Filter-reported CEP — nominal')
    ax.plot(t, np.median(res_dict['pert_cep'], axis=0), color='#d03b3b', lw=1.2, ls=':',
            label='Filter-reported CEP — perturbed (overlaps nominal)')
    ax.set(xlabel='Time [s]', ylabel='Error [m]', xlim=(0, T),
           title=f'Fig 4. "Confidently Wrong": filter CEP invariant while true error grows, '
                 f'N = {n_runs}')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(f'{outdir}/Figure_4.png', dpi=dpi); plt.close(fig)

    # ── Figure 5: NIS gate compliance ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    nc, pc = nis_compliance(res_dict)
    for ax, key, comp, color, title in [
        (axes[0], 'nom_nis',  nc, '#1baf7a', '(a) Nominal'),
        (axes[1], 'pert_nis', pc, '#2a78d6', '(b) Perturbed'),
    ]:
        data = res_dict[key]
        p5, p95 = np.percentile(data, [5, 95], axis=0)
        ax.fill_between(t, p5, p95, color=color, alpha=0.22)
        ax.plot(t, np.median(data, axis=0), color=color, lw=1.4, label='Median NIS')
        ax.axhline(CHI2_GATE, color='#d03b3b', ls=':', lw=1.4, label=f'χ²₃ gate = {CHI2_GATE}')
        ax.set(xlabel='Time [s]', ylabel='NIS', xlim=(0, T), title=title)
        ax.text(0.97, 0.96, f'Gate compliance: {comp*100:.1f}%', transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3)
    fig.suptitle(f'Fig 5. NIS gate compliance, N = {n_runs} (v{__version__})', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{outdir}/Figure_5.png', dpi=dpi); plt.close(fig)

    print(f"Figures 2–5 written to {outdir}/")


if __name__ == '__main__':
    # Usage: python ECT_3D_Simulation_v141.py [N_MC] [--no-figures]
    # Default N=500 reproduces the README results table (~2 min).
    args = [a for a in sys.argv[1:] if a != '--no-figures']
    n = int(args[0]) if args else N_MC
    res = run_mc(n_mc=n, verbose=True)
    print_summary(res)
    if '--no-figures' not in sys.argv:
        plt.switch_backend('Agg')
        make_figures(res)
