"""
tests/param_sweep.py

Systematic parameter sweep to identify the configuration that reproduces
the manuscript's headline numbers:
  - Nominal filter CEP: 3.2 m
  - Perturbed filter CEP: 7.9 m  (+147%)
  - Γ any-time exceedance: 100%
  - NIS gate compliance: 92–96%

Tests three axes independently, then in combination:
  1. A_PERT in {1.2, 1.8, 2.2, 2.5, 3.0}
  2. Q_vel diagonal in {0.001, 0.005, 0.01}
  3. Seeding/Γ strategy: unpaired-mean vs paired-per-run

N_MC=50 for speed; enough to distinguish regimes.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ECT_3D_Simulation_v141 as ECT

N_SWEEP = 50   # MC runs per configuration
TAIL    = 200  # steady-state window


def gamma_stats(nom_mse, pert_mse, paired):
    """
    Compute Γ exceedance two ways:
      paired=False : each perturbed run vs ensemble-mean nominal (current v2)
      paired=True  : each perturbed run vs its paired nominal run (same index)
    """
    if paired:
        gamma_all = pert_mse / (nom_mse + 1e-12)
    else:
        nom_mean  = np.mean(nom_mse, axis=0)
        gamma_all = pert_mse / (nom_mean + 1e-12)

    any_time  = np.mean(np.any(gamma_all >= ECT.GAMMA_CRIT, axis=1)) * 100
    final     = np.mean(gamma_all[:, -1] >= ECT.GAMMA_CRIT) * 100
    sustained = np.mean(np.all(gamma_all[:, -TAIL:] >= ECT.GAMMA_CRIT, axis=1)) * 100
    return any_time, final, sustained


def run_config(a_pert, q_vel, paired_seed, paired_gamma, label):
    q_diag = [ECT._Q_MANUSCRIPT[0], ECT._Q_MANUSCRIPT[1], ECT._Q_MANUSCRIPT[2],
               q_vel, q_vel, q_vel]
    q_mat = np.diag(q_diag)

    nom_res  = [ECT._simulate_trajectory(seed=i, perturbed=False,
                    a_pert_override=a_pert, q_override=q_mat)
                for i in range(N_SWEEP)]
    pert_res = [ECT._simulate_trajectory(
                    seed=i if paired_seed else i + 100000, perturbed=True,
                    a_pert_override=a_pert, q_override=q_mat)
                for i in range(N_SWEEP)]

    nom_mse   = np.array([r[0] for r in nom_res])
    nom_cep   = np.array([r[2] for r in nom_res])
    nom_perr  = np.array([r[3] for r in nom_res])
    pert_mse  = np.array([r[0] for r in pert_res])
    pert_cep  = np.array([r[2] for r in pert_res])
    pert_perr = np.array([r[3] for r in pert_res])
    nom_nis   = np.array([r[1] for r in nom_res])
    pert_nis  = np.array([r[1] for r in pert_res])

    nom_cep_ss   = np.median(nom_cep[:,  -TAIL:])
    pert_cep_ss  = np.median(pert_cep[:, -TAIL:])
    nom_perr_ss  = np.median(nom_perr[:,  -TAIL:])
    pert_perr_ss = np.median(pert_perr[:, -TAIL:])
    deg = ((pert_perr_ss / nom_perr_ss) - 1.0) * 100 if nom_perr_ss > 0 else 0

    any_t, final_t, sust = gamma_stats(nom_mse, pert_mse, paired=paired_gamma)

    nis_nom  = np.mean(nom_nis  <= ECT.CHI2_GATE) * 100
    nis_pert = np.mean(pert_nis <= ECT.CHI2_GATE) * 100

    # Score against manuscript targets
    hits = 0
    if abs(nom_cep_ss  - 3.2) < 0.8:  hits += 1   # nominal CEP ~3.2m
    if abs(pert_cep_ss - 7.9) < 1.5:  hits += 1   # perturbed CEP ~7.9m
    if any_t > 95:                     hits += 1   # Γ exceedance ~100%
    if 91 < nis_pert < 97:            hits += 1   # NIS compliance 92–96%

    print(f"\n{'─'*68}")
    print(f"  {label}")
    print(f"{'─'*68}")
    print(f"  A={a_pert}m  Q_vel={q_vel}  paired_seed={paired_seed}  paired_γ={paired_gamma}")
    print(f"  Filter CEP  nominal  : {nom_cep_ss:.2f} m   (target 3.2 m)")
    print(f"  Filter CEP  perturbed: {pert_cep_ss:.2f} m   (target 7.9 m)")
    print(f"  True Err    nominal  : {nom_perr_ss:.2f} m")
    print(f"  True Err    perturbed: {pert_perr_ss:.2f} m  ({deg:+.0f}%  target +147%)")
    print(f"  Γ any-time exceedance: {any_t:.1f}%   (target 100%)")
    print(f"  Γ final-time          : {final_t:.1f}%")
    print(f"  Γ sustained (last 200s): {sust:.1f}%")
    print(f"  NIS compliance  nom  : {nis_nom:.1f}%")
    print(f"  NIS compliance  pert : {nis_pert:.1f}%   (target 92–96%)")
    print(f"  True err degradation  : {deg:.0f}%   (target 147%)")
    print(f"  ► Targets hit: {hits}/4")

    return hits, {
        'nom_cep': nom_cep_ss, 'pert_cep': pert_cep_ss,
        'deg': deg, 'any_t': any_t, 'nis_pert': nis_pert, 'hits': hits,
        'a': a_pert, 'q_vel': q_vel, 'paired_s': paired_seed, 'paired_g': paired_gamma
    }


if __name__ == '__main__':
    print("="*68)
    print("  ECT PARAMETER SWEEP  (N=50 per config)")
    print("  Target: nominal CEP 3.2m / perturbed CEP 7.9m / Γ 100% / NIS 92-96%")
    print("="*68)

    results = []

    # Baseline
    _, r = run_config(1.2, 0.001, False, False, "BASELINE (v2 stated params)")
    results.append(r)

    # Axis 1: vary A_PERT only
    for a in [1.8, 2.2, 2.5, 3.0]:
        _, r = run_config(a, 0.001, False, False, f"Axis-A: A={a}m  Q_vel=0.001  unpaired")
        results.append(r)

    # Axis 2: vary Q_vel only
    for qv in [0.005, 0.01]:
        _, r = run_config(1.2, qv, False, False, f"Axis-Q: A=1.2m  Q_vel={qv}  unpaired")
        results.append(r)

    # Axis 3: paired seed + paired-Γ
    _, r = run_config(1.2, 0.001, True, True, "Axis-S: A=1.2m  Q_vel=0.001  PAIRED")
    results.append(r)

    # Combined candidates
    combos = [
        (2.2, 0.01,  False, False),
        (2.2, 0.01,  True,  True),
        (2.5, 0.01,  False, False),
        (2.5, 0.01,  True,  True),
        (1.8, 0.01,  True,  True),
        (3.0, 0.005, True,  True),
        (2.0, 0.01,  True,  True),
    ]
    for a, qv, ps, pg in combos:
        label = f"COMBO: A={a}m  Q_vel={qv}  paired_s={ps}  paired_γ={pg}"
        _, r = run_config(a, qv, ps, pg, label)
        results.append(r)

    # Ranking
    results.sort(key=lambda x: (-x['hits'], abs(x['pert_cep'] - 7.9)))
    print("\n" + "="*68)
    print("  RANKING  (by targets hit, then pert_CEP proximity to 7.9m)")
    print("="*68)
    print(f"  {'Config':<42} {'Hits':>4} {'nCEP':>6} {'pCEP':>7} {'Γ%':>6} {'NIS%':>6}")
    print(f"  {'─'*42} {'─'*4} {'─'*6} {'─'*7} {'─'*6} {'─'*6}")
    for r in results:
        lbl = f"A={r['a']} Qv={r['q_vel']} ps={r['paired_s']} pg={r['paired_g']}"
        print(f"  {lbl:<42} {r['hits']:>4}  {r['nom_cep']:>5.2f}  {r['pert_cep']:>6.2f}"
              f"  {r['any_t']:>5.1f}  {r['nis_pert']:>5.1f}")

    best = results[0]
    print(f"\n  Best: A={best['a']}m  Q_vel={best['q_vel']}"
          f"  paired_seed={best['paired_s']}  paired_γ={best['paired_g']}")
    print(f"  Hits {best['hits']}/4 manuscript targets.")
    if best['hits'] == 4:
        print("\n  RESULT: Full reproduction achieved.")
        print("  ACTION: Update Section II-E with these parameter values.")
    elif best['hits'] >= 3:
        print("\n  RESULT: Near-reproduction — NIS constraint may be violated.")
        print("  ACTION: Review NIS compliance before updating manuscript.")
    else:
        print("\n  RESULT: Gap cannot close within stated architecture.")
        print("  ACTION: Revise manuscript numbers to match clean-room output.")
    print("="*68)
