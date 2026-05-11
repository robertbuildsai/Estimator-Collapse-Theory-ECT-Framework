import pytest
import numpy as np
import sys
import os

# Ensure the parent directory is in the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ECT_3D_Simulation_v141 as ECT

def test_nominal_filter_cep():
    """
    Filter CEP at verified parameters (A=2.5m, Q_vel=0.01) converges to ~2.43m.
    """
    res = ECT.run_mc(n_mc=10, verbose=False)
    nom_c, _ = ECT.cep_steady(res)
    assert abs(nom_c - 2.43) < 0.4, f"Expected ~2.43m nominal filter CEP, got {nom_c:.2f}m"

def test_filter_cep_invariant_under_perturbation():
    """
    Core ECT finding: filter-reported CEP is structurally invariant under
    gate-compliant sinusoidal perturbation. Nominal and perturbed CEP must
    be within 0.05m of each other — the 'Confidently Wrong' signature.
    """
    res = ECT.run_mc(n_mc=10, verbose=False)
    nom_c, pert_c = ECT.cep_steady(res)
    delta = abs(pert_c - nom_c)
    assert delta < 0.05, f"Filter CEP changed under perturbation by {delta:.3f}m — expected ~0"

def test_true_position_error_degradation():
    """
    True position error degrades ~33% under verified parameters (A=2.5m,
    Q_vel=0.01, paired seeding). Filter reports none of this — Confidently Wrong.
    """
    res = ECT.run_mc(n_mc=10, verbose=False)
    nom_p, pert_p = ECT.pos_err_steady(res)
    deg = ((pert_p / nom_p) - 1.0) * 100
    assert 15.0 < deg < 55.0, f"Expected ~33% true error degradation, got {deg:.1f}%"

def test_gamma_exceedance():
    """
    Γ any-time exceedance reaches 100% with verified parameters.
    Final-time exceedance remains partial (oscillatory injection).
    """
    res = ECT.run_mc(n_mc=10, verbose=False)
    _, gamma_all = ECT.gamma_series(res)

    any_time = (np.any(gamma_all >= ECT.GAMMA_CRIT, axis=1)).mean() * 100
    final_time = (gamma_all[:, -1] >= ECT.GAMMA_CRIT).mean() * 100

    assert any_time >= 90.0, f"Expected ≥90% any-time exceedance, got {any_time:.1f}%"
    assert final_time < 30.0, f"Expected low final-time exceedance, got {final_time:.1f}%"

def test_nis_compliance():
    """
    NIS compliance stays in 92–96% range under perturbation — monitor
    cannot distinguish perturbed from nominal operation.
    nis_compliance() returns fractions in [0, 1].
    """
    res = ECT.run_mc(n_mc=10, verbose=False)
    nc, pc = ECT.nis_compliance(res)

    assert 0.90 < nc < 0.98, f"Nominal NIS compliance {nc:.3f} out of range"
    assert 0.90 < pc < 0.98, f"Perturbed NIS compliance {pc:.3f} out of range"

def test_version():
    """Confirm version string tracks honest implementation."""
    assert hasattr(ECT, '__version__')
    assert 'honest' in ECT.__version__

def test_manuscript_q_preserved():
    """
    Guard against parameter drift. Verified Q diagonal (v2.0.0):
    all six diagonals = 0.01 (position and velocity).
    """
    expected = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    actual = np.diag(ECT.Q).tolist()
    assert actual == expected, f"Q diagonal drifted from verified params: {actual}"
