import pytest
import numpy as np
import sys
import os

# Ensure the parent directory is in the path so we can import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ECT_3D_Simulation_v141 as ECT

def test_nominal_filter_cep():
    """
    Test that the clean-room implementation of the manuscript equations 
    produces the expected filter CEP. The manuscript claims 3.2m, but 
    the actual stated equations and parameters (Q=0.001) yield ~1.9m.
    """
    res = ECT.run_mc(n_mc=10, verbose=False)
    nom_c, _ = ECT.cep_steady(res)
    # Expected honest result without hidden scale factors is ~1.9m
    assert abs(nom_c - 1.9) < 0.5, f"Expected ~1.9m nominal filter CEP, got {nom_c:.2f}m"

def test_true_position_error_degradation():
    """
    Test that the true position error degradation with stated parameters 
    is minimal (~14%), rather than the 147% claimed in the paper.
    """
    res = ECT.run_mc(n_mc=10, verbose=False)
    nom_p, pert_p = ECT.pos_err_steady(res)
    deg = ((pert_p / nom_p) - 1.0) * 100
    assert deg < 30.0, f"Expected minimal degradation (<30%) with stated parameters, got {deg:.1f}%"

def test_gamma_exceedance():
    """
    Test that ANY-time gamma exceedance matches the manuscript's 100% claim,
    but final-time or sustained exceedance does not.
    """
    res = ECT.run_mc(n_mc=10, verbose=False)
    _, gamma_all = ECT.gamma_series(res)
    
    any_time = (np.any(gamma_all >= ECT.GAMMA_CRIT, axis=1)).mean() * 100
    final_time = (gamma_all[:, -1] >= ECT.GAMMA_CRIT).mean() * 100
    
    assert any_time > 80.0, f"Expected high any-time exceedance, got {any_time:.1f}%"
    assert final_time < 20.0, f"Expected low final-time exceedance, got {final_time:.1f}%"

def test_nis_compliance():
    """
    Test that the NIS compliance matches the 92-96% range claimed.
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
    Guard against accidental parameter drift. The manuscript's exact Q 
    diagonal must be [0.01, 0.01, 0.01, 0.001, 0.001, 0.001].
    """
    expected = [0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
    actual = np.diag(ECT.Q).tolist()
    assert actual == expected, f"Q diagonal drifted from manuscript: {actual}"
