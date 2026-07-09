"""
Verification tests for the v2.1 archival engine (ECT_3D_Simulation_v2_1.py).

These lock in what the v2.1 code *actually* produces at the manuscript-spec
parameter set (Q_vel = 0.001, sigma_range = 25 m, master seed 42), so the
numbers can never silently drift and the two known discrepancies with the
manuscript text stay documented in code:

  * Filter-reported CEP is 1.92 m, NOT the 2.43 m printed in Table I.
  * The per-run Gamma "any-time exceedance" statistic is vacuous — it fires
    for ~100% of *unperturbed* pairs too, so it cannot discriminate an attack.

Everything else (the +67% TPE growth, NIS ~94.8%, CEP invariance, MKI) is the
genuine, seed-robust result and is asserted here as a regression guard.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import ECT_3D_Simulation_v2_1 as V21

N = 25  # small ensemble; the v2.1 metrics are deterministic Riccati-anchored


@pytest.fixture(scope="module")
def metrics():
    arrays = V21.run_monte_carlo(n_runs=N, verbose=False)
    m = V21.compute_metrics(*arrays)
    # keep the raw nominal MSE for the false-positive test
    m["_mse_nom"] = arrays[0]
    return m


def test_true_error_growth_is_67pct(metrics):
    """Genuine, seed-robust: nominal ~2.6 m -> perturbed ~4.4 m (+67%)."""
    assert 58.0 < metrics["tpe_growth_pct"] < 76.0, \
        f"TPE growth {metrics['tpe_growth_pct']:.1f}% off the verified +67%"


def test_filter_cep_is_invariant(metrics):
    """The 'Confidently Wrong' core: filter CEP does not move under attack."""
    assert abs(metrics["cep_delta"]) < 0.05, \
        f"filter CEP shifted {metrics['cep_delta']:.3f} m — expected ~0"


def test_filter_cep_is_1p92_not_manuscript_2p43(metrics):
    """
    DOCUMENTED DISCREPANCY. Table I / Fig 4 / prose all state 2.43 m, but the
    v2.1 code produces 1.92 m at every seed (2.43 was the v2.0 Riccati point).
    This test passes by confirming the code does NOT produce 2.43 m — it must be
    fixed in the manuscript, not the code.
    """
    cep = metrics["mean_cep_pert_ss"]
    assert 1.75 < cep < 2.10, f"v2.1 filter CEP {cep:.3f} m outside the verified ~1.92 m band"
    assert abs(cep - 2.43) > 0.25, \
        f"filter CEP {cep:.3f} m unexpectedly near the manuscript's 2.43 m — recheck"


def test_nis_compliance_near_95(metrics):
    """Monitor sees nothing: NIS compliance ~95% nominal, ~94.8% perturbed."""
    assert 93.0 < metrics["nis_nom_compliance"] < 97.0
    assert 93.0 < metrics["nis_pert_compliance"] < 97.0


def test_mki_and_disclosed_nominal_baseline(metrics):
    """
    MKI = 0.29 at R_L=15 m (no kill) and 1.46 at R_L=3 m ('SMK'). But the
    nominal system already sits at ~0.87 at 3 m with no attack — that baseline
    must be disclosed alongside the 1.46 claim.
    """
    assert 0.24 < metrics["mki_conservative"] < 0.34, "MKI@15m off verified 0.29"
    assert 1.30 < metrics["mki_precision"] < 1.62, "MKI@3m off verified 1.46"
    nominal_mki_3m = metrics["mean_tpe_nom_ss"] / V21.R_L_PRECISION
    assert nominal_mki_3m > 0.75, \
        f"nominal MKI@3m {nominal_mki_3m:.2f} — the undisclosed baseline behind 'SMK confirmed'"


def test_gamma_any_time_statistic_is_vacuous(metrics):
    """
    The per-run instantaneous MSE-ratio 'any-time exceedance' fires for
    unperturbed pairs too. Score nominal run i vs run i-1 with the same rule:
    the false-positive rate must be high, proving the 100% claim non-diagnostic.
    """
    mse_nom = metrics["_mse_nom"]
    g_fp = mse_nom[1:] / np.maximum(mse_nom[:-1], 1e-9)
    fp_rate = (g_fp > V21.GAMMA_CRIT).any(axis=1).mean() * 100
    assert fp_rate > 50.0, \
        f"expected a high false-positive rate exposing the vacuous statistic, got {fp_rate:.0f}%"
