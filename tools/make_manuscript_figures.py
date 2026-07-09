"""
Generate the canonical manuscript figures (Figures/Figure_2..5.png) from the
v2.1 archival engine, rendered through the shared honest figure generator in
ECT_3D_Simulation_v141.make_figures.

This guarantees the committed figures are produced by the same code that reports
the v2.1 numbers — no hand-assembled or stale figures. Figure_1 (the EKF loop
schematic) is a hand-drawn diagram and is not regenerated here.

Usage:
    python tools/make_manuscript_figures.py [N]      # default N = 500
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import ECT_3D_Simulation_v2_1 as V21          # archival engine (the manuscript run)
import ECT_3D_Simulation_v141 as ECT          # shared, honest figure generator


def v21_results(n_runs):
    """Run the v2.1 archival Monte Carlo and pack it into the res_dict shape
    that ECT.make_figures expects. 'pos_err' carries the v2.1 True Position
    Error (3-D) trace; 'cep' is the filter-reported covariance CEP."""
    (mse_nom, mse_pert, tpe_nom, tpe_pert,
     cep_nom, cep_pert, nis_nom, nis_pert) = V21.run_monte_carlo(n_runs=n_runs, verbose=True)
    return {
        "t": np.arange(V21.N_STEPS) * V21.DT,
        "nom_mse": mse_nom, "pert_mse": mse_pert,
        "nom_nis": nis_nom, "pert_nis": nis_pert,
        "nom_cep": cep_nom, "pert_cep": cep_pert,
        "nom_pos_err": tpe_nom, "pert_pos_err": tpe_pert,
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"Rendering canonical manuscript figures from the v2.1 archival run (N={n})...")
    res = v21_results(n)
    ECT.plt.switch_backend("Agg")
    ECT.make_figures(res, outdir=os.path.join(ROOT, "Figures"),
                     err_label="True position error (TPE)")
    print("Canonical figures written to Figures/ (Figure_2..5).")
