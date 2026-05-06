# The Sophistication Paradox: A Systems-Theoretic Framework for Estimator Collapse in Precision-Guided Autonomous Navigation Architectures 📉

### **Formal Implementation of the Estimator Collapse Theory (ECT) Framework**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20037820.svg)](https://doi.org/10.5281/zenodo.20037820)

> **Note:** This repository hosts the code and formal proofs for the ECT framework as described in the 2026 manuscript of the same name.

---

### **Project Overview**
The **Estimator Collapse Theory (ECT)** framework formalizes the conditions under which traditional Bayesian estimators fail when encountering out-of-distribution (OOD) human postures.


# The Sophistication Paradox: A Systems-Theoretic Framework for Estimator Collapse in Precision-Guided Autonomous Navigation Architectures

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20037820.svg)](https://doi.org/10.5281/zenodo.20037820)
[![Version](https://img.shields.io/badge/version-v1.4.2-blue)](https://doi.org/10.5281/zenodo.20037820)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](requirements.txt)
[![Status](https://img.shields.io/badge/manuscript-in%20preparation-orange)]()

**Version:** v1.4.2 – Archival GA & Metadata Release (May 2026) | **Author:** N. Barua, AN Holdings Co. / Kobe Gakuin University | **ORCID:** [0000-0003-4641-0112](https://orcid.org/0000-0003-4641-0112)

---

<div align="center">
  <img src="Graphical_Abstract.png" width="900" alt="Graphical Abstract: Estimator Collapse Theory">
  <br><em>Graphical Abstract — Estimator Collapse Theory & The Sophistication Paradox</em>
</div>

---

## 📌 Overview

This repository provides the reference implementation, Monte Carlo validation scripts, and video demonstrations for **Estimator Collapse Theory (ECT)** — a systems-theoretic framework characterising how state-estimation destabilisation constitutes a formal failure pathway in precision autonomous navigation architectures.

ECT identifies a class of **sub-threshold structured perturbations** — bounded measurement disturbances calibrated to remain within standard statistical gating thresholds — that systematically corrupt the state estimate of an Extended Kalman Filter (EKF). In this regime, the filter reports high confidence in a position solution that no longer satisfies operational accuracy requirements. This failure mode is termed **"Confidently Wrong"**: actual position error diverges while onboard monitors report nominal behaviour throughout.

> *Software companion to:* **"The Sophistication Paradox: A Systems-Theoretic Framework for Estimator Collapse in Precision-Guided Autonomous Navigation Architectures"** — manuscript in preparation for submission to CEAS Aeronautical Journal (2026).

---

## 📺 Visual Demonstration: Operational Failure Transition

The video below evaluates the real-time transition from nominal state estimation to estimator collapse under structured perturbation.

https://github.com/Nick-Barua/Estimator-Collapse-Theory-ECT-Framework/blob/main/ECT_SMK_Conceptual_Overview.mp4.mp4

| Panel | Description |
|-------|-------------|
| **1 — Instability Number** | Contrasts actual estimation error with filter-reported covariance, illustrating divergence of estimator consistency |
| **2 — Telemetry** | Real-time tracking of the Mission Kill Index (MKI); failure indicated as $\Gamma(t)$ crosses $\Gamma_{crit} = 6.5$ |
| **3 — CEP** | Expansion of Circular Error Probable relative to operational tolerance $R_L$ |
| **4 — Information Theory** | Systematic growth of Shannon Entropy $h(X)$, representing maximal uncertainty growth in the guidance solution |

---

## 🧮 The Sophistication Paradox

ECT formalises the **Sophistication Paradox** via the Vulnerability Index $V_i$:

$$V_i = \frac{N_{\text{sensors}} \times f_{\text{update}}}{R_{\text{hardening}}}$$

This captures a counter-intuitive relationship: advanced multi-sensor fusion architectures improve nominal precision while simultaneously expanding the estimator's vulnerability surface. A single-sensor INS platform yields $V_i \approx 1$; a high-dynamics multi-sensor architecture yields $V_i \approx 67$–$320$.

| System Class | $N_s$ | $f$ (Hz) | $R_h$ | $V_i$ |
|---|---|---|---|---|
| Single-Sensor INS | 1 | 1 | 1.0 | ≈ 1 |
| Dual-Modal Midcourse | 3 | 10 | 1.0 | ≈ 30 |
| High-Precision Multi-Sensor | 4 | 20 | 1.2 | ≈ 67 |
| **High-Dynamics Platform** | 4 | 80 | 1.0 | **≈ 320** |

<div align="center">
  <img src="Figures/Figure_1.png" width="750" alt="EKF Loop and Attack Points">
  <br><em>Fig 1. Extended Kalman Filter loop and principal perturbation entry points (A: Measurement Bias, B: Covariance Inflation, C: GNSS Denial).</em>
</div>

---

## 🧪 3-D Monte Carlo Validation (v1.4.1)

The included `ECT_3D_Simulation_v141.py` reproduces all empirical results in Section II-E of the manuscript.

**Architecture:** 6-state 3-D kinematic EKF — state vector $x_k = [x, y, z, v_x, v_y, v_z]^T$, fusing GNSS and nonlinear range measurements. Constant-velocity dynamics with $Q = \text{diag}(0.01, 0.01, 0.01, 0.001, 0.001, 0.001)$.

**Perturbation:** Bounded sinusoidal $\delta z_k$ calibrated within $\chi^2_3 = 7.81$ innovation gate. $N = 500$ runs, 1,200-second trajectory.

**Key Results:**

| Metric | Result |
|--------|--------|
| Instability Rate ($\Gamma(t) > \Gamma_{crit} = 6.5$) | **100%** of 500 MC runs |
| NIS Gate Compliance during collapse | **92–96%** of epochs |
| CEP Degradation | **3.2 m → 7.9 m (+147%)** |
| Mission Kill Index ($R_L = 15$ m) | **MKI = 0.53**; confirmed SMK at $R_L \leq 7$ m |
| Anomalies detected by onboard monitor | **0** |

> **Core finding:** The filter appears healthy — NIS remains within compliance bounds — while navigation silently fails. Standard innovation monitors are insufficient to detect this class of estimator collapse.

<div align="center">
  <img src="Figures/Figure_2.png" width="850" alt="Estimator Instability Evolution">
  <br><em>Fig 2. Temporal evolution of the Estimator Instability Number Γ(t) across 500 Monte Carlo runs.</em>
</div>

<div align="center">
  <img src="Figures/Figure_3.png" width="420"> <img src="Figures/Figure_4.png" width="420">
  <br><em>Left: Fig 3. Γ(t) evolution across 500 runs. Right: Fig 4. CEP expansion establishing mission failure conditions.</em>
</div>

---

## 📊 Dimensionless Metrics

ECT introduces four primary metrics to quantify the estimator-collapse regime:

1. **Estimator Instability Number $\Gamma(t)$** — Ratio of actual MSE under perturbation to nominal MSE. Collapse confirmed when $\Gamma(t) > \Gamma_{crit}$.
2. **Mission Kill Index (MKI)** — Ratio of expanded CEP to operational failure threshold $R_L$. SMK confirmed when MKI $\geq 1$.
3. **Information-to-Energy Yield $\eta_{info}$** — Efficiency of uncertainty generation measured via differential Shannon entropy $\Delta h(X)$.
4. **Economic Reversal Ratio $R_{IE}$** — Cost-benefit metric for precision investment vs. vulnerability surface; reserved for future characterisation.

---

## 🛡️ Resilience and Mitigation

Standard innovation-based monitoring (NIS/NEES) is insufficient for this perturbation class. ECT identifies three mitigation layers:

- **Hardware:** Faraday cage / EM hardening — increases $R_{\text{hardening}}$, directly reducing $V_i$
- **Filter-level:** Multi-epoch SPRT monitors; adaptive Q/R estimation; IMM fusion; UKF sigma-point constraints
- **Architecture:** Cross-sensor residual correlation analysis; H∞ robust filtering; redundancy-aware sensor fusion design

<div align="center">
  <img src="Figures/Figure_5.png" width="750" alt="NIS Gate Compliance">
  <br><em>Fig 5. NIS gate compliance across nominal and perturbed runs, confirming the statistically inconspicuous nature of the perturbations.</em>
</div>

---

## 📂 Repository Structure

Estimator-Collapse-Theory-ECT-Framework/
├── ECT_3D_Simulation_v141.py               # Core 3-D dual-sensor Monte Carlo simulation
├── requirements.txt                         # Python dependencies
├── Barua_ECT_SupplementaryVideo_S1.mp4      # Simulation dashboard (Supplementary Video S1)
├── ECT_SMK_Conceptual_Overview.mp4.mp4      # SMK failure transition visualisation
├── Graphical_Abstract.png                   # Archival graphical abstract, 300 DPI
├── Figures/                                 # High-resolution validation figures (600 DPI)
│   ├── Figure_1.png                         # EKF loop and perturbation entry points
│   ├── Figure_2.png                         # Γ(t) temporal evolution
│   ├── Figure_3.png                         # Monte Carlo Γ(t) distribution
│   ├── Figure_4.png                         # CEP expansion results
│   └── Figure_5.png                         # NIS gate compliance
├── Citation                                 # Citation file
└── LICENSE                                  # Apache 2.0

---

## 🚀 Quick Start

```bash
git clone https://github.com/Nick-Barua/Estimator-Collapse-Theory-ECT-Framework.git
cd Estimator-Collapse-Theory-ECT-Framework
pip install -r requirements.txt
python ECT_3D_Simulation_v141.py
```

Expected runtime: ~2–5 minutes on a standard desktop. Reproduces all figures in the manuscript.

---

## 📖 Citation

```bibtex
@software{barua_ect_2026,
  author       = {Barua, Nick and Douglas, R. J.},
  title        = {{Estimator Collapse Theory (ECT) Framework}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {v1.4.2},
  doi          = {10.5281/zenodo.20037820},
  url          = {https://doi.org/10.5281/zenodo.20037820}
}
```

**Associated paper:**
> Barua, N., & Douglas, R. J. (2026). The Sophistication Paradox: A Systems-Theoretic Framework for Estimator Collapse in Precision-Guided Autonomous Navigation Architectures. Manuscript in preparation for submission to CEAS Aeronautical Journal.

**Concept DOI (all versions):** https://doi.org/10.5281/zenodo.19469720

---

## ⚠️ Scope and Limitations

This simulation constitutes a constructive plausibility proof that gate-compliant estimator divergence is mathematically reachable under representative autonomous navigation conditions. It is not a surrogate for full 6-DOF flight dynamics modelling. The kinematic EKF excludes closed-loop GNC feedback; 6-DOF aerodynamic validation is deferred to Phase I of the validation roadmap.

Future work must address: (1) multi-epoch detection evasion via coloured-noise injection sequences; (2) the adaptive filter race condition under simultaneous Q/R perturbation; (3) gate-compliance synthesis for UKF sigma-point architectures.

---

*© 2026 Nick Barua. Licensed under Apache 2.0.*
