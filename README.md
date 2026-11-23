# Thermodynamic- and Calendar-Aware Deep Ensembles for National and Divisional Peak Electricity Demand Forecasting in Bangladesh

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official reproducible codebase, dataset, results, and Elsevier (`elsarticle`) manuscript package for:

> **Thermodynamic- and Calendar-Aware Deep Ensembles for National and Divisional Peak Electricity Demand Forecasting in Bangladesh**  
> Mubtasim Fuad, Tanvir Anjum Abid, Payel Das, Md. Imam Hossain Rakib, and Mohammad Rashedur Rahman*  
> Department of Electrical and Computer Engineering, North South University, Dhaka, Bangladesh

---

## Key Highlights

- **Strict Leakage-Free Chronological Protocol:** 6.5 years of official BPDB daily dispatch records (`2,380` clean daily cycles, Dec 2019–Jun 2026), train-only Min–Max normalization, strictly past-only 14-day windows (`119` engineered channels), and 14-day split temporal embargoes.
- **16-Model, 5-Seed National Benchmark (`Seeds: 42, 123, 7, 2024, 999`):** Benchmarks Persistence, 7-Day Rolling Mean, strictly causal `SARIMAX(1,1,1)(1,0,1,7)`, XGBoost, LightGBM, DLinear, `PatchTST` (ICLR 2023), Transformer, CNN-BiGRU-Transformer, BiGRU, BiLSTM, BiLSTM-Attention, residual deep–XGBoost hybrids, Equal-Weight Ensemble, and our **Validation-Optimized SLSQP Weighted Deep Ensemble**.
- **Best Cross-Seed Stability & Multi-Year Generalization:**
  - **5-Seed Blind Test (462 days):** **`597.08 ± 4.35 MW MAE`**, **`855.34 ± 8.38 MW RMSE`**, **`4.46 ± 0.04% MAPE`** (**`95.54% Accuracy`**), **`R² = 0.82`**, with **3.5× lower cross-seed variance** than PatchTST (`584.99 ± 15.06 MW`, DM `p = 0.466` ns) and significant Newey–West HAC Diebold–Mariano gains (`p ≤ 0.002`) over **10 of 15 baselines** including SARIMAX (`−64.4 MW, p = 0.001`), LightGBM, DLinear, Transformer, and XGBoost.
  - **3-Fold Expanding-Window Rolling-Origin CV (2023–2026):** Cuts PatchTST's 3-year mean error by **94.1 MW** (`613.3` vs. `707.4 MW`) and achieves the **#1 Post-2023 (≥3.5 yr) mean MAE (`576.9 MW`)** over all models (beating PatchTST `603.0 MW` and SARIMAX `623.8 MW`).
- **Strictly Monotonic 5-Seed Ensemble Ablation:** `Base (617.51 ± 17.13)` → `+CDD/HDD (604.79 ± 7.32)` → `+3-Day Eid Lead/Lag Full No-PCA (597.08 ± 4.35 MW; std cut 75%; Holiday MAE −55.4 MW)` vs. `Full + PCA (629.81 ± 29.59 MW; +32.7 MW penalty & 6.8× variance)`.
- **Extreme-Peak (`>15 GW`, n = 160) Extrapolation:** **`549.6 MW MAE`** (`3.43% MAPE` / `96.57% Accuracy`) — a **49.9% error cut** vs. XGBoost (`1,097.7 MW`).
- **8-Division Demand & Strictly Causal 1–30d Multi-Horizon:** Beats Lag-1 Persistence in **8/8 divisions** and Yeaser et al. (2025) in **6/8 divisions** (`93.18%–94.34% Accuracy`), and beats Origin Persistence by **`+13.6%` to `+21.0%`** across `h = 1, 3, 7, 14, 30` days with zero future telemetry leakage.
- **Probabilistic Calibration, Causal Explainability & Deployment Speed:** Includes 80%/90% Split Conformal Prediction intervals, strictly causal train-only TreeSHAP (`CDD_Lag1: 120.9 MW`, `Temp_Lag1: 0.0 MW`), peak transmission loss quantification (`504.99 MW` mean), and **1.64 ms** CPU inference latency (`3.10 MB` fp32).

---

## Repository Structure

```text
Bangladesh-Peak-Demand-Forecasting/
├── version_5_extended_backup.xlsx          # Curated 6.5-year BPDB daily dataset (2019–2026)
├── run_part1_journal_suite.py              # 16 models × 5 seeds, SARIMAX, PatchTST, DM HAC, Conformal, Ablation, Bins
├── run_part1_rolling_cv.py                 # 3-Fold Expanding-Window Rolling-Origin CV (2023, 2024, 2025–2026)
├── run_part1_divisional_and_horizon.py     # Unified 8-division peak demand & causal direct 1–30d multi-horizon
├── generate_part1_extra_figures.py         # Architecture diagram, causal TreeSHAP surrogate, peak loss & deficits
├── grid_analysis_extensions.py             # Descriptive grid reliability & heat-load ratio plots
├── app.py                                  # Interactive Streamlit dashboard
├── requirements.txt                        # Exact Python dependencies
├── part1_journal_package/
│   ├── main.tex                            # Elsevier (elsarticle) Q1 manuscript
│   ├── references.bib                      # 29 verified BibTeX references
│   ├── highlights.txt                      # Elsevier Highlights (<= 85 chars per bullet)
│   ├── figures/                            # All 14 vector PDF publication figures
│   └── results/                            # All reproducible CSV & JSON benchmark outputs
└── Part1_Demand_Journal_Paper.zip          # Overleaf-ready submission archive
```

---

## Quickstart & Full Reproduction

### 1. Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce All Paper Tables, CSVs & 14 Vector PDF Figures
Run the 4 unified scripts sequentially (all outputs save to `part1_journal_package/results/` and `part1_journal_package/figures/`):
```bash
python run_part1_journal_suite.py
python run_part1_rolling_cv.py
python run_part1_divisional_and_horizon.py
python generate_part1_extra_figures.py
```

### 3. Launch Interactive Dashboard
```bash
streamlit run app.py
```

---

## Main Benchmark Summary (462 Blind Test Days, 5 Seeds)

| Model Architecture | MAE (MW) | RMSE (MW) | MAPE (%) | Accuracy (%) | R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| PatchTST | **584.99 ± 15.06** | 854.52 ± 27.33 | **4.42 ± 0.13** | **95.58 ± 0.13** | 0.82 |
| **Val-Opt Weighted Ensemble (Ours)** | **597.08 ± 4.35** | 855.34 ± 8.38 | 4.46 ± 0.04 | 95.54 ± 0.04 | 0.82 |
| Equal-Weight Ensemble | 598.22 ± 10.44 | **854.44 ± 8.70** | 4.47 ± 0.06 | 95.53 ± 0.06 | 0.82 |
| BiLSTM-Attention | 613.56 ± 16.71 | 873.51 ± 14.04 | 4.59 ± 0.12 | 95.41 ± 0.12 | 0.82 |
| BiLSTM | 618.40 ± 20.62 | 885.17 ± 24.85 | 4.61 ± 0.15 | 95.39 ± 0.15 | 0.81 |
| BiGRU | 634.63 ± 35.47 | 886.12 ± 37.42 | 4.71 ± 0.25 | 95.29 ± 0.25 | 0.81 |
| SARIMAX(1,1,1)(1,0,1,7) | 650.87 ± 0.00 | 907.53 ± 0.00 | 4.87 ± 0.00 | 95.13 ± 0.00 | 0.80 |
| LightGBM | 690.19 ± 0.00 | 954.45 ± 0.00 | 5.07 ± 0.00 | 94.93 ± 0.00 | 0.78 |
| Custom Transformer | 704.15 ± 56.00 | 941.65 ± 50.17 | 5.21 ± 0.40 | 94.79 ± 0.40 | 0.79 |
| Persistence (Lag-1) | 755.72 ± 0.00 | 1056.69 ± 0.00 | 5.71 ± 0.00 | 94.29 ± 0.00 | 0.73 |
| DLinear | 809.56 ± 21.96 | 1111.01 ± 22.88 | 5.97 ± 0.19 | 94.03 ± 0.19 | 0.70 |
| XGBoost | 825.17 ± 0.00 | 1126.84 ± 0.00 | 5.93 ± 0.00 | 94.07 ± 0.00 | 0.69 |

---

## Data Source & License
- Operational generation archives are publicly published by the **Bangladesh Power Development Board (BPDB)**: https://misc.bpdb.gov.bd/daily-generation
- Code and manuscript source released under the [MIT License](LICENSE).
