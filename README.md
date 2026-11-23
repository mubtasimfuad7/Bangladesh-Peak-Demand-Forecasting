# Bangladesh Peak Electricity Demand Forecasting (Q1 Journal Benchmark)

Reproducible multi-seed benchmark of **21 statistical, linear, tree-based, patched-Transformer, and recurrent architectures** with **Locally Normalized Adaptive Conformal Inference (LN-ACI)** on 6.5 years of daily operational records from the Bangladesh Power Development Board (BPDB, Nov 2019 – Jun 2026).

## Headline Verified Findings

- **2025–2026 Blind Test (462 days, 5 seeds, `Val-ES` / `Val-Cal` separated, train-only scaling, 14-day embargoes):**
  - **PatchTST-Ensemble (3×):** `574.53 ± 6.68 MW` MAE (`4.37%` MAPE / `95.63%` Acc, `444.7 MW` on `>15 GW` peaks, `366,915` params, `0.69 ms` CPU).
  - **Ridge Regression (`14 × 118` features, `α=10`):** `599.09 MW` MAE (`0.04 ms` CPU) — outperforming SARIMAX (`650.78 MW`), Equal-Weight RNN Ensemble (`607.08 ± 8.21 MW`), and Val-Opt SLSQP RNN Ensemble (`619.91 ± 17.60 MW`).
  - **Differenced XGBoost (`Δy`):** `641.99 MW` MAE (`−183.2 MW` rescue vs. level-space XGBoost `825.17 MW`, cutting `>15 GW` error from `1,097.7` to `601.8 MW`).
  - **Weekday-Adjusted Persistence:** `633.43 MW` MAE (`−122.3 MW` vs. raw Lag-1 Persistence `755.72 MW`).
- **3-Fold Expanding-Window Rolling-Origin CV (2023–2026) & Unconfounded Fixed-Test Learning Curve:**
  - **3-Fold Mean:** **Ridge (`567.79 MW`)** > SARIMAX (`598.32 MW`) > XGBoost-Diff (`609.63 MW`) > PatchTST-3× (`647.02 MW`) > Val-Opt RNN Ensemble (`652.86 MW`).
  - **Fixed 2025–26 Test Learning Curve (`1.5–4.5 yr`):** SARIMAX plateaus near `651 MW`, whereas **PatchTST-3×** leads at every history length (`633.0 → 604.5 → 582.8 → 574.5 MW`).
- **Adaptive Conformal Uncertainty (LN-ACI vs. Static Split Conformal):**
  - Static Split Conformal under-covers seasonal volatility (`67.5%` at 90% nominal; `51.2%` in Summer).
  - **LN-ACI** restores **`89.61%` empirical coverage** (`88.9%–90.1%` across all 4 seasons) and cuts the proper Winkler score by **`20.3%`** (`5,259 → 4,190 MW`).
- **Same-Split 8-Division & Strictly Causal 1–30 Day Horizons:**
  - Same-split 8-division RNN Ensemble beats Single BiGRU in `8/8` divisions and Ridge in `6/8` (Ridge leads Dhaka `251.7 MW` and Chattogram `74.2 MW`).
  - Direct multi-horizon models beat Climatology + Trend through `h = 14` days (`1,039.8` vs. `1,070.8 MW`); at `h = 30` days without future weather, Climatology + Trend (`1,071.9 MW`) leads.
- **Data Quality Audit & Direct Model Explainability:**
  - Audited 20 missing BPDB calendar days (14 gaps, max 6 days in Jul 2024) and excluded 16 physically impossible negative transmission-loss metering inversions (QA mean peak line loss: `510.50 MW`).
  - Direct group permutation importance on the deployed PyTorch ensemble quantifies true driver contributions (`Divisional Telemetry +943.6 MW`, `Lagged National Peak +337.9 MW`, `Day-of-Week/Season +148.1 MW`).

## Reproducing All Experiments & Dashboard

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run all 4 experiment & vector figure suites
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_part1_journal_suite.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_part1_rolling_cv.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_part1_divisional_and_horizon.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python generate_part1_extra_figures.py

# 2. Launch the interactive Streamlit dashboard
streamlit run app.py
```

## Manuscript Package
- LaTeX source & figures: `part1_journal_package/` (`main.tex`, `references.bib`, `highlights.txt`, `figures/*.pdf`, `results/*.csv`)
- Ready-to-upload Overleaf bundle: `Part1_Demand_Journal_Paper.zip`
