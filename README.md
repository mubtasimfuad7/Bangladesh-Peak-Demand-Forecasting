# Bangladesh Peak Electricity Demand Forecasting (Q1 Journal Benchmark)

Reproducible multi-seed benchmark of **24 statistical, linear, differenced-tree, MLP, channel-independent DLinear, patched-Transformer, and recurrent architectures** with a **5-way adaptive conformal prediction ablation** on 6.5 years of daily operational records from the Bangladesh Power Development Board (BPDB, Nov 2019 – Jun 2026).

## Headline Verified Findings

- **Tied Leading Group inside the 95% Hansen Model Confidence Set (Mar 2025 – Jun 2026, 462 blind test days, 5 seeds, raw `Year` dropped, train-only scaling, separated `Val-ES` / `Val-Cal`, 14-day embargoes):**
  - **PatchTST-Ensemble (3×)** (channel-mixing, no RevIN, `366,147` params, `1.40 MiB`, `1.05 ms` CPU): **`584.20 ± 10.71 MW` MAE**, **`841.60 ± 7.78 MW` RMSE**, `4.40%` MAPE, **`+7.8%` Skill vs. Weekday-Adjusted Persistence** (`p_MCS = 1.000` on MAE & RMSE).
  - **PatchTST-Diff (`Δy`, 3×):** `589.61 ± 6.23 MW` MAE, `890.85 MW` RMSE (`+6.9%` skill).
  - **Val-Opt RNN Ensemble:** `599.62 ± 13.50 MW` MAE, `846.25 MW` RMSE (`+5.3%` skill).
  - **Ridge (`14 × 117`, `α = 10`):** `602.29 MW` MAE, `862.98 MW` RMSE (`+4.9%` skill, `1,639` params, `6.40 KiB` fp32, `0.05 ms` CPU).
  - **Equal-Weight RNN Ens (`1/3` each):** `603.23 ± 6.66 MW` MAE, `845.23 MW` RMSE (`+4.8%` skill, half the cross-seed SD of SLSQP).
  - **Ridge-Diff (`Δy`, `α = 10`):** `607.07 MW` MAE, `872.05 MW` RMSE (`+4.2%` skill).
  - **HLN-DM (`Ref = PatchTST-3×`, Holm–Bonferroni):** Significantly beats `14/23` baselines on Seed 42 (`p_Holm ≤ 0.018`), while differences vs. `Ridge` (`p_Holm = 0.407 / 1.000`) and `Equal-Weight RNN` (`0.407 / 1.000`) are statistically non-significant.
  - **Standard CI-DLinear, MLP & Differenced Trees:** Standard univariate `DLinear (CI)` (`30` params, `0.12 KiB`) reaches `698.29 ± 2.74 MW`; `MLP` (`218k` params) reaches `719.51 ± 71.21 MW`; `XGBoost-Diff (Δy)` cuts level-XGBoost error by **`21.5%` overall** (`819.35 → 642.90 MW`) and **`44.2%` on `>15 GW` peaks** (`1,084.8 → 605.8 MW`).
- **3-Fold Expanding-Window Rolling-Origin CV (2023–2026) & Within-Period Sensitivity Curve:**
  - **3-Fold Mean:** **Weekday-Adjusted Persistence (`551.01 MW`)** > **Ridge-Diff (`562.76 MW`)** > **Ridge (`565.54 MW`)** > SARIMAX (`599.35 MW`) > XGBoost-Diff (`609.93 MW`) > Equal-Weight RNN (`662.96 MW`) > PatchTST-3× (`696.05 MW`, due to Fold-1 2023 degradation `907.72 MW`).
  - **Within-Period History Sensitivity (2025–26 fixed, `1.5–4.5 yr`):** SARIMAX plateaus near `651 MW`, whereas `PatchTST-3×` (`633.8 → 584.2 MW`), `Ridge` (`762.0 → 602.3 MW`), and `Equal-Weight RNN` (`739.6 → 603.2 MW`) improve monotonically.
- **5-Way Conformal Uncertainty Ablation on Untouched `Val-Cal` Residuals (`PatchTST-3×` & `Ridge`):**
  - Static Split Conformal fails under mild-to-hot shift (`70.8%` PICP at 90% nominal; `53.1%` in Summer; Winkler `4,869 MW`).
  - **Plain ACI** minimizes proper Winkler score (**`3,636 MW`**, `−25.3%` vs. static; `88.1%` PICP, `2,404 MW` mean width; `3,702 MW` on Ridge).
  - **LN-ACI** achieves the tightest seasonal coverage uniformity across all four seasons (**`88.1%–90.1%`**, lifting Summer PICP to **`90.1%`**; `89.4%` overall on `PatchTST-3×`, `89.6%` on `Ridge`, `2,878 MW` mean width).
- **Regimes, Same-Split 8 Divisions, Multi-Horizon Hybrids & Direct Explainability:**
  - **Holiday / Eid (`n = 41`):** **`Ridge` leads (`867.9 MW`)** ahead of `Equal-Weight RNN` (`892.6 MW`), `XGBoost-Diff` (`894.1 MW`), and `PatchTST-3×` (`926.7 MW`).
  - **Same-Split 8-Division:** **`Equal-Weight RNN` beats `PatchTST-3×` in `8/8` divisions** and leads 5 regional grids, while `Ridge` / `Weekday Persistence` tie or edge ahead in Dhaka and Chattogram.
  - **Causal Direct Multi-Horizon (`h = 1..30` days):** `PatchTST-3×` leads `h = 1, 3, 7` days; beyond one week, a **`Climatology+Ridge-Residual` hybrid** falls back gracefully to `Climatology+Trend` (`1,070.8 MW` at `h = 14`; `1,071.9 MW` at `h = 30`) instead of drifting.
  - **Direct Merged-Block Permutation Importance (`PatchTST-3×`):** `Autoregressive Load History (k = 43): +1,544.8 MW`, `DOW/Month: +151.7 MW`, `Holiday/Eid: +26.7 MW`, `Temp & Detrended CDD_25: +11.8 MW` (detrended `τ* = 25.0 °C`, `+387.2 MW/°C`).
  - **Robust Data QA (Appendix A):** Audited `20` missing calendar days, `16` negative metering inversions, and `3` upper IQR spikes `> 1,168 MW` (`robust_mean = 508.31 MW`, `median = 510.0 MW`, `IQR = 188.0 MW`).

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
