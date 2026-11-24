# Bangladesh Peak Electricity Demand Forecasting (Q1 Journal Benchmark)

Reproducible multi-seed benchmark of **25 statistical, linear, differenced-tree, MLP, channel-independent DLinear, channel-mixing & RevIN patched-Transformer, and recurrent architectures** with a **5-way adaptive conformal prediction ablation** on 6.5 years of daily operational records from the Bangladesh Power Development Board (BPDB, Nov 2019 – Jun 2026).

## Headline Verified Findings (v4 Revision)

- **25-Model, 5-Seed Blind Test (Mar 2025 – Jun 2026, `n = 462` days, raw `Year` dropped, train-only scaling, separated `Val-ES` / `Val-Cal`, 14-day embargoes):**
  - **`PatchTST-CM (3×)`** (channel-mixing, no RevIN, `366,147` params, `1.40 MiB`, `0.72 ms` Apple M1 CPU): **`576.94 ± 4.01 MW` MAE**, **`838.00 ± 6.70 MW` RMSE**, `4.35%` MAPE, **`+8.9%` Skill vs. Weekday-Adjusted Persistence** (`633.43 MW`; descriptive HLN-DM `p_Holm = 0.048` Seed 42, `0.014` 5-seed).
  - **`PatchTST-CM-Diff (Δy, 3×)`:** `591.67 ± 4.64 MW` MAE (`+6.6%` skill).
  - **`Equal-Weight RNN Ens` (`1/3` each):** `596.29 ± 10.73 MW` MAE, `844.81 MW` RMSE (`+5.9%` skill — beats `Val-Opt RNN Ensemble` `597.15 ± 11.81 MW` on MAE, RMSE, and SD).
  - **`Ridge (14 × 117, α = 10)`:** `602.29 MW` MAE, `862.98 MW` RMSE (`+4.9%` skill, `1,639` params, `6.40 KiB` fp32, `0.04 ms` CPU).
  - **Faithful univariate `PatchTST-RevIN (CI, 3×)`:** `667.99 ± 11.20 MW` (`−30.9 MW` ahead of univariate `DLinear (CI)` `698.84 ± 3.32 MW`).
  - **Low-Power 95% Hansen MCS:** Excludes only `2` of `48` model–loss entries among the top 12 models (`Weekday-Adj Persistence` on RMSE only; it stays in on MAE `p_MCS = 0.196 / 0.089`), confirming limited discrimination power at `n = 462` days.
- **Uniform 5-Seed 3-Fold Rolling-Origin CV (2023–2026) with Level-Shift Controls:**
  - Level `PatchTST-CM (3×)` degrades in Fold 1 (2023: `909.12 ± 113.29 MW`; 3-fold mean `693.11 MW`).
  - Faithful **`PatchTST-RevIN (CI, 3×)`** cuts Fold 1 by `−345.9 MW` to `563.24 ± 4.89 MW` (3-fold mean `611.05 MW`).
  - **`PatchTST-CM-Diff (Δy, 3×)`** cuts Fold 1 by `−392.8 MW` to `516.32 ± 17.02 MW`, wins Fold 2 (`530.79 ± 2.86 MW`), and achieves the **lowest 3-fold mean of any model (`546.26 MW`)**, ahead of Weekday Persistence (`551.01 MW`), `Ridge-Diff` (`562.76 MW`), `Ridge` (`565.54 MW`), and `SARIMAX` (`599.35 MW`).
- **5-Way Conformal Ablation with 95% Moving-Block Bootstrap CIs (`PatchTST-CM` & `Ridge`, 80% & 90%):**
  - Static 90% intervals cover `70.8% ± 5.6 pp` (`53.7%` Summer, Winkler `4,892 MW`).
  - **`Plain ACI`** achieves the lowest Winkler across both models and both nominal levels (**`3,674 MW`**, `−24.9%`, `2,450 MW` width on `PatchTST-CM`; **`3,702 MW`**, `2,385 MW` width on `Ridge`).
  - **`LN-Static`** (`3,921 / 3,788 MW` Winkler) and **`LN-ACI`** (`3,940 / 3,864 MW`, `87.8%–90.1%` across all four seasons, seasonal 95% CIs `±4.3–7.8 pp`) are nearly tied.
- **Downstream Analyses & Operational Audits:**
  - **Architecture-Dependent PCA Ablation:** 95% PCA degrades `PatchTST-CM` (`+79.5 MW`, `p = 0.001`) and `Ridge` (`+33.5 MW`) while leaving `Equal-Weight RNN` unchanged (`596.3 → 597.5 MW`, `p = 0.333`).
  - **Prior-Day `y(t−1)` Demand Bins & Holiday Paired Tests:** Removing target-selection bias puts `PatchTST-CM (3×)` first in `>15 GW` (`613.5` vs. `635.0 MW` Wkday Persist). Holiday/Eid differences (`n = 41`: `Ridge 867.9` vs. `PT-CM 910.9 MW`, `p = 0.366`) are not statistically significant.
  - **Same-Split 8 Divisions (5 Seeds + Paired DM):** `Equal-Weight RNN` significantly beats `PatchTST-CM (3×)` in `7/8` divisions (`p < 0.05`; Rajshahi `p = 0.379` ns), while `Ridge` / `Weekday Persistence` lead Dhaka and Chattogram.
  - **Multi-Horizon Convex Blend (`1–30 d`):** Direct skill over climatology dissipates by `h ≈ 7 d`; a validation-tuned `Climatology+Ridge` convex blend reaches `848.3 MW` (`h = 3`), `1,007.4 MW` (`h = 7`), `1,011.8 MW` (`h = 14`), and `1,071.9 MW` (`h = 30`).
  - **Protocol Audits:** Calendar-gap audit (`420` clean vs. `42` gap-spanning test windows), Eid `±1`-day lunar-sighting jitter (`+27–36 MW` on `+1 d` late shift), and full-year 2024 summer-inclusive validation sensitivity (`Ridge α = 10` unchanged; `PatchTST-CM` selects `lr = 2.5e-3 → 564.9 MW`).

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
