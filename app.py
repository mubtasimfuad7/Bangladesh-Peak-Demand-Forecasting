from pathlib import Path
import json
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Bangladesh Peak Electricity Demand Forecasting",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "version_5_extended_backup.xlsx"
RESULTS_DIR = ROOT_DIR / "part1_journal_package" / "results"
PAPER_ZIP_PATH = ROOT_DIR / "Part1_Demand_Journal_Paper.zip"

DIVISIONS = [
    "Dhaka",
    "Chattogram",
    "Rajshahi",
    "Khulna",
    "Mymensingh",
    "Rangpur",
    "Barishal",
    "Sylhet",
]


@st.cache_data(ttl="1h")
def load_grid_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Peak_Demand_MW"] = pd.to_numeric(
        df["Max. Demand at eve. peak (Generation end)"], errors="coerce"
    )
    df["Substation_Demand_MW"] = pd.to_numeric(
        df["Max. Demand at eve. peak (Sub-station end)"], errors="coerce"
    )
    raw_loss = df["Peak_Demand_MW"] - df["Substation_Demand_MW"]
    df["Raw_Loss_MW"] = raw_loss
    df["QA_Loss_MW"] = raw_loss.where(raw_loss >= 0, np.nan)
    df["QA_Loss_Pct"] = (df["QA_Loss_MW"] / df["Peak_Demand_MW"]) * 100.0
    df["Peak_Demand_GW"] = df["Peak_Demand_MW"] / 1000.0
    df["Month_Num"] = df["Date"].dt.month
    df["Season"] = df["Month_Num"].map(
        {
            12: "Winter (Dec–Feb)",
            1: "Winter (Dec–Feb)",
            2: "Winter (Dec–Feb)",
            3: "Summer (Mar–May)",
            4: "Summer (Mar–May)",
            5: "Summer (Mar–May)",
            6: "Monsoon (Jun–Sep)",
            7: "Monsoon (Jun–Sep)",
            8: "Monsoon (Jun–Sep)",
            9: "Monsoon (Jun–Sep)",
            10: "Post-Monsoon (Oct–Nov)",
            11: "Post-Monsoon (Oct–Nov)",
        }
    )
    return df


@st.cache_data(ttl="1h")
def load_journal_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(ttl="1h")
def load_meta() -> dict:
    p = RESULTS_DIR / "part1_meta.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`. Place `version_5_extended_backup.xlsx` in the project root.",
        icon=":material/error:",
    )
    st.stop()

df = load_grid_dataset(DATA_PATH)
meta = load_meta()
summary_df = load_journal_csv("part1_summary_metrics.csv")
per_seed_df = load_journal_csv("part1_per_seed_metrics.csv")
dm_df = load_journal_csv("part1_dm_tests.csv")
eff_df = load_journal_csv("part1_efficiency.csv")
rolling_df = load_journal_csv("part1_rolling_cv.csv")
lc_df = load_journal_csv("part1_learning_curve.csv")
ablation_df = load_journal_csv("part1_ablation.csv")
regime_df = load_journal_csv("part1_regime_analysis.csv")
bins_df = load_journal_csv("part1_demand_bins.csv")
div_df = load_journal_csv("part1_divisional_pytorch.csv")
horizon_df = load_journal_csv("part1_multihorizon.csv")
conformal_df = load_journal_csv("part1_conformal.csv")
perm_df = load_journal_csv("part1_permutation_importance.csv")

if not eff_df.empty and "Unnamed: 0" in eff_df.columns:
    eff_df = eff_df.rename(columns={"Unnamed: 0": "Model"})

# Header
st.title(":material/bolt: Bangladesh peak electricity demand forecasting")
st.caption(
    "21-model 5-seed benchmark (Statistical, Ridge, Differenced Trees, PatchTST-3×, Recurrent Ensembles), "
    "3-fold rolling CV + unconfounded fixed-test learning curve, Locally Normalized ACI, same-split 8-division forecasts, "
    "causal 1–30 day horizons, direct PyTorch permutation importance, and QA grid diagnostics."
)

# Sidebar
with st.sidebar:
    st.subheader(":material/tune: Protocol & data QA")
    st.markdown(
        ":green-badge[Train-only scaling] :blue-badge[Val-ES / Val-Cal split] :orange-badge[LN-ACI 89.6%]"
    )
    qa = meta.get("data_qa", {})
    st.caption(
        f"**Records:** {len(df):,} BPDB cycles across {qa.get('calendar_span_days', 2414):,} calendar days "
        f"({qa.get('missing_calendar_days', 20)} reporting gaps audited)  \n"
        f"**Peak range:** {df['Peak_Demand_MW'].min():,.0f}–{df['Peak_Demand_MW'].max():,.0f} MW "
        f"({df['Peak_Demand_GW'].min():.2f}–{df['Peak_Demand_GW'].max():.2f} GW)  \n"
        f"**Splits:** Train 1,652 · Val-ES 105 · Val-Cal 105 · Test 462  \n"
        f"**Cooling threshold:** τ* = {meta.get('tau_star_C', 26.0):.1f} °C (train-only; HDD dropped, 0% < 18 °C)  \n"
        f"**QA peak line loss:** {qa.get('qa_mean_loss_mw', 510.5):.2f} MW ({qa.get('negative_loss_records', 16)} negative anomalies excluded)"
    )

    st.subheader(":material/calendar_today: Historical date filter")
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    date_range = st.date_input(
        "Filter EDA & grid analytics",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d, end_d = min_date, max_date

    st.subheader(":material/link: Links")
    st.link_button(
        "GitHub repository",
        "https://github.com/mubtasimfuad7/Bangladesh-Peak-Demand-Forecasting",
        icon=":material/code:",
        width="stretch",
    )

filtered_df = df[
    (df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)
].copy()

# Top KPI row
recent_spark = df["Peak_Demand_MW"].tail(60).round(0).tolist()
pt3_row = (
    summary_df[summary_df["Model"] == "PatchTST-Ensemble (3x)"].iloc[0]
    if not summary_df.empty and (summary_df["Model"] == "PatchTST-Ensemble (3x)").any()
    else None
)
ridge_row = (
    summary_df[summary_df["Model"] == "Ridge (14xF)"].iloc[0]
    if not summary_df.empty and (summary_df["Model"] == "Ridge (14xF)").any()
    else None
)
aci90 = (
    conformal_df[(conformal_df["Method"] == "Locally Normalized ACI") & (conformal_df["Nominal_%"] == 90)].iloc[0]
    if not conformal_df.empty and "Method" in conformal_df.columns
    else None
)

with st.container(horizontal=True):
    st.metric(
        "Best 2025–26 test MAE",
        f"{pt3_row['MAE_mean']:.2f} ± {pt3_row['MAE_std']:.2f} MW" if pt3_row is not None else "574.53 ± 6.68 MW",
        "PatchTST-3× (0.69 ms CPU)",
        delta_color="off",
        border=True,
    )
    st.metric(
        "Best 3-fold CV mean",
        f"{rolling_df['Ridge_MAE'].mean():.2f} MW" if not rolling_df.empty and "Ridge_MAE" in rolling_df.columns else "567.79 MW",
        f"Ridge (Fold 3: {ridge_row['MAE_mean']:.1f} MW)" if ridge_row is not None else "Ridge (14×118)",
        delta_color="off",
        border=True,
    )
    st.metric(
        "90% LN-ACI coverage",
        f"{aci90['PICP_%']:.2f}%" if aci90 is not None else "89.61%",
        "−20.3% Winkler vs static",
        border=True,
    )
    st.metric(
        "Extreme peak (>15 GW)",
        f"{bins_df.loc[bins_df['Demand_Bin'].str.contains('Extreme'), 'PT3_MAE'].iloc[0]:.1f} MW"
        if not bins_df.empty and "PT3_MAE" in bins_df.columns else "444.7 MW",
        "PatchTST-3× (RNN 569.1 · XGB-Δy 601.8)",
        delta_color="off",
        border=True,
    )
    st.metric(
        "All-time gross peak",
        f"{df['Peak_Demand_MW'].max():,.0f} MW",
        f"{df['Peak_Demand_GW'].max():.2f} GW",
        chart_data=recent_spark,
        chart_type="line",
        border=True,
    )

tabs = st.tabs(
    [
        ":material/leaderboard: National 21-model benchmark",
        ":material/timeline: Rolling CV & learning curve",
        ":material/map: Same-split 8 divisions",
        ":material/schedule: Multi-horizon & LN-ACI",
        ":material/psychology: Direct explainability",
        ":material/analytics: QA grid analytics & data",
        ":material/tune: Scenario simulator & paper",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: 21-Model National Benchmark
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("National 21-model 5-seed benchmark (Mar 2025 – Jun 2026, n = 462 days)")
    st.caption(
        "Strict train-only scaling, 14-day split embargoes, separate Val-ES (105d) and Val-Cal (105d) partitions, "
        "and equal-budget 3-network comparisons (PatchTST-3× vs Equal-Weight & Val-Opt RNN Ensembles)."
    )

    if not summary_df.empty:
        metric_choice = st.segmented_control(
            "Select ranking metric",
            ["MAE (MW)", "RMSE (MW)", "MAPE (%)", "R²"],
            default="MAE (MW)",
            key="nat_metric_choice",
        )
        col_map = {
            "MAE (MW)": ("MAE_mean", "MAE_std", True),
            "RMSE (MW)": ("RMSE_mean", "RMSE_std", True),
            "MAPE (%)": ("MAPE_mean", "MAPE_std", True),
            "R²": ("R2_mean", "R2_std", False),
        }
        val_col, std_col, ascending = col_map[metric_choice or "MAE (MW)"]
        plot_df = summary_df.sort_values(val_col, ascending=ascending).copy()
        plot_df["Lower"] = plot_df[val_col] - plot_df[std_col]
        plot_df["Upper"] = plot_df[val_col] + plot_df[std_col]
        plot_df["Group"] = np.where(
            plot_df["Model"].str.contains("PatchTST"),
            "Patched Transformer",
            np.where(
                plot_df["Model"].str.contains("Ridge|SARIMAX"),
                "Linear / Statistical",
                np.where(plot_df["Model"].str.contains("RNN"), "RNN Ensemble", "Other Baseline"),
            ),
        )

        col_c, col_t = st.columns([1.25, 1.0])
        with col_c:
            with st.container(border=True):
                st.markdown(f"**21-model comparison by {metric_choice} (mean ± 1 SD across 5 seeds)**")
                bars = (
                    alt.Chart(plot_df)
                    .mark_bar(cornerRadiusEnd=4)
                    .encode(
                        x=alt.X(f"{val_col}:Q", title=metric_choice),
                        y=alt.Y("Model:N", sort=plot_df["Model"].tolist(), title=None),
                        color=alt.Color(
                            "Group:N",
                            scale=alt.Scale(
                                domain=["Patched Transformer", "Linear / Statistical", "RNN Ensemble", "Other Baseline"],
                                range=["#0D9488", "#D97706", "#2563EB", "#94A3B8"],
                            ),
                            legend=alt.Legend(orient="bottom", title=None),
                        ),
                        tooltip=[
                            "Model:N",
                            alt.Tooltip("MAE_mean:Q", title="MAE (MW)", format=".2f"),
                            alt.Tooltip("MAE_std:Q", title="SD (MW)", format=".2f"),
                            alt.Tooltip("RMSE_mean:Q", title="RMSE (MW)", format=".2f"),
                            alt.Tooltip("Acc_mean:Q", title="Accuracy (%)", format=".2f"),
                        ],
                    )
                )
                err = (
                    alt.Chart(plot_df)
                    .mark_errorbar(color="#1E293B", ticks=True)
                    .encode(
                        x=alt.X("Lower:Q", title=metric_choice),
                        x2="Upper:Q",
                        y=alt.Y("Model:N", sort=plot_df["Model"].tolist()),
                    )
                )
                st.altair_chart((bars + err).properties(height=540))

        with col_t:
            with st.container(border=True):
                st.markdown("**Full 21-model scorecard**")
                disp = summary_df.sort_values("MAE_mean").copy()
                disp["MAE (MW)"] = disp.apply(lambda r: f"{r['MAE_mean']:.2f} ± {r['MAE_std']:.2f}", axis=1)
                disp["RMSE (MW)"] = disp.apply(lambda r: f"{r['RMSE_mean']:.2f} ± {r['RMSE_std']:.2f}", axis=1)
                st.dataframe(
                    disp[["Model", "MAE (MW)", "RMSE (MW)", "MAPE_mean", "Acc_mean", "R2_mean"]],
                    column_config={
                        "Model": st.column_config.TextColumn("Model", pinned=True),
                        "MAPE_mean": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
                        "Acc_mean": st.column_config.ProgressColumn("Accuracy", format="%.2f%%", min_value=90, max_value=100),
                        "R2_mean": st.column_config.NumberColumn("R²", format="%.3f"),
                    },
                    hide_index=True,
                    height=540,
                )

    col_dm, col_eff = st.columns([1.15, 1.0])
    with col_dm:
        with st.container(border=True):
            st.markdown("**HLN-adjusted Diebold–Mariano tests (Holm–Bonferroni adjusted, 5-seed trajectories)**")
            st.caption(
                "Reference: Val-Opt RNN Ensemble (601.41 MW trajectory MAE). Significantly beats 8/20 baselines after Holm–Bonferroni correction (p_Holm ≤ 0.012); "
                "on a single-run 3-network Seed-42 comparison, PatchTST-3× beats Val-Opt RNN Ens by −56.8 MW (p < 0.001)."
            )
            if not dm_df.empty:
                dm_p = dm_df.sort_values("Delta_MAE_MW").copy()
                dm_p["Sig"] = np.where(dm_p["p_holm"] < 0.05, "p_Holm < 0.05", "Not significant")
                zr = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color="#64748B").encode(x="x:Q")
                ci = alt.Chart(dm_p).mark_rule(strokeWidth=2).encode(
                    x=alt.X("CI_lo:Q", title="ΔMAE vs baseline (MW, 95% bootstrap CI)"),
                    x2="CI_hi:Q",
                    y=alt.Y("Baseline:N", sort=dm_p["Baseline"].tolist(), title=None),
                    color=alt.Color("Sig:N", scale=alt.Scale(domain=["p_Holm < 0.05", "Not significant"], range=["#16A34A", "#94A3B8"]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                )
                pt = alt.Chart(dm_p).mark_point(filled=True, size=65).encode(
                    x="Delta_MAE_MW:Q",
                    y=alt.Y("Baseline:N", sort=dm_p["Baseline"].tolist(), title=None),
                    color="Sig:N",
                    tooltip=["Baseline:N", alt.Tooltip("Delta_MAE_MW:Q", format=".2f"), alt.Tooltip("p_val:Q", format=".4f"), alt.Tooltip("p_holm:Q", format=".4f")],
                )
                st.altair_chart((zr + ci + pt).properties(height=420))

    with col_eff:
        with st.container(border=True):
            st.markdown("**5-seed dispersion & CPU efficiency**")
            if not per_seed_df.empty:
                top_m = [
                    "PatchTST-Ensemble (3x)",
                    "PatchTST",
                    "Ridge (14xF)",
                    "Equal-Weight RNN Ens",
                    "Val-Opt RNN Ensemble",
                    "BiLSTM",
                    "BiLSTM-Attention",
                    "BiGRU",
                ]
                sub = per_seed_df[per_seed_df["Model"].isin(top_m)].copy()
                strip = (
                    alt.Chart(sub)
                    .mark_circle(size=75, opacity=0.85)
                    .encode(
                        x=alt.X("MAE:Q", title="Per-seed test MAE (MW)", scale=alt.Scale(zero=False)),
                        y=alt.Y("Model:N", sort=top_m, title=None),
                        color=alt.Color("Seed:N", legend=alt.Legend(orient="bottom", title="Seed")),
                        tooltip=["Model:N", "Seed:N", alt.Tooltip("MAE:Q", format=".2f")],
                    )
                    .properties(height=230)
                )
                st.altair_chart(strip)

            if not eff_df.empty and not summary_df.empty:
                em = eff_df.merge(summary_df[["Model", "MAE_mean"]], on="Model", how="left").sort_values("MAE_mean")
                st.dataframe(
                    em[["Model", "MAE_mean", "Params", "Size_KB", "CPU_ms"]],
                    column_config={
                        "MAE_mean": st.column_config.NumberColumn("MAE (MW)", format="%.2f"),
                        "Params": st.column_config.NumberColumn("Params", format="%d"),
                        "Size_KB": st.column_config.NumberColumn("Size (KB)", format="%.1f"),
                        "CPU_ms": st.column_config.NumberColumn("CPU (ms)", format="%.2f"),
                    },
                    hide_index=True,
                    height=180,
                )

# ---------------------------------------------------------------------------
# TAB 2: Rolling CV, Fixed-Test Learning Curve, Ablation & Regimes
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("3-fold rolling-origin CV & unconfounded fixed-test learning curve")
    st.caption(
        "Ridge leads the 3-fold rolling-origin mean (567.79 MW) ahead of SARIMAX (598.32 MW) and XGBoost-Diff (609.63 MW) due to Fold 1's post-2022 crisis shift. "
        "Holding the 2025–2026 test set fixed unconfounds data volume: SARIMAX plateaus at ~651 MW while PatchTST-3× leads at every history length (633.0 → 574.5 MW)."
    )

    col_cv, col_lc = st.columns(2)
    with col_cv:
        with st.container(border=True):
            st.markdown("**3-fold expanding-window rolling-origin CV (2023–2026)**")
            if not rolling_df.empty:
                cv_long = rolling_df.melt(
                    id_vars=["Fold"],
                    value_vars=["Ridge_MAE", "SARIMAX_MAE", "XGBDiff_MAE", "PatchTST3_MAE", "Ens_MAE_mean", "Persist_MAE"],
                    var_name="Model",
                    value_name="MAE_MW",
                )
                cv_long["Model"] = cv_long["Model"].replace(
                    {
                        "Ridge_MAE": "Ridge (14×F)",
                        "SARIMAX_MAE": "SARIMAX",
                        "XGBDiff_MAE": "XGBoost-Diff",
                        "PatchTST3_MAE": "PatchTST-3×",
                        "Ens_MAE_mean": "Val-Opt RNN Ens",
                        "Persist_MAE": "Lag-1 Persist",
                    }
                )
                st.altair_chart(
                    alt.Chart(cv_long)
                    .mark_line(point=True, strokeWidth=2.3)
                    .encode(
                        x=alt.X("Fold:N", sort=rolling_df["Fold"].tolist(), title=None),
                        y=alt.Y("MAE_MW:Q", title="Test MAE (MW)", scale=alt.Scale(zero=False)),
                        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=["Fold:N", "Model:N", alt.Tooltip("MAE_MW:Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                st.dataframe(
                    rolling_df[["Fold", "N_Train", "Ridge_MAE", "SARIMAX_MAE", "XGBDiff_MAE", "PatchTST3_MAE", "Ens_MAE_mean"]],
                    hide_index=True,
                )

    with col_lc:
        with st.container(border=True):
            st.markdown("**Unconfounded fixed-test learning curve (Test fixed = 2025–26)**")
            if not lc_df.empty:
                lc_long = lc_df.melt(
                    id_vars=["Train_Years"],
                    value_vars=["PatchTST3_MAE", "Ridge_MAE", "Ens_MAE_mean", "SARIMAX_MAE"],
                    var_name="Model",
                    value_name="MAE_MW",
                )
                lc_long["Model"] = lc_long["Model"].replace(
                    {
                        "PatchTST3_MAE": "PatchTST-3×",
                        "Ridge_MAE": "Ridge (14×F)",
                        "Ens_MAE_mean": "Val-Opt RNN Ens",
                        "SARIMAX_MAE": "SARIMAX",
                    }
                )
                st.altair_chart(
                    alt.Chart(lc_long)
                    .mark_line(point=True, strokeWidth=2.3)
                    .encode(
                        x=alt.X("Train_Years:O", title="Trailing training history (years)"),
                        y=alt.Y("MAE_MW:Q", title="Fixed 2025–26 Test MAE (MW)", scale=alt.Scale(zero=False)),
                        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=["Train_Years:O", "Model:N", alt.Tooltip("MAE_MW:Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                st.dataframe(lc_df, hide_index=True)

    col_ab, col_bn = st.columns(2)
    with col_ab:
        with st.container(border=True):
            st.markdown("**Validation-first 5-seed ablation (with paired p-values vs Base)**")
            st.caption(
                "No overall blind-test step is statistically significant at p < 0.05 on n = 5 seeds (p = 0.22–0.93). "
                "On Holiday/Eid days, Eid flags reduce error by −41.2 MW incrementally beyond CDD_26."
            )
            if not ablation_df.empty:
                st.dataframe(
                    ablation_df[["Config", "Dims", "Val_MAE_mean", "MAE_mean", "MAE_std", "Hol_MAE_mean", "p_paired_vs_Base"]],
                    column_config={
                        "Val_MAE_mean": st.column_config.NumberColumn("Val-Cal MAE", format="%.1f"),
                        "MAE_mean": st.column_config.NumberColumn("Test MAE", format="%.1f"),
                        "MAE_std": st.column_config.NumberColumn("SD", format="%.1f"),
                        "Hol_MAE_mean": st.column_config.NumberColumn("Hol MAE", format="%.1f"),
                        "p_paired_vs_Base": st.column_config.NumberColumn("Paired p", format="%.3f"),
                    },
                    hide_index=True,
                )

    with col_bn:
        with st.container(border=True):
            st.markdown("**Peak demand magnitude bins (level vs differenced trees & PatchTST-3×)**")
            if not bins_df.empty:
                st.dataframe(
                    bins_df[["Demand_Bin", "N", "PT3_MAE", "Ens_MAE", "XGBD_MAE", "XGB_MAE", "Persist_MAE"]],
                    column_config={
                        "PT3_MAE": st.column_config.NumberColumn("PatchTST-3×", format="%.1f"),
                        "Ens_MAE": st.column_config.NumberColumn("RNN Ens", format="%.1f"),
                        "XGBD_MAE": st.column_config.NumberColumn("XGB-Diff", format="%.1f"),
                        "XGB_MAE": st.column_config.NumberColumn("XGB-Level", format="%.1f"),
                        "Persist_MAE": st.column_config.NumberColumn("Lag-1", format="%.1f"),
                    },
                    hide_index=True,
                )

# ---------------------------------------------------------------------------
# TAB 3: Same-Split 8-Division Forecasts
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Same-split 8-division peak demand benchmark (no cross-dataset comparisons)")
    st.caption(
        "Val-Opt RNN Ensemble outperforms Single BiGRU in 8/8 divisions, Lag-7 in 8/8, Lag-1 Persistence in 7/8, and Ridge in 6/8; "
        "Ridge leads the two largest metropolitan grids (Dhaka 251.7 MW and Chattogram 74.2 MW)."
    )

    if not div_df.empty:
        col_d1, col_d2 = st.columns([1.2, 1.0])
        with col_d1:
            with st.container(border=True):
                st.markdown("**Same-split divisional MAPE comparison**")
                div_l = div_df.melt(
                    id_vars=["Division"],
                    value_vars=["MAPE_mean", "BiGRU_MAPE", "Ridge_MAPE", "Persist_MAPE"],
                    var_name="Method",
                    value_name="MAPE_Pct",
                )
                div_l["Method"] = div_l["Method"].replace(
                    {
                        "MAPE_mean": "Val-Opt RNN Ensemble",
                        "BiGRU_MAPE": "Single BiGRU (Same Split)",
                        "Ridge_MAPE": "Ridge (Same Split)",
                        "Persist_MAPE": "Lag-1 Persistence",
                    }
                )
                st.altair_chart(
                    alt.Chart(div_l)
                    .mark_bar(cornerRadiusEnd=3)
                    .encode(
                        x=alt.X("Division:N", sort=div_df["Division"].tolist(), title=None),
                        y=alt.Y("MAPE_Pct:Q", title="Blind test MAPE (%)"),
                        xOffset="Method:N",
                        color=alt.Color(
                            "Method:N",
                            scale=alt.Scale(
                                domain=["Val-Opt RNN Ensemble", "Single BiGRU (Same Split)", "Ridge (Same Split)", "Lag-1 Persistence"],
                                range=["#2563EB", "#0D9488", "#D97706", "#94A3B8"],
                            ),
                            legend=alt.Legend(orient="bottom", title=None),
                        ),
                        tooltip=["Division:N", "Method:N", alt.Tooltip("MAPE_Pct:Q", format=".2f")],
                    )
                    .properties(height=340)
                )

        with col_d2:
            with st.container(border=True):
                st.markdown("**Same-split 8-division scorecard**")
                st.dataframe(
                    div_df[["Division", "MAE_mean", "BiGRU_MAE", "Ridge_MAE", "Persist_MAE", "Acc_mean", "Gain_vs_BiGRU_%"]],
                    column_config={
                        "Division": st.column_config.TextColumn("Division", pinned=True),
                        "MAE_mean": st.column_config.NumberColumn("RNN Ens (MW)", format="%.1f"),
                        "BiGRU_MAE": st.column_config.NumberColumn("BiGRU (MW)", format="%.1f"),
                        "Ridge_MAE": st.column_config.NumberColumn("Ridge (MW)", format="%.1f"),
                        "Persist_MAE": st.column_config.NumberColumn("Persist (MW)", format="%.1f"),
                        "Acc_mean": st.column_config.NumberColumn("Acc", format="%.2f%%"),
                        "Gain_vs_BiGRU_%": st.column_config.NumberColumn("vs BiGRU", format="+%.1f%%"),
                    },
                    hide_index=True,
                    height=340,
                )

    with st.container(border=True):
        st.markdown("**Reported divisional demand vs supply (load-shedding allocation inspector)**")
        sel_div = st.pills("Select division", DIVISIONS, default="Dhaka", key="div_inspector_pill") or "Dhaka"
        dem_col, sup_col = f"{sel_div}_demand", f"{sel_div}_supply"
        ds_sub = filtered_df[["Date", dem_col, sup_col]].dropna().copy()
        ds_sub["Bal"] = ds_sub[sup_col] - ds_sub[dem_col]
        with st.container(horizontal=True):
            st.metric(f"{sel_div} mean demand", f"{ds_sub[dem_col].mean():,.1f} MW", border=True)
            st.metric(f"{sel_div} peak demand", f"{ds_sub[dem_col].max():,.1f} MW", border=True)
            st.metric(f"{sel_div} mean supply", f"{ds_sub[sup_col].mean():,.1f} MW", border=True)
            st.metric("Mean reported balance", f"{ds_sub['Bal'].mean():+,.2f} MW", "Reported allocation gap", delta_color="off", border=True)

# ---------------------------------------------------------------------------
# TAB 4: Multi-Horizon & LN-ACI Conformal
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Causal direct multi-horizon (1–30 days) & Locally Normalized ACI")
    col_mh, col_cp = st.columns([1.2, 1.0])

    with col_mh:
        with st.container(border=True):
            st.markdown("**Direct causal horizons vs Ridge, Climatology+Trend & Persistence**")
            st.caption(
                "Direct models beat both Origin Persistence and Climatology+Trend through h = 14 days; at h = 30 days (zero future weather), "
                "Climatology+Trend (1,071.9 MW) outperforms both direct RNN (1,230.7 MW) and direct Ridge (1,260.0 MW)."
            )
            if not horizon_df.empty:
                mh_l = horizon_df.melt(
                    id_vars=["Horizon_Days"],
                    value_vars=["Ens_MAE_mean", "Ridge_MAE", "Clim_Trend_MAE", "Origin_Persist_MAE"],
                    var_name="Model",
                    value_name="MAE_MW",
                )
                mh_l["Model"] = mh_l["Model"].replace(
                    {
                        "Ens_MAE_mean": "Val-Opt RNN Ens",
                        "Ridge_MAE": "Direct Ridge",
                        "Clim_Trend_MAE": "Climatology+Trend",
                        "Origin_Persist_MAE": "Origin Persistence",
                    }
                )
                st.altair_chart(
                    alt.Chart(mh_l)
                    .mark_line(point=True, strokeWidth=2.3)
                    .encode(
                        x=alt.X("Horizon_Days:O", title="Forecast horizon (days)"),
                        y=alt.Y("MAE_MW:Q", title="Blind test MAE (MW)"),
                        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=["Horizon_Days:O", "Model:N", alt.Tooltip("MAE_MW:Q", format=".1f")],
                    )
                    .properties(height=280)
                )
                st.dataframe(
                    horizon_df[["Horizon_Days", "Ens_MAE_mean", "Ridge_MAE", "Clim_Trend_MAE", "Origin_Persist_MAE", "Gain_vs_Clim_%"]],
                    column_config={
                        "Horizon_Days": st.column_config.NumberColumn("Horizon", format="%d d"),
                        "Ens_MAE_mean": st.column_config.NumberColumn("RNN Ens", format="%.1f"),
                        "Ridge_MAE": st.column_config.NumberColumn("Ridge", format="%.1f"),
                        "Clim_Trend_MAE": st.column_config.NumberColumn("Clim+Trend", format="%.1f"),
                        "Origin_Persist_MAE": st.column_config.NumberColumn("Persist", format="%.1f"),
                        "Gain_vs_Clim_%": st.column_config.NumberColumn("vs Clim", format="+%.1f%%"),
                    },
                    hide_index=True,
                )

    with col_cp:
        with st.container(border=True):
            st.markdown("**Static Split Conformal vs Locally Normalized ACI (LN-ACI)**")
            st.caption(
                "Static split conformal under-covers seasonal volatility (67.5% at 90% nominal; 51.2% in Summer). "
                "LN-ACI restores 89.6% empirical coverage (88.9%–90.1% across all 4 seasons) and cuts Winkler score by 20.3%."
            )
            if not conformal_df.empty:
                st.dataframe(
                    conformal_df,
                    column_config={
                        "Nominal_%": st.column_config.NumberColumn("Nominal", format="%d%%"),
                        "Mean_Half_Width_MW": st.column_config.NumberColumn("±Half-width", format="%.1f MW"),
                        "Mean_Width_MW": st.column_config.NumberColumn("Mean width", format="%.1f MW"),
                        "PICP_%": st.column_config.NumberColumn("Empirical PICP", format="%.2f%%"),
                        "PINAW_%": st.column_config.NumberColumn("PINAW", format="%.2f%%"),
                        "Winkler_MW": st.column_config.NumberColumn("Winkler", format="%.1f MW"),
                    },
                    hide_index=True,
                )
                season_cov = pd.DataFrame(
                    {
                        "Season": ["Winter", "Post-Monsoon", "Monsoon", "Summer"],
                        "Static Split PICP (%)": [88.9, 72.9, 70.2, 51.2],
                        "LN-ACI PICP (%)": [88.9, 89.8, 90.1, 89.5],
                        "LN-ACI Width (MW)": [2319, 1856, 2891, 4463],
                    }
                )
                st.markdown("**90% PI seasonal coverage breakdown**")
                st.dataframe(season_cov, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 5: Direct PyTorch Ensemble Group Permutation Importance
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Direct group permutation importance on the deployed PyTorch ensemble")
    st.caption(
        "Explains the actual deployed PyTorch ensemble (15 repeats across all 14 lookback steps) rather than a weaker tree surrogate. "
        "Lagged divisional and national load telemetry plus weekly seasonality dominate when live NWP temperature forecasts are unavailable."
    )

    col_p1, col_p2 = st.columns([1.05, 1.15])
    with col_p1:
        with st.container(border=True):
            st.markdown("**Ensemble ΔMAE upon causal block permutation (MW)**")
            if not perm_df.empty:
                ps = perm_df.sort_values("Delta_MAE_Mean_MW", ascending=False).copy()
                st.altair_chart(
                    alt.Chart(ps)
                    .mark_bar(cornerRadiusEnd=4, color="#2563EB")
                    .encode(
                        x=alt.X("Delta_MAE_Mean_MW:Q", title="Increase in ensemble MAE (MW)"),
                        y=alt.Y("Feature_Group:N", sort=ps["Feature_Group"].tolist(), title=None),
                        tooltip=[
                            "Feature_Group:N",
                            "Num_Features:Q",
                            alt.Tooltip("Delta_MAE_Mean_MW:Q", title="ΔMAE (MW)", format=".1f"),
                            alt.Tooltip("Delta_MAE_Std_MW:Q", title="SD (MW)", format=".1f"),
                        ],
                    )
                    .properties(height=340)
                )

    with col_p2:
        with st.container(border=True):
            st.markdown("**Subtropical thermal response (train-estimated τ* = 26.0 °C)**")
            st.caption(
                "Dhaka daily max temp never falls below 19.3 °C (0% < 18 °C → HDD omitted); piecewise slope above 26.0 °C is +470.9 MW/°C."
            )
            therm = filtered_df[["Date", "Dhaka_temp", "Peak_Demand_MW", "Season"]].dropna()
            rule = alt.Chart(pd.DataFrame({"t": [26.0]})).mark_rule(strokeDash=[5, 4], color="#DC2626", strokeWidth=1.8).encode(x="t:Q")
            sc = (
                alt.Chart(therm)
                .mark_circle(size=30, opacity=0.55)
                .encode(
                    x=alt.X("Dhaka_temp:Q", title="Dhaka daily max temperature (°C)", scale=alt.Scale(zero=False)),
                    y=alt.Y("Peak_Demand_MW:Q", title="Gross evening peak demand (MW)", scale=alt.Scale(zero=False)),
                    color=alt.Color("Season:N", legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["Date:T", alt.Tooltip("Dhaka_temp:Q", format=".1f"), alt.Tooltip("Peak_Demand_MW:Q", format=",.0f")],
                )
            )
            st.altair_chart((sc + rule).properties(height=340))

# ---------------------------------------------------------------------------
# TAB 6: QA Grid Analytics & Database Explorer
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Quality-assured grid diagnostics & BPDB database explorer")
    st.caption(
        "Transparent data audit: 20 missing BPDB calendar days (14 gaps, max 6d in Jul 2024) and 16 negative transmission-loss metering inversions excluded from QA loss metrics."
    )

    with st.container(horizontal=True):
        st.metric("QA mean peak line loss", f"{filtered_df['QA_Loss_MW'].mean():.2f} MW", "16 negative records excluded", delta_color="off", border=True)
        st.metric("Raw uncleaned mean loss", f"{filtered_df['Raw_Loss_MW'].mean():.2f} MW", f"Min raw {filtered_df['Raw_Loss_MW'].min():,.0f} MW", delta_color="off", border=True)
        st.metric("QA max peak line loss", f"{filtered_df['QA_Loss_MW'].max():,.0f} MW", border=True)
        st.metric("Filtered observations", f"{len(filtered_df):,} days", f"{start_d} → {end_d}", delta_color="off", border=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        with st.container(border=True):
            st.markdown("**National evening peak demand (MW)**")
            st.altair_chart(
                alt.Chart(filtered_df[["Date", "Peak_Demand_MW", "Season"]].dropna())
                .mark_line(strokeWidth=1.4, color="#2563EB")
                .encode(
                    x=alt.X("Date:T", title=None),
                    y=alt.Y("Peak_Demand_MW:Q", title="Peak demand (MW)", scale=alt.Scale(zero=False)),
                    tooltip=["Date:T", alt.Tooltip("Peak_Demand_MW:Q", format=",.0f"), "Season:N"],
                )
                .properties(height=270)
            )

    with col_g2:
        with st.container(border=True):
            st.markdown("**QA-filtered peak transmission loss (≥ 0 MW)**")
            st.altair_chart(
                alt.Chart(filtered_df[["Date", "QA_Loss_MW"]].dropna())
                .mark_area(opacity=0.35, color="#D97706", line={"color": "#D97706", "strokeWidth": 1.2})
                .encode(
                    x=alt.X("Date:T", title=None),
                    y=alt.Y("QA_Loss_MW:Q", title="QA peak line loss (MW)"),
                    tooltip=["Date:T", alt.Tooltip("QA_Loss_MW:Q", format=".1f")],
                )
                .properties(height=270)
            )

    with st.container(border=True):
        st.markdown("**BPDB dataset preview & CSV export**")
        n_rows = st.slider("Preview rows", 10, 200, 25, 5)
        st.dataframe(filtered_df.tail(n_rows), hide_index=True)
        st.download_button(
            "Download filtered BPDB dataset (CSV)",
            data=filtered_df.to_csv(index=False).encode("utf-8"),
            file_name="BPDB_Bangladesh_Peak_Demand_QA.csv",
            mime="text/csv",
            icon=":material/download:",
        )

# ---------------------------------------------------------------------------
# TAB 7: Scenario Simulator & Manuscript Package
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Interactive peak demand scenario simulator & Overleaf package")
    col_in, col_out = st.columns([1.1, 1.0])

    with col_in:
        with st.container(border=True):
            st.markdown("**Operational scenario inputs**")
            prev_peak = st.slider("Yesterday's peak demand (MW)", 7500, 18000, 14500, 100)
            dhaka_temp = st.slider("Dhaka daily max temperature (°C)", 19.0, 42.0, 33.0, 0.5)
            dow = st.selectbox("Day of week", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"], index=2)
            season_sel = st.selectbox("Season (sets LN-ACI adaptive width)", ["Summer (Mar–May)", "Monsoon (Jun–Sep)", "Winter (Dec–Feb)", "Post-Monsoon (Oct–Nov)"])
            cal_regime = st.selectbox("Calendar regime", ["Regular working day", "Single-day public holiday", "Eid / continuous festival window", "Ramadan evening peak"])

    cdd26 = max(0.0, dhaka_temp - 26.0)
    day_adj = {"Friday": -780.0, "Saturday": -240.0, "Sunday": 110.0, "Monday": 160.0, "Tuesday": 220.0, "Wednesday": 180.0, "Thursday": 90.0}[dow]
    hol_adj = {"Regular working day": 0.0, "Single-day public holiday": -520.0, "Eid / continuous festival window": -2100.0, "Ramadan evening peak": 680.0}[cal_regime]
    sim_mw = float(np.clip(0.84 * prev_peak + 0.16 * 12800.0 + cdd26 * 125.0 + day_adj + hol_adj - 350.0, 7000.0, 19500.0))
    aci_half = {"Summer (Mar–May)": 2231.5, "Monsoon (Jun–Sep)": 1445.5, "Winter (Dec–Feb)": 1159.5, "Post-Monsoon (Oct–Nov)": 928.0}[season_sel]

    with col_out:
        with st.container(border=True):
            st.markdown("**Simulated day-ahead forecast & season-adaptive 90% LN-ACI envelope**")
            with st.container(horizontal=True):
                st.metric("Point forecast", f"{sim_mw:,.0f} MW", f"{sim_mw / 1000:.2f} GW", delta_color="off", border=True)
                st.metric("90% LN-ACI band", f"{sim_mw - aci_half:,.0f} – {sim_mw + aci_half:,.0f} MW", f"±{aci_half:,.0f} MW ({season_sel.split(' ')[0]})", delta_color="off", border=True)
            st.caption(f"Subtropical CDD (base 26.0 °C): **{cdd26:.1f} °C** · Historical 90% LN-ACI coverage: **88.9%–90.1%** across all four seasons.")

    with st.container(border=True):
        st.markdown("**Revised Q1 manuscript package (Elsevier `elsarticle 5p,twocolumn`)**")
        st.markdown(
            "- **16 tables, 14 vector PDF figures, 30 verified references** — addressing all 13 senior-reviewer methodological points.\n"
            "- **Open repository:** [github.com/mubtasimfuad7/Bangladesh-Peak-Demand-Forecasting](https://github.com/mubtasimfuad7/Bangladesh-Peak-Demand-Forecasting)"
        )
        if PAPER_ZIP_PATH.exists():
            st.download_button(
                "Download revised Overleaf LaTeX package (ZIP)",
                data=PAPER_ZIP_PATH.read_bytes(),
                file_name="Part1_Demand_Journal_Paper.zip",
                mime="application/zip",
                icon=":material/download:",
            )
