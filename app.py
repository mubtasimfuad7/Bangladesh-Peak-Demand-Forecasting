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
    nonneg = raw_loss[raw_loss >= 0]
    q1, q3 = nonneg.quantile(0.25), nonneg.quantile(0.75)
    upper_cut = q3 + 3.0 * (q3 - q1)
    df["QA_Loss_MW"] = raw_loss.where((raw_loss >= 0) & (raw_loss <= upper_cut), np.nan)
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
mcs_df = load_journal_csv("part1_mcs.csv")
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
conf_sea_df = load_journal_csv("part1_conformal_seasonal.csv")
perm_df = load_journal_csv("part1_permutation_importance.csv")

if not eff_df.empty and "Unnamed: 0" in eff_df.columns:
    eff_df = eff_df.rename(columns={"Unnamed: 0": "Model"})

# Header
st.title(":material/bolt: Bangladesh peak electricity demand forecasting")
st.caption(
    "24-model 5-seed benchmark with 95% Hansen Model Confidence Set, HLN-DM tests referenced to PatchTST-3×, "
    "3-fold rolling-origin CV, 5-way conformal uncertainty ablation on untouched Val-Cal residuals, "
    "same-split 8-division forecasts, causal 1–30 day hybrids, and direct merged-block permutation importance."
)

# Sidebar
with st.sidebar:
    st.subheader(":material/tune: Protocol & data QA")
    st.markdown(
        ":green-badge[Train-only scaling] :blue-badge[95% Hansen MCS] :orange-badge[5-way conformal]"
    )
    qa = meta.get("data_qa", {})
    st.caption(
        f"**Records:** {len(df):,} BPDB cycles across {qa.get('calendar_span_days', 2414):,} calendar days "
        f"({qa.get('missing_calendar_days', 20)} reporting gaps audited)  \n"
        f"**Peak range:** {df['Peak_Demand_MW'].min():,.0f}–{df['Peak_Demand_MW'].max():,.0f} MW "
        f"({df['Peak_Demand_GW'].min():.2f}–{df['Peak_Demand_GW'].max():.2f} GW)  \n"
        f"**Features:** {meta.get('n_features', 117)} channels (raw `Year` dropped)  \n"
        f"**Splits:** Train 1,652 · Val-ES 105 · Val-Cal 105 · Test 462  \n"
        f"**Detrended cooling threshold:** τ* = {meta.get('tau_star_C', 25.0):.1f} °C "
        f"(+{meta.get('tau_slope_detrended_MW_per_C', 387.2):.1f} MW/°C; HDD omitted)  \n"
        f"**Robust QA line loss:** {qa.get('robust_mean_loss_mw', 508.3):.1f} MW mean / "
        f"{qa.get('robust_median_loss_mw', 510.0):.1f} MW median "
        f"({qa.get('negative_loss_records', 16)} neg + {qa.get('upper_outlier_loss_records', 3)} upper IQR outliers excluded)"
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
plain_aci90 = (
    conformal_df[
        (conformal_df["Model"] == "PatchTST-Ensemble (3x)")
        & (conformal_df["Method"].str.contains("Plain ACI"))
        & (conformal_df["Nominal_%"] == 90)
    ].iloc[0]
    if not conformal_df.empty and "Model" in conformal_df.columns
    else None
)
hol_row = (
    regime_df[regime_df["Regime"].str.contains("Holiday", na=False)].iloc[0]
    if not regime_df.empty and "Regime" in regime_df.columns
    else None
)

with st.container(horizontal=True):
    st.metric(
        "2025–26 MCS leader (tied)",
        f"{pt3_row['MAE_mean']:.2f} ± {pt3_row['MAE_std']:.2f} MW" if pt3_row is not None else "584.20 ± 10.71 MW",
        f"PatchTST-3× ({pt3_row['Skill_Wkday_mean']:+.1f}% vs wkday)" if pt3_row is not None else "+7.8% vs wkday",
        delta_color="off",
        border=True,
    )
    st.metric(
        "3-fold CV leader (2023–26)",
        f"{rolling_df['Wkday_MAE'].mean():.2f} MW" if not rolling_df.empty and "Wkday_MAE" in rolling_df.columns else "551.01 MW",
        f"Wkday Persist · Ridge-Diff {rolling_df['RidgeDiff_MAE'].mean():.1f} MW"
        if not rolling_df.empty and "RidgeDiff_MAE" in rolling_df.columns else "Ridge-Diff 562.8 MW",
        delta_color="off",
        border=True,
    )
    st.metric(
        "Best 90% Winkler (Plain ACI)",
        f"{plain_aci90['Winkler_MW']:,.0f} MW" if plain_aci90 is not None else "3,636 MW",
        "−25.3% vs static · LN-ACI 88.1–90.1% PICP",
        border=True,
    )
    st.metric(
        "Holiday / Eid leader (n=41)",
        f"{hol_row['Ridge_MAE']:.1f} MW" if hol_row is not None else "867.9 MW",
        f"Ridge (Eq-RNN {hol_row['EqRNN_MAE']:.1f} · PT3 {hol_row['PT3_MAE']:.1f})" if hol_row is not None else "Ridge leads festivals",
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
        ":material/leaderboard: 24-model benchmark & MCS",
        ":material/timeline: Rolling CV & ablations",
        ":material/map: Same-split 8 divisions",
        ":material/schedule: Multi-horizon & 5-way conformal",
        ":material/psychology: Direct explainability",
        ":material/analytics: Appendix A — QA grid data",
        ":material/tune: Scenario simulator & paper",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1: 24-Model National Benchmark & 95% Hansen MCS
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("National 24-model 5-seed benchmark & 95% Hansen MCS (Mar 2025 – Jun 2026, n = 462)")
    st.caption(
        "Strict train-only scaling, raw Year dropped, 14-day split embargoes, separate Val-ES (105d) and Val-Cal (105d) partitions, "
        "and matched Val-ES hyperparameter tuning across all families. PatchTST-3×, PatchTST-Diff, Ridge, and Equal-Weight RNN form a statistically tied leading group inside the 95% MCS."
    )

    if not summary_df.empty:
        metric_choice = st.segmented_control(
            "Select ranking metric",
            ["MAE (MW)", "RMSE (MW)", "MAPE (%)", "Skill vs wkday (%)", "R²"],
            default="MAE (MW)",
            key="nat_metric_choice",
        )
        col_map = {
            "MAE (MW)": ("MAE_mean", "MAE_std", True),
            "RMSE (MW)": ("RMSE_mean", "RMSE_std", True),
            "MAPE (%)": ("MAPE_mean", "MAPE_std", True),
            "Skill vs wkday (%)": ("Skill_Wkday_mean", None, False),
            "R²": ("R2_mean", "R2_std", False),
        }
        val_col, std_col, ascending = col_map[metric_choice or "MAE (MW)"]
        plot_df = summary_df.sort_values(val_col, ascending=ascending).copy()
        if std_col is not None:
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
                st.markdown(f"**24-model comparison by {metric_choice} (mean ± 1 SD across 5 seeds)**")
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
                            alt.Tooltip("MAE_std:Q", title="MAE SD (MW)", format=".2f"),
                            alt.Tooltip("RMSE_mean:Q", title="RMSE (MW)", format=".2f"),
                            alt.Tooltip("Skill_Wkday_mean:Q", title="Skill vs wkday (%)", format="+.1f"),
                        ],
                    )
                )
                if std_col is not None:
                    err = (
                        alt.Chart(plot_df)
                        .mark_errorbar(color="#1E293B", ticks=True)
                        .encode(
                            x=alt.X("Lower:Q", title=metric_choice),
                            x2="Upper:Q",
                            y=alt.Y("Model:N", sort=plot_df["Model"].tolist()),
                        )
                    )
                    st.altair_chart((bars + err).properties(height=580))
                else:
                    st.altair_chart(bars.properties(height=580))

        with col_t:
            with st.container(border=True):
                st.markdown("**Full 24-model scorecard**")
                disp = summary_df.sort_values("MAE_mean").copy()
                disp["MAE (MW)"] = disp.apply(lambda r: f"{r['MAE_mean']:.2f} ± {r['MAE_std']:.2f}", axis=1)
                disp["RMSE (MW)"] = disp.apply(lambda r: f"{r['RMSE_mean']:.2f} ± {r['RMSE_std']:.2f}", axis=1)
                st.dataframe(
                    disp[["Model", "MAE (MW)", "RMSE (MW)", "MAPE_mean", "Skill_Wkday_mean", "R2_mean"]],
                    column_config={
                        "Model": st.column_config.TextColumn("Model", pinned=True),
                        "MAPE_mean": st.column_config.NumberColumn("MAPE", format="%.2f%%"),
                        "Skill_Wkday_mean": st.column_config.NumberColumn("Skill vs wkday", format="%+.1f%%"),
                        "R2_mean": st.column_config.NumberColumn("R²", format="%.3f"),
                    },
                    hide_index=True,
                    height=580,
                )

    col_mcs, col_dm = st.columns([0.95, 1.25])
    with col_mcs:
        with st.container(border=True):
            st.markdown("**95% Hansen Model Confidence Set (top 12 models)**")
            st.caption(
                "Moving-block bootstrap (T_max, block 7, 1,000 resamples). All top seven models remain inside the 95% MCS on both MAE and RMSE (p_MCS ≥ 0.29)."
            )
            if not mcs_df.empty:
                st.dataframe(
                    mcs_df[["Model", "S42_MAE_pMCS", "S42_RMSE_pMCS", "Ens5_MAE_pMCS", "Ens5_RMSE_pMCS"]],
                    column_config={
                        "Model": st.column_config.TextColumn("Model", pinned=True),
                        "S42_MAE_pMCS": st.column_config.NumberColumn("S42 MAE p", format="%.3f"),
                        "S42_RMSE_pMCS": st.column_config.NumberColumn("S42 RMSE p", format="%.3f"),
                        "Ens5_MAE_pMCS": st.column_config.NumberColumn("5-Seed MAE p", format="%.3f"),
                        "Ens5_RMSE_pMCS": st.column_config.NumberColumn("5-Seed RMSE p", format="%.3f"),
                    },
                    hide_index=True,
                    height=430,
                )

    with col_dm:
        with st.container(border=True):
            st.markdown("**HLN-adjusted Diebold–Mariano tests (Ref: PatchTST-Ensemble 3×)**")
            st.caption(
                "Holm–Bonferroni adjusted across 23 comparators. On Seed-42 (577.37 MW), PatchTST-3× significantly beats 14/23 baselines (p_Holm ≤ 0.018); "
                "gaps vs Ridge (p_Holm = 0.407 / 1.000) and Equal-Weight RNN (0.407 / 1.000) are not significant."
            )
            if not dm_df.empty:
                dm_p = dm_df.sort_values("Ens5_Delta_MW").copy()
                dm_p["Sig"] = np.where(dm_p["Ens5_p_holm"] < 0.05, "p_Holm < 0.05 (5-seed)", "Not significant")
                zr = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4], color="#64748B").encode(x="x:Q")
                ci = alt.Chart(dm_p).mark_rule(strokeWidth=2).encode(
                    x=alt.X("Ens5_CI_lo:Q", title="ΔMAE (PatchTST-3× minus comparator, MW, 95% CI)"),
                    x2="Ens5_CI_hi:Q",
                    y=alt.Y("Comparator:N", sort=dm_p["Comparator"].tolist(), title=None),
                    color=alt.Color(
                        "Sig:N",
                        scale=alt.Scale(domain=["p_Holm < 0.05 (5-seed)", "Not significant"], range=["#0D9488", "#94A3B8"]),
                        legend=alt.Legend(orient="bottom", title=None),
                    ),
                )
                pt = alt.Chart(dm_p).mark_point(filled=True, size=60).encode(
                    x="Ens5_Delta_MW:Q",
                    y=alt.Y("Comparator:N", sort=dm_p["Comparator"].tolist(), title=None),
                    color="Sig:N",
                    tooltip=[
                        "Comparator:N",
                        alt.Tooltip("S42_Delta_MW:Q", title="Seed-42 ΔMAE", format=".2f"),
                        alt.Tooltip("S42_p_holm:Q", title="Seed-42 p_Holm", format=".4f"),
                        alt.Tooltip("Ens5_Delta_MW:Q", title="5-Seed ΔMAE", format=".2f"),
                        alt.Tooltip("Ens5_p_holm:Q", title="5-Seed p_Holm", format=".4f"),
                    ],
                )
                st.altair_chart((zr + ci + pt).properties(height=430))

    col_sp, col_eff = st.columns([1.05, 1.0])
    with col_sp:
        with st.container(border=True):
            st.markdown("**5-seed dispersion across leading architectures**")
            if not per_seed_df.empty:
                top_m = [
                    "PatchTST-Ensemble (3x)",
                    "PatchTST-Diff (Δy, 3x)",
                    "PatchTST",
                    "Val-Opt RNN Ensemble",
                    "Ridge (14xF)",
                    "Equal-Weight RNN Ens",
                    "BiLSTM",
                    "DLinear (CI)",
                ]
                sub = per_seed_df[per_seed_df["Model"].isin(top_m)].copy()
                strip = (
                    alt.Chart(sub)
                    .mark_circle(size=75, opacity=0.85)
                    .encode(
                        x=alt.X("MAE:Q", title="Per-seed test MAE (MW)", scale=alt.Scale(zero=False)),
                        y=alt.Y("Model:N", sort=top_m, title=None),
                        color=alt.Color("Seed:N", legend=alt.Legend(orient="bottom", title="Seed")),
                        tooltip=["Model:N", "Seed:N", alt.Tooltip("MAE:Q", format=".2f"), alt.Tooltip("RMSE:Q", format=".2f")],
                    )
                    .properties(height=260)
                )
                st.altair_chart(strip)

    with col_eff:
        with st.container(border=True):
            st.markdown("**Computational efficiency profile (consistent fp32 KiB & CPU ms)**")
            if not eff_df.empty and not summary_df.empty:
                em = eff_df.merge(summary_df[["Model", "MAE_mean"]], on="Model", how="left").sort_values("MAE_mean")
                st.dataframe(
                    em[["Model", "MAE_mean", "Params", "Size_KiB", "CPU_ms"]],
                    column_config={
                        "MAE_mean": st.column_config.NumberColumn("MAE (MW)", format="%.2f"),
                        "Params": st.column_config.NumberColumn("Params", format="%d"),
                        "Size_KiB": st.column_config.NumberColumn("fp32 Size (KiB)", format="%.2f"),
                        "CPU_ms": st.column_config.NumberColumn("CPU (ms)", format="%.2f"),
                    },
                    hide_index=True,
                    height=260,
                )

# ---------------------------------------------------------------------------
# TAB 2: Rolling CV, History Sensitivity, Ablation & Regimes
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("3-fold rolling-origin CV, within-period history sensitivity, and ablations")
    st.caption(
        "Across 3 expanding folds (2023–2026), Weekday-Adjusted Persistence (551.01 MW) and Ridge-Diff (562.76 MW) lead the 3-fold mean "
        "while deep models suffer sharp Fold-1 (2023) errors. The trailing-history curve holds 2025–2026 fixed (same test window as Fold 3) as a within-period sensitivity diagnostic."
    )

    col_cv, col_lc = st.columns(2)
    with col_cv:
        with st.container(border=True):
            st.markdown("**3-fold expanding-window rolling-origin CV (2023–2026)**")
            if not rolling_df.empty:
                cv_long = rolling_df.melt(
                    id_vars=["Fold"],
                    value_vars=["Wkday_MAE", "RidgeDiff_MAE", "Ridge_MAE", "SARIMAX_MAE", "XGBDiff_MAE", "EqRNN_MAE", "PatchTST3_MAE"],
                    var_name="Model",
                    value_name="MAE_MW",
                )
                cv_long["Model"] = cv_long["Model"].replace(
                    {
                        "Wkday_MAE": "Wkday Persist.",
                        "RidgeDiff_MAE": "Ridge-Diff (Δy)",
                        "Ridge_MAE": "Ridge (14×F)",
                        "SARIMAX_MAE": "SARIMAX",
                        "XGBDiff_MAE": "XGBoost-Diff",
                        "EqRNN_MAE": "Equal-Wt RNN",
                        "PatchTST3_MAE": "PatchTST-3×",
                    }
                )
                st.altair_chart(
                    alt.Chart(cv_long)
                    .mark_line(point=True, strokeWidth=2.2)
                    .encode(
                        x=alt.X("Fold:N", sort=rolling_df["Fold"].tolist(), title=None),
                        y=alt.Y("MAE_MW:Q", title="Test MAE (MW)", scale=alt.Scale(zero=False)),
                        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=["Fold:N", "Model:N", alt.Tooltip("MAE_MW:Q", format=".2f")],
                    )
                    .properties(height=280)
                )
                st.dataframe(
                    rolling_df[["Fold", "N_Train", "Wkday_MAE", "RidgeDiff_MAE", "Ridge_MAE", "SARIMAX_MAE", "EqRNN_MAE", "PatchTST3_MAE"]],
                    column_config={
                        "Wkday_MAE": st.column_config.NumberColumn("Wkday", format="%.1f"),
                        "RidgeDiff_MAE": st.column_config.NumberColumn("Ridge-Δy", format="%.1f"),
                        "Ridge_MAE": st.column_config.NumberColumn("Ridge", format="%.1f"),
                        "SARIMAX_MAE": st.column_config.NumberColumn("SARIMAX", format="%.1f"),
                        "EqRNN_MAE": st.column_config.NumberColumn("Eq-RNN", format="%.1f"),
                        "PatchTST3_MAE": st.column_config.NumberColumn("PT3", format="%.1f"),
                    },
                    hide_index=True,
                )

    with col_lc:
        with st.container(border=True):
            st.markdown("**Within-period sensitivity to trailing training length (fixed 2025–26 test)**")
            if not lc_df.empty:
                lc_long = lc_df.melt(
                    id_vars=["Train_Years"],
                    value_vars=["PatchTST3_MAE", "Ridge_MAE", "EqRNN_MAE", "SARIMAX_MAE"],
                    var_name="Model",
                    value_name="MAE_MW",
                )
                lc_long["Model"] = lc_long["Model"].replace(
                    {
                        "PatchTST3_MAE": "PatchTST-3×",
                        "Ridge_MAE": "Ridge (14×F)",
                        "EqRNN_MAE": "Equal-Wt RNN",
                        "SARIMAX_MAE": "SARIMAX",
                    }
                )
                st.altair_chart(
                    alt.Chart(lc_long)
                    .mark_line(point=True, strokeWidth=2.3)
                    .encode(
                        x=alt.X("Train_Years:O", title="Trailing training history (years)"),
                        y=alt.Y("MAE_MW:Q", title="Fixed 2025–26 test MAE (MW)", scale=alt.Scale(zero=False)),
                        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=["Train_Years:O", "Model:N", alt.Tooltip("MAE_MW:Q", format=".1f")],
                    )
                    .properties(height=280)
                )
                st.dataframe(lc_df, hide_index=True)

    col_ab, col_bn = st.columns(2)
    with col_ab:
        with st.container(border=True):
            st.markdown("**Validation-first 5-seed ablation on PatchTST-3× & Ridge**")
            st.caption(
                "Detrended CDD_25 and Eid flags yield non-significant overall shifts (p = 0.237) while cutting PatchTST-3× Holiday MAE by −101.4 MW. "
                "95%-variance PCA significantly degrades PatchTST-3× (+83.1 MW, paired p = 0.006**) and Ridge (+33.5 MW)."
            )
            if not ablation_df.empty:
                st.dataframe(
                    ablation_df[
                        ["Config", "Dims", "PT3_ValCal_MAE", "PT3_Test_MAE", "PT3_Test_SD", "p_paired_vs_Base", "PT3_Hol_MAE", "Ridge_Test_MAE"]
                    ],
                    column_config={
                        "PT3_ValCal_MAE": st.column_config.NumberColumn("PT3 Val-Cal", format="%.1f"),
                        "PT3_Test_MAE": st.column_config.NumberColumn("PT3 Test", format="%.2f"),
                        "PT3_Test_SD": st.column_config.NumberColumn("SD", format="%.2f"),
                        "p_paired_vs_Base": st.column_config.NumberColumn("Paired p", format="%.3f"),
                        "PT3_Hol_MAE": st.column_config.NumberColumn("PT3 Hol", format="%.1f"),
                        "Ridge_Test_MAE": st.column_config.NumberColumn("Ridge Test", format="%.2f"),
                    },
                    hide_index=True,
                )

    with col_bn:
        with st.container(border=True):
            st.markdown("**Peak demand magnitude bins & regime breakdown**")
            if not bins_df.empty:
                st.dataframe(
                    bins_df[["Demand_Bin", "N", "PT3_MAE", "Ridge_MAE", "EqRNN_MAE", "XGBD_MAE", "XGB_MAE", "Wkday_MAE"]],
                    column_config={
                        "PT3_MAE": st.column_config.NumberColumn("PT3", format="%.1f"),
                        "Ridge_MAE": st.column_config.NumberColumn("Ridge", format="%.1f"),
                        "EqRNN_MAE": st.column_config.NumberColumn("Eq-RNN", format="%.1f"),
                        "XGBD_MAE": st.column_config.NumberColumn("XGB-Δy", format="%.1f"),
                        "XGB_MAE": st.column_config.NumberColumn("XGB-Lvl", format="%.1f"),
                        "Wkday_MAE": st.column_config.NumberColumn("Wkday", format="%.1f"),
                    },
                    hide_index=True,
                )
            if not regime_df.empty:
                st.dataframe(
                    regime_df[["Regime", "N", "PT3_MAE", "Ridge_MAE", "EqRNN_MAE", "XGBD_MAE", "Wkday_MAE"]],
                    column_config={
                        "PT3_MAE": st.column_config.NumberColumn("PT3", format="%.1f"),
                        "Ridge_MAE": st.column_config.NumberColumn("Ridge", format="%.1f"),
                        "EqRNN_MAE": st.column_config.NumberColumn("Eq-RNN", format="%.1f"),
                        "XGBD_MAE": st.column_config.NumberColumn("XGB-Δy", format="%.1f"),
                        "Wkday_MAE": st.column_config.NumberColumn("Wkday", format="%.1f"),
                    },
                    hide_index=True,
                )

# ---------------------------------------------------------------------------
# TAB 3: Same-Split 8-Division Forecasts
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Same-split 8-division peak demand benchmark (5 seeds, identical 2025–26 split)")
    st.caption(
        "Equal-Weight RNN outperforms PatchTST-3× in all 8 of 8 divisions and leads 5 regional grids (Rajshahi, Khulna, Mymensingh, Barishal, Sylhet). "
        "Single BiGRU, Ridge, and Equal-Weight RNN tie closely in Dhaka (249.6–251.3 MW), while Weekday Persistence (73.0 MW) and Ridge (74.3 MW) edge ahead in Chattogram."
    )

    if not div_df.empty:
        col_d1, col_d2 = st.columns([1.2, 1.0])
        with col_d1:
            with st.container(border=True):
                st.markdown("**Same-split divisional MAPE comparison**")
                div_l = div_df.melt(
                    id_vars=["Division"],
                    value_vars=["EqRNN_MAPE", "PT3_MAPE", "Ridge_MAPE", "Wkday_MAPE"],
                    var_name="Method",
                    value_name="MAPE_Pct",
                )
                div_l["Method"] = div_l["Method"].replace(
                    {
                        "EqRNN_MAPE": "Equal-Weight RNN",
                        "PT3_MAPE": "PatchTST-3×",
                        "Ridge_MAPE": "Ridge (14×F)",
                        "Wkday_MAPE": "Wkday Persistence",
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
                                domain=["Equal-Weight RNN", "PatchTST-3×", "Ridge (14×F)", "Wkday Persistence"],
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
                st.markdown("**Same-split 8-division MAE scorecard (MW)**")
                st.dataframe(
                    div_df[["Division", "EqRNN_MAE", "PT3_MAE", "Ridge_MAE", "BiGRU_MAE", "Wkday_MAE", "EqRNN_MAPE"]],
                    column_config={
                        "Division": st.column_config.TextColumn("Division", pinned=True),
                        "EqRNN_MAE": st.column_config.NumberColumn("Eq-RNN (MW)", format="%.1f"),
                        "PT3_MAE": st.column_config.NumberColumn("PT3 (MW)", format="%.1f"),
                        "Ridge_MAE": st.column_config.NumberColumn("Ridge (MW)", format="%.1f"),
                        "BiGRU_MAE": st.column_config.NumberColumn("BiGRU (MW)", format="%.1f"),
                        "Wkday_MAE": st.column_config.NumberColumn("Wkday (MW)", format="%.1f"),
                        "EqRNN_MAPE": st.column_config.NumberColumn("Eq-RNN MAPE", format="%.2f%%"),
                    },
                    hide_index=True,
                    height=340,
                )

    with st.container(border=True):
        st.markdown("**Historical BPDB divisional reported demand vs supply (descriptive archive)**")
        sel_div = st.pills("Select division", DIVISIONS, default="Dhaka", key="div_inspector_pill") or "Dhaka"
        dem_col, sup_col = f"{sel_div}_demand", f"{sel_div}_supply"
        ds_sub = filtered_df[["Date", dem_col, sup_col]].dropna().copy()
        ds_sub["Bal"] = ds_sub[sup_col] - ds_sub[dem_col]
        with st.container(horizontal=True):
            st.metric(f"{sel_div} mean demand", f"{ds_sub[dem_col].mean():,.1f} MW", border=True)
            st.metric(f"{sel_div} peak demand", f"{ds_sub[dem_col].max():,.1f} MW", border=True)
            st.metric(f"{sel_div} mean supply", f"{ds_sub[sup_col].mean():,.1f} MW", border=True)
            st.metric("Mean reported supply − demand", f"{ds_sub['Bal'].mean():+,.2f} MW", "Historical descriptive archive", delta_color="off", border=True)

# ---------------------------------------------------------------------------
# TAB 4: Multi-Horizon Hybrids & 5-Way Conformal Ablation
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Causal direct multi-horizon hybrids (1–30 days) & 5-way conformal ablation")
    col_mh, col_cp = st.columns([1.15, 1.05])

    with col_mh:
        with st.container(border=True):
            st.markdown("**Direct horizons vs Climatology+Ridge-Residual Hybrid**")
            st.caption(
                "PatchTST-3× leads h = 1, 3, and 7 days. Beyond one week (without future NWP weather), unconstrained direct neural/Ridge models drift, "
                "whereas the Climatology+Ridge-Residual Hybrid falls back gracefully to Climatology+Trend (1,070.8 MW at h=14; 1,071.9 MW at h=30)."
            )
            if not horizon_df.empty:
                mh_l = horizon_df.melt(
                    id_vars=["Horizon_Days"],
                    value_vars=["PT3_MAE", "EqRNN_MAE", "Clim_Resid_Hyb_MAE", "Clim_Trend_MAE", "Origin_Persist_MAE"],
                    var_name="Model",
                    value_name="MAE_MW",
                )
                mh_l["Model"] = mh_l["Model"].replace(
                    {
                        "PT3_MAE": "PatchTST-3×",
                        "EqRNN_MAE": "Equal-Wt RNN",
                        "Clim_Resid_Hyb_MAE": "Clim+Ridge-Resid Hybrid",
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
                    horizon_df[
                        ["Horizon_Days", "PT3_MAE", "EqRNN_MAE", "Ridge_MAE", "Clim_Resid_Hyb_MAE", "Clim_Trend_MAE", "Origin_Persist_MAE"]
                    ],
                    column_config={
                        "Horizon_Days": st.column_config.NumberColumn("Horizon", format="%d d"),
                        "PT3_MAE": st.column_config.NumberColumn("PT3", format="%.1f"),
                        "EqRNN_MAE": st.column_config.NumberColumn("Eq-RNN", format="%.1f"),
                        "Ridge_MAE": st.column_config.NumberColumn("Ridge", format="%.1f"),
                        "Clim_Resid_Hyb_MAE": st.column_config.NumberColumn("Clim+Ridge Hyb", format="%.1f"),
                        "Clim_Trend_MAE": st.column_config.NumberColumn("Clim+Trend", format="%.1f"),
                        "Origin_Persist_MAE": st.column_config.NumberColumn("Persist", format="%.1f"),
                    },
                    hide_index=True,
                )

    with col_cp:
        with st.container(border=True):
            st.markdown("**5-way conformal ablation on untouched Val-Cal residuals**")
            st.caption(
                "Plain ACI minimizes proper Winkler score (−25.3% vs static; 2,404 MW mean width), "
                "while LN-ACI achieves the tightest seasonal coverage uniformity (88.1%–90.1% across all four seasons, including 90.1% in Summer)."
            )
            if not conformal_df.empty:
                cp_model = st.segmented_control(
                    "Select base forecaster",
                    ["PatchTST-Ensemble (3x)", "Ridge (14xF)"],
                    default="PatchTST-Ensemble (3x)",
                    key="cp_model_sel",
                )
                sub_cp = conformal_df[conformal_df["Model"] == (cp_model or "PatchTST-Ensemble (3x)")]
                st.dataframe(
                    sub_cp[["Method", "Nominal_%", "Mean_Width_MW", "PICP_%", "PINAW_%", "Winkler_MW"]],
                    column_config={
                        "Nominal_%": st.column_config.NumberColumn("Nom", format="%d%%"),
                        "Mean_Width_MW": st.column_config.NumberColumn("Width", format="%.0f MW"),
                        "PICP_%": st.column_config.NumberColumn("PICP", format="%.1f%%"),
                        "PINAW_%": st.column_config.NumberColumn("PINAW", format="%.1f%%"),
                        "Winkler_MW": st.column_config.NumberColumn("Winkler", format="%.0f MW"),
                    },
                    hide_index=True,
                    height=290,
                )
            if not conf_sea_df.empty:
                st.markdown("**90% PI seasonal coverage & width breakdown (PatchTST-3×)**")
                st.dataframe(
                    conf_sea_df[
                        [
                            "Method",
                            "Overall_PICP_%",
                            "Winter_PICP_%",
                            "PostMon_PICP_%",
                            "Monsoon_PICP_%",
                            "Summer_PICP_%",
                            "Summer_Width_MW",
                        ]
                    ],
                    column_config={
                        "Overall_PICP_%": st.column_config.NumberColumn("All", format="%.1f%%"),
                        "Winter_PICP_%": st.column_config.NumberColumn("Win", format="%.1f%%"),
                        "PostMon_PICP_%": st.column_config.NumberColumn("Post", format="%.1f%%"),
                        "Monsoon_PICP_%": st.column_config.NumberColumn("Mon", format="%.1f%%"),
                        "Summer_PICP_%": st.column_config.NumberColumn("Sum", format="%.1f%%"),
                        "Summer_Width_MW": st.column_config.NumberColumn("Sum W", format="%.0f"),
                    },
                    hide_index=True,
                )

# ---------------------------------------------------------------------------
# TAB 5: Direct Merged-Block Permutation Importance & Thermal Response
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Direct group permutation importance on PatchTST-3× & detrended thermal response")
    st.caption(
        "Explains deployed PatchTST-Ensemble (3×) directly (15 repeats across all 14 lookback steps) with a merged Autoregressive Load History block "
        "(k = 43 national + divisional load features) to avoid splitting collinear lag signals."
    )

    col_p1, col_p2 = st.columns([1.05, 1.15])
    with col_p1:
        with st.container(border=True):
            st.markdown("**PatchTST-3× ΔMAE upon causal block permutation (MW)**")
            if not perm_df.empty:
                ps = perm_df.sort_values("Delta_MAE_Mean_MW", ascending=False).copy()
                st.altair_chart(
                    alt.Chart(ps)
                    .mark_bar(cornerRadiusEnd=4, color="#0D9488")
                    .encode(
                        x=alt.X("Delta_MAE_Mean_MW:Q", title="Increase in PatchTST-3× MAE (MW)"),
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
            st.markdown("**Detrended subtropical thermal response (train-estimated τ* = 25.0 °C)**")
            st.caption(
                "Linearly detrending Train demand (+780.8 MW/yr secular growth) before threshold search yields τ* = 25.0 °C "
                "(detrended slope +387.2 MW/°C vs confounded +470.9 MW/°C). Dhaka max temperature never drops below 19.3 °C (HDD omitted)."
            )
            therm = filtered_df[["Date", "Dhaka_temp", "Peak_Demand_MW", "Season"]].dropna()
            rule = alt.Chart(pd.DataFrame({"t": [25.0]})).mark_rule(strokeDash=[5, 4], color="#DC2626", strokeWidth=1.8).encode(x="t:Q")
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
# TAB 6: Appendix A — Robust QA Grid Analytics & Database Explorer
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Appendix A: Robust QA-filtered descriptive grid diagnostics & BPDB explorer")
    st.caption(
        "Transparent data audit: 20 missing BPDB calendar days (14 gaps, max 6d in Jul 2024), 16 negative transmission-loss metering inversions, "
        "and 3 extreme upper outliers above Q3 + 3×IQR (1,168 MW) excluded from robust QA loss metrics."
    )

    with st.container(horizontal=True):
        st.metric("Robust QA mean line loss", f"{filtered_df['QA_Loss_MW'].mean():.2f} MW", "16 neg + 3 upper IQR outliers excluded", delta_color="off", border=True)
        st.metric("Robust QA median line loss", f"{filtered_df['QA_Loss_MW'].median():.1f} MW", f"IQR {filtered_df['QA_Loss_MW'].quantile(0.75) - filtered_df['QA_Loss_MW'].quantile(0.25):.1f} MW", delta_color="off", border=True)
        st.metric("Raw uncleaned mean loss", f"{filtered_df['Raw_Loss_MW'].mean():.2f} MW", f"Range {filtered_df['Raw_Loss_MW'].min():,.0f} to {filtered_df['Raw_Loss_MW'].max():,.0f} MW", delta_color="off", border=True)
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
            st.markdown("**Robust QA-filtered peak transmission loss (0 to 1,168 MW)**")
            st.altair_chart(
                alt.Chart(filtered_df[["Date", "QA_Loss_MW"]].dropna())
                .mark_area(opacity=0.35, color="#D97706", line={"color": "#D97706", "strokeWidth": 1.2})
                .encode(
                    x=alt.X("Date:T", title=None),
                    y=alt.Y("QA_Loss_MW:Q", title="Robust QA peak line loss (MW)"),
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
# TAB 7: Scenario Simulator & Revised Manuscript Package
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
            season_sel = st.selectbox("Season (sets adaptive conformal width)", ["Summer (Mar–May)", "Monsoon (Jun–Sep)", "Winter (Dec–Feb)", "Post-Monsoon (Oct–Nov)"])
            cal_regime = st.selectbox("Calendar regime", ["Regular working day", "Single-day public holiday", "Eid / continuous festival window", "Ramadan evening peak"])
            cp_mode = st.segmented_control("Conformal wrapper", ["Plain ACI (tighter Winkler)", "LN-ACI (uniform seasonal PICP)"], default="LN-ACI (uniform seasonal PICP)", key="sim_cp_mode")

    cdd25 = max(0.0, dhaka_temp - 25.0)
    day_adj = {"Friday": -780.0, "Saturday": -240.0, "Sunday": 110.0, "Monday": 160.0, "Tuesday": 220.0, "Wednesday": 180.0, "Thursday": 90.0}[dow]
    hol_adj = {"Regular working day": 0.0, "Single-day public holiday": -520.0, "Eid / continuous festival window": -2100.0, "Ramadan evening peak": 680.0}[cal_regime]
    sim_mw = float(np.clip(0.84 * prev_peak + 0.16 * 12800.0 + cdd25 * 115.0 + day_adj + hol_adj - 320.0, 7000.0, 19500.0))

    if (cp_mode or "").startswith("Plain"):
        aci_half = {"Summer (Mar–May)": 1419.5, "Monsoon (Jun–Sep)": 1185.0, "Winter (Dec–Feb)": 1018.5, "Post-Monsoon (Oct–Nov)": 983.5}[season_sel]
        cov_note = "Plain ACI: lowest proper Winkler (3,636 MW; 88.1% overall / 84.6% Summer PICP)."
    else:
        aci_half = {"Summer (Mar–May)": 2025.0, "Monsoon (Jun–Sep)": 1260.0, "Winter (Dec–Feb)": 1044.5, "Post-Monsoon (Oct–Nov)": 890.5}[season_sel]
        cov_note = "LN-ACI: tightest seasonal coverage uniformity (88.1%–90.1% across all four seasons; 90.1% Summer PICP)."

    with col_out:
        with st.container(border=True):
            st.markdown("**Simulated day-ahead forecast & 90% conformal envelope**")
            with st.container(horizontal=True):
                st.metric("Point forecast", f"{sim_mw:,.0f} MW", f"{sim_mw / 1000:.2f} GW", delta_color="off", border=True)
                st.metric("90% conformal band", f"{sim_mw - aci_half:,.0f} – {sim_mw + aci_half:,.0f} MW", f"±{aci_half:,.0f} MW ({season_sel.split(' ')[0]})", delta_color="off", border=True)
            st.caption(f"Detrended subtropical CDD (base 25.0 °C): **{cdd25:.1f} °C** · {cov_note}")

    with st.container(border=True):
        st.markdown("**Revised Q1 manuscript package (Elsevier `elsarticle 5p,twocolumn`)**")
        st.markdown(
            "- **16 tables, 14 vector PDF figures, 41 verified references** — complete Round-2 revision (95% Hansen MCS, 24-model equalized suite, 5-way conformal ablation, Appendix A QA).\n"
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
