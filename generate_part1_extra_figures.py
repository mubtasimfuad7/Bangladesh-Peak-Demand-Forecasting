"""
Round-2 Data-Driven Supplementary Figure Generator:
  1. figures/fig_architecture.pdf       — Dual-Model Leading Group Pipeline (PatchTST-3x + Ridge + 5-Way Conformal/LN-ACI)
                                          dynamically populated with exact verified CSV/JSON numbers & consistent MiB units.
  2. figures/fig10_grid_reliability.pdf — Appendix A: Robust QA-Filtered Peak Transmission Loss (excluding 16 negatives
                                          & 3 upper IQR outliers > 1,168 MW; reporting robust mean & median) +
                                          Reported Divisional Supply Minus Demand (MW).
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

FIG_DIR = "part1_journal_package/figures"
OUT_DIR = "part1_journal_package/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})


def build_architecture_figure():
    meta = json.load(open(f"{OUT_DIR}/part1_meta.json"))
    s5 = pd.read_csv(f"{OUT_DIR}/part1_summary_metrics.csv", index_col=0)
    cp = pd.read_csv(f"{OUT_DIR}/part1_conformal.csv")
    rc = pd.read_csv(f"{OUT_DIR}/part1_rolling_cv.csv")
    mh = pd.read_csv(f"{OUT_DIR}/part1_multihorizon.csv")
    reg = pd.read_csv(f"{OUT_DIR}/part1_regime_analysis.csv")
    hol_xgbd = reg[reg["Regime"] == "Holiday / Eid"]["XGBD_MAE"].iloc[0]

    n_f = meta["n_features"]
    tau = meta["tau_star_C"]
    eff = meta["efficiency"]
    pt3_mae, pt3_sd = s5.loc["PatchTST-Ensemble (3x)", "MAE_mean"], s5.loc["PatchTST-Ensemble (3x)", "MAE_std"]
    pt3_ss = s5.loc["PatchTST-Ensemble (3x)", "Skill_Wkday_mean"]
    ridge_mae = s5.loc["Ridge (14xF)", "MAE_mean"]
    ridge_cv = rc["Ridge_MAE"].mean()
    eq_mae, eq_sd = s5.loc["Equal-Weight RNN Ens", "MAE_mean"], s5.loc["Equal-Weight RNN Ens", "MAE_std"]
    aci90 = cp[(cp["Model"] == "PatchTST-Ensemble (3x)") & (cp["Method"].str.contains("LN-ACI")) & (cp["Nominal_%"] == 90)].iloc[0]
    qa = meta["data_qa"]

    pt3_mib = eff["PatchTST-Ensemble (3x)"]["Size_KiB"] / 1024.0
    ridge_kib = eff["Ridge (14xF)"]["Size_KiB"]
    rnn_mib = eff["Equal-Weight RNN Ens"]["Size_KiB"] / 1024.0

    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.7, 7.95)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f8fafc", ec="#1d3557", bold=False, fs=7.8):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.3
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color="#111827")

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.3))

    box(0.4, 6.75, 9.2, 0.92,
        f"CAUSAL MULTIVARIATE INPUT WINDOW  X ∈ R^(14 × {n_f})  [Raw Year Dropped]\n"
        f"[Past-Only Peak Demand & Lags (3/7/14d) | Detrended CDD (τ*={tau:.0f}°C) | Holiday ±3d | 8-Div Telemetry | Train-Only MinMax]",
        fc="#e8f0fe", ec="#1d3557", bold=True, fs=7.8)

    box(0.4, 4.72, 2.85, 1.35,
        f"PatchTST-Ensemble (3×)\n3× [Patch(P=4,S=2) → 2×Trans.]\n{eff['PatchTST-Ensemble (3x)']['Params']:,} params | {pt3_mib:.2f} MiB\n{eff['PatchTST-Ensemble (3x)']['CPU_ms']:.2f} ms CPU | Test: {pt3_mae:.1f}±{pt3_sd:.1f} MW\n(Best 2025–26 MAE; +{pt3_ss:.1f}% vs Wkday)",
        fc="#d1fae5", ec="#047857", fs=7.4)

    box(3.58, 4.72, 2.85, 1.35,
        f"Ridge Regression (14×{n_f})\nL2 Linear (α={meta['ridge_alpha']:g}, Val-ES tuned)\n{eff['Ridge (14xF)']['Params']:,} params | {ridge_kib:.2f} KiB (fp32)\n{eff['Ridge (14xF)']['CPU_ms']:.2f} ms CPU | Test: {ridge_mae:.1f} MW\n(Best 3-Fold CV Mean: {ridge_cv:.1f} MW)",
        fc="#fef3c7", ec="#b45309", fs=7.4)

    box(6.75, 4.72, 2.85, 1.35,
        f"Equal-Weight RNN Ens (3×)\n1/3 (BiGRU + BiLSTM-Attn + BiLSTM)\n{eff['Equal-Weight RNN Ens']['Params']:,} params | {rnn_mib:.2f} MiB\n{eff['Equal-Weight RNN Ens']['CPU_ms']:.2f} ms CPU | Test: {eq_mae:.1f}±{eq_sd:.1f} MW\n(Tied in 95% MCS; beats SLSQP)",
        fc="#e0e7ff", ec="#1d3557", fs=7.4)

    arrow(2.2, 6.75, 1.82, 6.15)
    arrow(5.0, 6.75, 5.0, 6.15)
    arrow(7.8, 6.75, 8.17, 6.15)

    box(1.3, 2.98, 7.4, 1.04,
        "Separated Validation Protocol & 5-Way Adaptive Conformal Uncertainty Wrapper\n"
        "Val-ES (105d): Early Stopping & α/LR Tuning  |  14d Embargo  |  Val-Cal (105d): Untouched Conformal Calibration\n"
        f"LN-ACI (14d local scale + online α_t ∈ [0.02, 0.50]): 90% PICP = {aci90['PICP_%']:.1f}% | Winkler = {aci90['Winkler_MW']:.0f} MW",
        fc="#dbeafe", ec="#1d3557", bold=True, fs=7.4)

    arrow(1.82, 4.72, 3.6, 4.10)
    arrow(5.0, 4.72, 5.0, 4.10)
    arrow(8.17, 4.72, 6.4, 4.10)

    box(0.45, 1.0, 2.8, 1.2,
        f"24-Model MCS & Rolling CV\n95% Hansen MCS (MAE & RMSE)\n3-Fold CV + Sensitivity Curve\nXGB-Diff best on Eid ({hol_xgbd:.1f} MW)",
        fc="#f3f4f6", ec="#374151", fs=7.3)

    box(3.6, 1.0, 2.8, 1.2,
        f"Divisional & 1–30d Horizons\nSame-Split 8-Div Benchmark\nClim.+Ridge-Resid Hybrid\n(h=30: {mh.loc[mh['Horizon_Days']==30, 'Clim_Resid_Hyb_MAE'].iloc[0]:.0f} MW vs Clim {mh.loc[mh['Horizon_Days']==30, 'Clim_Trend_MAE'].iloc[0]:.0f})",
        fc="#f3f4f6", ec="#374151", fs=7.3)

    box(6.75, 1.0, 2.8, 1.2,
        f"Direct XAI & Appendix A QA\nPatchTST-3× Merged Perm. Imp.\n20 BPDB Date Gaps Audited\nRobust Loss Med: {qa['robust_median_loss_mw']:.0f} MW",
        fc="#f3f4f6", ec="#374151", fs=7.3)

    arrow(3.5, 2.98, 1.85, 2.28)
    arrow(5.0, 2.98, 5.0, 2.28)
    arrow(6.5, 2.98, 8.15, 2.28)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved Round-2 data-driven fig_architecture.pdf")


def build_grid_reliability_figure(df_raw):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))

    ax = axes[0]
    raw_loss = (df_raw["Max. Demand at eve. peak (Generation end)"] - df_raw["Max. Demand at eve. peak (Sub-station end)"]).dropna()
    n_neg = int((raw_loss < 0).sum())
    nonneg = raw_loss[raw_loss >= 0]
    q1, q3 = np.nanpercentile(nonneg, [25, 75])
    hi_cut = q3 + 3.0 * (q3 - q1)
    n_hi = int((raw_loss > hi_cut).sum())
    rob_loss = raw_loss[(raw_loss >= 0) & (raw_loss <= hi_cut)]
    loss_roll = rob_loss.rolling(30).mean()

    ax.plot(rob_loss.index, rob_loss.values, color="#9ca3af", alpha=0.35, lw=0.6, label="QA Daily Loss")
    ax.plot(loss_roll.index, loss_roll.values, color="#9b2226", lw=1.6, label="30-Day Moving Avg")
    ax.axhline(np.median(rob_loss), color="#1d3557", ls="--", lw=1.2, label=f"Median: {np.median(rob_loss):.1f} MW (Mean {rob_loss.mean():.1f})")
    ax.set_ylabel("Peak Transmission Loss (MW)")
    ax.set_ylim(0, hi_cut * 1.08)
    ax.set_title(f"(a) Robust QA Peak System Loss ({n_neg} Neg. & {n_hi} Upper Outliers Excluded)", pad=28)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=7.1)

    ax = axes[1]
    divs = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Mymensingh", "Rangpur", "Barishal", "Sylhet"]
    deficits = [(df_raw[f"{d}_supply"] - df_raw[f"{d}_demand"]).mean() for d in divs]
    s_def = pd.Series(deficits, index=divs).sort_values()
    ax.barh(s_def.index, s_def.values, color="#d62828")
    ax.axvline(0, color="black", lw=0.9)
    for i, v in enumerate(s_def.values):
        ax.text(v - 1.2, i, f"{v:.1f}", va="center", ha="right", fontsize=7.8)
    ax.set_xlim(-47, 8)
    ax.set_xlabel("Mean Reported Balance: Supply − Demand (MW)")
    ax.set_title("(b) Reported Divisional Supply Minus Demand (2019–2026)")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig10_grid_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved robust Appendix A fig10_grid_reliability.pdf (mean={rob_loss.mean():.2f}, med={np.median(rob_loss):.2f})")


if __name__ == "__main__":
    df_raw = pd.read_excel("version_5_extended_backup.xlsx")
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").set_index("Date")
    if "Dhaka_temp" in df_raw.columns:
        df_raw["Maximum Temperature in Dhaka was"] = df_raw["Dhaka_temp"]

    build_architecture_figure()
    build_grid_reliability_figure(df_raw)
    print("All Round-2 supplementary figures generated cleanly!")
