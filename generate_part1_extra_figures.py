"""
Clean, data-driven generator for the 2 dedicated Part 1 supplementary figures:
  1. figures/fig_architecture.pdf       — System architecture dynamically populated from verified CSVs/JSON (zero hardcoded drift!)
  2. figures/fig10_grid_reliability.pdf — QA-filtered Peak Transmission Loss (excluding 16 negative BPDB reporting anomalies)
                                          & Reported Divisional Net Supply-Demand Balance (load-shedding allocation)
Note: fig9_shap_attribution.pdf is now generated directly by run_part1_journal_suite.py
      as Direct Group Permutation Importance on the actual deployed PyTorch Ensemble.
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
    div = pd.read_csv(f"{OUT_DIR}/part1_divisional_pytorch.csv")
    mh = pd.read_csv(f"{OUT_DIR}/part1_multihorizon.csv")
    cp = pd.read_csv(f"{OUT_DIR}/part1_conformal.csv")

    w_mean = meta["learned_weights_mean"]
    eff = meta["efficiency"]
    n_f = meta["n_features"]
    tau = meta["tau_star_C"]
    ens_mae = s5.loc["Val-Opt RNN Ensemble", "MAE_mean"]
    ens_sd = s5.loc["Val-Opt RNN Ensemble", "MAE_std"]
    mh_min_gain = mh["Gain_vs_Persist_%"].min()
    mh_max_gain = mh["Gain_vs_Persist_%"].max()
    div_acc_min, div_acc_max = div["Acc_mean"].min(), div["Acc_mean"].max()
    div_wins_p = int((div["Gain_vs_Persist_%"] > 0).sum())
    aci90 = cp[(cp["Method"] == "Locally Normalized ACI") & (cp["Nominal_%"] == 90)].iloc[0]
    qa_loss = meta["data_qa"]["qa_mean_loss_mw"]

    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0.7, 7.95)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#f8fafc", ec="#1d3557", bold=False, fs=7.9):
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
        f"CAUSAL MULTIVARIATE INPUT WINDOW  X ∈ R^(14 × {n_f})\n"
        f"[Past-Only Peak Demand & Lags (3/7/14d) | Subtropical CDD (τ*={tau:.0f}°C) | Holiday ±3d | 8-Div Telemetry | Train-Only MinMax]",
        fc="#e8f0fe", ec="#1d3557", bold=True, fs=7.9)

    box(0.4, 4.75, 2.8, 1.32,
        f"Branch 1: BiGRU\n(Mean Val-Cal w₁ = {w_mean[0]:.3f})\nBidirectional GRU (64×2)\nFlatten → Dense(128, ReLU)\nDropout(0.20) → Dense(1)\n[{eff['BiGRU']['Params']:,} params]",
        fc="#fef3c7", ec="#b45309", fs=7.7)

    box(3.6, 4.75, 2.8, 1.32,
        f"Branch 2: BiLSTM-Attention\n(Mean Val-Cal w₂ = {w_mean[1]:.3f})\nBidirectional LSTM (96×2)\nScaled Dot-Product Self-Attn\nContext → Dropout → Dense(1)\n[{eff['BiLSTM-Attention']['Params']:,} params]",
        fc="#d1fae5", ec="#047857", fs=7.7)

    box(6.8, 4.75, 2.8, 1.32,
        f"Branch 3: BiLSTM\n(Mean Val-Cal w₃ = {w_mean[2]:.3f})\nBidirectional LSTM (64×2)\nFlatten → Dense(128, ReLU)\nDropout(0.20) → Dense(1)\n[{eff['BiLSTM']['Params']:,} params]",
        fc="#fce7f3", ec="#9d174d", fs=7.7)

    arrow(2.2, 6.75, 1.8, 6.15)
    arrow(5.0, 6.75, 5.0, 6.15)
    arrow(7.8, 6.75, 8.2, 6.15)

    tot_p = eff["Val-Opt RNN Ensemble"]["Params"]
    tot_mb = eff["Val-Opt RNN Ensemble"]["Size_KB"] / 1024.0
    tot_ms = eff["Val-Opt RNN Ensemble"]["CPU_ms"]
    box(1.5, 3.0, 7.0, 1.02,
        "Val-Cal SLSQP Fusion (with Equal-Weight Benchmark) & Inverse Scaling\n"
        f"ŷ_ens = w₁·ŷ_BiGRU + w₂·ŷ_BiLSTM-Attn + w₃·ŷ_BiLSTM  (Val-ES early stop; Val-Cal fusion)\n"
        f"[Total: {tot_p:,} params | {tot_mb:.2f} MB (fp32) | {tot_ms:.2f} ms CPU | Benchmarked vs PatchTST-3× & Ridge]",
        fc="#dbeafe", ec="#1d3557", bold=True, fs=7.5)

    arrow(1.8, 4.75, 3.6, 4.10)
    arrow(5.0, 4.75, 5.0, 4.10)
    arrow(8.2, 4.75, 6.4, 4.10)

    box(0.45, 1.0, 2.8, 1.2,
        f"National Peak & Rolling CV\n5-Seed MAE: {ens_mae:.2f}±{ens_sd:.2f} MW\n3-Fold CV + Fixed-Test Curve\nCausal 1–30d (+{mh_min_gain:.1f}–{mh_max_gain:.1f}% vs P)",
        fc="#f3f4f6", ec="#374151", fs=7.4)

    box(3.6, 1.0, 2.8, 1.2,
        f"Same-Split 8-Division Suite\n3-Seed Val-Opt RNN Ensemble\nDivisional Acc: {div_acc_min:.1f}–{div_acc_max:.1f}%\nBeats Persist in {div_wins_p}/8 Divs",
        fc="#f3f4f6", ec="#374151", fs=7.4)

    box(6.75, 1.0, 2.8, 1.2,
        f"LN-ACI, XAI & QA Analytics\n90% LN-ACI PICP: {aci90['PICP_%']:.1f}%\nDirect Ensemble Perm. Imp.\nQA Peak Line Loss: {qa_loss:.1f} MW",
        fc="#f3f4f6", ec="#374151", fs=7.4)

    arrow(3.5, 3.0, 1.85, 2.28)
    arrow(5.0, 3.0, 5.0, 2.28)
    arrow(6.5, 3.0, 8.15, 2.28)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_architecture.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved data-driven fig_architecture.pdf")


def build_grid_reliability_figure(df_raw):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))

    ax = axes[0]
    raw_loss = (df_raw["Max. Demand at eve. peak (Generation end)"] - df_raw["Max. Demand at eve. peak (Sub-station end)"]).dropna()
    n_neg = int((raw_loss < 0).sum())
    qa_loss = raw_loss[raw_loss >= 0]
    loss_roll = qa_loss.rolling(30).mean()
    ax.plot(qa_loss.index, qa_loss.values, color="#9ca3af", alpha=0.35, lw=0.6, label="QA Daily Peak Loss (≥0)")
    ax.plot(loss_roll.index, loss_roll.values, color="#9b2226", lw=1.6, label="30-Day Moving Average")
    ax.axhline(qa_loss.mean(), color="#1d3557", ls="--", lw=1.2, label=f"QA Mean: {qa_loss.mean():.2f} MW")
    ax.set_ylabel("Peak Transmission Loss (MW)")
    ax.set_ylim(0, qa_loss.max() * 1.12)
    ax.set_title(f"(a) QA-Filtered Peak System Loss ({n_neg} Negative Records Excluded)", pad=24)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False, fontsize=7.4)

    ax = axes[1]
    divs = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Mymensingh", "Rangpur", "Barishal", "Sylhet"]
    deficits = [(df_raw[f"{d}_supply"] - df_raw[f"{d}_demand"]).mean() for d in divs]
    s_def = pd.Series(deficits, index=divs).sort_values()
    c_def = ["#d62828" if v < 0 else "#2a9d8f" for v in s_def.values]
    ax.barh(s_def.index, s_def.values, color=c_def)
    ax.axvline(0, color="black", lw=0.9)
    for i, v in enumerate(s_def.values):
        ax.text(v - 1.2 if v < 0 else v + 1.2, i, f"{v:.1f}", va="center",
                ha="right" if v < 0 else "left", fontsize=7.8)
    ax.set_xlim(min(-46, s_def.min() - 10), max(12, s_def.max() + 8))
    ax.set_xlabel("Mean Reported Balance: Supply − Demand (MW)")
    ax.set_title("(b) Reported Divisional Net Balance (Load-Shedding Allocation)")
    ax.grid(axis="x", alpha=0.25)

    print(f"QA Loss: mean={qa_loss.mean():.2f} MW (raw={raw_loss.mean():.2f} MW, n_neg={n_neg})")
    print("Divisional mean net balances (Supply - Demand):")
    for k, v in s_def.items():
        print(f"  {k:12s}: {v:+.2f} MW")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig10_grid_reliability.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved QA-filtered fig10_grid_reliability.pdf")


if __name__ == "__main__":
    df_raw = pd.read_excel("version_5_extended_backup.xlsx")
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").set_index("Date")
    if "Dhaka_temp" in df_raw.columns:
        df_raw["Maximum Temperature in Dhaka was"] = df_raw["Dhaka_temp"]

    build_architecture_figure()
    build_grid_reliability_figure(df_raw)
    print("\nAll revised supplementary figures generated cleanly!")
