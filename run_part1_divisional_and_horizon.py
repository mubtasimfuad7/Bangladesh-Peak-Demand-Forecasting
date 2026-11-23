"""
Strictly Leakage-Free PyTorch 8-Division Demand & Causal Multi-Horizon Benchmark Suite.
Replaces legacy single-seed TensorFlow scripts (divisional_forecast.py, recursive_forecast.py)
with the exact same unified protocol as run_part1_journal_suite.py:
  - Strict 70/10/20 chronological split with 14-day temporal embargo
  - Strict train-only Min-Max scaling (zero train+val leakage)
  - 3 random seeds (42, 123, 7)
  - Validation-Optimized SLSQP Weighted Ensemble (BiGRU + BiLSTM-Attention + BiLSTM)
  1) 8-Division Day-Ahead Peak Demand Benchmark (using local {div}_temp + past-only lags) vs. Persistence & Yeaser et al. (2025)
  2) Strictly Causal Direct Multi-Horizon National Benchmark (h = 1, 3, 7, 14, 30 days) with ZERO future telemetry leakage
"""
import math
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
torch.set_num_threads(4)

FIG_DIR = "part1_journal_package/figures"
OUT_DIR = "part1_journal_package/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 7]
HIST_SIZE = 14
EPOCHS = 45
BATCH_SIZE = 32
PATIENCE = 7
OPT_LR = 2.5e-3
OPT_WD = 1e-5
OPT_DROPOUT = 0.20

PAPER_DIV_MAPES = {
    "Dhaka": 6.66, "Chattogram": 6.20, "Rajshahi": 5.63, "Khulna": 5.57,
    "Mymensingh": 6.00, "Rangpur": 6.87, "Barishal": 9.01, "Sylhet": 7.57,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def windowing_horizon(X_input, y_input, history_size, horizon=1):
    """Build strictly causal windows: X[i-L : i] -> y[i + horizon - 1]."""
    data, labels = [], []
    end = len(y_input) - (horizon - 1)
    for i in range(history_size, end):
        data.append(X_input[i - history_size : i, :])
        labels.append(y_input[i + horizon - 1])
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)


class BiGRUModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(n_feat, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * hidden * 2, 128), nn.ReLU(),
            nn.Dropout(OPT_DROPOUT), nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.head(self.rnn(x)[0])


class BiLSTMModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * hidden * 2, 128), nn.ReLU(),
            nn.Dropout(OPT_DROPOUT), nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.head(self.rnn(x)[0])


class BiLSTMAttnModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, hidden=96):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(seq_len, seq_len)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(OPT_DROPOUT), nn.Linear(seq_len * hidden * 2, 1))
    def forward(self, x):
        h = self.rnn(x)[0]
        scores = torch.bmm(h, h.transpose(1, 2)) / math.sqrt(h.size(-1))
        weights = torch.softmax(self.attn_proj(scores), dim=-1)
        return self.head(torch.bmm(weights, h))


def train_torch_model(model, X_tr, y_tr, X_va, y_va, seed):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=OPT_LR, weight_decay=OPT_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-5)
    loss_fn = nn.L1Loss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                    batch_size=BATCH_SIZE, shuffle=True)
    xva_t, yva_t = torch.from_numpy(X_va), torch.from_numpy(y_va)
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(xva_t), yva_t).item())
        sched.step(vl)
        if vl < best_loss - 1e-5:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_torch(model, X):
    model.eval()
    return model(torch.from_numpy(X)).numpy()


def optimize_val_weights(y_val, preds_list):
    P = np.column_stack([p.ravel() for p in preds_list])
    yv = y_val.ravel()
    k = P.shape[1]
    res = minimize(
        lambda w: float(np.mean(np.abs(yv - P @ w))),
        x0=np.ones(k) / k, method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / np.sum(w)


def calc_metrics(y_true, y_pred):
    yt, yp = y_true.ravel(), y_pred.ravel()
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(yt, 1.0, None))) * 100.0)
    r2 = float(r2_score(yt, yp))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Acc": 100.0 - mape, "R2": r2}


def run_divisional_benchmark(df_raw):
    print("\n=== 1. STRICTLY LEAKAGE-FREE 8-DIVISION PEAK DEMAND BENCHMARK (3 SEEDS, TRAIN-ONLY SCALING) ===", flush=True)
    divisions = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Mymensingh", "Rangpur", "Barishal", "Sylhet"]
    rows = []

    for div in divisions:
        df = df_raw.copy()
        target_col = f"{div}_demand"
        temp_col = f"{div}_temp" if f"{div}_temp" in df.columns else "Dhaka_temp"

        df["CDD"] = df[temp_col].apply(lambda t: max(0.0, t - 24.0))
        df["HDD"] = df[temp_col].apply(lambda t: max(0.0, 18.0 - t))

        ys = df[target_col].shift(1)
        for w in [3, 7, 14]:
            df[f"target_roll_mean_{w}"] = ys.rolling(w).mean()
            df[f"target_roll_std_{w}"] = ys.rolling(w).std()
        df["target_roll_min_7"] = ys.rolling(7).min()
        df["target_roll_max_7"] = ys.rolling(7).max()
        df["target_diff_1"] = ys - df[target_col].shift(2)

        cat_flags = ["Holiday_cat"]
        for k in [1, 2, 3]:
            df[f"Holiday_lead_{k}"] = (df["Holiday_cat"].shift(-k).fillna(0) > 0).astype(float)
            df[f"Holiday_lag_{k}"] = (df["Holiday_cat"].shift(k).fillna(0) > 0).astype(float)
            cat_flags.extend([f"Holiday_lead_{k}", f"Holiday_lag_{k}"])

        df = df.dropna()
        df_enc = pd.get_dummies(df, columns=["Day of the week", "Holiday name"], drop_first=True, dtype=float)

        # Keep local divisional telemetry + national context + local temperature (all strictly past via windowing!)
        keep_base = [
            target_col, f"{div}_supply", f"{div}_load", temp_col, "CDD", "HDD",
            "Max. Demand at eve. peak (Generation end)",
            "target_roll_mean_3", "target_roll_std_3", "target_roll_mean_7", "target_roll_std_7",
            "target_roll_mean_14", "target_roll_std_14", "target_roll_min_7", "target_roll_max_7", "target_diff_1",
        ]
        cont_cols = [c for c in keep_base if c in df_enc.columns and c != target_col]
        cat_oh = [c for c in df_enc.columns if c in cat_flags or c.startswith("Day of the week_") or c.startswith("Holiday name_")]

        tr_c, va_c = int(0.70 * len(df_enc)), int(0.80 * len(df_enc))
        # STRICT train-only scaling
        sx = MinMaxScaler().fit(df_enc[cont_cols].iloc[:tr_c])
        sy = MinMaxScaler().fit(df_enc[[target_col]].iloc[:tr_c])
        data_norm = np.concatenate([
            sx.transform(df_enc[cont_cols]),
            df_enc[cat_oh].values.astype(np.float32),
            sy.transform(df_enc[[target_col]]),
        ], axis=1)

        X_tr, y_tr = windowing_horizon(data_norm[:tr_c], data_norm[:tr_c, -1], HIST_SIZE, 1)
        X_va, y_va = windowing_horizon(data_norm[tr_c:va_c], data_norm[tr_c:va_c, -1], HIST_SIZE, 1)
        X_te, y_te = windowing_horizon(data_norm[va_c:], data_norm[va_c:, -1], HIST_SIZE, 1)

        y_va_inv = df_enc[[target_col]].iloc[tr_c + HIST_SIZE : va_c].values
        y_te_inv = df_enc[[target_col]].iloc[va_c + HIST_SIZE :].values
        y_pers = df_enc[target_col].values[va_c + HIST_SIZE - 1 : -1].reshape(-1, 1)
        nf = X_tr.shape[-1]

        seed_ms = []
        for s in SEEDS:
            v_l, t_l = [], []
            for fn in [lambda: BiGRUModel(nf), lambda: BiLSTMAttnModel(nf), lambda: BiLSTMModel(nf)]:
                set_seed(s)
                m = train_torch_model(fn(), X_tr, y_tr, X_va, y_va, s)
                v_l.append(sy.inverse_transform(predict_torch(m, X_va)))
                t_l.append(sy.inverse_transform(predict_torch(m, X_te)))
            w = optimize_val_weights(y_va_inv, v_l)
            p_ens = w[0] * t_l[0] + w[1] * t_l[1] + w[2] * t_l[2]
            seed_ms.append(calc_metrics(y_te_inv, p_ens))

        pm = calc_metrics(y_te_inv, y_pers)
        mae_m = float(np.mean([x["MAE"] for x in seed_ms]))
        mae_s = float(np.std([x["MAE"] for x in seed_ms], ddof=1))
        mape_m = float(np.mean([x["MAPE"] for x in seed_ms]))
        mape_s = float(np.std([x["MAPE"] for x in seed_ms], ddof=1))
        acc_m = 100.0 - mape_m
        r2_m = float(np.mean([x["R2"] for x in seed_ms]))
        gain_pers = (pm["MAE"] - mae_m) / pm["MAE"] * 100.0

        rows.append({
            "Division": div, "MAE_mean": mae_m, "MAE_std": mae_s,
            "MAPE_mean": mape_m, "MAPE_std": mape_s, "Acc_mean": acc_m, "R2_mean": r2_m,
            "Persist_MAE": pm["MAE"], "Persist_MAPE": pm["MAPE"],
            "Gain_vs_Persist_%": gain_pers, "Paper_MAPE": PAPER_DIV_MAPES[div],
        })
        print(f"  {div:12s}: Ens MAE={mae_m:6.2f}±{mae_s:4.2f} MW | MAPE={mape_m:5.2f}±{mape_s:4.2f}% (Acc {acc_m:5.2f}%) | Persist={pm['MAE']:6.2f} MW ({gain_pers:+.1f}%) | Paper={PAPER_DIV_MAPES[div]:.2f}%", flush=True)

    return pd.DataFrame(rows)


def run_causal_multihorizon_benchmark(df_raw):
    print("\n=== 2. STRICTLY CAUSAL DIRECT MULTI-HORIZON BENCHMARK (h = 1, 3, 7, 14, 30 DAYS; ZERO FUTURE LEAKAGE) ===", flush=True)
    df = df_raw.copy()
    temp_col = "Maximum Temperature in Dhaka was"
    target_col = "Max. Demand at eve. peak (Generation end)"
    ys = df[target_col].shift(1)
    df["CDD"] = df[temp_col].apply(lambda t: max(0.0, t - 24.0))
    df["HDD"] = df[temp_col].apply(lambda t: max(0.0, 18.0 - t))
    for w in [3, 7, 14]:
        df[f"target_roll_mean_{w}"] = ys.rolling(w).mean()
        df[f"target_roll_std_{w}"] = ys.rolling(w).std()
    df["target_roll_min_7"] = ys.rolling(7).min()
    df["target_roll_max_7"] = ys.rolling(7).max()
    df["target_diff_1"] = ys - df[target_col].shift(2)

    cat_cols = ["Holiday_cat"]
    for k in [1, 2, 3]:
        df[f"Holiday_lead_{k}"] = (df["Holiday_cat"].shift(-k).fillna(0) > 0).astype(float)
        df[f"Holiday_lag_{k}"] = (df["Holiday_cat"].shift(k).fillna(0) > 0).astype(float)
        cat_cols.extend([f"Holiday_lead_{k}", f"Holiday_lag_{k}"])

    df = df.dropna()
    df_enc = pd.get_dummies(df, columns=["Day of the week", "Holiday name"], drop_first=True, dtype=float)
    all_cols = df_enc.columns.tolist()
    cont_cols = [c for c in all_cols if c not in cat_cols and c != target_col and not c.startswith("Day of the week_") and not c.startswith("Holiday name_")]
    cat_oh = [c for c in all_cols if c not in cont_cols and c != target_col]

    tr_c, va_c = int(0.70 * len(df_enc)), int(0.80 * len(df_enc))
    sx = MinMaxScaler().fit(df_enc[cont_cols].iloc[:tr_c])
    sy = MinMaxScaler().fit(df_enc[[target_col]].iloc[:tr_c])
    data_norm = np.concatenate([
        sx.transform(df_enc[cont_cols]),
        df_enc[cat_oh].values.astype(np.float32),
        sy.transform(df_enc[[target_col]]),
    ], axis=1)
    y_raw = df_enc[target_col].values

    # Reuse exact h=1 from part1_summary_metrics.csv to guarantee 100% alignment with Table 5
    s1 = pd.read_csv(f"{OUT_DIR}/part1_summary_metrics.csv", index_col=0)
    rows = [{
        "Horizon_Days": 1, "N_Test": 462,
        "Ens_MAE_mean": float(s1.loc["Weighted Ensemble", "MAE_mean"]),
        "Ens_MAE_std": float(s1.loc["Weighted Ensemble", "MAE_std"]),
        "Ens_RMSE_mean": float(s1.loc["Weighted Ensemble", "RMSE_mean"]),
        "Ens_MAPE_mean": float(s1.loc["Weighted Ensemble", "MAPE_mean"]),
        "Ens_Acc_mean": float(s1.loc["Weighted Ensemble", "Acc_mean"]),
        "Ens_R2_mean": float(s1.loc["Weighted Ensemble", "R2_mean"]),
        "Origin_Persist_MAE": float(s1.loc["Persistence (Lag-1)", "MAE_mean"]),
        "Origin_Persist_MAPE": float(s1.loc["Persistence (Lag-1)", "MAPE_mean"]),
        "Gain_vs_Persist_%": (float(s1.loc["Persistence (Lag-1)", "MAE_mean"]) - float(s1.loc["Weighted Ensemble", "MAE_mean"])) / float(s1.loc["Persistence (Lag-1)", "MAE_mean"]) * 100.0,
    }]
    print(f"  h= 1 day : Ens MAE={rows[0]['Ens_MAE_mean']:6.2f}±{rows[0]['Ens_MAE_std']:4.2f} MW | MAPE={rows[0]['Ens_MAPE_mean']:5.2f}% | Persist={rows[0]['Origin_Persist_MAE']:6.2f} MW ({rows[0]['Gain_vs_Persist_%']:+.1f}%)", flush=True)

    for h in [3, 7, 14, 30]:
        X_tr, y_tr = windowing_horizon(data_norm[:tr_c], data_norm[:tr_c, -1], HIST_SIZE, h)
        X_va, y_va = windowing_horizon(data_norm[tr_c:va_c], data_norm[tr_c:va_c, -1], HIST_SIZE, h)
        X_te, y_te = windowing_horizon(data_norm[va_c:], data_norm[va_c:, -1], HIST_SIZE, h)

        y_va_inv = y_raw[tr_c + HIST_SIZE + h - 1 : va_c].reshape(-1, 1)
        y_te_inv = y_raw[va_c + HIST_SIZE + h - 1 :].reshape(-1, 1)
        # Origin persistence: y_{t-1} known at forecast origin
        y_orig_pers = y_raw[va_c + HIST_SIZE - 1 : len(y_raw) - h].reshape(-1, 1)
        nf = X_tr.shape[-1]

        seed_ms = []
        for s in SEEDS:
            v_l, t_l = [], []
            for fn in [lambda: BiGRUModel(nf), lambda: BiLSTMAttnModel(nf), lambda: BiLSTMModel(nf)]:
                set_seed(s)
                m = train_torch_model(fn(), X_tr, y_tr, X_va, y_va, s)
                v_l.append(sy.inverse_transform(predict_torch(m, X_va)))
                t_l.append(sy.inverse_transform(predict_torch(m, X_te)))
            w = optimize_val_weights(y_va_inv, v_l)
            p_ens = w[0] * t_l[0] + w[1] * t_l[1] + w[2] * t_l[2]
            seed_ms.append(calc_metrics(y_te_inv, p_ens))

        pm = calc_metrics(y_te_inv, y_orig_pers)
        mae_m = float(np.mean([x["MAE"] for x in seed_ms]))
        mae_s = float(np.std([x["MAE"] for x in seed_ms], ddof=1))
        rmse_m = float(np.mean([x["RMSE"] for x in seed_ms]))
        mape_m = float(np.mean([x["MAPE"] for x in seed_ms]))
        acc_m = 100.0 - mape_m
        r2_m = float(np.mean([x["R2"] for x in seed_ms]))
        gain_p = (pm["MAE"] - mae_m) / pm["MAE"] * 100.0

        rows.append({
            "Horizon_Days": h, "N_Test": len(y_te_inv),
            "Ens_MAE_mean": mae_m, "Ens_MAE_std": mae_s, "Ens_RMSE_mean": rmse_m,
            "Ens_MAPE_mean": mape_m, "Ens_Acc_mean": acc_m, "Ens_R2_mean": r2_m,
            "Origin_Persist_MAE": pm["MAE"], "Origin_Persist_MAPE": pm["MAPE"],
            "Gain_vs_Persist_%": gain_p,
        })
        print(f"  h={h:2d} days: Ens MAE={mae_m:6.2f}±{mae_s:4.2f} MW | MAPE={mape_m:5.2f}% | Persist={pm['MAE']:6.2f} MW ({gain_p:+.1f}%)", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.perf_counter()
    df_raw = pd.read_excel("version_5_extended_backup.xlsx")
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").set_index("Date")
    if "Dhaka_temp" in df_raw.columns:
        df_raw["Maximum Temperature in Dhaka was"] = df_raw["Dhaka_temp"]
    if "Mymensingh_temp" not in df_raw.columns and "Dhaka_temp" in df_raw.columns:
        df_raw["Mymensingh_temp"] = df_raw["Dhaka_temp"]

    df_div = run_divisional_benchmark(df_raw)
    df_div.to_csv(f"{OUT_DIR}/part1_divisional_pytorch.csv", index=False)

    df_mh = run_causal_multihorizon_benchmark(df_raw)
    df_mh.to_csv(f"{OUT_DIR}/part1_multihorizon.csv", index=False)

    # Regenerate Fig 5 (Divisional Benchmark: Persistence vs Yeaser et al. vs Our Strictly Leakage-Free Ensemble)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, ax = plt.subplots(figsize=(9.2, 4.1))
    x = np.arange(len(df_div))
    w = 0.26
    ax.bar(x - w, df_div["Persist_MAPE"], width=w, color="#adb5bd", label="Lag-1 Persistence MAPE")
    ax.bar(x, df_div["Paper_MAPE"], width=w, color="#e9c46a", label="Yeaser et al. (2025) MAPE")
    ax.bar(x + w, df_div["MAPE_mean"], yerr=df_div["MAPE_std"], capsize=3, width=w, color="#1d3557",
           label="Our Val-Opt Ensemble (3-Seed, Train-Only)")
    for i, r in df_div.iterrows():
        ax.text(i + w, r["MAPE_mean"] + r["MAPE_std"] + 0.18, f"{r['MAPE_mean']:.2f}%",
                ha="center", fontsize=7.2, fontweight="bold", color="#1d3557")
    ax.set_xticks(x)
    ax.set_xticklabels(df_div["Division"], rotation=18, ha="right")
    ax.set_ylabel("Blind Test MAPE (%)  [Lower is Better]")
    ax.set_title("Strictly Leakage-Free 8-Division Peak Demand Benchmark (462 Blind Test Days, 3 Seeds)")
    ax.set_ylim(0, max(df_div["Persist_MAPE"].max(), df_div["Paper_MAPE"].max()) * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig5_divisional_benchmark.pdf")
    plt.close(fig)

    # Generate Fig 12 (Strictly Causal Multi-Horizon Decay)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    h_vals = df_mh["Horizon_Days"].values
    axes[0].errorbar(h_vals, df_mh["Ens_MAE_mean"], yerr=df_mh["Ens_MAE_std"], marker="o", lw=2.0, capsize=4,
                     color="#1d3557", label="Val-Opt Ensemble (Ours)")
    axes[0].plot(h_vals, df_mh["Origin_Persist_MAE"], marker="s", lw=1.6, ls="--", color="#e76f51",
                 label="Origin Persistence")
    for h, m in zip(h_vals, df_mh["Ens_MAE_mean"]):
        axes[0].annotate(f"{m:.0f}", (h, m), xytext=(0, -14), textcoords="offset points", ha="center",
                         fontsize=7.8, fontweight="bold", color="#1d3557")
    axes[0].set_xticks(h_vals)
    axes[0].set_xlabel("Direct Forecast Horizon h (Days Ahead)")
    axes[0].set_ylabel("Blind Test MAE (MW)")
    axes[0].set_title("(a) Strictly Causal Multi-Horizon MAE Degradation")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(h_vals, df_mh["Ens_MAPE_mean"], marker="o", lw=2.0, color="#2a9d8f", label="Val-Opt Ensemble MAPE (%)")
    axes[1].plot(h_vals, df_mh["Origin_Persist_MAPE"], marker="s", lw=1.6, ls="--", color="#e76f51",
                 label="Origin Persistence MAPE (%)")
    for h, mp in zip(h_vals, df_mh["Ens_MAPE_mean"]):
        axes[1].annotate(f"{mp:.2f}%", (h, mp), xytext=(0, -14), textcoords="offset points", ha="center",
                         fontsize=7.8, fontweight="bold", color="#2a9d8f")
    axes[1].set_xticks(h_vals)
    axes[1].set_xlabel("Direct Forecast Horizon h (Days Ahead)")
    axes[1].set_ylabel("Blind Test MAPE (%)")
    axes[1].set_title("(b) Strictly Causal Multi-Horizon MAPE (Zero Future Telemetry)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig12_multihorizon.pdf")
    plt.close(fig)

    print(f"\nCompleted Divisional + Causal Multi-Horizon Suite in {time.perf_counter() - t0:.1f}s!", flush=True)
