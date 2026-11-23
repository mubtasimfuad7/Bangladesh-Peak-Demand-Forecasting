"""
Strictly Same-Split 8-Division Demand & Causal Direct Multi-Horizon Benchmark Suite (Revised).
Addresses Reviewer Points 8 & 9:
  1) 8-Division Benchmark: DROPS the invalid cross-dataset Yeaser et al. (2025) published-number
     comparison and evaluates strictly on the SAME 462-day blind test split against:
       - Lag-1 Persistence
       - Weekday-Aware Naive (Lag-7)
       - Ridge Regression (on identical divisional features)
       - Single BiGRU (same-split replication of the Yeaser et al. recurrent baseline, 3 seeds)
       - Val-Opt RNN Ensemble (3 seeds, Val-ES / Val-Cal split, subtropical CDD_26)
  2) Strictly Causal Direct Multi-Horizon (h = 1, 3, 7, 14, 30 days): adds
       - Climatological Day-of-Year + Linear Trend baseline (fit strictly on train!)
       - Ridge (14 x F) direct linear baseline
     alongside Origin Persistence and Val-Opt RNN Ensemble.
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
from sklearn.linear_model import Ridge
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
TAU_COOL = 26.0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def windowing_horizon(X_input, y_input, history_size, horizon=1):
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
    def forward(self, x): return self.head(self.rnn(x)[0])


class BiLSTMModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(seq_len * hidden * 2, 128), nn.ReLU(),
            nn.Dropout(OPT_DROPOUT), nn.Linear(128, 1)
        )
    def forward(self, x): return self.head(self.rnn(x)[0])


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


def train_torch_model(model, X_tr, y_tr, X_va_es, y_va_es, seed):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=OPT_LR, weight_decay=OPT_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-5)
    loss_fn = nn.L1Loss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                    batch_size=BATCH_SIZE, shuffle=True)
    xva_t, yva_t = torch.from_numpy(X_va_es), torch.from_numpy(y_va_es)
    best_loss, best_state, wait = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = float(loss_fn(model(xva_t), yva_t).item())
        sched.step(vl)
        if vl < best_loss - 1e-5:
            best_loss, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE: break
    model.load_state_dict(best_state); model.eval()
    return model


@torch.no_grad()
def predict_torch(model, X):
    model.eval()
    return model(torch.from_numpy(X)).numpy()


def optimize_val_weights(y_val, preds_list):
    P = np.column_stack([p.ravel() for p in preds_list])
    yv = y_val.ravel()
    k = P.shape[1]
    res = minimize(lambda w: float(np.mean(np.abs(yv - P @ w))), x0=np.ones(k) / k, method="SLSQP",
                   bounds=[(0.0, 1.0)] * k, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    w = np.clip(res.x, 0.0, 1.0)
    return w / np.sum(w)


def calc_metrics(y_true, y_pred):
    yt, yp = y_true.ravel(), y_pred.ravel()
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(yt, 1.0, None))) * 100.0)
    r2 = float(r2_score(yt, yp))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Acc": 100.0 - mape, "R2": r2}


def fit_ridge_val(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_te_2d, sy):
    best_mae, best_mod = float("inf"), None
    for a in [1.0, 10.0, 50.0, 100.0, 300.0, 1000.0]:
        m = Ridge(alpha=a).fit(X_tr_2d, y_tr.ravel())
        mae = mean_absolute_error(y_va_es.ravel(), m.predict(X_va_es_2d))
        if mae < best_mae:
            best_mae, best_mod = mae, m
    return sy.inverse_transform(best_mod.predict(X_te_2d).reshape(-1, 1))


def fit_climatology_trend_baseline(dates_tr, y_tr_raw, dates_te):
    """
    Strong seasonal + linear-growth baseline for medium/long horizons (h >= 7, 14, 30):
    y(t) = b0 + b1 * t_days + sum_{k=1..3} [a_k sin(2*pi*k*doy/365.25) + c_k cos(2*pi*k*doy/365.25)] + weekday dummies.
    Fitted strictly on training partition!
    """
    t0 = dates_tr[0]
    def make_design(dates):
        t_yr = ((dates - t0).days.values / 365.25).astype(float)
        doy = dates.dayofyear.values.astype(float)
        cols = [np.ones_like(t_yr), t_yr]
        for k in [1, 2, 3]:
            cols.append(np.sin(2.0 * np.pi * k * doy / 365.25))
            cols.append(np.cos(2.0 * np.pi * k * doy / 365.25))
        dow = dates.dayofweek.values
        for d in range(6):
            cols.append((dow == d).astype(float))
        return np.column_stack(cols)

    X_tr = make_design(dates_tr)
    coef, _, _, _ = np.linalg.lstsq(X_tr, y_tr_raw.ravel(), rcond=None)
    return (make_design(dates_te) @ coef).reshape(-1, 1)


def run_divisional_benchmark(df_raw):
    print("\n=== 1. SAME-SPLIT 8-DIVISION BENCHMARK (3 SEEDS: PERSIST, LAG-7, RIDGE, BIGRU, VAL-OPT ENS) ===", flush=True)
    divisions = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Mymensingh", "Rangpur", "Barishal", "Sylhet"]
    rows = []

    for div in divisions:
        df = df_raw.copy()
        target_col = f"{div}_demand"
        temp_col = f"{div}_temp" if f"{div}_temp" in df.columns else "Dhaka_temp"

        df["CDD"] = np.maximum(0.0, df[temp_col].values - TAU_COOL)
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

        keep_base = [
            target_col, f"{div}_supply", f"{div}_load", temp_col, "CDD",
            "Max. Demand at eve. peak (Generation end)",
            "target_roll_mean_3", "target_roll_std_3", "target_roll_mean_7", "target_roll_std_7",
            "target_roll_mean_14", "target_roll_std_14", "target_roll_min_7", "target_roll_max_7", "target_diff_1",
        ]
        cont_cols = [c for c in keep_base if c in df_enc.columns and c != target_col]
        cat_oh = [c for c in df_enc.columns if c in cat_flags or c.startswith("Day of the week_") or c.startswith("Holiday name_")]

        tr_c, va_c = int(0.70 * len(df_enc)), int(0.80 * len(df_enc))
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

        n_va = len(X_va)
        va_es_end = (n_va - HIST_SIZE) // 2
        va_cal_start = va_es_end + HIST_SIZE
        X_va_es, y_va_es = X_va[:va_es_end], y_va[:va_es_end]
        X_va_cal = X_va[va_cal_start:]

        y_va_inv = df_enc[[target_col]].iloc[tr_c + HIST_SIZE : va_c].values
        y_va_cal_inv = y_va_inv[va_cal_start:]
        y_te_inv = df_enc[[target_col]].iloc[va_c + HIST_SIZE :].values
        y_pers = df_enc[target_col].values[va_c + HIST_SIZE - 1 : -1].reshape(-1, 1)
        y_lag7 = df_enc[target_col].values[va_c + HIST_SIZE - 7 : len(df_enc) - 7].reshape(-1, 1)
        nf = X_tr.shape[-1]

        p_ridge = fit_ridge_val(X_tr.reshape(len(X_tr), -1), y_tr, X_va_es.reshape(len(X_va_es), -1), y_va_es, X_te.reshape(len(X_te), -1), sy)
        rm = calc_metrics(y_te_inv, p_ridge)

        ens_ms, bigru_ms = [], []
        for s in SEEDS:
            v_l, t_l = [], []
            for fn in [lambda: BiGRUModel(nf), lambda: BiLSTMAttnModel(nf), lambda: BiLSTMModel(nf)]:
                m = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, s)
                v_l.append(sy.inverse_transform(predict_torch(m, X_va_cal)))
                t_l.append(sy.inverse_transform(predict_torch(m, X_te)))
            bigru_ms.append(calc_metrics(y_te_inv, t_l[0]))
            w = optimize_val_weights(y_va_cal_inv, v_l)
            p_ens = w[0] * t_l[0] + w[1] * t_l[1] + w[2] * t_l[2]
            ens_ms.append(calc_metrics(y_te_inv, p_ens))

        pm = calc_metrics(y_te_inv, y_pers)
        l7m = calc_metrics(y_te_inv, y_lag7)

        mae_m = float(np.mean([x["MAE"] for x in ens_ms]))
        mae_s = float(np.std([x["MAE"] for x in ens_ms], ddof=1))
        mape_m = float(np.mean([x["MAPE"] for x in ens_ms]))
        mape_s = float(np.std([x["MAPE"] for x in ens_ms], ddof=1))
        bg_mae_m = float(np.mean([x["MAE"] for x in bigru_ms]))
        bg_mape_m = float(np.mean([x["MAPE"] for x in bigru_ms]))
        bg_mape_s = float(np.std([x["MAPE"] for x in bigru_ms], ddof=1))

        rows.append({
            "Division": div,
            "MAE_mean": mae_m, "MAE_std": mae_s,
            "MAPE_mean": mape_m, "MAPE_std": mape_s,
            "Acc_mean": 100.0 - mape_m,
            "R2_mean": float(np.mean([x["R2"] for x in ens_ms])),
            "BiGRU_MAE": bg_mae_m, "BiGRU_MAPE": bg_mape_m, "BiGRU_MAPE_std": bg_mape_s,
            "Ridge_MAE": rm["MAE"], "Ridge_MAPE": rm["MAPE"],
            "Lag7_MAE": l7m["MAE"], "Lag7_MAPE": l7m["MAPE"],
            "Persist_MAE": pm["MAE"], "Persist_MAPE": pm["MAPE"],
            "Gain_vs_Persist_%": (pm["MAE"] - mae_m) / pm["MAE"] * 100.0,
            "Gain_vs_BiGRU_%": (bg_mae_m - mae_m) / bg_mae_m * 100.0,
        })
        print(f"  {div:12s}: Ens={mae_m:6.2f}±{mae_s:4.2f} ({mape_m:.2f}%) | BiGRU={bg_mae_m:6.2f} ({bg_mape_m:.2f}%) | Ridge={rm['MAE']:6.2f} | Lag7={l7m['MAE']:6.2f} | Persist={pm['MAE']:6.2f}", flush=True)

    return pd.DataFrame(rows)


def run_causal_multihorizon_benchmark(df_raw):
    print("\n=== 2. CAUSAL DIRECT MULTI-HORIZON BENCHMARK (h = 1, 3, 7, 14, 30 DAYS; WITH CLIMATOLOGY+TREND & RIDGE) ===", flush=True)
    df = df_raw.copy()
    temp_col = "Maximum Temperature in Dhaka was"
    target_col = "Max. Demand at eve. peak (Generation end)"
    ys = df[target_col].shift(1)
    df["CDD"] = np.maximum(0.0, df[temp_col].values - TAU_COOL)
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
    dates_tr = df_enc.index[:tr_c]

    s1 = pd.read_csv(f"{OUT_DIR}/part1_summary_metrics.csv", index_col=0)
    rows = []
    for h in [1, 3, 7, 14, 30]:
        X_tr, y_tr = windowing_horizon(data_norm[:tr_c], data_norm[:tr_c, -1], HIST_SIZE, h)
        X_va, y_va = windowing_horizon(data_norm[tr_c:va_c], data_norm[tr_c:va_c, -1], HIST_SIZE, h)
        X_te, y_te = windowing_horizon(data_norm[va_c:], data_norm[va_c:, -1], HIST_SIZE, h)

        n_va = len(X_va)
        va_es_end = (n_va - HIST_SIZE) // 2
        va_cal_start = va_es_end + HIST_SIZE
        X_va_es, y_va_es = X_va[:va_es_end], y_va[:va_es_end]
        X_va_cal = X_va[va_cal_start:]

        y_va_inv = y_raw[tr_c + HIST_SIZE + h - 1 : va_c].reshape(-1, 1)
        y_va_cal_inv = y_va_inv[va_cal_start:]
        y_te_inv = y_raw[va_c + HIST_SIZE + h - 1 :].reshape(-1, 1)
        dates_te = df_enc.index[va_c + HIST_SIZE + h - 1 :]
        y_orig_pers = y_raw[va_c + HIST_SIZE - 1 : len(y_raw) - h].reshape(-1, 1)
        y_clim = fit_climatology_trend_baseline(dates_tr, y_raw[:tr_c], dates_te)
        cm = calc_metrics(y_te_inv, y_clim)
        pm = calc_metrics(y_te_inv, y_orig_pers)

        p_ridge = fit_ridge_val(X_tr.reshape(len(X_tr), -1), y_tr, X_va_es.reshape(len(X_va_es), -1), y_va_es, X_te.reshape(len(X_te), -1), sy)
        rm = calc_metrics(y_te_inv, p_ridge)

        if h == 1:
            mae_m = float(s1.loc["Val-Opt RNN Ensemble", "MAE_mean"])
            mae_s = float(s1.loc["Val-Opt RNN Ensemble", "MAE_std"])
            rmse_m = float(s1.loc["Val-Opt RNN Ensemble", "RMSE_mean"])
            mape_m = float(s1.loc["Val-Opt RNN Ensemble", "MAPE_mean"])
            acc_m = float(s1.loc["Val-Opt RNN Ensemble", "Acc_mean"])
            r2_m = float(s1.loc["Val-Opt RNN Ensemble", "R2_mean"])
        else:
            nf = X_tr.shape[-1]
            seed_ms = []
            for s in SEEDS:
                v_l, t_l = [], []
                for fn in [lambda: BiGRUModel(nf), lambda: BiLSTMAttnModel(nf), lambda: BiLSTMModel(nf)]:
                    m = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, s)
                    v_l.append(sy.inverse_transform(predict_torch(m, X_va_cal)))
                    t_l.append(sy.inverse_transform(predict_torch(m, X_te)))
                w = optimize_val_weights(y_va_cal_inv, v_l)
                p_ens = w[0] * t_l[0] + w[1] * t_l[1] + w[2] * t_l[2]
                seed_ms.append(calc_metrics(y_te_inv, p_ens))
            mae_m = float(np.mean([x["MAE"] for x in seed_ms]))
            mae_s = float(np.std([x["MAE"] for x in seed_ms], ddof=1))
            rmse_m = float(np.mean([x["RMSE"] for x in seed_ms]))
            mape_m = float(np.mean([x["MAPE"] for x in seed_ms]))
            acc_m = 100.0 - mape_m
            r2_m = float(np.mean([x["R2"] for x in seed_ms]))

        rows.append({
            "Horizon_Days": h, "N_Test": len(y_te_inv),
            "Ens_MAE_mean": mae_m, "Ens_MAE_std": mae_s, "Ens_RMSE_mean": rmse_m,
            "Ens_MAPE_mean": mape_m, "Ens_Acc_mean": acc_m, "Ens_R2_mean": r2_m,
            "Ridge_MAE": rm["MAE"], "Clim_Trend_MAE": cm["MAE"], "Clim_Trend_MAPE": cm["MAPE"],
            "Origin_Persist_MAE": pm["MAE"], "Origin_Persist_MAPE": pm["MAPE"],
            "Gain_vs_Persist_%": (pm["MAE"] - mae_m) / pm["MAE"] * 100.0,
            "Gain_vs_Clim_%": (cm["MAE"] - mae_m) / cm["MAE"] * 100.0,
        })
        print(f"  h={h:2d}d: Ens={mae_m:6.1f}±{mae_s:4.1f} | Ridge={rm['MAE']:6.1f} | Clim+Trend={cm['MAE']:6.1f} | Persist={pm['MAE']:6.1f} (vsP {(pm['MAE']-mae_m)/pm['MAE']*100:+.1f}%, vsC {(cm['MAE']-mae_m)/cm['MAE']*100:+.1f}%)", flush=True)

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

    # Fig 5: Same-Split 8-Division Benchmark (with error bars on multi-seed baselines, ZERO legend overlap!)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    x = np.arange(len(df_div))
    w = 0.17
    ax.bar(x - 2 * w, df_div["Persist_MAPE"], width=w, color="#adb5bd", label="Lag-1 Persistence")
    ax.bar(x - w, df_div["Lag7_MAPE"], width=w, color="#cbd5e1", label="Weekday Naive (Lag-7)")
    ax.bar(x, df_div["Ridge_MAPE"], width=w, color="#e9c46a", label="Ridge (Same Split)")
    ax.bar(x + w, df_div["BiGRU_MAPE"], yerr=df_div["BiGRU_MAPE_std"], capsize=2.5, width=w, color="#2a9d8f", label="Single BiGRU (3-Seed)")
    ax.bar(x + 2 * w, df_div["MAPE_mean"], yerr=df_div["MAPE_std"], capsize=2.5, width=w, color="#1d3557", label="Val-Opt RNN Ensemble (3-Seed)")
    for i, r in df_div.iterrows():
        y_top = max(r["MAPE_mean"] + r["MAPE_std"], r["BiGRU_MAPE"] + r["BiGRU_MAPE_std"]) + 0.30
        ax.text(i + 2 * w, y_top, f"{r['MAPE_mean']:.2f}%",
                ha="center", fontsize=6.7, fontweight="bold", color="#1d3557")
    ax.set_xticks(x)
    ax.set_xticklabels(df_div["Division"], rotation=15, ha="right")
    ax.set_ylabel("Blind Test MAPE (%)  [Lower is Better]")
    ax.set_title("Same-Split 8-Division Peak Demand Benchmark (462 Blind Test Days, 3 Seeds)", pad=28)
    ax.set_ylim(0, df_div[["Persist_MAPE", "Lag7_MAPE", "Ridge_MAPE", "BiGRU_MAPE"]].values.max() * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False, fontsize=7.6)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig5_divisional_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 12: Strictly Causal Direct Multi-Horizon (with Climatology+Trend & Ridge; clean top legends!)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    h_vals = df_mh["Horizon_Days"].values

    axes[0].errorbar(h_vals, df_mh["Ens_MAE_mean"], yerr=df_mh["Ens_MAE_std"], marker="o", lw=2.0, capsize=4, color="#1d3557", label="Val-Opt RNN Ens")
    axes[0].plot(h_vals, df_mh["Ridge_MAE"], marker="^", lw=1.6, ls="-.", color="#2a9d8f", label="Ridge (Direct)")
    axes[0].plot(h_vals, df_mh["Clim_Trend_MAE"], marker="D", lw=1.6, ls=":", color="#6d597a", label="Climatology+Trend")
    axes[0].plot(h_vals, df_mh["Origin_Persist_MAE"], marker="s", lw=1.6, ls="--", color="#e76f51", label="Origin Persistence")
    axes[0].set_xticks(h_vals)
    axes[0].set_xlabel("Direct Forecast Horizon h (Days Ahead)")
    axes[0].set_ylabel("Blind Test MAE (MW)")
    axes[0].set_title("(a) Multi-Horizon MAE vs. Seasonal & Linear Baselines", pad=38)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.6)

    axes[1].plot(h_vals, df_mh["Ens_MAPE_mean"], marker="o", lw=2.0, color="#1d3557", label="Val-Opt RNN Ens MAPE (%)")
    axes[1].plot(h_vals, df_mh["Clim_Trend_MAPE"], marker="D", lw=1.6, ls=":", color="#6d597a", label="Climatology+Trend MAPE (%)")
    axes[1].plot(h_vals, df_mh["Origin_Persist_MAPE"], marker="s", lw=1.6, ls="--", color="#e76f51", label="Origin Persistence MAPE (%)")
    axes[1].set_xticks(h_vals)
    axes[1].set_xlabel("Direct Forecast Horizon h (Days Ahead)")
    axes[1].set_ylabel("Blind Test MAPE (%)")
    axes[1].set_title("(b) Causal Multi-Horizon MAPE (Zero Future Telemetry)", pad=38)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.6)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig12_multihorizon.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nCompleted Revised Divisional + Causal Multi-Horizon Suite in {time.perf_counter() - t0:.1f}s!", flush=True)
