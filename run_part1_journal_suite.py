"""
Part 1 Q1-Journal Gold-Standard Reproducible Benchmark Suite (5 Seeds, 16 Models).
Strict leakage-free chronological protocol (train-only MinMax scaling + 14-day split embargo):
  1. Multi-seed benchmark (5 seeds: 42, 123, 7, 2024, 999) across 16 models:
     - Classical / Statistical: Persistence (Lag-1), 7-Day Rolling Mean, SARIMAX(1,1,1)(1,0,1,7)
     - GBDTs: XGBoost, LightGBM
     - Linear & Patched Transformer: DLinear (AAAI 2023), PatchTST (ICLR 2023)
     - Deep Sequential/Attention: BiGRU, BiLSTM, BiLSTM-Attention, Transformer, CNN-BiGRU-Transformer
     - Residual Hybrids: Hybrid BiGRU-XGBoost, Hybrid BiLSTM-Attn-XGBoost
     - Ensembles: Equal-Weight Ensemble (1/3,1/3,1/3), Validation-Optimized Weighted Ensemble (SLSQP on Val MAE)
  2. Diebold-Mariano (Newey-West HAC, lag 14) + Moving-Block Bootstrap 95% CIs (5-seed averaged trajectories)
  3. Split Conformal Prediction (80% & 90% intervals calibrated strictly on Val residuals: PICP, Width, Winkler)
  4. Controlled 5-Seed Feature Ablation on the Val-Opt Ensemble (Overall MAE + Holiday/Eid-Subset MAE)
  5. 5-Seed Averaged Regime Analysis (Season, Calendar) & Peak Demand-Bin Stratification
  6. Computational Efficiency Profile & Vector PDF Figures -> part1_journal_package/figures/
"""
import json
import math
import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from scipy.optimize import minimize
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

FIG_DIR = "part1_journal_package/figures"
OUT_DIR = "part1_journal_package/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 7, 2024, 999]
HIST_SIZE = 14
EPOCHS = 55
BATCH_SIZE = 32
PATIENCE = 8
OPT_LR = 2.5e-3
OPT_WD = 1e-5
OPT_DROPOUT = 0.20


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def windowing(X_input, y_input, history_size):
    data, labels = [], []
    for i in range(history_size, len(y_input)):
        data.append(X_input[i - history_size : i, :])
        labels.append(y_input[i])
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)


def prepare_dataset(use_cdd=True, use_eid=True, use_pca=False):
    df = pd.read_excel("version_5_extended_backup.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    if "Dhaka_temp" in df.columns:
        df["Maximum Temperature in Dhaka was"] = df["Dhaka_temp"]
    if "Mymensingh_temp" not in df.columns and "Dhaka_temp" in df.columns:
        df["Mymensingh_temp"] = df["Dhaka_temp"]

    temp_col = "Maximum Temperature in Dhaka was"
    target_col = "Max. Demand at eve. peak (Generation end)"
    y_shifted = df[target_col].shift(1)

    if use_cdd:
        df["CDD"] = df[temp_col].apply(lambda t: max(0.0, t - 24.0))
        df["HDD"] = df[temp_col].apply(lambda t: max(0.0, 18.0 - t))

    df["target_roll_mean_3"] = y_shifted.rolling(3).mean()
    df["target_roll_std_3"] = y_shifted.rolling(3).std()
    df["target_roll_mean_7"] = y_shifted.rolling(7).mean()
    df["target_roll_std_7"] = y_shifted.rolling(7).std()
    df["target_roll_mean_14"] = y_shifted.rolling(14).mean()
    df["target_roll_std_14"] = y_shifted.rolling(14).std()
    df["target_roll_min_7"] = y_shifted.rolling(7).min()
    df["target_roll_max_7"] = y_shifted.rolling(7).max()
    df["target_diff_1"] = y_shifted - df[target_col].shift(2)

    cat_cols = ["Holiday_cat"]
    if use_eid:
        for k in [1, 2, 3]:
            df[f"Holiday_lead_{k}"] = (df["Holiday_cat"].shift(-k).fillna(0) > 0).astype(float)
            df[f"Holiday_lag_{k}"] = (df["Holiday_cat"].shift(k).fillna(0) > 0).astype(float)
            cat_cols.extend([f"Holiday_lead_{k}", f"Holiday_lag_{k}"])

    df = df.dropna()
    meta = df[["Holiday_cat", "Day of the week", "Holiday name", "Month", target_col, temp_col]].copy()
    df_enc = pd.get_dummies(df, columns=["Day of the week", "Holiday name"], drop_first=True, dtype=float)

    all_cols = df_enc.columns.tolist()
    cont_cols = [
        c for c in all_cols
        if c not in cat_cols and c != target_col
        and not c.startswith("Day of the week_") and not c.startswith("Holiday name_")
    ]
    cat_onehot_cols = [c for c in all_cols if c not in cont_cols and c != target_col]

    train_cut = int(0.70 * len(df_enc))
    val_cut = int(0.80 * len(df_enc))

    # STRICT train-only scaling
    sx = MinMaxScaler().fit(df_enc[cont_cols].iloc[:train_cut])
    X_cont = sx.transform(df_enc[cont_cols])
    sy = MinMaxScaler().fit(df_enc[[target_col]].iloc[:train_cut])
    y_norm = sy.transform(df_enc[[target_col]])

    X_cat = df_enc[cat_onehot_cols].values.astype(np.float32)
    X_all = np.concatenate((X_cont, X_cat), axis=1)

    if use_pca:
        pca = PCA(n_components=0.95, random_state=42).fit(X_all[:train_cut])
        X_all = pca.transform(X_all)

    data_norm = np.concatenate((X_all, y_norm), axis=1)
    X_tr, y_tr = windowing(data_norm[:train_cut], data_norm[:train_cut, -1], HIST_SIZE)
    X_va, y_va = windowing(data_norm[train_cut:val_cut], data_norm[train_cut:val_cut, -1], HIST_SIZE)
    X_te, y_te = windowing(data_norm[val_cut:], data_norm[val_cut:, -1], HIST_SIZE)

    y_va_inv = df_enc[[target_col]].iloc[train_cut + HIST_SIZE : val_cut].values
    y_te_inv = df_enc[[target_col]].iloc[val_cut + HIST_SIZE :].values
    te_dates = df_enc.index[val_cut + HIST_SIZE :]
    te_meta = meta.iloc[val_cut + HIST_SIZE :].copy()

    return {
        "X_tr": X_tr, "y_tr": y_tr, "X_va": X_va, "y_va": y_va, "X_te": X_te, "y_te": y_te,
        "y_va_inv": y_va_inv, "y_te_inv": y_te_inv, "sy": sy, "te_dates": te_dates, "te_meta": te_meta,
        "meta_full": meta,
        "train_dates": (df_enc.index[0], df_enc.index[train_cut - 1]),
        "val_dates": (df_enc.index[train_cut], df_enc.index[val_cut - 1]),
        "test_dates": (te_dates[0], te_dates[-1]),
        "n_raw": len(df_enc), "n_features": data_norm.shape[1],
        "df_raw_target": df_enc[target_col], "val_cut": val_cut, "train_cut": train_cut
    }


def run_sarimax_baseline(ds):
    """Strictly causal SARIMAX(1,1,1)(1,0,1,7) fitted on train only, filtered one-step-ahead on blind test."""
    meta = ds["meta_full"].copy()
    target_col = "Max. Demand at eve. peak (Generation end)"
    temp_col = "Maximum Temperature in Dhaka was"
    exog = pd.DataFrame(index=meta.index)
    t_lag1 = meta[temp_col].shift(1).bfill()
    exog["CDD_lag1"] = np.maximum(0.0, t_lag1 - 24.0)
    exog["HDD_lag1"] = np.maximum(0.0, 18.0 - t_lag1)
    exog["Is_Weekend"] = meta["Day of the week"].isin(["Friday", "Saturday"]).astype(float)
    exog["Holiday"] = (meta["Holiday_cat"] > 0).astype(float)

    tr_c, va_c = ds["train_cut"], ds["val_cut"]
    y_all = meta[target_col].values.astype(float)
    ex_all = exog.values.astype(float)

    t0 = time.perf_counter()
    mod = SARIMAX(
        y_all[:tr_c], exog=ex_all[:tr_c],
        order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit_res = mod.fit(disp=False, maxiter=80)
    # Apply fixed train-estimated parameters across the full timeline (strictly causal Kalman 1-step-ahead)
    full_res = fit_res.apply(y_all, exog=ex_all)
    preds_all = full_res.fittedvalues
    p_te = preds_all[va_c + HIST_SIZE :].reshape(-1, 1)
    lat_ms = (time.perf_counter() - t0) / len(y_all) * 1000.0
    n_params = len(fit_res.params)
    return p_te, n_params, round(lat_ms, 2)


# --- PyTorch Model Architectures ---
class DLinearModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, kernel_size=5):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.linear_seasonal = nn.Linear(seq_len * n_feat, 1)
        self.linear_trend = nn.Linear(seq_len * n_feat, 1)
    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        trend = self.avg(x_pad.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return self.linear_seasonal(seasonal.reshape(x.size(0), -1)) + self.linear_trend(trend.reshape(x.size(0), -1))


class PatchTSTModel(nn.Module):
    """PatchTST (Nie et al., ICLR 2023): overlapping temporal patching + learnable positional encoding + Transformer."""
    def __init__(self, n_feat, seq_len=14, patch_len=4, stride=2, d_model=64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = (seq_len - patch_len) // stride + 1  # (14-4)//2 + 1 = 6 patches
        self.patch_proj = nn.Linear(patch_len * n_feat, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=OPT_DROPOUT, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(OPT_DROPOUT),
            nn.Linear(self.n_patches * d_model, 64), nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, x):
        # x: (B, L, C) -> unfold along L: (B, n_patches, C, patch_len) -> (B, n_patches, patch_len*C)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(x.size(0), self.n_patches, -1)
        z = self.patch_proj(patches) + self.pos_emb
        z = self.encoder(z)
        return self.head(z)


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


class TransformerModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, d_model=64):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128,
                                         dropout=OPT_DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * d_model, 128), nn.ReLU(),
                                  nn.Dropout(OPT_DROPOUT), nn.Linear(128, 1))
    def forward(self, x):
        return self.head(self.encoder(self.proj(x)))


class CNNBiGRUTransModel(nn.Module):
    def __init__(self, n_feat, seq_len=14):
        super().__init__()
        self.conv = nn.Conv1d(n_feat, 32, kernel_size=3, padding=1)
        self.drop = nn.Dropout(0.25)
        self.rnn = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.mha = nn.MultiheadAttention(64, num_heads=2, dropout=0.25, batch_first=True)
        self.ln = nn.LayerNorm(64)
        self.head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.25), nn.Linear(32, 1))
    def forward(self, x):
        c = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        h = self.rnn(self.drop(c))[0]
        a = self.mha(h, h, h)[0]
        z = self.ln(h + self.drop(a)).mean(dim=1)
        return self.head(z)


def train_torch_model(model, X_tr, y_tr, X_va, y_va, seed, epochs=EPOCHS, patience=PATIENCE, lr=OPT_LR):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=OPT_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-5)
    loss_fn = nn.L1Loss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                    batch_size=BATCH_SIZE, shuffle=True)
    xva_t, yva_t = torch.from_numpy(X_va), torch.from_numpy(y_va)
    best_loss, best_state, wait = float("inf"), None, 0
    val_hist = []
    for _ in range(epochs):
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
        val_hist.append(vl)
        sched.step(vl)
        if vl < best_loss - 1e-5:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, val_hist


@torch.no_grad()
def predict_torch(model, X):
    model.eval()
    return model(torch.from_numpy(X)).numpy()


@torch.no_grad()
def measure_torch_latency(model, x_sample, n_runs=60):
    model.eval()
    xt = torch.from_numpy(x_sample)
    _ = model(xt)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(xt)
    return (time.perf_counter() - t0) / n_runs * 1000.0


def calc_metrics(y_true, y_pred):
    yt, yp = y_true.ravel(), y_pred.ravel()
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)
    r2 = float(r2_score(yt, yp))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Acc": 100.0 - mape, "R2": r2}


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


def conformal_metrics(y_val, pred_val, y_test, pred_test, alpha):
    cal_scores = np.abs(y_val.ravel() - pred_val.ravel())
    n_cal = len(cal_scores)
    q_level = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
    q = float(np.quantile(cal_scores, q_level, method="higher"))
    lo = pred_test.ravel() - q
    hi = pred_test.ravel() + q
    yt = y_test.ravel()
    picp = float(np.mean((yt >= lo) & (yt <= hi)) * 100.0)
    width = float(np.mean(hi - lo))
    pinaw = float(width / (np.max(yt) - np.min(yt)) * 100.0)
    penalty = np.where(yt < lo, (2.0 / alpha) * (lo - yt),
              np.where(yt > hi, (2.0 / alpha) * (yt - hi), 0.0))
    winkler = float(np.mean((hi - lo) + penalty))
    return {"Nominal_%": int((1.0 - alpha) * 100), "Half_Width_MW": q, "Mean_Width_MW": width,
            "PICP_%": picp, "PINAW_%": pinaw, "Winkler_MW": winkler, "lo": lo, "hi": hi}


def dm_and_bootstrap(y_true, pred_ens, pred_base, max_lag=14, block_len=7, n_boot=1000, seed=42):
    e_ens = np.abs(y_true.ravel() - pred_ens.ravel())
    e_base = np.abs(y_true.ravel() - pred_base.ravel())
    d = e_ens - e_base
    T = len(d)
    d_mean = float(np.mean(d))
    var_d = np.var(d, ddof=1)
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)
        var_d += 2.0 * w * np.cov(d[k:], d[:-k], ddof=1)[0, 1]
    dm_stat = float(d_mean / np.sqrt(max(var_d, 1e-12) / T))
    p_val = float(2.0 * stats.t.sf(np.abs(dm_stat), df=T - 1))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(T / block_len))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T - block_len + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:T]
        diffs[b] = np.mean(d[idx])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return dm_stat, p_val, d_mean, float(ci_lo), float(ci_hi)


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("Preparing strictly leakage-free Part 1 dataset (5 Seeds, 16 Models)...", flush=True)
    ds = prepare_dataset(use_cdd=True, use_eid=True, use_pca=False)
    print(f"Total clean rows: {ds['n_raw']} | Features: {ds['n_features']} | Train/Val/Test: {len(ds['X_tr'])}/{len(ds['X_va'])}/{len(ds['X_te'])}", flush=True)

    X_tr, y_tr = ds["X_tr"], ds["y_tr"]
    X_va, y_va, y_va_inv = ds["X_va"], ds["y_va"], ds["y_va_inv"]
    X_te, y_te_inv, sy = ds["X_te"], ds["y_te_inv"], ds["sy"]
    X_tr_2d = X_tr.reshape(len(X_tr), -1)
    X_va_2d = X_va.reshape(len(X_va), -1)
    X_te_2d = X_te.reshape(len(X_te), -1)
    n_f = X_tr.shape[-1]

    y_full = ds["df_raw_target"].values
    vc = ds["val_cut"]
    p_persist = y_full[vc + HIST_SIZE - 1 : -1].reshape(-1, 1)
    p_roll7 = pd.Series(y_full).shift(1).rolling(7).mean().values[vc + HIST_SIZE :].reshape(-1, 1)

    print("Fitting strictly causal SARIMAX(1,1,1)(1,0,1,7) on Train split...", flush=True)
    p_sarimax, sarimax_p, sarimax_ms = run_sarimax_baseline(ds)
    msx = calc_metrics(y_te_inv, p_sarimax)
    print(f"  SARIMAX(1,1,1)(1,0,1,7): MAE={msx['MAE']:6.1f} MW | RMSE={msx['RMSE']:6.1f} | Acc={msx['Acc']:.2f}%", flush=True)

    all_seed_metrics = []
    seed_preds = {s: {} for s in SEEDS}
    seed_val_preds = {s: {} for s in SEEDS}
    learned_weights = []
    val_histories = {m: [] for m in ["DLinear", "PatchTST", "BiGRU", "BiLSTM", "BiLSTM-Attention", "Transformer", "CNN-BiGRU-Transformer"]}
    eff_profile = {"SARIMAX": {"Params": sarimax_p, "Size_KB": round(sarimax_p * 8 / 1024, 2), "CPU_ms": sarimax_ms}}

    for s in SEEDS:
        print(f"\n--- Running Seed {s} ---", flush=True)
        seed_preds[s]["Persistence (Lag-1)"] = p_persist
        seed_preds[s]["7-Day Rolling Mean"] = p_roll7
        seed_preds[s]["SARIMAX"] = p_sarimax
        for bn, bp in [("Persistence (Lag-1)", p_persist), ("7-Day Rolling Mean", p_roll7), ("SARIMAX", p_sarimax)]:
            all_seed_metrics.append({"Seed": s, "Model": bn, **calc_metrics(y_te_inv, bp)})

        xm = xgb.XGBRegressor(random_state=s, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
        xm.fit(X_tr_2d, y_tr.ravel(), eval_set=[(X_va_2d, y_va.ravel())], verbose=False)
        p_xgb = sy.inverse_transform(xm.predict(X_te_2d).reshape(-1, 1))
        seed_preds[s]["XGBoost"] = p_xgb
        all_seed_metrics.append({"Seed": s, "Model": "XGBoost", **calc_metrics(y_te_inv, p_xgb)})
        if s == SEEDS[0]:
            t0 = time.perf_counter()
            for _ in range(60): _ = xm.predict(X_te_2d[:1])
            eff_profile["XGBoost"] = {"Params": 0, "Size_KB": 0.0, "CPU_ms": round((time.perf_counter() - t0) / 60 * 1000, 2)}

        lm = lgb.LGBMRegressor(objective="regression_l1", random_state=s, n_estimators=500,
                               learning_rate=0.05, num_leaves=31, n_jobs=1, verbose=-1)
        lm.fit(X_tr_2d, y_tr.ravel(), eval_set=[(X_va_2d, y_va.ravel())],
               callbacks=[lgb.early_stopping(8, verbose=False)])
        p_lgb = sy.inverse_transform(lm.predict(X_te_2d).reshape(-1, 1))
        seed_preds[s]["LightGBM"] = p_lgb
        all_seed_metrics.append({"Seed": s, "Model": "LightGBM", **calc_metrics(y_te_inv, p_lgb)})
        if s == SEEDS[0]:
            t0 = time.perf_counter()
            for _ in range(60): _ = lm.predict(X_te_2d[:1])
            eff_profile["LightGBM"] = {"Params": 0, "Size_KB": 0.0, "CPU_ms": round((time.perf_counter() - t0) / 60 * 1000, 2)}

        dl_builders = [
            ("DLinear", lambda: DLinearModel(n_f)),
            ("PatchTST", lambda: PatchTSTModel(n_f)),
            ("BiGRU", lambda: BiGRUModel(n_f)),
            ("BiLSTM", lambda: BiLSTMModel(n_f)),
            ("BiLSTM-Attention", lambda: BiLSTMAttnModel(n_f)),
            ("Transformer", lambda: TransformerModel(n_f)),
            ("CNN-BiGRU-Transformer", lambda: CNNBiGRUTransModel(n_f)),
        ]
        trained_dl = {}
        for name, fn in dl_builders:
            set_seed(s)
            mod, vh = train_torch_model(fn(), X_tr, y_tr, X_va, y_va, s)
            val_histories[name].append(vh)
            pred_va = sy.inverse_transform(predict_torch(mod, X_va))
            pred_te = sy.inverse_transform(predict_torch(mod, X_te))
            seed_val_preds[s][name] = pred_va
            seed_preds[s][name] = pred_te
            trained_dl[name] = mod
            mr = calc_metrics(y_te_inv, pred_te)
            print(f"  [Seed {s}] {name:22s}: MAE={mr['MAE']:6.1f} MW | Acc={mr['Acc']:.2f}%", flush=True)
            all_seed_metrics.append({"Seed": s, "Model": name, **mr})
            if s == SEEDS[0]:
                np_cnt = sum(p.numel() for p in mod.parameters())
                eff_profile[name] = {
                    "Params": np_cnt, "Size_KB": round(np_cnt * 4 / 1024, 1),
                    "CPU_ms": round(measure_torch_latency(mod, X_te[:1]), 2)
                }

        for base_name, h_name in [("BiGRU", "Hybrid BiGRU-XGBoost"), ("BiLSTM-Attention", "Hybrid BiLSTM-Attn-XGBoost")]:
            bm = trained_dl[base_name]
            res_tr = y_tr - predict_torch(bm, X_tr)
            res_va = y_va - predict_torch(bm, X_va)
            hx = xgb.XGBRegressor(random_state=s, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
            hx.fit(X_tr_2d, res_tr.ravel(), eval_set=[(X_va_2d, res_va.ravel())], verbose=False)
            pred_h = sy.inverse_transform(predict_torch(bm, X_te) + hx.predict(X_te_2d).reshape(-1, 1))
            seed_preds[s][h_name] = pred_h
            all_seed_metrics.append({"Seed": s, "Model": h_name, **calc_metrics(y_te_inv, pred_h)})

        p_eq_te = (seed_preds[s]["BiGRU"] + seed_preds[s]["BiLSTM-Attention"] + seed_preds[s]["BiLSTM"]) / 3.0
        seed_preds[s]["Equal-Weight Ensemble"] = p_eq_te
        all_seed_metrics.append({"Seed": s, "Model": "Equal-Weight Ensemble", **calc_metrics(y_te_inv, p_eq_te)})

        w_opt = optimize_val_weights(
            y_va_inv,
            [seed_val_preds[s]["BiGRU"], seed_val_preds[s]["BiLSTM-Attention"], seed_val_preds[s]["BiLSTM"]]
        )
        learned_weights.append(w_opt.tolist())
        p_opt_va = w_opt[0] * seed_val_preds[s]["BiGRU"] + w_opt[1] * seed_val_preds[s]["BiLSTM-Attention"] + w_opt[2] * seed_val_preds[s]["BiLSTM"]
        p_opt_te = w_opt[0] * seed_preds[s]["BiGRU"] + w_opt[1] * seed_preds[s]["BiLSTM-Attention"] + w_opt[2] * seed_preds[s]["BiLSTM"]
        seed_val_preds[s]["Weighted Ensemble"] = p_opt_va
        seed_preds[s]["Weighted Ensemble"] = p_opt_te
        me = calc_metrics(y_te_inv, p_opt_te)
        print(f"  [Seed {s}] {'Weighted Ensemble':22s}: MAE={me['MAE']:6.1f} MW | Acc={me['Acc']:.2f}% | w={np.round(w_opt, 3)}", flush=True)
        all_seed_metrics.append({"Seed": s, "Model": "Weighted Ensemble", **me})

        if s == SEEDS[0]:
            tot_p = eff_profile["BiGRU"]["Params"] + eff_profile["BiLSTM-Attention"]["Params"] + eff_profile["BiLSTM"]["Params"]
            tot_ms = eff_profile["BiGRU"]["CPU_ms"] + eff_profile["BiLSTM-Attention"]["CPU_ms"] + eff_profile["BiLSTM"]["CPU_ms"]
            eff_profile["Weighted Ensemble"] = {"Params": tot_p, "Size_KB": round(tot_p * 4 / 1024, 1), "CPU_ms": round(tot_ms, 2)}

    mean_w = np.mean(learned_weights, axis=0)
    print(f"\nLearned Validation-Optimized Weights (mean across 5 seeds) [BiGRU, BiLSTM-Attn, BiLSTM]: {np.round(mean_w, 3)}")

    df_seeds = pd.DataFrame(all_seed_metrics)
    df_summary = df_seeds.groupby("Model").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
        Acc_mean=("Acc", "mean"), Acc_std=("Acc", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    ).sort_values("MAE_mean")
    print("\n=== 1. PART 1 MULTI-SEED BENCHMARK (16 MODELS, 5 SEEDS) ===")
    print(df_summary.round(2).to_string())

    # 5-seed averaged forecast trajectories
    mean_preds = {m: np.mean([seed_preds[s][m] for s in SEEDS], axis=0) for m in df_summary.index}
    mean_val_ens = np.mean([seed_val_preds[s]["Weighted Ensemble"] for s in SEEDS], axis=0)
    ens_5s_mae = float(mean_absolute_error(y_te_inv, mean_preds["Weighted Ensemble"]))

    dm_rows = []
    for m in df_summary.index:
        if m == "Weighted Ensemble":
            continue
        base_5s_mae = float(mean_absolute_error(y_te_inv, mean_preds[m]))
        dm_s, pv, d_m, lo, hi = dm_and_bootstrap(y_te_inv, mean_preds["Weighted Ensemble"], mean_preds[m])
        dm_rows.append({
            "Baseline": m, "Ens_5Seed_MAE": ens_5s_mae, "Base_5Seed_MAE": base_5s_mae,
            "DM_stat": dm_s, "p_val": pv, "Delta_MAE_MW": d_m, "CI_lo": lo, "CI_hi": hi
        })
    df_dm = pd.DataFrame(dm_rows).sort_values("Delta_MAE_MW")
    print("\n=== 2. DIEBOLD-MARIANO (HAC LAG=14) & 95% BLOCK-BOOTSTRAP CI (5-SEED AVERAGED TRAJECTORIES) ===")
    print(df_dm.round(3).to_string(index=False))

    # Split Conformal Prediction (80% & 90%)
    cp_80 = conformal_metrics(y_va_inv, mean_val_ens, y_te_inv, mean_preds["Weighted Ensemble"], alpha=0.20)
    cp_90 = conformal_metrics(y_va_inv, mean_val_ens, y_te_inv, mean_preds["Weighted Ensemble"], alpha=0.10)
    df_cp = pd.DataFrame([
        {k: v for k, v in cp_80.items() if k not in ("lo", "hi")},
        {k: v for k, v in cp_90.items() if k not in ("lo", "hi")},
    ])
    print("\n=== 3. SPLIT CONFORMAL PREDICTION INTERVALS (CALIBRATED ON 224 VAL DAYS) ===")
    print(df_cp.round(2).to_string(index=False))

    # Controlled 5-Seed Feature Ablation on the Val-Opt Ensemble
    print("\nRunning Controlled 5-Seed Feature Ablation on the Val-Opt Ensemble...", flush=True)
    ablation_configs = [
        ("1. Base (No CDD/HDD, No Eid Flags)", dict(use_cdd=False, use_eid=False, use_pca=False)),
        ("2. + Thermodynamic CDD/HDD", dict(use_cdd=True, use_eid=False, use_pca=False)),
        ("3. + Eid Lead/Lag (Full No-PCA)", dict(use_cdd=True, use_eid=True, use_pca=False)),
        ("4. Full + PCA (95% Variance)", dict(use_cdd=True, use_eid=True, use_pca=True)),
    ]
    hol_mask = (ds["te_meta"]["Holiday_cat"].values > 0)
    ablation_rows = []
    for label, cfg in ablation_configs:
        if cfg == dict(use_cdd=True, use_eid=True, use_pca=False):
            ms = [calc_metrics(y_te_inv, seed_preds[s]["Weighted Ensemble"]) for s in SEEDS]
            hol_maes = [float(mean_absolute_error(y_te_inv[hol_mask], seed_preds[s]["Weighted Ensemble"][hol_mask])) for s in SEEDS]
            dims = ds["n_features"]
        else:
            dsi = prepare_dataset(**cfg)
            dims = dsi["n_features"]
            ms, hol_maes = [], []
            hm = (dsi["te_meta"]["Holiday_cat"].values > 0)
            for s in SEEDS:
                v_list, t_list = [], []
                for fn in [lambda: BiGRUModel(dims), lambda: BiLSTMAttnModel(dims), lambda: BiLSTMModel(dims)]:
                    set_seed(s)
                    mod, _ = train_torch_model(fn(), dsi["X_tr"], dsi["y_tr"], dsi["X_va"], dsi["y_va"], s)
                    v_list.append(dsi["sy"].inverse_transform(predict_torch(mod, dsi["X_va"])))
                    t_list.append(dsi["sy"].inverse_transform(predict_torch(mod, dsi["X_te"])))
                wi = optimize_val_weights(dsi["y_va_inv"], v_list)
                pr = wi[0] * t_list[0] + wi[1] * t_list[1] + wi[2] * t_list[2]
                ms.append(calc_metrics(dsi["y_te_inv"], pr))
                hol_maes.append(float(mean_absolute_error(dsi["y_te_inv"][hm], pr[hm])))
        ablation_rows.append({
            "Config": label, "Dims": dims,
            "MAE_mean": float(np.mean([x["MAE"] for x in ms])),
            "MAE_std": float(np.std([x["MAE"] for x in ms], ddof=1)),
            "Hol_MAE_mean": float(np.mean(hol_maes)),
            "MAPE_mean": float(np.mean([x["MAPE"] for x in ms])),
            "Acc_mean": float(np.mean([x["Acc"] for x in ms])),
        })
    df_abl = pd.DataFrame(ablation_rows)
    print("\n=== 4. CONTROLLED ENSEMBLE FEATURE ABLATION (5 SEEDS) ===")
    print(df_abl.round(2).to_string(index=False))

    # Regime Analysis & 5-Seed Averaged Demand Bins
    te_m = ds["te_meta"].copy()
    te_m["Actual"] = y_te_inv.ravel()
    te_m["Pred_Ens"] = mean_preds["Weighted Ensemble"].ravel()
    te_m["Pred_XGB"] = mean_preds["XGBoost"].ravel()
    te_m["Pred_Persist"] = mean_preds["Persistence (Lag-1)"].ravel()

    def map_season(m):
        if m in [3, 4, 5]: return "Summer (Mar-May)"
        if m in [6, 7, 8, 9]: return "Monsoon (Jun-Sep)"
        if m in [10, 11]: return "Post-Monsoon (Oct-Nov)"
        return "Winter (Dec-Feb)"

    te_m["Season"] = te_m["Month"].apply(map_season)
    te_m["Is_Weekend"] = te_m["Day of the week"].isin(["Friday", "Saturday"])
    te_m["Cal_Regime"] = np.where(te_m["Holiday_cat"] > 0, "Holiday / Eid",
                         np.where(te_m["Is_Weekend"], "Weekend (Fri-Sat)", "Working Weekday"))

    regime_rows = []
    for col in ["Season", "Cal_Regime"]:
        for grp, g in te_m.groupby(col):
            me = calc_metrics(g["Actual"].values, g["Pred_Ens"].values)
            mx = calc_metrics(g["Actual"].values, g["Pred_XGB"].values)
            mp = calc_metrics(g["Actual"].values, g["Pred_Persist"].values)
            regime_rows.append({
                "Axis": col, "Regime": grp, "N": len(g),
                "Ens_MAE": me["MAE"], "Ens_MAPE": me["MAPE"],
                "XGB_MAE": mx["MAE"], "Persist_MAE": mp["MAE"]
            })
    df_reg = pd.DataFrame(regime_rows)

    bins = [0, 11000, 13000, 15000, 25000]
    bin_labels = ["Low (<11 GW)", "Moderate (11-13 GW)", "High (13-15 GW)", "Extreme Peak (>15 GW)"]
    te_m["Bin"] = pd.cut(te_m["Actual"], bins=bins, labels=bin_labels)
    bin_rows = []
    for lbl in bin_labels:
        g = te_m[te_m["Bin"] == lbl]
        bin_rows.append({
            "Demand_Bin": lbl, "N": len(g),
            "Ens_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Ens"])),
            "XGB_MAE": float(mean_absolute_error(g["Actual"], g["Pred_XGB"])),
            "Persist_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Persist"])),
            "Ens_MAPE": float(np.mean(np.abs((g["Actual"] - g["Pred_Ens"]) / g["Actual"])) * 100.0),
        })
    df_bins = pd.DataFrame(bin_rows)
    print("\n=== 5. 5-SEED AVERAGED DEMAND-BIN BREAKDOWN ===")
    print(df_bins.round(2).to_string(index=False))

    # ==========================================================================
    # REGENERATE UPDATED VECTOR PDF FIGURES
    # ==========================================================================
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(ds["te_dates"], y_te_inv, color="#111111", lw=1.4, label="Actual National Peak Demand")
    ax.plot(ds["te_dates"], mean_preds["Weighted Ensemble"], color="#1d3557", lw=1.3, label="Val-Opt Ensemble (5-Seed)")
    ax.plot(ds["te_dates"], mean_preds["XGBoost"], color="#e76f51", lw=1.0, ls="--", alpha=0.8, label="XGBoost")
    ax.set_ylabel("Peak Demand (MW)")
    ax.set_title("Bangladesh National Evening Peak Demand — Blind Test Set Comparison (462 Days)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_forecast_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.1))
    colors = {"BiGRU": "#1d3557", "BiLSTM-Attention": "#2a9d8f", "BiLSTM": "#e9c46a",
              "PatchTST": "#f4a261", "DLinear": "#6d597a", "Transformer": "#e76f51", "CNN-BiGRU-Transformer": "#9b2226"}
    for name, hlist in val_histories.items():
        max_len = max(len(h) for h in hlist)
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in hlist])
        mu, sd = padded.mean(axis=0), padded.std(axis=0)
        ep = np.arange(1, max_len + 1)
        ax.plot(ep, mu, lw=1.7, color=colors[name], label=name)
        ax.fill_between(ep, mu - sd, mu + sd, color=colors[name], alpha=0.12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss (Normalized MAE)")
    ax.set_title("Multi-Seed Validation Convergence (Mean ± Std across 5 Seeds)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_convergence_curves.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    y_pos = np.arange(len(df_dm))
    c_bars = ["#1d3557" if p < 0.01 else ("#457b9d" if p < 0.05 else "#adb5bd") for p in df_dm["p_val"]]
    ax.axvline(0, color="black", ls="--", lw=1.0)
    for i, (_, r) in enumerate(df_dm.iterrows()):
        ax.plot([r["CI_lo"], r["CI_hi"]], [i, i], color=c_bars[i], lw=2.2)
        ax.plot(r["Delta_MAE_MW"], i, "o", color=c_bars[i], ms=6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_dm["Baseline"])
    ax.set_xlabel("MAE Difference: Val-Opt Ensemble − Baseline (MW)  [Negative = Ensemble Better]")
    ax.set_title("Diebold–Mariano Significance & 95% Block-Bootstrap CIs (5-Seed Trajectories, 15 Baselines)")
    ax.grid(axis="x", alpha=0.25)
    p1 = mpatches.Patch(color="#1d3557", label="p < 0.01")
    p2 = mpatches.Patch(color="#457b9d", label="p < 0.05")
    p3 = mpatches.Patch(color="#adb5bd", label="p ≥ 0.05 (ns)")
    ax.legend(handles=[p1, p2, p3], loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig3_dm_forest.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax = axes[0]
    short_lbl = ["Base", "+CDD/HDD", "+Eid Flags\n(No-PCA)", "Full + PCA\n(95%)"]
    ax.bar(short_lbl, df_abl["MAE_mean"], yerr=df_abl["MAE_std"], capsize=4,
           color=["#adb5bd", "#457b9d", "#1d3557", "#e76f51"], width=0.55)
    for i, v in enumerate(df_abl["MAE_mean"]):
        ax.text(i, v + df_abl["MAE_std"].iloc[i] + 8, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Test MAE (MW)")
    ax.set_title("(a) Controlled Ensemble Feature Ablation (5 Seeds)")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    s_reg = df_reg[df_reg["Axis"] == "Season"]
    x = np.arange(len(s_reg))
    w = 0.27
    ax.bar(x - w, s_reg["Persist_MAE"], width=w, color="#adb5bd", label="Persistence")
    ax.bar(x, s_reg["XGB_MAE"], width=w, color="#e76f51", label="XGBoost")
    ax.bar(x + w, s_reg["Ens_MAE"], width=w, color="#1d3557", label="Val-Opt Ensemble")
    ax.set_xticks(x)
    ax.set_xticklabels([s.split(" ")[0] for s in s_reg["Regime"]], fontsize=8.5)
    ax.set_ylabel("Test MAE (MW)")
    ax.set_title("(b) Seasonal Regime Error Breakdown (5-Seed)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_ablation_regime.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(df_bins))
    w = 0.26
    ax.bar(x - w, df_bins["Persist_MAE"], width=w, color="#adb5bd", label="Persistence (Lag-1)")
    ax.bar(x, df_bins["XGB_MAE"], width=w, color="#e76f51", label="XGBoost")
    ax.bar(x + w, df_bins["Ens_MAE"], width=w, color="#1d3557", label="Val-Opt Ensemble (Ours)")
    for i, r in df_bins.iterrows():
        ax.text(i + w, r["Ens_MAE"] + 18, f"{r['Ens_MAE']:.0f}", ha="center", fontsize=7.5, fontweight="bold", color="#1d3557")
        ax.text(i, r["XGB_MAE"] + 18, f"{r['XGB_MAE']:.0f}", ha="center", fontsize=7.5, color="#b54428")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(df_bins["Demand_Bin"], df_bins["N"])], fontsize=8.5)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_title("Peak Demand-Level Error Stratification (5-Seed Averaged Trajectories)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig6_demand_bins.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    w1_mask = (ds["te_dates"] >= "2025-12-15") & (ds["te_dates"] <= "2026-01-14")
    w2_mask = (ds["te_dates"] >= "2025-05-20") & (ds["te_dates"] <= "2025-06-19")
    for ax, mask, title in [
        (axes[0], w1_mask, "(a) Stable Winter Regime (Dec 2025–Jan 2026)"),
        (axes[1], w2_mask, "(b) Volatile Summer & Eid-ul-Adha Window (May–Jun 2025)"),
    ]:
        d_sub = ds["te_dates"][mask]
        yt_s = y_te_inv.ravel()[mask]
        ye_s = mean_preds["Weighted Ensemble"].ravel()[mask]
        yx_s = mean_preds["XGBoost"].ravel()[mask]
        yp_s = mean_preds["Persistence (Lag-1)"].ravel()[mask]
        ax.plot(d_sub, yt_s, color="#111111", lw=1.7, label="Actual")
        ax.plot(d_sub, ye_s, color="#1d3557", lw=1.5, label=f"Ensemble ({mean_absolute_error(yt_s, ye_s):.0f} MW)")
        ax.plot(d_sub, yx_s, color="#e76f51", lw=1.1, ls="--", label=f"XGBoost ({mean_absolute_error(yt_s, yx_s):.0f} MW)")
        ax.plot(d_sub, yp_s, color="#adb5bd", lw=1.0, ls=":", label=f"Persist ({mean_absolute_error(yt_s, yp_s):.0f} MW)")
        ax.set_title(title, fontsize=9.5)
        ax.set_ylabel("Peak Demand (MW)")
        ax.tick_params(axis="x", rotation=25, labelsize=7.5)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig7_qualitative_events.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.1))
    nn_names = ["CNN-BiGRU-Transformer", "DLinear", "PatchTST", "BiLSTM-Attention", "Transformer", "BiGRU", "BiLSTM", "Weighted Ensemble"]
    pts_x = [eff_profile[m]["Params"] / 1000.0 for m in nn_names]
    pts_y = [float(df_summary.loc[m, "MAE_mean"]) for m in nn_names]
    cols_p = ["#2a9d8f" if m != "Weighted Ensemble" else "#1d3557" for m in nn_names]
    sizes_p = [70 if m != "Weighted Ensemble" else 140 for m in nn_names]
    axes[0].scatter(pts_x, pts_y, c=cols_p, s=sizes_p, zorder=4, edgecolors="white")
    offsets = {
        "DLinear": (10, -10), "CNN-BiGRU-Transformer": (10, 6), "PatchTST": (10, -12),
        "BiLSTM-Attention": (10, -14), "Transformer": (10, 4),
        "BiGRU": (10, 6), "BiLSTM": (10, -12), "Weighted Ensemble": (-118, 8)
    }
    for m, px, py in zip(nn_names, pts_x, pts_y):
        dx, dy = offsets.get(m, (8, 4))
        axes[0].annotate(m, (px, py), xytext=(dx, dy), textcoords="offset points",
                         fontsize=7.6, fontweight="bold" if m == "Weighted Ensemble" else "normal")
    axes[0].set_xlabel("Trainable Parameters (Thousands)")
    axes[0].set_ylabel("Blind Test MAE (MW)")
    axes[0].set_title("(a) Accuracy vs. Parameter Count (5-Seed)")
    axes[0].set_xlim(-30, 880)
    axes[0].grid(alpha=0.25)

    lat_models = ["XGBoost", "LightGBM", "DLinear", "PatchTST", "Transformer", "BiGRU", "BiLSTM", "BiLSTM-Attention", "CNN-BiGRU-Transformer", "Weighted Ensemble"]
    lats = [eff_profile[m]["CPU_ms"] for m in lat_models]
    bar_c = ["#1d3557" if m == "Weighted Ensemble" else "#457b9d" for m in lat_models]
    axes[1].barh(lat_models, lats, color=bar_c, height=0.6)
    for i, v in enumerate(lats):
        axes[1].text(v + 0.03, i, f"{v:.2f} ms", va="center", fontsize=7.6)
    axes[1].set_xlabel("Single-Sample CPU Inference Latency (ms)")
    axes[1].set_title("(b) CPU Inference Latency Profile")
    axes[1].set_xlim(0, max(lats) * 1.25)
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig8_efficiency_pareto.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.2, 3.8))
    ax.fill_between(ds["te_dates"], cp_90["lo"], cp_90["hi"], color="#a8dadc", alpha=0.45,
                    label=f"90% Conformal PI (PICP={cp_90['PICP_%']:.1f}%, ±{cp_90['Half_Width_MW']:.0f} MW)")
    ax.fill_between(ds["te_dates"], cp_80["lo"], cp_80["hi"], color="#457b9d", alpha=0.35,
                    label=f"80% Conformal PI (PICP={cp_80['PICP_%']:.1f}%, ±{cp_80['Half_Width_MW']:.0f} MW)")
    ax.plot(ds["te_dates"], y_te_inv, color="#111111", lw=1.1, label="Actual National Peak Demand")
    ax.plot(ds["te_dates"], mean_preds["Weighted Ensemble"], color="#1d3557", lw=1.1, label="Val-Opt Ensemble (5-Seed)")
    ax.set_ylabel("Peak Demand (MW)")
    ax.set_title("Split Conformal Prediction Intervals on the 462-Day Blind Test Set (Calibrated Strictly on Validation Set)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig11_conformal_intervals.pdf")
    plt.close(fig)

    df_seeds.to_csv(f"{OUT_DIR}/part1_per_seed_metrics.csv", index=False)
    df_summary.to_csv(f"{OUT_DIR}/part1_summary_metrics.csv")
    df_dm.to_csv(f"{OUT_DIR}/part1_dm_tests.csv", index=False)
    df_cp.to_csv(f"{OUT_DIR}/part1_conformal.csv", index=False)
    df_abl.to_csv(f"{OUT_DIR}/part1_ablation.csv", index=False)
    df_reg.to_csv(f"{OUT_DIR}/part1_regime_analysis.csv", index=False)
    df_bins.to_csv(f"{OUT_DIR}/part1_demand_bins.csv", index=False)
    pd.DataFrame(eff_profile).T.to_csv(f"{OUT_DIR}/part1_efficiency.csv")

    meta_out = {
        "n_clean": ds["n_raw"], "n_features": ds["n_features"],
        "n_train": len(X_tr), "n_val": len(X_va), "n_test": len(X_te),
        "seeds": SEEDS,
        "train_span": [str(ds["train_dates"][0].date()), str(ds["train_dates"][1].date())],
        "val_span": [str(ds["val_dates"][0].date()), str(ds["val_dates"][1].date())],
        "test_span": [str(ds["test_dates"][0].date()), str(ds["test_dates"][1].date())],
        "learned_weights_per_seed": {str(s): [round(x, 4) for x in w] for s, w in zip(SEEDS, learned_weights)},
        "learned_weights_mean": [round(float(x), 4) for x in mean_w],
        "ens_5seed_mae": round(ens_5s_mae, 2),
        "efficiency": eff_profile,
    }
    with open(f"{OUT_DIR}/part1_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nALL 5-SEED GOLD BENCHMARK EXPERIMENTS COMPLETED IN {time.perf_counter() - t_start:.1f}s!", flush=True)
