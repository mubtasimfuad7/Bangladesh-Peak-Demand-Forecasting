"""
Part 1 Q1-Journal Reproducible Benchmark Suite (Round-2 Revision — Addressing All 17 Reviewer Items).
Key upgrades:
  1. Drops raw extrapolating `Year` column; estimates cooling threshold tau* = 25.0 C on LINEARLY DETRENDED
     training demand (unconfounded by 780.8 MW/yr secular growth); applies robust IQR transmission-loss QA.
  2. Equalized 24-model 5-seed benchmark (seeds 42, 123, 7, 2024, 999):
     - Naive: Persistence (Lag-1), Weekday-Adj Persistence, Seasonal Naive (Lag-7), 7-Day Rolling Mean
     - Statistical & Linear: SARIMAX(1,1,1)(1,0,1,7), Ridge (14xF), Ridge-Diff (Delta-y)
     - GBDTs: XGBoost (Level), XGBoost-Diff (Delta-y), LightGBM
     - Standard & Tabular Neural: Standard Channel-Independent DLinear (Zeng et al. 2023, 30 params),
       2-Layer MLP (14xF), PatchTST (Single, channel-mixing, no RevIN), PatchTST-Ensemble (3x),
       PatchTST-Diff (Delta-y, 3x)
     - Deep Recurrent/Attention/Hybrid: BiGRU, BiLSTM, BiLSTM-Attention, Transformer, CNN-BiGRU-Transformer,
       Hybrid BiGRU-XGBoost, Hybrid BiLSTM-Attn-XGBoost, Equal-Weight RNN Ens, Val-Opt RNN Ensemble
     - Matched Val-ES learning-rate selection {1e-3, 2.5e-3} per neural family on Seed 42.
  3. Primary Metrics: MAE (MW), RMSE (MW), MAPE (%), Skill Score vs Weekday-Adj Persistence (%), R^2.
  4. Significance & Model Confidence Set (MCS):
     - PatchTST-Ensemble (3x) as reference in HLN-DM + Holm-Bonferroni on BOTH matched single-run (Seed 42)
       and 5-seed ensemble averages.
     - Formal 95% Hansen Model Confidence Set (MCS) via moving-block bootstrap on both MAE and RMSE.
  5. 5-Way Conformal Prediction Ablation on UNTOUCHED Val-Cal residuals of BOTH PatchTST-3x and Ridge:
     Static Split vs Rolling-Window (120d) vs Locally Normalized Static vs Plain ACI vs LN-ACI
     (safe alpha_t in [0.02, 0.50] over 120 scores), plus full Seasonal Coverage & Width table.
  6. Direct Group Permutation Importance on PatchTST-3x with MERGED Autoregressive Load History block.
  7. Validation-first 5-seed Feature & PCA Ablation on PatchTST-3x and Ridge.
  8. Consistent fp32 KiB / MiB memory accounting (1 MiB = 1024 KiB) across all models.
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
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


def estimate_detrended_cooling_threshold(temp_series, demand_series):
    """Estimate piecewise cooling hinge tau* on LINEARLY DETRENDED training demand (unconfounded by secular growth)."""
    t = temp_series.values.astype(float)
    y = demand_series.values.astype(float)
    idx = np.arange(len(y), dtype=float)
    trend_slope, trend_int = np.polyfit(idx, y, 1)
    y_det = y - (trend_slope * idx + trend_int)

    best_rmse, best_tau, best_slope = float("inf"), 25.0, 0.0
    for tau in np.arange(20.0, 34.5, 0.5):
        h = np.maximum(0.0, t - tau)
        X = np.column_stack([np.ones_like(h), h])
        coef, _, _, _ = np.linalg.lstsq(X, y_det, rcond=None)
        rmse = float(np.sqrt(np.mean((y_det - X @ coef) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_tau, best_slope = rmse, float(tau), float(coef[1])
    return best_tau, best_rmse, best_slope, float(trend_slope * 365.25)


def prepare_dataset(use_cdd=True, use_eid=True, use_pca=False, tau_override=None):
    df = pd.read_excel("version_5_extended_backup.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    full_span_days = int((df["Date"].max() - df["Date"].min()).days + 1)
    n_missing_calendar = full_span_days - len(df)
    date_gaps = df["Date"].diff().dt.days.fillna(1.0)
    df["Post_Gap_Flag"] = (date_gaps > 1.0).astype(int)

    # Robust transmission-loss QA (flag both negatives and upper extreme spikes > Q3 + 3*IQR)
    gen_col = "Max. Demand at eve. peak (Generation end)"
    sub_col = "Max. Demand at eve. peak (Sub-station end)"
    raw_loss = df[gen_col] - df[sub_col]
    n_neg_loss = int((raw_loss < 0).sum())
    nonneg_loss = raw_loss[raw_loss >= 0]
    q1, q3 = np.nanpercentile(nonneg_loss, [25, 75])
    hi_cut = float(q3 + 3.0 * (q3 - q1))
    n_hi_outliers = int((raw_loss > hi_cut).sum())
    robust_loss = raw_loss[(raw_loss >= 0) & (raw_loss <= hi_cut)]

    df = df.set_index("Date")
    if "Dhaka_temp" in df.columns:
        df["Maximum Temperature in Dhaka was"] = df["Dhaka_temp"]
    if "Mymensingh_temp" not in df.columns and "Dhaka_temp" in df.columns:
        df["Mymensingh_temp"] = df["Dhaka_temp"]

    temp_col = "Maximum Temperature in Dhaka was"
    target_col = gen_col
    y_shifted = df[target_col].shift(1)

    # Drop raw extrapolating Year column!
    if "Year" in df.columns:
        df = df.drop(columns=["Year"])

    pre_train_cut = int(0.70 * len(df))
    if tau_override is None:
        tau_star, tau_rmse, tau_slope, trend_yr = estimate_detrended_cooling_threshold(
            df[temp_col].iloc[:pre_train_cut], df[target_col].iloc[:pre_train_cut]
        )
    else:
        tau_star, tau_rmse, tau_slope, trend_yr = float(tau_override), 0.0, 0.0, 0.0

    if use_cdd:
        df["CDD"] = np.maximum(0.0, df[temp_col].values - tau_star)

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

    post_gap_series = df["Post_Gap_Flag"].copy()
    df = df.drop(columns=["Post_Gap_Flag"]).dropna()
    post_gap_series = post_gap_series.loc[df.index]

    meta = df[["Holiday_cat", "Day of the week", "Holiday name", "Month", target_col, temp_col]].copy()
    meta["Post_Gap_Flag"] = post_gap_series.values
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

    sx = MinMaxScaler().fit(df_enc[cont_cols].iloc[:train_cut])
    X_cont = sx.transform(df_enc[cont_cols])
    sy = MinMaxScaler().fit(df_enc[[target_col]].iloc[:train_cut])
    y_norm = sy.transform(df_enc[[target_col]])

    X_cat = df_enc[cat_onehot_cols].values.astype(np.float32)
    X_all = np.concatenate((X_cont, X_cat), axis=1)
    feat_names = cont_cols + cat_onehot_cols + [target_col]

    if use_pca:
        pca = PCA(n_components=0.95, random_state=42).fit(X_all[:train_cut])
        X_all = pca.transform(X_all)
        feat_names = [f"PC_{i+1}" for i in range(X_all.shape[1])] + [target_col]

    data_norm = np.concatenate((X_all, y_norm), axis=1)
    X_tr, y_tr = windowing(data_norm[:train_cut], data_norm[:train_cut, -1], HIST_SIZE)
    X_va, y_va = windowing(data_norm[train_cut:val_cut], data_norm[train_cut:val_cut, -1], HIST_SIZE)
    X_te, y_te = windowing(data_norm[val_cut:], data_norm[val_cut:, -1], HIST_SIZE)

    n_va = len(X_va)
    va_es_end = (n_va - HIST_SIZE) // 2
    va_cal_start = va_es_end + HIST_SIZE

    y_va_inv = df_enc[[target_col]].iloc[train_cut + HIST_SIZE : val_cut].values
    y_te_inv = df_enc[[target_col]].iloc[val_cut + HIST_SIZE :].values
    va_dates = df_enc.index[train_cut + HIST_SIZE : val_cut]
    te_dates = df_enc.index[val_cut + HIST_SIZE :]
    va_meta = meta.iloc[train_cut + HIST_SIZE : val_cut].copy()
    te_meta = meta.iloc[val_cut + HIST_SIZE :].copy()

    # Lag-1 target in MW for each window (for Delta-y models!)
    y_full = df_enc[target_col].values
    y_tr_lag1 = y_full[HIST_SIZE - 1 : train_cut - 1].reshape(-1, 1)
    y_tr_mw = y_full[HIST_SIZE : train_cut].reshape(-1, 1)
    y_va_lag1 = y_full[train_cut + HIST_SIZE - 1 : val_cut - 1].reshape(-1, 1)
    y_te_lag1 = y_full[val_cut + HIST_SIZE - 1 : -1].reshape(-1, 1)

    dy_tr_mw = y_tr_mw - y_tr_lag1
    dy_va_mw = y_va_inv - y_va_lag1
    s_dy = StandardScaler().fit(dy_tr_mw)
    dy_tr_norm = s_dy.transform(dy_tr_mw).astype(np.float32)
    dy_va_es_norm = s_dy.transform(dy_va_mw[:va_es_end]).astype(np.float32)

    return {
        "X_tr": X_tr, "y_tr": y_tr, "y_tr_mw": y_tr_mw, "y_tr_lag1": y_tr_lag1,
        "dy_tr_mw": dy_tr_mw, "dy_tr_norm": dy_tr_norm, "s_dy": s_dy,
        "X_va": X_va, "y_va": y_va, "y_va_inv": y_va_inv, "y_va_lag1": y_va_lag1,
        "va_dates": va_dates, "va_meta": va_meta,
        "X_va_es": X_va[:va_es_end], "y_va_es": y_va[:va_es_end], "y_va_es_inv": y_va_inv[:va_es_end],
        "y_va_es_lag1": y_va_lag1[:va_es_end], "dy_va_es_mw": dy_va_mw[:va_es_end], "dy_va_es_norm": dy_va_es_norm,
        "X_va_cal": X_va[va_cal_start:], "y_va_cal": y_va[va_cal_start:], "y_va_cal_inv": y_va_inv[va_cal_start:],
        "y_va_cal_lag1": y_va_lag1[va_cal_start:],
        "X_te": X_te, "y_te": y_te, "y_te_inv": y_te_inv, "y_te_lag1": y_te_lag1,
        "te_dates": te_dates, "te_meta": te_meta,
        "sy": sy, "meta_full": meta, "feat_names": feat_names,
        "train_dates": (df_enc.index[0], df_enc.index[train_cut - 1]),
        "val_dates": (df_enc.index[train_cut], df_enc.index[val_cut - 1]),
        "test_dates": (te_dates[0], te_dates[-1]),
        "n_raw": len(df_enc), "n_features": data_norm.shape[1],
        "df_raw_target": df_enc[target_col], "val_cut": val_cut, "train_cut": train_cut,
        "tau_star": tau_star, "tau_slope": tau_slope, "trend_mw_per_yr": trend_yr,
        "data_qa": {
            "calendar_span_days": full_span_days,
            "raw_records": int(full_span_days - n_missing_calendar),
            "missing_calendar_days": int(n_missing_calendar),
            "negative_loss_records": n_neg_loss,
            "upper_outlier_loss_records": n_hi_outliers,
            "upper_iqr_cutoff_mw": round(hi_cut, 1),
            "raw_mean_loss_mw": round(float(raw_loss.mean()), 2),
            "robust_mean_loss_mw": round(float(robust_loss.mean()), 2),
            "robust_median_loss_mw": round(float(np.median(robust_loss)), 2),
            "robust_iqr_loss_mw": round(float(np.percentile(robust_loss, 75) - np.percentile(robust_loss, 25)), 2),
        },
    }


def run_sarimax_baseline(ds):
    meta = ds["meta_full"].copy()
    target_col = "Max. Demand at eve. peak (Generation end)"
    temp_col = "Maximum Temperature in Dhaka was"
    tau = ds["tau_star"]
    exog = pd.DataFrame(index=meta.index)
    t_lag1 = meta[temp_col].shift(1).bfill()
    exog["CDD_lag1"] = np.maximum(0.0, t_lag1 - tau)
    exog["Is_Weekend"] = meta["Day of the week"].isin(["Friday", "Saturday"]).astype(float)
    exog["Holiday"] = (meta["Holiday_cat"] > 0).astype(float)

    tr_c, va_c = ds["train_cut"], ds["val_cut"]
    y_all = meta[target_col].values.astype(float)
    ex_all = exog.values.astype(float)

    t0 = time.perf_counter()
    mod = SARIMAX(y_all[:tr_c], exog=ex_all[:tr_c], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                  enforce_stationarity=False, enforce_invertibility=False)
    fit_res = mod.fit(disp=False, maxiter=80)
    full_res = fit_res.apply(y_all, exog=ex_all)
    preds_all = full_res.fittedvalues
    p_te = preds_all[va_c + HIST_SIZE :].reshape(-1, 1)
    p_va = preds_all[tr_c + HIST_SIZE : va_c].reshape(-1, 1)
    lat_ms = (time.perf_counter() - t0) / len(y_all) * 1000.0
    return p_va, p_te, len(fit_res.params), round(lat_ms, 2)


def run_ridge_baseline(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_va_cal_2d, X_te_2d, sy):
    best_alpha, best_mae, best_mod = 10.0, float("inf"), None
    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]:
        mod = Ridge(alpha=alpha).fit(X_tr_2d, y_tr.ravel())
        mae = float(mean_absolute_error(y_va_es.ravel(), mod.predict(X_va_es_2d)))
        if mae < best_mae:
            best_mae, best_alpha, best_mod = mae, alpha, mod
    t0 = time.perf_counter()
    for _ in range(60): _ = best_mod.predict(X_te_2d[:1])
    lat_ms = round((time.perf_counter() - t0) / 60 * 1000.0, 2)
    p_vc = sy.inverse_transform(best_mod.predict(X_va_cal_2d).reshape(-1, 1))
    p_te = sy.inverse_transform(best_mod.predict(X_te_2d).reshape(-1, 1))
    return p_vc, p_te, best_alpha, int(X_tr_2d.shape[1] + 1), lat_ms


def run_ridge_diff_baseline(X_tr_2d, dy_tr_mw, X_va_es_2d, dy_va_es_mw, X_va_cal_2d, y_va_cal_lag1, X_te_2d, y_te_lag1):
    best_alpha, best_mae, best_mod = 100.0, float("inf"), None
    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]:
        mod = Ridge(alpha=alpha).fit(X_tr_2d, dy_tr_mw.ravel())
        mae = float(mean_absolute_error(dy_va_es_mw.ravel(), mod.predict(X_va_es_2d)))
        if mae < best_mae:
            best_mae, best_alpha, best_mod = mae, alpha, mod
    p_vc = y_va_cal_lag1 + best_mod.predict(X_va_cal_2d).reshape(-1, 1)
    p_te = y_te_lag1 + best_mod.predict(X_te_2d).reshape(-1, 1)
    return p_vc, p_te, best_alpha


# --- PyTorch Model Architectures ---
class StandardDLinearModel(nn.Module):
    """Standard Channel-Independent DLinear (Zeng et al., AAAI 2023) on the 14-day target lag channel (30 params)."""
    def __init__(self, seq_len=14, kernel_size=5):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.linear_seasonal = nn.Linear(seq_len, 1)
        self.linear_trend = nn.Linear(seq_len, 1)
    def forward(self, x):
        # Strictly target channel x[:, :, -1:] -> (B, L, 1)
        y_seq = x[:, :, -1:]
        front = y_seq[:, 0:1, :].repeat(1, self.pad, 1)
        end = y_seq[:, -1:, :].repeat(1, self.pad, 1)
        y_pad = torch.cat([front, y_seq, end], dim=1)
        trend = self.avg(y_pad.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        seasonal = y_seq.squeeze(-1) - trend
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)


class MLPModel(nn.Module):
    """2-layer Feed-Forward MLP on the identical flattened 14 x F lag matrix."""
    def __init__(self, n_feat, seq_len=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_feat, 128), nn.ReLU(), nn.Dropout(OPT_DROPOUT),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(OPT_DROPOUT),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)


class PatchTSTModel(nn.Module):
    """Channel-mixing PatchTST (no RevIN; 6 overlapping patches P=4, S=2; 2 Pre-Norm Transformer layers, d=64, 4 heads)."""
    def __init__(self, n_feat, seq_len=14, patch_len=4, stride=2, d_model=64):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len - patch_len) // stride + 1
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
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.reshape(x.size(0), self.n_patches, -1)
        z = self.patch_proj(patches) + self.pos_emb
        return self.head(self.encoder(z))


class BiGRUModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, hidden=64):
        super().__init__()
        self.rnn = nn.GRU(n_feat, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * hidden * 2, 128), nn.ReLU(),
                                  nn.Dropout(OPT_DROPOUT), nn.Linear(128, 1))
    def forward(self, x): return self.head(self.rnn(x)[0])


class BiLSTMModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, hidden=64):
        super().__init__()
        self.rnn = nn.LSTM(n_feat, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * hidden * 2, 128), nn.ReLU(),
                                  nn.Dropout(OPT_DROPOUT), nn.Linear(128, 1))
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


class TransformerModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, d_model=64):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128,
                                         dropout=OPT_DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * d_model, 128), nn.ReLU(),
                                  nn.Dropout(OPT_DROPOUT), nn.Linear(128, 1))
    def forward(self, x): return self.head(self.encoder(self.proj(x)))


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
        return self.head(self.ln(h + self.drop(a)).mean(dim=1))


def train_torch_model(model, X_tr, y_tr, X_va_es, y_va_es, seed, epochs=EPOCHS, patience=PATIENCE, lr=2.5e-3):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=OPT_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-5)
    loss_fn = nn.L1Loss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                    batch_size=BATCH_SIZE, shuffle=True)
    xva_t, yva_t = torch.from_numpy(X_va_es), torch.from_numpy(y_va_es)
    best_loss, best_state, wait = float("inf"), None, 0
    val_hist = []
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad(): vl = float(loss_fn(model(xva_t), yva_t).item())
        val_hist.append(vl)
        sched.step(vl)
        if vl < best_loss - 1e-5:
            best_loss, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state); model.eval()
    return model, val_hist, best_loss


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
    for _ in range(n_runs): _ = model(xt)
    return (time.perf_counter() - t0) / n_runs * 1000.0


def calc_metrics(y_true, y_pred, wkday_mae=None):
    yt, yp = y_true.ravel(), y_pred.ravel()
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / yt)) * 100.0)
    r2 = float(r2_score(yt, yp))
    out = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}
    if wkday_mae is not None:
        out["Skill_Wkday_%"] = (1.0 - mae / wkday_mae) * 100.0
    return out


def optimize_val_weights(y_val_cal, preds_list):
    P = np.column_stack([p.ravel() for p in preds_list])
    yv = y_val_cal.ravel()
    k = P.shape[1]
    res = minimize(lambda w: float(np.mean(np.abs(yv - P @ w))), x0=np.ones(k) / k, method="SLSQP",
                   bounds=[(0.0, 1.0)] * k, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    w = np.clip(res.x, 0.0, 1.0)
    return w / np.sum(w)


# --- 5-Way Conformal Prediction Ablation (Untouched Val-Cal Residuals!) ---
def eval_interval(method_name, model_name, yt, lo, hi, alpha):
    picp = float(np.mean((yt >= lo) & (yt <= hi)) * 100.0)
    width = float(np.mean(hi - lo))
    pinaw = float(width / (np.max(yt) - np.min(yt)) * 100.0)
    penalty = np.where(yt < lo, (2.0 / alpha) * (lo - yt),
              np.where(yt > hi, (2.0 / alpha) * (yt - hi), 0.0))
    winkler = float(np.mean((hi - lo) + penalty))
    return {
        "Model": model_name, "Method": method_name, "Nominal_%": int((1.0 - alpha) * 100),
        "Mean_Half_Width_MW": width / 2.0, "Mean_Width_MW": width,
        "PICP_%": picp, "PINAW_%": pinaw, "Winkler_MW": winkler, "lo": lo, "hi": hi
    }


def run_5way_conformal(model_name, y_cal, pred_cal, y_test, pred_test, alpha, gamma=0.05, scale_win=14, score_win=120):
    """
    5-way conformal ablation on strictly untouched Val-Cal residuals:
      1. Static Split Conformal (fixed Val-Cal quantile)
      2. Rolling-Window Conformal (120-day unweighted sliding window, no alpha update)
      3. Locally Normalized Static (14-day causal local scale sigma_t, fixed Val-Cal normalized quantile)
      4. Plain ACI (Gibbs & Candes online alpha_t in [0.02, 0.50] over 120 raw residuals, no local scale)
      5. Locally Normalized ACI (both 14-day local scale sigma_t and online alpha_t in [0.02, 0.50] over 120 scores)
    """
    res_cal = np.abs(y_cal.ravel() - pred_cal.ravel())
    yt = y_test.ravel()
    yp = pred_test.ravel()
    n_cal = len(res_cal)
    floor_sigma = max(100.0, float(np.percentile(res_cal, 25)))

    # 1. Static Split
    q_lvl = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
    q_stat = float(np.quantile(res_cal, q_lvl, method="higher"))
    r1 = eval_interval("1. Static Split Conformal", model_name, yt, yp - q_stat, yp + q_stat, alpha)

    # 2. Rolling-Window Conformal (120d)
    hist_raw = list(res_cal)
    lo2, hi2 = np.empty(len(yt)), np.empty(len(yt))
    for t in range(len(yt)):
        w_scores = np.array(hist_raw[-score_win:])
        q_t = float(np.quantile(w_scores, min(1.0, math.ceil((len(w_scores) + 1) * (1.0 - alpha)) / len(w_scores)), method="higher"))
        lo2[t], hi2[t] = yp[t] - q_t, yp[t] + q_t
        hist_raw.append(abs(yt[t] - yp[t]))
    r2 = eval_interval("2. Rolling-Window Conformal (120d)", model_name, yt, lo2, hi2, alpha)

    # Build causal normalized calibration scores
    norm_cal = []
    for i in range(n_cal):
        sig = max(floor_sigma, float(np.mean(res_cal[: max(1, i + 1)])) if i < 5 else float(np.mean(res_cal[max(0, i - scale_win) : i])))
        norm_cal.append(res_cal[i] / sig)
    q_norm_stat = float(np.quantile(norm_cal, q_lvl, method="higher"))

    # 3. Locally Normalized Static (isolates local scaling without online alpha update)
    hist_r3 = list(res_cal)
    lo3, hi3 = np.empty(len(yt)), np.empty(len(yt))
    for t in range(len(yt)):
        sig_t = max(floor_sigma, float(np.mean(hist_r3[-scale_win:])))
        hw = q_norm_stat * sig_t
        lo3[t], hi3[t] = yp[t] - hw, yp[t] + hw
        hist_r3.append(abs(yt[t] - yp[t]))
    r3 = eval_interval("3. Locally Normalized Static", model_name, yt, lo3, hi3, alpha)

    # 4. Plain ACI (isolates online alpha_t feedback without local scaling; safe alpha_t in [0.02, 0.50])
    hist_r4 = list(res_cal)
    alpha_t = alpha
    lo4, hi4 = np.empty(len(yt)), np.empty(len(yt))
    for t in range(len(yt)):
        w_scores = np.array(hist_r4[-score_win:])
        q_t = float(np.quantile(w_scores, np.clip(1.0 - alpha_t, 0.0, 1.0)))
        lo4[t], hi4[t] = yp[t] - q_t, yp[t] + q_t
        err_t = 1.0 if (yt[t] < lo4[t] or yt[t] > hi4[t]) else 0.0
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - err_t), 0.02, 0.50))
        hist_r4.append(abs(yt[t] - yp[t]))
    r4 = eval_interval("4. Plain ACI (No Local Scale)", model_name, yt, lo4, hi4, alpha)

    # 5. Locally Normalized ACI (LN-ACI)
    hist_r5 = list(res_cal)
    norm_scores = list(norm_cal)
    alpha_t = alpha
    lo5, hi5 = np.empty(len(yt)), np.empty(len(yt))
    for t in range(len(yt)):
        sig_t = max(floor_sigma, float(np.mean(hist_r5[-scale_win:])))
        w_scores = np.array(norm_scores[-score_win:])
        q_norm = float(np.quantile(w_scores, np.clip(1.0 - alpha_t, 0.0, 1.0)))
        hw = q_norm * sig_t
        lo5[t], hi5[t] = yp[t] - hw, yp[t] + hw
        err_t = 1.0 if (yt[t] < lo5[t] or yt[t] > hi5[t]) else 0.0
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - err_t), 0.02, 0.50))
        r_t = abs(yt[t] - yp[t])
        hist_r5.append(r_t)
        norm_scores.append(r_t / sig_t)
    r5 = eval_interval("5. Locally Normalized ACI (LN-ACI)", model_name, yt, lo5, hi5, alpha)
    return [r1, r2, r3, r4, r5]


def dm_hln_and_bootstrap(y_true, pred_ref, pred_other, h=1, max_lag=14, block_len=7, n_boot=1000, seed=42):
    e_a = np.abs(y_true.ravel() - pred_ref.ravel())
    e_b = np.abs(y_true.ravel() - pred_other.ravel())
    d = e_a - e_b
    T = len(d)
    d_mean = float(np.mean(d))
    var_d = np.var(d, ddof=1)
    for k in range(1, max_lag + 1):
        w = 1.0 - k / (max_lag + 1.0)
        var_d += 2.0 * w * np.cov(d[k:], d[:-k], ddof=1)[0, 1]
    dm_raw = float(d_mean / np.sqrt(max(var_d, 1e-12) / T))
    hln_factor = math.sqrt((T + 1.0 - 2.0 * h + (h * (h - 1.0)) / T) / T)
    dm_hln = dm_raw * hln_factor
    p_val = float(2.0 * stats.t.sf(np.abs(dm_hln), df=T - 1))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(T / block_len))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T - block_len + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:T]
        diffs[b] = np.mean(d[idx])
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    return dm_hln, p_val, d_mean, float(ci_lo), float(ci_hi)


def holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, max(prev, pvals[idx] * (m - rank)))
        adj[idx] = val
        prev = val
    return adj


def model_confidence_set(loss_matrix, model_names, alpha=0.05, block_len=7, n_boot=1000, seed=42):
    """
    Hansen, Lunde & Nason (2011) Model Confidence Set (MCS) via T_max statistic & moving-block bootstrap.
    Returns dict {model: mcs_p_value} where p_MCS >= alpha indicates membership in the 95% MCS.
    """
    T, M = loss_matrix.shape
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(T / block_len))
    boot_idx = np.empty((n_boot, T), dtype=int)
    for b in range(n_boot):
        starts = rng.integers(0, T - block_len + 1, size=n_blocks)
        boot_idx[b] = np.concatenate([np.arange(s, s + block_len) for s in starts])[:T]

    active = list(range(M))
    p_mcs = {}
    max_p_so_far = 0.0
    while len(active) > 1:
        L = loss_matrix[:, active]
        l_bar = L.mean(axis=0)
        l_dot = l_bar.mean()
        d_i = l_bar - l_dot
        # Bootstrap distribution of d_i
        d_boot = np.empty((n_boot, len(active)))
        for b in range(n_boot):
            lb = L[boot_idx[b]].mean(axis=0)
            d_boot[b] = (lb - lb.mean()) - d_i
        se = np.sqrt(np.maximum(np.var(d_boot, axis=0, ddof=1), 1e-12))
        t_stat = d_i / se
        t_max_obs = float(np.max(t_stat))
        t_max_boot = np.max(d_boot / se[None, :], axis=1)
        pv = float(np.mean(t_max_boot >= t_max_obs))
        max_p_so_far = max(max_p_so_far, pv)
        worst_pos = int(np.argmax(t_stat))
        worst_idx = active.pop(worst_pos)
        p_mcs[model_names[worst_idx]] = max_p_so_far
    p_mcs[model_names[active[0]]] = 1.0
    return p_mcs


def compute_patchtst3_group_permutation_importance(pt_models, X_eval, y_eval_inv, sy, feat_names, n_repeats=15, seed=42):
    """
    Direct Group Permutation Importance on the deployed PatchTST-Ensemble (3x)
    with MERGED collinear Autoregressive Load History (National + Divisional).
    """
    def eval_pt3(X_in):
        pred = np.mean([sy.inverse_transform(predict_torch(m, X_in)) for m in pt_models], axis=0)
        return float(mean_absolute_error(y_eval_inv, pred))

    base_mae = eval_pt3(X_eval)
    groups = {
        "Autoregressive Load History (National + Divisional)": [
            i for i, c in enumerate(feat_names)
            if c == "Max. Demand at eve. peak (Generation end)"
            or c.startswith("target_roll_") or c == "target_diff_1"
            or "Generation" in c or "Sub-station" in c
            or any(c.startswith(d + "_") and not c.endswith("_temp")
                   for d in ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Mymensingh", "Rangpur", "Barishal", "Sylhet", "Cumilla"])
        ],
        "Day-of-Week & Monthly Seasonality": [
            i for i, c in enumerate(feat_names)
            if c.startswith("Day of the week_") or c == "Month"
        ],
        "Holiday & Eid Lead/Lag Flags": [
            i for i, c in enumerate(feat_names)
            if c.startswith("Holiday")
        ],
        "Ambient Temperature & Detrended CDD (25°C)": [
            i for i, c in enumerate(feat_names)
            if "temp" in c.lower() or "Temperature" in c or c == "CDD"
        ],
        "Generation Constraints & Plant Maintenance": [
            i for i, c in enumerate(feat_names)
            if any(k in c for k in ["Gas", "Coal", "Kaptai", "Plants under"])
        ],
    }

    rng = np.random.default_rng(seed)
    rows = []
    for gname, idxs in groups.items():
        if not idxs: continue
        deltas = []
        for _ in range(n_repeats):
            Xp = X_eval.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, :, idxs] = Xp[perm][:, :, idxs]
            deltas.append(eval_pt3(Xp) - base_mae)
        rows.append({
            "Feature_Group": gname, "Num_Features": len(idxs),
            "Delta_MAE_Mean_MW": float(np.mean(deltas)),
            "Delta_MAE_Std_MW": float(np.std(deltas, ddof=1)),
        })
    return pd.DataFrame(rows).sort_values("Delta_MAE_Mean_MW", ascending=False)


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("=== ROUND-2 REVISION: 24 MODELS x 5 SEEDS, MCS, 5-WAY CONFORMAL ABLATION, PATCHTST-3x DOWNSTREAM ===", flush=True)
    ds = prepare_dataset(use_cdd=True, use_eid=True, use_pca=False)
    print(f"Detrended train-only tau* = {ds['tau_star']:.1f} C (slope = +{ds['tau_slope']:.1f} MW/C, trend = +{ds['trend_mw_per_yr']:.1f} MW/yr)", flush=True)
    print(f"Data QA: {ds['data_qa']}", flush=True)
    print(f"Rows: {ds['n_raw']} | Features: {ds['n_features']} (Year dropped!) | Train: {len(ds['X_tr'])} | Val-ES: {len(ds['X_va_es'])} | Val-Cal: {len(ds['X_va_cal'])} | Test: {len(ds['X_te'])}", flush=True)

    X_tr, y_tr = ds["X_tr"], ds["y_tr"]
    X_va_es, y_va_es = ds["X_va_es"], ds["y_va_es"]
    X_va_cal, y_va_cal_inv = ds["X_va_cal"], ds["y_va_cal_inv"]
    X_te, y_te_inv, sy = ds["X_te"], ds["y_te_inv"], ds["sy"]
    X_tr_2d, X_va_es_2d, X_va_cal_2d, X_te_2d = (
        X_tr.reshape(len(X_tr), -1), X_va_es.reshape(len(X_va_es), -1),
        X_va_cal.reshape(len(X_va_cal), -1), X_te.reshape(len(X_te), -1)
    )
    n_f = X_tr.shape[-1]

    # Baselines: Lag-1, Weekday-Adj Persistence, Lag-7, 7-Day Mean
    y_full = ds["df_raw_target"].values
    tc, vc = ds["train_cut"], ds["val_cut"]
    p_persist_te = ds["y_te_lag1"]
    p_lag7_te = y_full[vc + HIST_SIZE - 7 : len(y_full) - 7].reshape(-1, 1)
    p_roll7_te = pd.Series(y_full).shift(1).rolling(7).mean().values[vc + HIST_SIZE :].reshape(-1, 1)

    tr_meta_full = ds["meta_full"].iloc[:tc]
    dow_means = tr_meta_full.groupby("Day of the week")["Max. Demand at eve. peak (Generation end)"].mean().to_dict()
    all_dows = ds["meta_full"]["Day of the week"].values
    dow_step_te = np.array([dow_means[all_dows[i]] - dow_means[all_dows[i - 1]] for i in range(vc + HIST_SIZE, len(y_full))]).reshape(-1, 1)
    p_wkday_pers_te = p_persist_te + dow_step_te
    wkday_mae = float(mean_absolute_error(y_te_inv, p_wkday_pers_te))

    print("Fitting SARIMAX, Ridge (Level), and Ridge-Diff (Delta-y)...", flush=True)
    p_sarimax_va, p_sarimax_te, sarimax_p, sarimax_ms = run_sarimax_baseline(ds)
    p_ridge_vc, p_ridge_te, ridge_alpha, ridge_p, ridge_ms = run_ridge_baseline(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_va_cal_2d, X_te_2d, sy)
    p_ridged_vc, p_ridged_te, ridged_alpha = run_ridge_diff_baseline(
        X_tr_2d, ds["dy_tr_mw"], X_va_es_2d, ds["dy_va_es_mw"], X_va_cal_2d, ds["y_va_cal_lag1"], X_te_2d, ds["y_te_lag1"]
    )
    print(f"  SARIMAX:             MAE={mean_absolute_error(y_te_inv, p_sarimax_te):6.2f} MW", flush=True)
    print(f"  Ridge (alpha={ridge_alpha:g}):    MAE={mean_absolute_error(y_te_inv, p_ridge_te):6.2f} MW", flush=True)
    print(f"  Ridge-Diff (a={ridged_alpha:g}):  MAE={mean_absolute_error(y_te_inv, p_ridged_te):6.2f} MW", flush=True)
    print(f"  Weekday-Adj Persist: MAE={wkday_mae:6.2f} MW", flush=True)

    # Matched Val-ES LR tuning per neural family (Seed 42 only, strictly on Val-ES!)
    dl_builders = [
        ("DLinear (CI)", lambda: StandardDLinearModel(HIST_SIZE)),
        ("MLP (14xF)", lambda: MLPModel(n_f)),
        ("PatchTST", lambda: PatchTSTModel(n_f)),
        ("BiGRU", lambda: BiGRUModel(n_f)),
        ("BiLSTM", lambda: BiLSTMModel(n_f)),
        ("BiLSTM-Attention", lambda: BiLSTMAttnModel(n_f)),
        ("Transformer", lambda: TransformerModel(n_f)),
        ("CNN-BiGRU-Transformer", lambda: CNNBiGRUTransModel(n_f)),
    ]
    print("\nTuning learning rate per neural family strictly on Val-ES (Seed 42)...", flush=True)
    tuned_lrs = {}
    for name, fn in dl_builders:
        cands = [5e-3, 1e-2] if name == "DLinear (CI)" else [1e-3, 2.5e-3]
        best_vl, best_lr = float("inf"), cands[0]
        for lr_c in cands:
            _, _, vl = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, 42, lr=lr_c)
            if vl < best_vl:
                best_vl, best_lr = vl, lr_c
        tuned_lrs[name] = best_lr
        print(f"  {name:24s} -> best Val-ES lr = {best_lr:g} (val L1={best_vl:.4f})", flush=True)

    # Tune PatchTST-Diff LR on Val-ES
    best_vl_d, lr_pt_diff = float("inf"), 1e-3
    for lr_c in [1e-3, 2.5e-3]:
        _, _, vl = train_torch_model(PatchTSTModel(n_f), X_tr, ds["dy_tr_norm"], X_va_es, ds["dy_va_es_norm"], 42, lr=lr_c)
        if vl < best_vl_d:
            best_vl_d, lr_pt_diff = vl, lr_c
    tuned_lrs["PatchTST-Diff (3x)"] = lr_pt_diff
    print(f"  {'PatchTST-Diff (3x)':24s} -> best Val-ES lr = {lr_pt_diff:g}", flush=True)

    all_seed_metrics = []
    seed_preds = {s: {} for s in SEEDS}
    seed_val_cal_preds = {s: {} for s in SEEDS}
    learned_weights = []
    val_histories = {n: [] for n, _ in dl_builders}
    eff_profile = {
        "SARIMAX": {"Params": sarimax_p, "Size_KiB": round(sarimax_p * 4 / 1024, 3), "CPU_ms": sarimax_ms},
        "Ridge (14xF)": {"Params": ridge_p, "Size_KiB": round(ridge_p * 4 / 1024, 2), "CPU_ms": ridge_ms},
        "Ridge-Diff (Δy)": {"Params": ridge_p, "Size_KiB": round(ridge_p * 4 / 1024, 2), "CPU_ms": ridge_ms},
    }
    seed0_pt3_models = []

    for s in SEEDS:
        print(f"\n--- Running Seed {s} ---", flush=True)
        det_baselines = [
            ("Persistence (Lag-1)", p_persist_te),
            ("Weekday-Adj Persistence", p_wkday_pers_te),
            ("Seasonal Naive (Lag-7)", p_lag7_te),
            ("7-Day Rolling Mean", p_roll7_te),
            ("SARIMAX", p_sarimax_te),
            ("Ridge (14xF)", p_ridge_te),
            ("Ridge-Diff (Δy)", p_ridged_te),
        ]
        for bn, bp in det_baselines:
            seed_preds[s][bn] = bp
            all_seed_metrics.append({"Seed": s, "Model": bn, **calc_metrics(y_te_inv, bp, wkday_mae)})
        seed_val_cal_preds[s]["Ridge (14xF)"] = p_ridge_vc
        seed_val_cal_preds[s]["Ridge-Diff (Δy)"] = p_ridged_vc

        # GBDTs
        xm = xgb.XGBRegressor(random_state=s, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
        xm.fit(X_tr_2d, y_tr.ravel(), eval_set=[(X_va_es_2d, y_va_es.ravel())], verbose=False)
        p_xgb = sy.inverse_transform(xm.predict(X_te_2d).reshape(-1, 1))
        seed_preds[s]["XGBoost (Level)"] = p_xgb
        all_seed_metrics.append({"Seed": s, "Model": "XGBoost (Level)", **calc_metrics(y_te_inv, p_xgb, wkday_mae)})
        if s == SEEDS[0]:
            t0 = time.perf_counter()
            for _ in range(60): _ = xm.predict(X_te_2d[:1])
            eff_profile["XGBoost (Level)"] = {"Params": 0, "Size_KiB": 0.0, "CPU_ms": round((time.perf_counter() - t0) / 60 * 1000, 2)}

        xmd = xgb.XGBRegressor(random_state=s, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
        xmd.fit(X_tr_2d, ds["dy_tr_mw"].ravel(), eval_set=[(X_va_es_2d, ds["dy_va_es_mw"].ravel())], verbose=False)
        p_xgbd = ds["y_te_lag1"] + xmd.predict(X_te_2d).reshape(-1, 1)
        seed_preds[s]["XGBoost-Diff (Δy)"] = p_xgbd
        all_seed_metrics.append({"Seed": s, "Model": "XGBoost-Diff (Δy)", **calc_metrics(y_te_inv, p_xgbd, wkday_mae)})
        if s == SEEDS[0]:
            eff_profile["XGBoost-Diff (Δy)"] = eff_profile["XGBoost (Level)"].copy()

        lm = lgb.LGBMRegressor(objective="regression_l1", random_state=s, n_estimators=500,
                               learning_rate=0.05, num_leaves=31, n_jobs=1, verbose=-1)
        lm.fit(X_tr_2d, y_tr.ravel(), eval_set=[(X_va_es_2d, y_va_es.ravel())],
               callbacks=[lgb.early_stopping(8, verbose=False)])
        p_lgb = sy.inverse_transform(lm.predict(X_te_2d).reshape(-1, 1))
        seed_preds[s]["LightGBM"] = p_lgb
        all_seed_metrics.append({"Seed": s, "Model": "LightGBM", **calc_metrics(y_te_inv, p_lgb, wkday_mae)})
        if s == SEEDS[0]:
            t0 = time.perf_counter()
            for _ in range(60): _ = lm.predict(X_te_2d[:1])
            eff_profile["LightGBM"] = {"Params": 0, "Size_KiB": 0.0, "CPU_ms": round((time.perf_counter() - t0) / 60 * 1000, 2)}

        # Neural models with matched Val-ES tuned LR
        trained_dl = {}
        for name, fn in dl_builders:
            mod, vh, _ = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, s, lr=tuned_lrs[name])
            val_histories[name].append(vh)
            pred_vc = sy.inverse_transform(predict_torch(mod, X_va_cal))
            pred_te = sy.inverse_transform(predict_torch(mod, X_te))
            seed_val_cal_preds[s][name] = pred_vc
            seed_preds[s][name] = pred_te
            trained_dl[name] = mod
            mr = calc_metrics(y_te_inv, pred_te, wkday_mae)
            print(f"  [Seed {s}] {name:24s}: MAE={mr['MAE']:6.1f} | RMSE={mr['RMSE']:6.1f} | SS={mr['Skill_Wkday_%']:+5.1f}%", flush=True)
            all_seed_metrics.append({"Seed": s, "Model": name, **mr})
            if s == SEEDS[0]:
                np_cnt = sum(p.numel() for p in mod.parameters())
                eff_profile[name] = {
                    "Params": np_cnt, "Size_KiB": round(np_cnt * 4 / 1024, 2),
                    "CPU_ms": round(measure_torch_latency(mod, X_te[:1]), 2)
                }

        # Equal-Budget 3-Member PatchTST-Ensemble (Level)
        pt_vc_list = [seed_val_cal_preds[s]["PatchTST"]]
        pt_te_list = [seed_preds[s]["PatchTST"]]
        if s == SEEDS[0]:
            seed0_pt3_models = [trained_dl["PatchTST"]]
        for offset in [1000, 2000]:
            m_pt_extra, _, _ = train_torch_model(PatchTSTModel(n_f), X_tr, y_tr, X_va_es, y_va_es, s + offset, lr=tuned_lrs["PatchTST"])
            pt_vc_list.append(sy.inverse_transform(predict_torch(m_pt_extra, X_va_cal)))
            pt_te_list.append(sy.inverse_transform(predict_torch(m_pt_extra, X_te)))
            if s == SEEDS[0]:
                seed0_pt3_models.append(m_pt_extra)
        p_pt3_vc = np.mean(pt_vc_list, axis=0)
        p_pt3_te = np.mean(pt_te_list, axis=0)
        seed_val_cal_preds[s]["PatchTST-Ensemble (3x)"] = p_pt3_vc
        seed_preds[s]["PatchTST-Ensemble (3x)"] = p_pt3_te
        m_pt3 = calc_metrics(y_te_inv, p_pt3_te, wkday_mae)
        print(f"  [Seed {s}] {'PatchTST-Ensemble (3x)':24s}: MAE={m_pt3['MAE']:6.1f} | RMSE={m_pt3['RMSE']:6.1f} | SS={m_pt3['Skill_Wkday_%']:+5.1f}%", flush=True)
        all_seed_metrics.append({"Seed": s, "Model": "PatchTST-Ensemble (3x)", **m_pt3})
        if s == SEEDS[0]:
            eff_profile["PatchTST-Ensemble (3x)"] = {
                "Params": eff_profile["PatchTST"]["Params"] * 3,
                "Size_KiB": round(eff_profile["PatchTST"]["Params"] * 3 * 4 / 1024, 2),
                "CPU_ms": round(eff_profile["PatchTST"]["CPU_ms"] * 3, 2),
            }

        # PatchTST-Diff (Delta-y, 3-member ensemble — gives neural models the same Delta-y option!)
        ptd_vc_list, ptd_te_list = [], []
        for offset in [0, 1000, 2000]:
            m_ptd, _, _ = train_torch_model(PatchTSTModel(n_f), X_tr, ds["dy_tr_norm"], X_va_es, ds["dy_va_es_norm"], s + offset, lr=lr_pt_diff)
            ptd_vc_list.append(ds["y_va_cal_lag1"] + ds["s_dy"].inverse_transform(predict_torch(m_ptd, X_va_cal)))
            ptd_te_list.append(ds["y_te_lag1"] + ds["s_dy"].inverse_transform(predict_torch(m_ptd, X_te)))
        p_ptd_vc = np.mean(ptd_vc_list, axis=0)
        p_ptd_te = np.mean(ptd_te_list, axis=0)
        seed_val_cal_preds[s]["PatchTST-Diff (Δy, 3x)"] = p_ptd_vc
        seed_preds[s]["PatchTST-Diff (Δy, 3x)"] = p_ptd_te
        m_ptd = calc_metrics(y_te_inv, p_ptd_te, wkday_mae)
        print(f"  [Seed {s}] {'PatchTST-Diff (Δy, 3x)':24s}: MAE={m_ptd['MAE']:6.1f} | RMSE={m_ptd['RMSE']:6.1f} | SS={m_ptd['Skill_Wkday_%']:+5.1f}%", flush=True)
        all_seed_metrics.append({"Seed": s, "Model": "PatchTST-Diff (Δy, 3x)", **m_ptd})
        if s == SEEDS[0]:
            eff_profile["PatchTST-Diff (Δy, 3x)"] = eff_profile["PatchTST-Ensemble (3x)"].copy()

        # Residual Hybrids
        for base_name, h_name in [("BiGRU", "Hybrid BiGRU-XGBoost"), ("BiLSTM-Attention", "Hybrid BiLSTM-Attn-XGBoost")]:
            bm = trained_dl[base_name]
            res_tr = y_tr - predict_torch(bm, X_tr)
            res_va = y_va_es - predict_torch(bm, X_va_es)
            hx = xgb.XGBRegressor(random_state=s, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
            hx.fit(X_tr_2d, res_tr.ravel(), eval_set=[(X_va_es_2d, res_va.ravel())], verbose=False)
            pred_h = sy.inverse_transform(predict_torch(bm, X_te) + hx.predict(X_te_2d).reshape(-1, 1))
            seed_preds[s][h_name] = pred_h
            all_seed_metrics.append({"Seed": s, "Model": h_name, **calc_metrics(y_te_inv, pred_h, wkday_mae)})

        # Equal-Weight RNN Ensemble (3-network)
        p_eq_vc = (seed_val_cal_preds[s]["BiGRU"] + seed_val_cal_preds[s]["BiLSTM-Attention"] + seed_val_cal_preds[s]["BiLSTM"]) / 3.0
        p_eq_te = (seed_preds[s]["BiGRU"] + seed_preds[s]["BiLSTM-Attention"] + seed_preds[s]["BiLSTM"]) / 3.0
        seed_val_cal_preds[s]["Equal-Weight RNN Ens"] = p_eq_vc
        seed_preds[s]["Equal-Weight RNN Ens"] = p_eq_te
        meq = calc_metrics(y_te_inv, p_eq_te, wkday_mae)
        print(f"  [Seed {s}] {'Equal-Weight RNN Ens':24s}: MAE={meq['MAE']:6.1f} | RMSE={meq['RMSE']:6.1f} | SS={meq['Skill_Wkday_%']:+5.1f}%", flush=True)
        all_seed_metrics.append({"Seed": s, "Model": "Equal-Weight RNN Ens", **meq})

        # Val-Cal SLSQP Weighted RNN Ensemble
        w_opt = optimize_val_weights(y_va_cal_inv, [seed_val_cal_preds[s]["BiGRU"], seed_val_cal_preds[s]["BiLSTM-Attention"], seed_val_cal_preds[s]["BiLSTM"]])
        learned_weights.append(w_opt.tolist())
        p_opt_vc = w_opt[0] * seed_val_cal_preds[s]["BiGRU"] + w_opt[1] * seed_val_cal_preds[s]["BiLSTM-Attention"] + w_opt[2] * seed_val_cal_preds[s]["BiLSTM"]
        p_opt_te = w_opt[0] * seed_preds[s]["BiGRU"] + w_opt[1] * seed_preds[s]["BiLSTM-Attention"] + w_opt[2] * seed_preds[s]["BiLSTM"]
        seed_val_cal_preds[s]["Val-Opt RNN Ensemble"] = p_opt_vc
        seed_preds[s]["Val-Opt RNN Ensemble"] = p_opt_te
        all_seed_metrics.append({"Seed": s, "Model": "Val-Opt RNN Ensemble", **calc_metrics(y_te_inv, p_opt_te, wkday_mae)})

        if s == SEEDS[0]:
            tot_p = eff_profile["BiGRU"]["Params"] + eff_profile["BiLSTM-Attention"]["Params"] + eff_profile["BiLSTM"]["Params"]
            tot_ms = eff_profile["BiGRU"]["CPU_ms"] + eff_profile["BiLSTM-Attention"]["CPU_ms"] + eff_profile["BiLSTM"]["CPU_ms"]
            eff_profile["Equal-Weight RNN Ens"] = {"Params": tot_p, "Size_KiB": round(tot_p * 4 / 1024, 2), "CPU_ms": round(tot_ms, 2)}
            eff_profile["Val-Opt RNN Ensemble"] = eff_profile["Equal-Weight RNN Ens"].copy()

    df_seeds = pd.DataFrame(all_seed_metrics)
    df_summary = df_seeds.groupby("Model").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
        Skill_Wkday_mean=("Skill_Wkday_%", "mean"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    ).sort_values("MAE_mean")
    print("\n=== 1. ROUND-2 24-MODEL 5-SEED BENCHMARK ===")
    print(df_summary.round(2).to_string())

    # 5-seed ensemble averages + Matched Seed-42 HLN-DM with PatchTST-Ensemble (3x) as Reference!
    mean_preds = {m: np.mean([seed_preds[s][m] for s in SEEDS], axis=0) for m in df_summary.index}
    mean_vc_pt3 = np.mean([seed_val_cal_preds[s]["PatchTST-Ensemble (3x)"] for s in SEEDS], axis=0)
    ref_name = "PatchTST-Ensemble (3x)"

    dm_rows = []
    for m in df_summary.index:
        if m == ref_name: continue
        dm_s0, p_s0, d_s0, lo_s0, hi_s0 = dm_hln_and_bootstrap(y_te_inv, seed_preds[42][ref_name], seed_preds[42][m])
        dm_5s, p_5s, d_5s, lo_5s, hi_5s = dm_hln_and_bootstrap(y_te_inv, mean_preds[ref_name], mean_preds[m])
        dm_rows.append({
            "Comparator": m,
            "S42_MAE": float(mean_absolute_error(y_te_inv, seed_preds[42][m])),
            "S42_Delta_MW": d_s0, "S42_DM_HLN": dm_s0, "S42_p": p_s0,
            "Ens5_MAE": float(mean_absolute_error(y_te_inv, mean_preds[m])),
            "Ens5_Delta_MW": d_5s, "Ens5_DM_HLN": dm_5s, "Ens5_p": p_5s,
            "Ens5_CI_lo": lo_5s, "Ens5_CI_hi": hi_5s,
        })
    df_dm = pd.DataFrame(dm_rows).sort_values("Ens5_Delta_MW").reset_index(drop=True)
    df_dm["S42_p_holm"] = holm_bonferroni(df_dm["S42_p"].values)
    df_dm["Ens5_p_holm"] = holm_bonferroni(df_dm["Ens5_p"].values)
    print(f"\n=== 2. HLN-DM TESTS (REF = {ref_name}: S42={mean_absolute_error(y_te_inv, seed_preds[42][ref_name]):.2f}, 5-Seed={mean_absolute_error(y_te_inv, mean_preds[ref_name]):.2f} MW) ===")
    print(df_dm[["Comparator", "S42_MAE", "S42_Delta_MW", "S42_p_holm", "Ens5_MAE", "Ens5_Delta_MW", "Ens5_p_holm"]].round(4).to_string(index=False))

    # Formal 95% Model Confidence Set (MCS) across top 12 models on BOTH MAE and RMSE (Seed 42 & 5-Seed)
    mcs_models = list(df_summary.index[:12])
    L_mae_s42 = np.column_stack([np.abs(y_te_inv.ravel() - seed_preds[42][m].ravel()) for m in mcs_models])
    L_mse_s42 = np.column_stack([(y_te_inv.ravel() - seed_preds[42][m].ravel()) ** 2 for m in mcs_models])
    L_mae_5s = np.column_stack([np.abs(y_te_inv.ravel() - mean_preds[m].ravel()) for m in mcs_models])
    L_mse_5s = np.column_stack([(y_te_inv.ravel() - mean_preds[m].ravel()) ** 2 for m in mcs_models])

    mcs_mae_s42 = model_confidence_set(L_mae_s42, mcs_models)
    mcs_mse_s42 = model_confidence_set(L_mse_s42, mcs_models)
    mcs_mae_5s = model_confidence_set(L_mae_5s, mcs_models)
    mcs_mse_5s = model_confidence_set(L_mse_5s, mcs_models)
    df_mcs = pd.DataFrame([{
        "Model": m,
        "S42_MAE_pMCS": mcs_mae_s42[m], "S42_RMSE_pMCS": mcs_mse_s42[m],
        "Ens5_MAE_pMCS": mcs_mae_5s[m], "Ens5_RMSE_pMCS": mcs_mse_5s[m],
        "In_95_MCS_MAE_S42": mcs_mae_s42[m] >= 0.05,
        "In_95_MCS_RMSE_S42": mcs_mse_s42[m] >= 0.05,
    } for m in mcs_models])
    print("\n=== 3. 95% HANSEN MODEL CONFIDENCE SET (MCS, BLOCK BOOTSTRAP) ===")
    print(df_mcs.round(3).to_string(index=False))

    # 5-Way Conformal Prediction Ablation on UNTOUCHED Val-Cal residuals of BOTH PatchTST-3x and Ridge!
    cp_rows = []
    cp_store_pt3 = {}
    for m_lbl, vc_pred, te_pred in [
        ("PatchTST-Ensemble (3x)", mean_vc_pt3, mean_preds["PatchTST-Ensemble (3x)"]),
        ("Ridge (14xF)", p_ridge_vc, p_ridge_te),
    ]:
        for alpha in [0.20, 0.10]:
            res_list = run_5way_conformal(m_lbl, y_va_cal_inv, vc_pred, y_te_inv, te_pred, alpha)
            for r in res_list:
                cp_rows.append({k: v for k, v in r.items() if k not in ("lo", "hi")})
                if m_lbl == "PatchTST-Ensemble (3x)" and alpha == 0.10:
                    cp_store_pt3[r["Method"]] = r
    df_cp = pd.DataFrame(cp_rows)
    print("\n=== 4. 5-WAY CONFORMAL ABLATION (UNTOUCHED VAL-CAL RESIDUALS: PATCHTST-3x & RIDGE) ===")
    print(df_cp.round(2).to_string(index=False))

    # Seasonal Coverage & Width Table (90% Nominal on PatchTST-3x across all 5 conformal methods!)
    def map_season(m):
        if m in [3, 4, 5]: return "Summer (Mar-May)"
        if m in [6, 7, 8, 9]: return "Monsoon (Jun-Sep)"
        if m in [10, 11]: return "Post-Monsoon (Oct-Nov)"
        return "Winter (Dec-Feb)"

    te_seasons = ds["te_meta"]["Month"].apply(map_season).values
    yt_flat = y_te_inv.ravel()
    sea_rows = []
    for m_name, r_obj in cp_store_pt3.items():
        row = {"Method": m_name, "Overall_PICP_%": r_obj["PICP_%"], "Overall_Width_MW": r_obj["Mean_Width_MW"], "Winkler_MW": r_obj["Winkler_MW"]}
        for sea, short in [("Winter (Dec-Feb)", "Winter"), ("Post-Monsoon (Oct-Nov)", "PostMon"), ("Monsoon (Jun-Sep)", "Monsoon"), ("Summer (Mar-May)", "Summer")]:
            ms = te_seasons == sea
            row[f"{short}_PICP_%"] = float(np.mean((yt_flat[ms] >= r_obj["lo"][ms]) & (yt_flat[ms] <= r_obj["hi"][ms])) * 100.0)
            row[f"{short}_Width_MW"] = float(np.mean(r_obj["hi"][ms] - r_obj["lo"][ms]))
        sea_rows.append(row)
    df_cp_sea = pd.DataFrame(sea_rows)
    print("\n=== 5. SEASONAL 90% CONFORMAL COVERAGE & WIDTH (PATCHTST-3x) ===")
    print(df_cp_sea.round(1).to_string(index=False))

    # Direct Group Permutation Importance on deployed PatchTST-Ensemble (3x) (MERGED load history block!)
    print("\nComputing Direct Group Permutation Importance on PatchTST-Ensemble (3x)...", flush=True)
    df_perm = compute_patchtst3_group_permutation_importance(seed0_pt3_models, X_te, y_te_inv, sy, ds["feat_names"], n_repeats=15, seed=42)
    print(df_perm.round(2).to_string(index=False))

    # Validation-First 5-Seed Feature & PCA Ablation on PatchTST-Ensemble (3x) AND Ridge!
    print("\nRunning Validation-First 5-Seed Feature Ablation on PatchTST-Ensemble (3x) & Ridge...", flush=True)
    ablation_configs = [
        ("1. Base (No CDD_25, No Eid Flags)", dict(use_cdd=False, use_eid=False, use_pca=False)),
        ("2. + Detrended CDD_25", dict(use_cdd=True, use_eid=False, use_pca=False)),
        ("3. + Eid Lead/Lag (Full No-PCA)", dict(use_cdd=True, use_eid=True, use_pca=False)),
        ("4. Full + PCA (95% Variance)", dict(use_cdd=True, use_eid=True, use_pca=True)),
    ]
    hol_mask = (ds["te_meta"]["Holiday_cat"].values > 0)
    abl_rows, abl_seed_maes = [], {}
    for label, cfg in ablation_configs:
        if cfg == dict(use_cdd=True, use_eid=True, use_pca=False):
            vc_maes = [float(mean_absolute_error(y_va_cal_inv, seed_val_cal_preds[s]["PatchTST-Ensemble (3x)"])) for s in SEEDS]
            te_maes = [float(mean_absolute_error(y_te_inv, seed_preds[s]["PatchTST-Ensemble (3x)"])) for s in SEEDS]
            hol_maes = [float(mean_absolute_error(y_te_inv[hol_mask], seed_preds[s]["PatchTST-Ensemble (3x)"][hol_mask])) for s in SEEDS]
            r_vc_mae = float(mean_absolute_error(y_va_cal_inv, p_ridge_vc))
            r_te_mae = float(mean_absolute_error(y_te_inv, p_ridge_te))
            dims = ds["n_features"]
        else:
            dsi = prepare_dataset(**cfg, tau_override=ds["tau_star"])
            dims = dsi["n_features"]
            hm = (dsi["te_meta"]["Holiday_cat"].values > 0)
            p_rv, p_rt, _, _, _ = run_ridge_baseline(
                dsi["X_tr"].reshape(len(dsi["X_tr"]), -1), dsi["y_tr"],
                dsi["X_va_es"].reshape(len(dsi["X_va_es"]), -1), dsi["y_va_es"],
                dsi["X_va_cal"].reshape(len(dsi["X_va_cal"]), -1), dsi["X_te"].reshape(len(dsi["X_te"]), -1), dsi["sy"]
            )
            r_vc_mae = float(mean_absolute_error(dsi["y_va_cal_inv"], p_rv))
            r_te_mae = float(mean_absolute_error(dsi["y_te_inv"], p_rt))
            vc_maes, te_maes, hol_maes = [], [], []
            for s in SEEDS:
                v_l, t_l = [], []
                for off in [0, 1000, 2000]:
                    m, _, _ = train_torch_model(PatchTSTModel(dims), dsi["X_tr"], dsi["y_tr"], dsi["X_va_es"], dsi["y_va_es"], s + off, lr=tuned_lrs["PatchTST"])
                    v_l.append(dsi["sy"].inverse_transform(predict_torch(m, dsi["X_va_cal"])))
                    t_l.append(dsi["sy"].inverse_transform(predict_torch(m, dsi["X_te"])))
                pv, pt = np.mean(v_l, axis=0), np.mean(t_l, axis=0)
                vc_maes.append(float(mean_absolute_error(dsi["y_va_cal_inv"], pv)))
                te_maes.append(float(mean_absolute_error(dsi["y_te_inv"], pt)))
                hol_maes.append(float(mean_absolute_error(dsi["y_te_inv"][hm], pt[hm])))
        abl_seed_maes[label] = te_maes
        abl_rows.append({
            "Config": label, "Dims": dims,
            "PT3_ValCal_MAE": float(np.mean(vc_maes)),
            "PT3_Test_MAE": float(np.mean(te_maes)), "PT3_Test_SD": float(np.std(te_maes, ddof=1)),
            "PT3_Hol_MAE": float(np.mean(hol_maes)),
            "Ridge_ValCal_MAE": r_vc_mae, "Ridge_Test_MAE": r_te_mae,
        })
    df_abl = pd.DataFrame(abl_rows)
    base_vec = np.array(abl_seed_maes[ablation_configs[0][0]])
    df_abl["p_paired_vs_Base"] = [1.0] + [float(stats.ttest_rel(abl_seed_maes[lbl], base_vec)[1]) for lbl, _ in ablation_configs[1:]]
    print("\n=== 6. VALIDATION-FIRST 5-SEED ABLATION (PATCHTST-3x & RIDGE) ===")
    print(df_abl.round(3).to_string(index=False))

    # Regime & Demand-Bin Stratification (including PatchTST-3x, Ridge, Eq-Weight RNN, XGB-Diff, Wkday Persist!)
    te_m = ds["te_meta"].copy()
    te_m["Actual"] = y_te_inv.ravel()
    te_m["Pred_PT3"] = mean_preds["PatchTST-Ensemble (3x)"].ravel()
    te_m["Pred_Ridge"] = mean_preds["Ridge (14xF)"].ravel()
    te_m["Pred_EqRNN"] = mean_preds["Equal-Weight RNN Ens"].ravel()
    te_m["Pred_XGBD"] = mean_preds["XGBoost-Diff (Δy)"].ravel()
    te_m["Pred_XGB"] = mean_preds["XGBoost (Level)"].ravel()
    te_m["Pred_Wkday"] = mean_preds["Weekday-Adj Persistence"].ravel()
    te_m["Pred_Persist"] = mean_preds["Persistence (Lag-1)"].ravel()
    te_m["Season"] = te_m["Month"].apply(map_season)
    te_m["Is_Weekend"] = te_m["Day of the week"].isin(["Friday", "Saturday"])
    te_m["Cal_Regime"] = np.where(te_m["Holiday_cat"] > 0, "Holiday / Eid",
                         np.where(te_m["Is_Weekend"], "Weekend (Fri-Sat)", "Working Weekday"))

    regime_rows = []
    for col in ["Season", "Cal_Regime"]:
        for grp, g in te_m.groupby(col):
            regime_rows.append({
                "Axis": col, "Regime": grp, "N": len(g),
                "PT3_MAE": float(mean_absolute_error(g["Actual"], g["Pred_PT3"])),
                "Ridge_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Ridge"])),
                "EqRNN_MAE": float(mean_absolute_error(g["Actual"], g["Pred_EqRNN"])),
                "XGBD_MAE": float(mean_absolute_error(g["Actual"], g["Pred_XGBD"])),
                "Wkday_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Wkday"])),
                "Persist_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Persist"])),
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
            "PT3_MAE": float(mean_absolute_error(g["Actual"], g["Pred_PT3"])),
            "Ridge_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Ridge"])),
            "EqRNN_MAE": float(mean_absolute_error(g["Actual"], g["Pred_EqRNN"])),
            "XGBD_MAE": float(mean_absolute_error(g["Actual"], g["Pred_XGBD"])),
            "XGB_MAE": float(mean_absolute_error(g["Actual"], g["Pred_XGB"])),
            "Wkday_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Wkday"])),
        })
    df_bins = pd.DataFrame(bin_rows)
    print("\n=== 7. REGIME & DEMAND-BIN BREAKDOWN ===")
    print(df_reg.round(1).to_string(index=False))
    print(df_bins.round(1).to_string(index=False))

    # ==========================================================================
    # REGENERATE CLEAN, ZERO-OVERLAP VECTOR PDF FIGURES
    # ==========================================================================
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    # Fig 1: Forecast comparison (Tied Leading Group: PatchTST-3x, Ridge, Equal-Weight RNN)
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.plot(ds["te_dates"], y_te_inv, color="#111111", lw=1.35, label="Actual Peak Demand")
    ax.plot(ds["te_dates"], mean_preds["PatchTST-Ensemble (3x)"], color="#0d9488", lw=1.25, label="PatchTST-3×")
    ax.plot(ds["te_dates"], mean_preds["Ridge (14xF)"], color="#d97706", lw=1.1, ls="--", alpha=0.85, label="Ridge (14×F)")
    ax.plot(ds["te_dates"], mean_preds["Equal-Weight RNN Ens"], color="#1d3557", lw=1.1, ls="-.", alpha=0.85, label="Equal-Weight RNN Ens")
    ax.set_ylabel("Peak Demand (MW)")
    ax.set_title("National Evening Peak Demand — Leading Group Comparison (462 Blind Test Days)", pad=28)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=8.2)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig1_forecast_comparison.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 2: Convergence curves (8 neural families)
    fig, ax = plt.subplots(figsize=(8.6, 4.7))
    colors = {"DLinear (CI)": "#6d597a", "MLP (14xF)": "#84a98c", "PatchTST": "#0284c7", "BiGRU": "#1d3557",
              "BiLSTM": "#d97706", "BiLSTM-Attention": "#2a9d8f", "Transformer": "#e76f51", "CNN-BiGRU-Transformer": "#9b2226"}
    for name, hlist in val_histories.items():
        max_len = max(len(h) for h in hlist)
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in hlist])
        mu, sd = padded.mean(axis=0), padded.std(axis=0)
        ep = np.arange(1, max_len + 1)
        ax.plot(ep, mu, lw=1.6, color=colors[name], label=name)
        ax.fill_between(ep, mu - sd, mu + sd, color=colors[name], alpha=0.10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val-ES Loss (Normalized MAE)")
    ax.set_title("Multi-Seed Val-ES Convergence Across Neural Families (Mean ± 1 SD, 5 Seeds)", pad=38)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=7.6)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig2_convergence_curves.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 3: HLN-DM Forest Plot (Ref = PatchTST-Ensemble 3x!)
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    y_pos = np.arange(len(df_dm))
    c_bars = ["#0d9488" if p < 0.01 else ("#2563eb" if p < 0.05 else "#94a3b8") for p in df_dm["Ens5_p_holm"]]
    ax.axvline(0, color="black", ls="--", lw=1.0)
    for i, (_, r) in enumerate(df_dm.iterrows()):
        ax.plot([r["Ens5_CI_lo"], r["Ens5_CI_hi"]], [i, i], color=c_bars[i], lw=2.1)
        ax.plot(r["Ens5_Delta_MW"], i, "o", color=c_bars[i], ms=5.2)
    ax.set_yticks(y_pos); ax.set_yticklabels(df_dm["Comparator"], fontsize=8.0)
    ax.set_xlabel("ΔMAE: PatchTST-Ensemble (3×) − Comparator (MW)  [Negative = PatchTST-3× Lower Error]")
    ax.set_title("HLN-Adjusted Diebold–Mariano Tests (Ref: PatchTST-3×; Holm–Bonferroni Adjusted)", pad=28)
    ax.grid(axis="x", alpha=0.25)
    p1 = mpatches.Patch(color="#0d9488", label="p_Holm < 0.01")
    p2 = mpatches.Patch(color="#2563eb", label="p_Holm < 0.05")
    p3 = mpatches.Patch(color="#94a3b8", label="p_Holm ≥ 0.05 (Tied / ns)")
    ax.legend(handles=[p1, p2, p3], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig3_dm_forest.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 4: Feature Ablation (PatchTST-3x & Ridge) & Seasonal Regime Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))
    ax = axes[0]
    short_lbl = ["Base", "+CDD_25", "+Eid Flags\n(Full)", "Full + PCA\n(95%)"]
    x_a = np.arange(len(short_lbl)); wa = 0.34
    ax.bar(x_a - wa/2, df_abl["PT3_Test_MAE"], yerr=df_abl["PT3_Test_SD"], capsize=3.5, width=wa, color="#0d9488", label="PatchTST-3× (5-Seed)")
    ax.bar(x_a + wa/2, df_abl["Ridge_Test_MAE"], width=wa, color="#d97706", label="Ridge (14×F)")
    for i, r in df_abl.iterrows():
        ax.text(i - wa/2, r["PT3_Test_MAE"] + r["PT3_Test_SD"] + 8, f"{r['PT3_Test_MAE']:.0f}", ha="center", fontsize=7.5, color="#0f766e")
        ax.text(i + wa/2, r["Ridge_Test_MAE"] + 8, f"{r['Ridge_Test_MAE']:.0f}", ha="center", fontsize=7.5, color="#b45309")
    ax.set_xticks(x_a); ax.set_xticklabels(short_lbl, fontsize=8.3)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_ylim(0, max(df_abl["PT3_Test_MAE"].max(), df_abl["Ridge_Test_MAE"].max()) * 1.18)
    ax.set_title("(a) Validation-First Feature & PCA Ablation", pad=28)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False, fontsize=7.8)

    ax = axes[1]
    s_reg = df_reg[df_reg["Axis"] == "Season"]
    x = np.arange(len(s_reg)); w = 0.19
    ax.bar(x - 1.5*w, s_reg["Wkday_MAE"], width=w, color="#94a3b8", label="Wkday Persist")
    ax.bar(x - 0.5*w, s_reg["Ridge_MAE"], width=w, color="#d97706", label="Ridge (14×F)")
    ax.bar(x + 0.5*w, s_reg["EqRNN_MAE"], width=w, color="#1d3557", label="Eq-Wt RNN Ens")
    ax.bar(x + 1.5*w, s_reg["PT3_MAE"], width=w, color="#0d9488", label="PatchTST-3×")
    ax.set_xticks(x); ax.set_xticklabels([s.split(" ")[0] for s in s_reg["Regime"]], fontsize=8.5)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_ylim(0, s_reg[["Wkday_MAE", "Ridge_MAE", "EqRNN_MAE", "PT3_MAE"]].values.max() * 1.15)
    ax.set_title("(b) Seasonal Regime Error Breakdown", pad=28)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=7.4)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig4_ablation_regime.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 6: Demand Bins
    fig, ax = plt.subplots(figsize=(9.4, 4.4))
    x = np.arange(len(df_bins)); w = 0.15
    ax.bar(x - 2*w, df_bins["Wkday_MAE"], width=w, color="#94a3b8", label="Wkday Persist")
    ax.bar(x - w, df_bins["XGB_MAE"], width=w, color="#e76f51", label="XGBoost (Level)")
    ax.bar(x, df_bins["XGBD_MAE"], width=w, color="#f4a261", label="XGBoost-Diff (Δy)")
    ax.bar(x + w, df_bins["Ridge_MAE"], width=w, color="#d97706", label="Ridge (14×F)")
    ax.bar(x + 2*w, df_bins["PT3_MAE"], width=w, color="#0d9488", label="PatchTST-3×")
    for i, r in df_bins.iterrows():
        ax.text(i + 2*w, r["PT3_MAE"] + 16, f"{r['PT3_MAE']:.0f}", ha="center", fontsize=7.2, fontweight="bold", color="#0f766e")
        ax.text(i, r["XGBD_MAE"] + 16, f"{r['XGBD_MAE']:.0f}", ha="center", fontsize=7.0, color="#b45309")
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(df_bins["Demand_Bin"], df_bins["N"])], fontsize=8.5)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_ylim(0, df_bins[["Wkday_MAE", "XGB_MAE", "XGBD_MAE", "Ridge_MAE", "PT3_MAE"]].values.max() * 1.15)
    ax.set_title("Peak Demand-Level Error Stratification Across Leading Models & Differenced Trees", pad=28)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False, fontsize=7.6)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig6_demand_bins.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 7: Qualitative events (highlighting Holiday/Eid XGB-Diff strength too!)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    w1_mask = (ds["te_dates"] >= "2025-12-15") & (ds["te_dates"] <= "2026-01-14")
    w2_mask = (ds["te_dates"] >= "2025-05-20") & (ds["te_dates"] <= "2025-06-19")
    for ax, mask, title in [
        (axes[0], w1_mask, "(a) Stable Winter Regime (Dec 2025–Jan 2026)"),
        (axes[1], w2_mask, "(b) Volatile Summer & Eid-ul-Adha Window (May–Jun 2025)"),
    ]:
        d_sub = ds["te_dates"][mask]
        yt_s = y_te_inv.ravel()[mask]
        ypt_s = mean_preds["PatchTST-Ensemble (3x)"].ravel()[mask]
        yr_s = mean_preds["Ridge (14xF)"].ravel()[mask]
        yxd_s = mean_preds["XGBoost-Diff (Δy)"].ravel()[mask]
        ax.plot(d_sub, yt_s, color="#111111", lw=1.7, label="Actual")
        ax.plot(d_sub, ypt_s, color="#0d9488", lw=1.45, label=f"PatchTST-3× ({mean_absolute_error(yt_s, ypt_s):.0f})")
        ax.plot(d_sub, yr_s, color="#d97706", lw=1.2, ls="-.", label=f"Ridge ({mean_absolute_error(yt_s, yr_s):.0f})")
        ax.plot(d_sub, yxd_s, color="#e76f51", lw=1.1, ls="--", label=f"XGB-Diff ({mean_absolute_error(yt_s, yxd_s):.0f})")
        ax.set_title(title, fontsize=9.3, pad=36)
        ax.set_ylabel("Peak Demand (MW)")
        ax.tick_params(axis="x", rotation=25, labelsize=7.5)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig7_qualitative_events.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 8: Efficiency Pareto (consistent fp32 KiB & CPU ms)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    nn_names = ["Ridge (14xF)", "DLinear (CI)", "CNN-BiGRU-Transformer", "PatchTST", "BiLSTM-Attention",
                "Transformer", "MLP (14xF)", "BiGRU", "BiLSTM", "PatchTST-Ensemble (3x)", "Equal-Weight RNN Ens"]
    pts_x = [eff_profile[m]["Params"] / 1000.0 for m in nn_names]
    pts_y = [float(df_summary.loc[m, "MAE_mean"]) for m in nn_names]
    cols_p = ["#0d9488" if "PatchTST" in m else ("#d97706" if "Ridge" in m else ("#1d3557" if "RNN" in m else "#64748b")) for m in nn_names]
    sizes_p = [130 if m in ("PatchTST-Ensemble (3x)", "Ridge (14xF)", "Equal-Weight RNN Ens") else 65 for m in nn_names]
    axes[0].scatter(pts_x, pts_y, c=cols_p, s=sizes_p, zorder=4, edgecolors="white")
    offsets = {
        "Ridge (14xF)": (-2, 12), "DLinear (CI)": (10, 4), "CNN-BiGRU-Transformer": (10, 4),
        "PatchTST": (10, -10), "BiLSTM-Attention": (-42, 9), "Transformer": (10, 4),
        "MLP (14xF)": (10, -4), "BiGRU": (-12, 10), "BiLSTM": (10, -2),
        "PatchTST-Ensemble (3x)": (12, 2), "Equal-Weight RNN Ens": (-128, 8)
    }
    for m, px, py in zip(nn_names, pts_x, pts_y):
        dx, dy = offsets.get(m, (8, 4))
        axes[0].annotate(m, (px, py), xytext=(dx, dy), textcoords="offset points",
                         fontsize=7.2, fontweight="bold" if m in ("PatchTST-Ensemble (3x)", "Ridge (14xF)", "Equal-Weight RNN Ens") else "normal")
    axes[0].set_xlabel("Trainable Parameters (Thousands)"); axes[0].set_ylabel("Blind Test MAE (MW)")
    axes[0].set_title("(a) Accuracy vs. Parameter Count (5-Seed)")
    axes[0].set_xlim(-35, 900); axes[0].set_ylim(545, max(pts_y) + 25); axes[0].grid(alpha=0.25)

    lat_models = ["Ridge (14xF)", "DLinear (CI)", "XGBoost-Diff (Δy)", "MLP (14xF)", "LightGBM", "PatchTST",
                  "Transformer", "BiLSTM", "BiGRU", "BiLSTM-Attention", "PatchTST-Ensemble (3x)", "Equal-Weight RNN Ens"]
    lats = [eff_profile[m]["CPU_ms"] for m in lat_models]
    bar_c = ["#0d9488" if "PatchTST" in m else ("#d97706" if "Ridge" in m else ("#1d3557" if "RNN" in m else "#64748b")) for m in lat_models]
    axes[1].barh(lat_models, lats, color=bar_c, height=0.62)
    for i, v in enumerate(lats): axes[1].text(v + 0.025, i, f"{v:.2f} ms", va="center", fontsize=7.3)
    axes[1].set_xlabel("Single-Sample CPU Inference Latency (ms)"); axes[1].set_title("(b) CPU Inference Latency Profile")
    axes[1].set_xlim(0, max(lats) * 1.25); axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig8_efficiency_pareto.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 9: Direct Group Permutation Importance on PatchTST-3x (Merged Load History Block!)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    df_p_plot = df_perm.sort_values("Delta_MAE_Mean_MW", ascending=True)
    ax.barh(df_p_plot["Feature_Group"], df_p_plot["Delta_MAE_Mean_MW"],
            xerr=df_p_plot["Delta_MAE_Std_MW"], capsize=3, color="#0d9488", height=0.56)
    for i, (_, r) in enumerate(df_p_plot.iterrows()):
        x_txt = max(0.0, r["Delta_MAE_Mean_MW"]) + r["Delta_MAE_Std_MW"] + 20
        ax.text(x_txt, i, f"{r['Delta_MAE_Mean_MW']:+.1f} MW (k={int(r['Num_Features'])})", va="center", fontsize=7.8)
    ax.set_xlabel("Increase in Deployed PatchTST-3× MAE upon Block Permutation (MW)")
    ax.set_title("Direct Group Permutation Importance on PatchTST-Ensemble (3×) (Merged Load Block, 15 Repeats)")
    ax.set_xlim(-25, (df_p_plot["Delta_MAE_Mean_MW"] + df_p_plot["Delta_MAE_Std_MW"]).max() * 1.28)
    ax.axvline(0, color="black", lw=0.8); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig9_shap_attribution.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 11: Conformal Intervals on PatchTST-3x (Static Split vs Rolling-Window vs LN-ACI)
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.6), sharex=True)
    panels = [
        (axes[0], cp_store_pt3["1. Static Split Conformal"], "#94a3b8",
         f"(a) Static Split Conformal — 90% PICP={cp_store_pt3['1. Static Split Conformal']['PICP_%']:.1f}%, Winkler={cp_store_pt3['1. Static Split Conformal']['Winkler_MW']:.0f} MW"),
        (axes[1], cp_store_pt3["2. Rolling-Window Conformal (120d)"], "#d97706",
         f"(b) Rolling-Window Conformal (120d) — 90% PICP={cp_store_pt3['2. Rolling-Window Conformal (120d)']['PICP_%']:.1f}%, Winkler={cp_store_pt3['2. Rolling-Window Conformal (120d)']['Winkler_MW']:.0f} MW"),
        (axes[2], cp_store_pt3["5. Locally Normalized ACI (LN-ACI)"], "#0d9488",
         f"(c) Locally Normalized ACI (LN-ACI) — 90% PICP={cp_store_pt3['5. Locally Normalized ACI (LN-ACI)']['PICP_%']:.1f}%, Winkler={cp_store_pt3['5. Locally Normalized ACI (LN-ACI)']['Winkler_MW']:.0f} MW"),
    ]
    for ax, obj, colr, ttl in panels:
        ax.fill_between(ds["te_dates"], obj["lo"], obj["hi"], color=colr, alpha=0.28,
                        label=f"90% PI (Mean Width {obj['Mean_Width_MW']:.0f} MW)")
        ax.plot(ds["te_dates"], y_te_inv, color="#111111", lw=1.0, label="Actual Demand")
        ax.plot(ds["te_dates"], mean_preds["PatchTST-Ensemble (3x)"], color="#0f766e", lw=0.95, label="PatchTST-3×")
        ax.set_ylabel("Peak (MW)")
        ax.set_title(ttl, fontsize=9.1, pad=24)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=7.8)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig11_conformal_intervals.pdf", bbox_inches="tight"); plt.close(fig)

    # Save CSVs & metadata
    df_seeds.to_csv(f"{OUT_DIR}/part1_per_seed_metrics.csv", index=False)
    df_summary.to_csv(f"{OUT_DIR}/part1_summary_metrics.csv")
    df_dm.to_csv(f"{OUT_DIR}/part1_dm_tests.csv", index=False)
    df_mcs.to_csv(f"{OUT_DIR}/part1_mcs.csv", index=False)
    df_cp.to_csv(f"{OUT_DIR}/part1_conformal.csv", index=False)
    df_cp_sea.to_csv(f"{OUT_DIR}/part1_conformal_seasonal.csv", index=False)
    df_abl.to_csv(f"{OUT_DIR}/part1_ablation.csv", index=False)
    df_reg.to_csv(f"{OUT_DIR}/part1_regime_analysis.csv", index=False)
    df_bins.to_csv(f"{OUT_DIR}/part1_demand_bins.csv", index=False)
    df_perm.to_csv(f"{OUT_DIR}/part1_permutation_importance.csv", index=False)
    pd.DataFrame(eff_profile).T.to_csv(f"{OUT_DIR}/part1_efficiency.csv")

    mean_w = np.mean(learned_weights, axis=0)
    meta_out = {
        "n_clean": ds["n_raw"], "n_features": ds["n_features"], "year_dropped": True,
        "tau_star_C": ds["tau_star"], "tau_slope_detrended_MW_per_C": round(ds["tau_slope"], 2),
        "secular_trend_MW_per_yr": round(ds["trend_mw_per_yr"], 1),
        "ridge_alpha": ridge_alpha, "ridge_diff_alpha": ridged_alpha,
        "tuned_neural_lrs": tuned_lrs,
        "n_train": len(X_tr), "n_val_total": len(ds["X_va"]),
        "n_val_es": len(X_va_es), "n_val_cal": len(X_va_cal), "n_test": len(X_te),
        "seeds": SEEDS,
        "wkday_persist_mae": round(wkday_mae, 2),
        "pt3_s42_mae": round(float(mean_absolute_error(y_te_inv, seed_preds[42][ref_name])), 2),
        "pt3_5seed_mae": round(float(mean_absolute_error(y_te_inv, mean_preds[ref_name])), 2),
        "learned_weights_per_seed": {str(s): [round(x, 4) for x in w] for s, w in zip(SEEDS, learned_weights)},
        "learned_weights_mean": [round(float(x), 4) for x in mean_w],
        "data_qa": ds["data_qa"],
        "efficiency": eff_profile,
    }
    with open(f"{OUT_DIR}/part1_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nALL ROUND-2 BENCHMARK EXPERIMENTS COMPLETED IN {time.perf_counter() - t_start:.1f}s!", flush=True)
