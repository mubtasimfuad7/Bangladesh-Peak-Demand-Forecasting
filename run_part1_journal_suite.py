"""
Part 1 Q1-Journal Reproducible Benchmark Suite (v4 Revision — Complete Resolution of v3 Review):
  1. Drops raw extrapolating `Year`; detrended cooling threshold tau* = 25.0 C; robust IQR loss QA.
  2. 25-Model, 5-Seed Benchmark (seeds 42, 123, 7, 2024, 999):
     - Explicit naming: `PatchTST-CM` (channel-mixing, no RevIN) vs. faithful `PatchTST-RevIN (3x)`
       (channel-independent target patching + Reversible Instance Normalization, Nie et al. 2023).
     - Includes Naive (Lag-1, Weekday-Adj, Lag-7, 7d Mean), SARIMAX, Ridge, Ridge-Diff, XGBoost (Level/Diff),
       LightGBM, DLinear (CI, 30p), MLP (14xF), PatchTST-CM (1x/3x/Diff-3x), PatchTST-RevIN (3x),
       BiGRU, BiLSTM, BiLSTM-Attn, Transformer, CNN-BiGRU-T, Hybrids, Equal-Weight RNN, Val-Opt RNN.
  3. 95% Hansen MCS (MAE & RMSE, Seed 42 & 5-Seed) + HLN-DM (Ref = PatchTST-CM 3x, Holm-Bonferroni).
  4. 5-Way Conformal Ablation on BOTH PatchTST-CM (3x) AND Ridge (14xF) at 80% & 90% nominal,
     WITH 95% moving-block bootstrap CIs (block length 7, 1,000 resamples) on overall AND seasonal PICP!
  5. Architecture-dependent 5-Seed Feature & PCA Ablation across PatchTST-CM (3x), Equal-Weight RNN, & Ridge.
  6. Lagged-demand (y_{t-1}) bins (no realized-target selection bias) + paired Holiday/Eid tests (n=41).
  7. Gap-handling audit (420 gap-free vs. 42 gap-spanning test windows; 10 gap-spanning Val-ES windows).
  8. Eid +/-1-day lunar-sighting jitter sensitivity test on frozen trained models.
  9. Summer-inclusive validation sensitivity check (tuning on full-year 2024 Fold 2 vs. Jul-Nov Val-ES).
 10. Exact CPU hardware & thread metadata + Direct merged-block permutation importance on PatchTST-CM (3x).
"""
import json
import math
import os
import platform
import subprocess
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
TORCH_THREADS = 2
torch.set_num_threads(TORCH_THREADS)

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


def get_cpu_brand():
    try:
        out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        ncpu = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
        return f"{out} ({ncpu}-core ARM64)"
    except Exception:
        return platform.processor() or "Unknown CPU"


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


def prepare_dataset(use_cdd=True, use_eid=True, use_pca=False, tau_override=None, eid_jitter_test=0):
    df = pd.read_excel("version_5_extended_backup.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    full_span_days = int((df["Date"].max() - df["Date"].min()).days + 1)
    n_missing_calendar = full_span_days - len(df)
    date_gaps = df["Date"].diff().dt.days.fillna(1.0)
    df["Post_Gap_Flag"] = (date_gaps > 1.0).astype(int)

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

    if "Year" in df.columns:
        df = df.drop(columns=["Year"])

    # Optional test-only Eid +/-1-day lunar-sighting jitter (applied strictly to post-val test rows)
    if eid_jitter_test != 0:
        approx_te_start = int(0.80 * len(df))
        hol_s = df["Holiday_cat"].copy()
        name_s = df["Holiday name"].copy()
        te_mask = np.arange(len(df)) >= approx_te_start
        eid_mask = te_mask & name_s.astype(str).str.contains("Eid", case=False, na=False)
        if eid_mask.any():
            hol_te = hol_s.iloc[approx_te_start:].copy()
            name_te = name_s.iloc[approx_te_start:].copy()
            # Shift only Eid indicator positions by eid_jitter_test days
            is_eid_te = name_te.astype(str).str.contains("Eid", case=False, na=False)
            new_hol_te = hol_te.copy()
            new_hol_te[is_eid_te] = 0
            shifted_idx = np.clip(np.where(is_eid_te.values)[0] + eid_jitter_test, 0, len(hol_te) - 1)
            new_hol_te.iloc[shifted_idx] = hol_te.values[is_eid_te.values]
            df.iloc[approx_te_start:, df.columns.get_loc("Holiday_cat")] = new_hol_te.values

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

    # Track exact windows spanning any calendar gap (> 14 calendar days from window start to target)
    te_all_dates = df_enc.index[val_cut:]
    te_gap_span = np.array([(te_all_dates[i] - te_all_dates[i - HIST_SIZE]).days > HIST_SIZE for i in range(HIST_SIZE, len(te_all_dates))])
    va_all_dates = df_enc.index[train_cut:val_cut]
    va_gap_span = np.array([(va_all_dates[i] - va_all_dates[i - HIST_SIZE]).days > HIST_SIZE for i in range(HIST_SIZE, len(va_all_dates))])

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
        "va_dates": va_dates, "va_meta": va_meta, "va_gap_span": va_gap_span,
        "X_va_es": X_va[:va_es_end], "y_va_es": y_va[:va_es_end], "y_va_es_inv": y_va_inv[:va_es_end],
        "y_va_es_lag1": y_va_lag1[:va_es_end], "dy_va_es_mw": dy_va_mw[:va_es_end], "dy_va_es_norm": dy_va_es_norm,
        "X_va_cal": X_va[va_cal_start:], "y_va_cal": y_va[va_cal_start:], "y_va_cal_inv": y_va_inv[va_cal_start:],
        "y_va_cal_lag1": y_va_lag1[va_cal_start:],
        "X_te": X_te, "y_te": y_te, "y_te_inv": y_te_inv, "y_te_lag1": y_te_lag1,
        "te_dates": te_dates, "te_meta": te_meta, "te_gap_span": te_gap_span,
        "sy": sy, "meta_full": meta, "feat_names": feat_names, "df_enc": df_enc,
        "cont_cols": cont_cols, "cat_onehot_cols": cat_onehot_cols,
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
            "val_es_gap_spanning_windows": int(va_gap_span[:va_es_end].sum()),
            "val_cal_gap_spanning_windows": int(va_gap_span[va_cal_start:].sum()),
            "test_gap_spanning_windows": int(te_gap_span.sum()),
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
    return p_vc, p_te, best_alpha, int(X_tr_2d.shape[1] + 1), lat_ms, best_mod


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
    """Standard Channel-Independent DLinear (Zeng et al., AAAI 2023) on target channel (30 params)."""
    def __init__(self, seq_len=14, kernel_size=5):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.linear_seasonal = nn.Linear(seq_len, 1)
        self.linear_trend = nn.Linear(seq_len, 1)
    def forward(self, x):
        y_seq = x[:, :, -1:]
        front = y_seq[:, 0:1, :].repeat(1, self.pad, 1)
        end = y_seq[:, -1:, :].repeat(1, self.pad, 1)
        y_pad = torch.cat([front, y_seq, end], dim=1)
        trend = self.avg(y_pad.transpose(1, 2)).transpose(1, 2).squeeze(-1)
        seasonal = y_seq.squeeze(-1) - trend
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)


class MLPModel(nn.Module):
    def __init__(self, n_feat, seq_len=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_feat, 128), nn.ReLU(), nn.Dropout(OPT_DROPOUT),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(OPT_DROPOUT),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x)


class PatchTSTCMModel(nn.Module):
    """Channel-Mixing Patched Transformer (no RevIN; P=4, S=2, 6 patches, 2 Pre-Norm layers, d=64, 4 heads)."""
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


class PatchTSTRevINModel(nn.Module):
    """
    Faithful Channel-Independent + RevIN PatchTST (Nie et al., ICLR 2023):
    Applies Reversible Instance Normalization (RevIN) to the 14-day target load sequence,
    patches it channel-independently (P=4, S=2), passes through a 2-layer Pre-Norm Transformer,
    and denormalizes the scalar prediction back via the instance mean and std.
    """
    def __init__(self, seq_len=14, patch_len=4, stride=2, d_model=64, eps=1e-5):
        super().__init__()
        self.patch_len, self.stride, self.eps = patch_len, stride, eps
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.revin_weight = nn.Parameter(torch.ones(1, 1))
        self.revin_bias = nn.Parameter(torch.zeros(1, 1))
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=128,
            dropout=OPT_DROPOUT, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(OPT_DROPOUT),
            nn.Linear(self.n_patches * d_model, 1)
        )
    def forward(self, x):
        y_seq = x[:, :, -1]  # (B, L)
        mu = y_seq.mean(dim=1, keepdim=True)
        sd = torch.sqrt(y_seq.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        y_norm = ((y_seq - mu) / sd) * self.revin_weight + self.revin_bias
        patches = y_norm.unfold(dimension=1, size=self.patch_len, step=self.stride)  # (B, n_patches, P)
        z = self.patch_proj(patches) + self.pos_emb
        out_norm = self.head(self.encoder(z))
        return ((out_norm - self.revin_bias) / (self.revin_weight + self.eps)) * sd + mu


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


def block_bootstrap_picp_ci(covered, block_len=7, n_boot=1000, seed=42):
    cov = np.asarray(covered, dtype=float)
    T = len(cov)
    bl = min(block_len, max(1, T // 3))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(T / bl))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T - bl + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + bl) for s in starts])[:T]
        boots[b] = np.mean(cov[idx]) * 100.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi), float((hi - lo) / 2.0)


def eval_interval(method_name, model_name, yt, lo, hi, alpha):
    cov = (yt >= lo) & (yt <= hi)
    picp = float(np.mean(cov) * 100.0)
    ci_lo, ci_hi, ci_hw = block_bootstrap_picp_ci(cov, block_len=7, n_boot=1000, seed=42)
    width = float(np.mean(hi - lo))
    pinaw = float(width / (np.max(yt) - np.min(yt)) * 100.0)
    penalty = np.where(yt < lo, (2.0 / alpha) * (lo - yt),
              np.where(yt > hi, (2.0 / alpha) * (yt - hi), 0.0))
    winkler = float(np.mean((hi - lo) + penalty))
    return {
        "Model": model_name, "Method": method_name, "Nominal_%": int((1.0 - alpha) * 100),
        "Mean_Half_Width_MW": width / 2.0, "Mean_Width_MW": width,
        "PICP_%": picp, "PICP_CI_lo": ci_lo, "PICP_CI_hi": ci_hi, "PICP_CI_hw": ci_hw,
        "PINAW_%": pinaw, "Winkler_MW": winkler, "lo": lo, "hi": hi
    }


def run_5way_conformal(model_name, y_cal, pred_cal, y_test, pred_test, alpha, gamma=0.05, scale_win=14, score_win=120):
    res_cal = np.abs(y_cal.ravel() - pred_cal.ravel())
    yt = y_test.ravel()
    yp = pred_test.ravel()
    n_cal = len(res_cal)
    floor_sigma = max(100.0, float(np.percentile(res_cal, 25)))

    q_lvl = min(1.0, math.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal)
    q_stat = float(np.quantile(res_cal, q_lvl, method="higher"))
    r1 = eval_interval("1. Static Split Conformal", model_name, yt, yp - q_stat, yp + q_stat, alpha)

    hist_raw = list(res_cal)
    lo2, hi2 = np.empty(len(yt)), np.empty(len(yt))
    for t in range(len(yt)):
        w_scores = np.array(hist_raw[-score_win:])
        q_t = float(np.quantile(w_scores, min(1.0, math.ceil((len(w_scores) + 1) * (1.0 - alpha)) / len(w_scores)), method="higher"))
        lo2[t], hi2[t] = yp[t] - q_t, yp[t] + q_t
        hist_raw.append(abs(yt[t] - yp[t]))
    r2 = eval_interval("2. Rolling-Window Conformal (120d)", model_name, yt, lo2, hi2, alpha)

    norm_cal = []
    for i in range(n_cal):
        sig = max(floor_sigma, float(np.mean(res_cal[: max(1, i + 1)])) if i < 5 else float(np.mean(res_cal[max(0, i - scale_win) : i])))
        norm_cal.append(res_cal[i] / sig)
    q_norm_stat = float(np.quantile(norm_cal, q_lvl, method="higher"))

    hist_r3 = list(res_cal)
    lo3, hi3 = np.empty(len(yt)), np.empty(len(yt))
    for t in range(len(yt)):
        sig_t = max(floor_sigma, float(np.mean(hist_r3[-scale_win:])))
        hw = q_norm_stat * sig_t
        lo3[t], hi3[t] = yp[t] - hw, yp[t] + hw
        hist_r3.append(abs(yt[t] - yp[t]))
    r3 = eval_interval("3. Locally Normalized Static", model_name, yt, lo3, hi3, alpha)

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
    ml = min(max_lag, max(1, T // 4))
    bl = min(block_len, max(1, T // 3))
    d_mean = float(np.mean(d))
    var_d = np.var(d, ddof=1)
    for k in range(1, ml + 1):
        w = 1.0 - k / (ml + 1.0)
        var_d += 2.0 * w * np.cov(d[k:], d[:-k], ddof=1)[0, 1]
    dm_raw = float(d_mean / np.sqrt(max(var_d, 1e-12) / T))
    hln_factor = math.sqrt((T + 1.0 - 2.0 * h + (h * (h - 1.0)) / T) / T)
    dm_hln = dm_raw * hln_factor
    p_val = float(2.0 * stats.t.sf(np.abs(dm_hln), df=T - 1))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(T / bl))
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T - bl + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + bl) for s in starts])[:T]
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
    cpu_brand = get_cpu_brand()
    print(f"=== v4 REVISION SUITE (25 MODELS x 5 SEEDS) | CPU: {cpu_brand} | torch_threads={TORCH_THREADS} ===", flush=True)
    ds = prepare_dataset(use_cdd=True, use_eid=True, use_pca=False)
    print(f"Detrended tau* = {ds['tau_star']:.1f} C (+{ds['tau_slope']:.1f} MW/C, trend +{ds['trend_mw_per_yr']:.1f} MW/yr)", flush=True)
    print(f"Data QA: {ds['data_qa']}", flush=True)

    X_tr, y_tr = ds["X_tr"], ds["y_tr"]
    X_va_es, y_va_es = ds["X_va_es"], ds["y_va_es"]
    X_va_cal, y_va_cal_inv = ds["X_va_cal"], ds["y_va_cal_inv"]
    X_te, y_te_inv, sy = ds["X_te"], ds["y_te_inv"], ds["sy"]
    X_tr_2d, X_va_es_2d, X_va_cal_2d, X_te_2d = (
        X_tr.reshape(len(X_tr), -1), X_va_es.reshape(len(X_va_es), -1),
        X_va_cal.reshape(len(X_va_cal), -1), X_te.reshape(len(X_te), -1)
    )
    n_f = X_tr.shape[-1]

    # Baselines: Lag-1, Weekday-Adj Persistence, Lag-7 (row & calendar-aware check), 7-Day Mean
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

    print("Fitting SARIMAX, Ridge, and Ridge-Diff...", flush=True)
    p_sarimax_va, p_sarimax_te, sarimax_p, sarimax_ms = run_sarimax_baseline(ds)
    p_ridge_vc, p_ridge_te, ridge_alpha, ridge_p, ridge_ms, ridge_mod = run_ridge_baseline(
        X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_va_cal_2d, X_te_2d, sy
    )
    p_ridged_vc, p_ridged_te, ridged_alpha = run_ridge_diff_baseline(
        X_tr_2d, ds["dy_tr_mw"], X_va_es_2d, ds["dy_va_es_mw"], X_va_cal_2d, ds["y_va_cal_lag1"], X_te_2d, ds["y_te_lag1"]
    )
    print(f"  SARIMAX={mean_absolute_error(y_te_inv, p_sarimax_te):.2f} | Ridge(a={ridge_alpha:g})={mean_absolute_error(y_te_inv, p_ridge_te):.2f} | Ridge-Diff(a={ridged_alpha:g})={mean_absolute_error(y_te_inv, p_ridged_te):.2f} | Wkday={wkday_mae:.2f}", flush=True)

    # Val-ES LR selection per neural family (Seed 42)
    dl_builders = [
        ("DLinear (CI)", lambda: StandardDLinearModel(HIST_SIZE)),
        ("MLP (14xF)", lambda: MLPModel(n_f)),
        ("PatchTST-CM (Single)", lambda: PatchTSTCMModel(n_f)),
        ("PatchTST-RevIN (Single)", lambda: PatchTSTRevINModel(HIST_SIZE)),
        ("BiGRU", lambda: BiGRUModel(n_f)),
        ("BiLSTM", lambda: BiLSTMModel(n_f)),
        ("BiLSTM-Attention", lambda: BiLSTMAttnModel(n_f)),
        ("Transformer", lambda: TransformerModel(n_f)),
        ("CNN-BiGRU-Transformer", lambda: CNNBiGRUTransModel(n_f)),
    ]
    print("\nSelecting LR per neural family on Val-ES (Seed 42)...", flush=True)
    tuned_lrs = {}
    for name, fn in dl_builders:
        cands = [5e-3, 1e-2] if name == "DLinear (CI)" else [1e-3, 2.5e-3]
        best_vl, best_lr = float("inf"), cands[0]
        for lr_c in cands:
            _, _, vl = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, 42, lr=lr_c)
            if vl < best_vl:
                best_vl, best_lr = vl, lr_c
        tuned_lrs[name] = best_lr
        print(f"  {name:25s} -> lr = {best_lr:g} (Val-ES L1={best_vl:.4f})", flush=True)

    best_vl_d, lr_pt_diff = float("inf"), 1e-3
    for lr_c in [1e-3, 2.5e-3]:
        _, _, vl = train_torch_model(PatchTSTCMModel(n_f), X_tr, ds["dy_tr_norm"], X_va_es, ds["dy_va_es_norm"], 42, lr=lr_c)
        if vl < best_vl_d:
            best_vl_d, lr_pt_diff = vl, lr_c
    tuned_lrs["PatchTST-CM-Diff (3x)"] = lr_pt_diff
    print(f"  {'PatchTST-CM-Diff (3x)':25s} -> lr = {lr_pt_diff:g}", flush=True)

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
    seed0_rnn_models = []

    for s in SEEDS:
        print(f"\n--- Seed {s} ---", flush=True)
        for bn, bp in [
            ("Persistence (Lag-1)", p_persist_te),
            ("Weekday-Adj Persistence", p_wkday_pers_te),
            ("Seasonal Naive (Lag-7)", p_lag7_te),
            ("7-Day Rolling Mean", p_roll7_te),
            ("SARIMAX", p_sarimax_te),
            ("Ridge (14xF)", p_ridge_te),
            ("Ridge-Diff (Δy)", p_ridged_te),
        ]:
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

        # Neural models
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
            if name != "PatchTST-RevIN (Single)":
                all_seed_metrics.append({"Seed": s, "Model": name, **mr})
            if s == SEEDS[0]:
                np_cnt = sum(p.numel() for p in mod.parameters())
                eff_profile[name] = {
                    "Params": np_cnt, "Size_KiB": round(np_cnt * 4 / 1024, 2),
                    "CPU_ms": round(measure_torch_latency(mod, X_te[:1]), 2)
                }

        # 3-Member PatchTST-CM (3x)
        pt_vc_list = [seed_val_cal_preds[s]["PatchTST-CM (Single)"]]
        pt_te_list = [seed_preds[s]["PatchTST-CM (Single)"]]
        if s == SEEDS[0]:
            seed0_pt3_models = [trained_dl["PatchTST-CM (Single)"]]
            seed0_rnn_models = [trained_dl["BiGRU"], trained_dl["BiLSTM-Attention"], trained_dl["BiLSTM"]]
        for offset in [1000, 2000]:
            m_pt_extra, _, _ = train_torch_model(PatchTSTCMModel(n_f), X_tr, y_tr, X_va_es, y_va_es, s + offset, lr=tuned_lrs["PatchTST-CM (Single)"])
            pt_vc_list.append(sy.inverse_transform(predict_torch(m_pt_extra, X_va_cal)))
            pt_te_list.append(sy.inverse_transform(predict_torch(m_pt_extra, X_te)))
            if s == SEEDS[0]:
                seed0_pt3_models.append(m_pt_extra)
        p_pt3_vc, p_pt3_te = np.mean(pt_vc_list, axis=0), np.mean(pt_te_list, axis=0)
        seed_val_cal_preds[s]["PatchTST-CM (3x)"] = p_pt3_vc
        seed_preds[s]["PatchTST-CM (3x)"] = p_pt3_te
        m_pt3 = calc_metrics(y_te_inv, p_pt3_te, wkday_mae)
        all_seed_metrics.append({"Seed": s, "Model": "PatchTST-CM (3x)", **m_pt3})
        if s == SEEDS[0]:
            eff_profile["PatchTST-CM (3x)"] = {
                "Params": eff_profile["PatchTST-CM (Single)"]["Params"] * 3,
                "Size_KiB": round(eff_profile["PatchTST-CM (Single)"]["Params"] * 3 * 4 / 1024, 2),
                "CPU_ms": round(eff_profile["PatchTST-CM (Single)"]["CPU_ms"] * 3, 2),
            }

        # 3-Member Faithful PatchTST-RevIN (CI + RevIN, 3x)
        rev_vc_list = [seed_val_cal_preds[s]["PatchTST-RevIN (Single)"]]
        rev_te_list = [seed_preds[s]["PatchTST-RevIN (Single)"]]
        for offset in [1000, 2000]:
            m_rev_ex, _, _ = train_torch_model(PatchTSTRevINModel(HIST_SIZE), X_tr, y_tr, X_va_es, y_va_es, s + offset, lr=tuned_lrs["PatchTST-RevIN (Single)"])
            rev_vc_list.append(sy.inverse_transform(predict_torch(m_rev_ex, X_va_cal)))
            rev_te_list.append(sy.inverse_transform(predict_torch(m_rev_ex, X_te)))
        p_rev_vc, p_rev_te = np.mean(rev_vc_list, axis=0), np.mean(rev_te_list, axis=0)
        seed_val_cal_preds[s]["PatchTST-RevIN (CI, 3x)"] = p_rev_vc
        seed_preds[s]["PatchTST-RevIN (CI, 3x)"] = p_rev_te
        m_rev = calc_metrics(y_te_inv, p_rev_te, wkday_mae)
        all_seed_metrics.append({"Seed": s, "Model": "PatchTST-RevIN (CI, 3x)", **m_rev})
        if s == SEEDS[0]:
            eff_profile["PatchTST-RevIN (CI, 3x)"] = {
                "Params": eff_profile["PatchTST-RevIN (Single)"]["Params"] * 3,
                "Size_KiB": round(eff_profile["PatchTST-RevIN (Single)"]["Params"] * 3 * 4 / 1024, 2),
                "CPU_ms": round(eff_profile["PatchTST-RevIN (Single)"]["CPU_ms"] * 3, 2),
            }

        # 3-Member PatchTST-CM-Diff (Delta-y, 3x)
        ptd_vc_list, ptd_te_list = [], []
        for offset in [0, 1000, 2000]:
            m_ptd, _, _ = train_torch_model(PatchTSTCMModel(n_f), X_tr, ds["dy_tr_norm"], X_va_es, ds["dy_va_es_norm"], s + offset, lr=lr_pt_diff)
            ptd_vc_list.append(ds["y_va_cal_lag1"] + ds["s_dy"].inverse_transform(predict_torch(m_ptd, X_va_cal)))
            ptd_te_list.append(ds["y_te_lag1"] + ds["s_dy"].inverse_transform(predict_torch(m_ptd, X_te)))
        p_ptd_vc, p_ptd_te = np.mean(ptd_vc_list, axis=0), np.mean(ptd_te_list, axis=0)
        seed_val_cal_preds[s]["PatchTST-CM-Diff (Δy, 3x)"] = p_ptd_vc
        seed_preds[s]["PatchTST-CM-Diff (Δy, 3x)"] = p_ptd_te
        all_seed_metrics.append({"Seed": s, "Model": "PatchTST-CM-Diff (Δy, 3x)", **calc_metrics(y_te_inv, p_ptd_te, wkday_mae)})
        if s == SEEDS[0]:
            eff_profile["PatchTST-CM-Diff (Δy, 3x)"] = eff_profile["PatchTST-CM (3x)"].copy()

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

        # Equal-Weight & Val-Opt RNN Ensembles
        p_eq_vc = (seed_val_cal_preds[s]["BiGRU"] + seed_val_cal_preds[s]["BiLSTM-Attention"] + seed_val_cal_preds[s]["BiLSTM"]) / 3.0
        p_eq_te = (seed_preds[s]["BiGRU"] + seed_preds[s]["BiLSTM-Attention"] + seed_preds[s]["BiLSTM"]) / 3.0
        seed_val_cal_preds[s]["Equal-Weight RNN Ens"] = p_eq_vc
        seed_preds[s]["Equal-Weight RNN Ens"] = p_eq_te
        all_seed_metrics.append({"Seed": s, "Model": "Equal-Weight RNN Ens", **calc_metrics(y_te_inv, p_eq_te, wkday_mae)})

        w_opt = optimize_val_weights(y_va_cal_inv, [seed_val_cal_preds[s]["BiGRU"], seed_val_cal_preds[s]["BiLSTM-Attention"], seed_val_cal_preds[s]["BiLSTM"]])
        learned_weights.append(w_opt.tolist())
        p_opt_vc = w_opt[0] * seed_val_cal_preds[s]["BiGRU"] + w_opt[1] * seed_val_cal_preds[s]["BiLSTM-Attention"] + w_opt[2] * seed_val_cal_preds[s]["BiLSTM"]
        p_opt_te = w_opt[0] * seed_preds[s]["BiGRU"] + w_opt[1] * seed_preds[s]["BiLSTM-Attention"] + w_opt[2] * seed_preds[s]["BiLSTM"]
        seed_val_cal_preds[s]["Val-Opt RNN Ensemble"] = p_opt_vc
        seed_preds[s]["Val-Opt RNN Ensemble"] = p_opt_te
        all_seed_metrics.append({"Seed": s, "Model": "Val-Opt RNN Ensemble", **calc_metrics(y_te_inv, p_opt_te, wkday_mae)})
        print(f"  [Seed {s}] PT-CM(3x)={m_pt3['MAE']:.1f} | PT-RevIN(3x)={m_rev['MAE']:.1f} | EqRNN={mean_absolute_error(y_te_inv, p_eq_te):.1f}", flush=True)

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
    print("\n=== 1. v4 25-MODEL 5-SEED BENCHMARK ===")
    print(df_summary.round(2).to_string())

    mean_preds = {m: np.mean([seed_preds[s][m] for s in SEEDS], axis=0) for m in df_summary.index}
    mean_vc_pt3 = np.mean([seed_val_cal_preds[s]["PatchTST-CM (3x)"] for s in SEEDS], axis=0)
    ref_name = "PatchTST-CM (3x)"

    # 2. Descriptive HLN-DM (Ref = PatchTST-CM 3x)
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
    print(f"\n=== 2. DESCRIPTIVE HLN-DM (REF={ref_name}: S42={mean_absolute_error(y_te_inv, seed_preds[42][ref_name]):.2f}, 5-Seed={mean_absolute_error(y_te_inv, mean_preds[ref_name]):.2f}) ===")
    print(df_dm[["Comparator", "S42_MAE", "S42_Delta_MW", "S42_p_holm", "Ens5_MAE", "Ens5_Delta_MW", "Ens5_p_holm"]].round(4).to_string(index=False))

    # 3. 95% Hansen MCS (Top 12)
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
    print("\n=== 3. 95% HANSEN MCS (TOP 12) ===")
    print(df_mcs.round(3).to_string(index=False))

    # 4. 5-Way Conformal Ablation WITH 95% Block-Bootstrap CIs for BOTH PatchTST-CM (3x) AND Ridge (80% & 90%, + Seasonal for BOTH!)
    def map_season(m):
        if m in [3, 4, 5]: return "Summer (Mar-May)"
        if m in [6, 7, 8, 9]: return "Monsoon (Jun-Sep)"
        if m in [10, 11]: return "Post-Monsoon (Oct-Nov)"
        return "Winter (Dec-Feb)"

    te_seasons = ds["te_meta"]["Month"].apply(map_season).values
    yt_flat = y_te_inv.ravel()
    cp_rows, sea_rows = [], []
    cp_store_pt3 = {}
    for m_lbl, vc_pred, te_pred in [
        ("PatchTST-CM (3x)", mean_vc_pt3, mean_preds["PatchTST-CM (3x)"]),
        ("Ridge (14xF)", p_ridge_vc, p_ridge_te),
    ]:
        for alpha in [0.20, 0.10]:
            res_list = run_5way_conformal(m_lbl, y_va_cal_inv, vc_pred, y_te_inv, te_pred, alpha)
            for r in res_list:
                cp_rows.append({k: v for k, v in r.items() if k not in ("lo", "hi")})
                if alpha == 0.10:
                    if m_lbl == "PatchTST-CM (3x)":
                        cp_store_pt3[r["Method"]] = r
                    srow = {
                        "Model": m_lbl, "Method": r["Method"],
                        "Overall_PICP_%": r["PICP_%"], "Overall_CI_hw": r["PICP_CI_hw"],
                        "Overall_Width_MW": r["Mean_Width_MW"], "Winkler_MW": r["Winkler_MW"],
                    }
                    for sea, short in [("Winter (Dec-Feb)", "Winter"), ("Post-Monsoon (Oct-Nov)", "PostMon"),
                                       ("Monsoon (Jun-Sep)", "Monsoon"), ("Summer (Mar-May)", "Summer")]:
                        ms = te_seasons == sea
                        cov_s = (yt_flat[ms] >= r["lo"][ms]) & (yt_flat[ms] <= r["hi"][ms])
                        _, _, hw_s = block_bootstrap_picp_ci(cov_s, block_len=7, n_boot=1000, seed=42)
                        srow[f"{short}_PICP_%"] = float(np.mean(cov_s) * 100.0)
                        srow[f"{short}_CI_hw"] = hw_s
                        srow[f"{short}_Width_MW"] = float(np.mean(r["hi"][ms] - r["lo"][ms]))
                    sea_rows.append(srow)
    df_cp = pd.DataFrame(cp_rows)
    df_cp_sea = pd.DataFrame(sea_rows)
    print("\n=== 4. 5-WAY CONFORMAL ABLATION (BOTH MODELS, 80% & 90%, WITH BLOCK-BOOTSTRAP 95% CIs) ===")
    print(df_cp[["Model", "Method", "Nominal_%", "Mean_Width_MW", "PICP_%", "PICP_CI_hw", "Winkler_MW"]].round(2).to_string(index=False))
    print("\n=== 5. SEASONAL 90% CONFORMAL COVERAGE + 95% BLOCK-BOOTSTRAP +/-pp CIs (BOTH MODELS) ===")
    print(df_cp_sea[["Model", "Method", "Overall_PICP_%", "Overall_CI_hw", "Winter_PICP_%", "Winter_CI_hw", "PostMon_PICP_%", "Monsoon_PICP_%", "Summer_PICP_%", "Summer_CI_hw", "Summer_Width_MW"]].round(1).to_string(index=False))

    # 6. Direct Group Permutation Importance on PatchTST-CM (3x)
    print("\nComputing Direct Group Permutation Importance on PatchTST-CM (3x)...", flush=True)
    df_perm = compute_patchtst3_group_permutation_importance(seed0_pt3_models, X_te, y_te_inv, sy, ds["feat_names"], n_repeats=15, seed=42)
    print(df_perm.round(2).to_string(index=False))

    # 7. Architecture-Dependent 5-Seed Feature & PCA Ablation (PatchTST-CM 3x + Equal-Weight RNN + Ridge!)
    print("\nRunning 5-Seed Feature & PCA Ablation on PatchTST-CM (3x), Equal-Weight RNN, & Ridge...", flush=True)
    ablation_configs = [
        ("1. Base (No CDD_25, No Eid Flags)", dict(use_cdd=False, use_eid=False, use_pca=False)),
        ("2. + Detrended CDD_25", dict(use_cdd=True, use_eid=False, use_pca=False)),
        ("3. + Eid Lead/Lag (Full No-PCA)", dict(use_cdd=True, use_eid=True, use_pca=False)),
        ("4. Full + PCA (95% Variance)", dict(use_cdd=True, use_eid=True, use_pca=True)),
    ]
    hol_mask = (ds["te_meta"]["Holiday_cat"].values > 0)
    abl_rows, abl_pt3_seeds, abl_rnn_seeds = [], {}, {}
    for label, cfg in ablation_configs:
        if cfg == dict(use_cdd=True, use_eid=True, use_pca=False):
            vc_pt = [float(mean_absolute_error(y_va_cal_inv, seed_val_cal_preds[s]["PatchTST-CM (3x)"])) for s in SEEDS]
            te_pt = [float(mean_absolute_error(y_te_inv, seed_preds[s]["PatchTST-CM (3x)"])) for s in SEEDS]
            hol_pt = [float(mean_absolute_error(y_te_inv[hol_mask], seed_preds[s]["PatchTST-CM (3x)"][hol_mask])) for s in SEEDS]
            vc_rnn = [float(mean_absolute_error(y_va_cal_inv, seed_val_cal_preds[s]["Equal-Weight RNN Ens"])) for s in SEEDS]
            te_rnn = [float(mean_absolute_error(y_te_inv, seed_preds[s]["Equal-Weight RNN Ens"])) for s in SEEDS]
            r_vc_mae = float(mean_absolute_error(y_va_cal_inv, p_ridge_vc))
            r_te_mae = float(mean_absolute_error(y_te_inv, p_ridge_te))
            dims = ds["n_features"]
        else:
            dsi = prepare_dataset(**cfg, tau_override=ds["tau_star"])
            dims = dsi["n_features"]
            hm = (dsi["te_meta"]["Holiday_cat"].values > 0)
            p_rv, p_rt, _, _, _, _ = run_ridge_baseline(
                dsi["X_tr"].reshape(len(dsi["X_tr"]), -1), dsi["y_tr"],
                dsi["X_va_es"].reshape(len(dsi["X_va_es"]), -1), dsi["y_va_es"],
                dsi["X_va_cal"].reshape(len(dsi["X_va_cal"]), -1), dsi["X_te"].reshape(len(dsi["X_te"]), -1), dsi["sy"]
            )
            r_vc_mae = float(mean_absolute_error(dsi["y_va_cal_inv"], p_rv))
            r_te_mae = float(mean_absolute_error(dsi["y_te_inv"], p_rt))
            vc_pt, te_pt, hol_pt, vc_rnn, te_rnn = [], [], [], [], []
            for s in SEEDS:
                v_l, t_l = [], []
                for off in [0, 1000, 2000]:
                    m, _, _ = train_torch_model(PatchTSTCMModel(dims), dsi["X_tr"], dsi["y_tr"], dsi["X_va_es"], dsi["y_va_es"], s + off, lr=tuned_lrs["PatchTST-CM (Single)"])
                    v_l.append(dsi["sy"].inverse_transform(predict_torch(m, dsi["X_va_cal"])))
                    t_l.append(dsi["sy"].inverse_transform(predict_torch(m, dsi["X_te"])))
                pv, pt = np.mean(v_l, axis=0), np.mean(t_l, axis=0)
                vc_pt.append(float(mean_absolute_error(dsi["y_va_cal_inv"], pv)))
                te_pt.append(float(mean_absolute_error(dsi["y_te_inv"], pt)))
                hol_pt.append(float(mean_absolute_error(dsi["y_te_inv"][hm], pt[hm])))

                rv_l, rt_l = [], []
                for nm, fn in [("BiGRU", lambda: BiGRUModel(dims)), ("BiLSTM-Attention", lambda: BiLSTMAttnModel(dims)), ("BiLSTM", lambda: BiLSTMModel(dims))]:
                    mrn, _, _ = train_torch_model(fn(), dsi["X_tr"], dsi["y_tr"], dsi["X_va_es"], dsi["y_va_es"], s, lr=tuned_lrs[nm])
                    rv_l.append(dsi["sy"].inverse_transform(predict_torch(mrn, dsi["X_va_cal"])))
                    rt_l.append(dsi["sy"].inverse_transform(predict_torch(mrn, dsi["X_te"])))
                vc_rnn.append(float(mean_absolute_error(dsi["y_va_cal_inv"], np.mean(rv_l, axis=0))))
                te_rnn.append(float(mean_absolute_error(dsi["y_te_inv"], np.mean(rt_l, axis=0))))
        abl_pt3_seeds[label] = te_pt
        abl_rnn_seeds[label] = te_rnn
        abl_rows.append({
            "Config": label, "Dims": dims,
            "PT3_ValCal_MAE": float(np.mean(vc_pt)),
            "PT3_Test_MAE": float(np.mean(te_pt)), "PT3_Test_SD": float(np.std(te_pt, ddof=1)),
            "PT3_Hol_MAE": float(np.mean(hol_pt)),
            "EqRNN_ValCal_MAE": float(np.mean(vc_rnn)),
            "EqRNN_Test_MAE": float(np.mean(te_rnn)), "EqRNN_Test_SD": float(np.std(te_rnn, ddof=1)),
            "Ridge_ValCal_MAE": r_vc_mae, "Ridge_Test_MAE": r_te_mae,
        })
    df_abl = pd.DataFrame(abl_rows)
    base_pt = np.array(abl_pt3_seeds[ablation_configs[0][0]])
    base_rnn = np.array(abl_rnn_seeds[ablation_configs[0][0]])
    df_abl["PT3_p_vs_Base"] = [1.0] + [float(stats.ttest_rel(abl_pt3_seeds[lbl], base_pt)[1]) for lbl, _ in ablation_configs[1:]]
    df_abl["EqRNN_p_vs_Base"] = [1.0] + [float(stats.ttest_rel(abl_rnn_seeds[lbl], base_rnn)[1]) for lbl, _ in ablation_configs[1:]]
    print("\n=== 6. ARCHITECTURE-DEPENDENT 5-SEED ABLATION (PATCHTST-CM, EQ-RNN, RIDGE) ===")
    print(df_abl.round(3).to_string(index=False))

    # 8. Regimes, Holiday/Eid Paired Tests, and Lagged-Demand (y_{t-1}) Bins (NO realized-target selection bias!)
    te_m = ds["te_meta"].copy()
    te_m["Actual"] = y_te_inv.ravel()
    te_m["Lag1_Demand"] = ds["y_te_lag1"].ravel()
    te_m["Pred_PT3"] = mean_preds["PatchTST-CM (3x)"].ravel()
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

    # Paired HLN-DM tests on Holiday / Eid subset (n = 41, Ref = Ridge)
    g_hol = te_m[te_m["Cal_Regime"] == "Holiday / Eid"]
    hol_test_rows = []
    for lbl, col in [("PatchTST-CM (3x)", "Pred_PT3"), ("Equal-Weight RNN Ens", "Pred_EqRNN"),
                     ("XGBoost-Diff (Δy)", "Pred_XGBD"), ("Weekday-Adj Persistence", "Pred_Wkday")]:
        dm_h, p_h, d_h, lo_h, hi_h = dm_hln_and_bootstrap(
            g_hol["Actual"].values, g_hol["Pred_Ridge"].values, g_hol[col].values, max_lag=5, block_len=3
        )
        hol_test_rows.append({
            "Comparison": f"Ridge vs {lbl}", "N": len(g_hol), "Ridge_MAE": float(mean_absolute_error(g_hol["Actual"], g_hol["Pred_Ridge"])),
            "Other_MAE": float(mean_absolute_error(g_hol["Actual"], g_hol[col])),
            "Delta_MW": d_h, "CI_lo": lo_h, "CI_hi": hi_h, "DM_HLN": dm_h, "p_val": p_h,
        })
    df_hol_tests = pd.DataFrame(hol_test_rows)
    print("\n=== 7A. HOLIDAY / EID PAIRED TESTS (n=41, REF=RIDGE) ===")
    print(df_hol_tests.round(3).to_string(index=False))

    # Stratify Table 13 by KNOWN lagged peak demand y_{t-1} (eliminates target-selection bias!)
    bins = [0, 11000, 13000, 15000, 25000]
    bin_labels = ["Low (<11 GW)", "Moderate (11-13 GW)", "High (13-15 GW)", "Extreme Peak (>15 GW)"]
    te_m["Lag1_Bin"] = pd.cut(te_m["Lag1_Demand"], bins=bins, labels=bin_labels)
    bin_rows = []
    for lbl in bin_labels:
        g = te_m[te_m["Lag1_Bin"] == lbl]
        bin_rows.append({
            "Lag1_Demand_Bin": lbl, "N": len(g),
            "PT3_MAE": float(mean_absolute_error(g["Actual"], g["Pred_PT3"])),
            "Ridge_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Ridge"])),
            "EqRNN_MAE": float(mean_absolute_error(g["Actual"], g["Pred_EqRNN"])),
            "XGBD_MAE": float(mean_absolute_error(g["Actual"], g["Pred_XGBD"])),
            "XGB_MAE": float(mean_absolute_error(g["Actual"], g["Pred_XGB"])),
            "Wkday_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Wkday"])),
        })
    df_bins = pd.DataFrame(bin_rows)
    print("\n=== 7B. LAGGED-DEMAND (y_{t-1}) BINS (NO REALIZED-TARGET SELECTION BIAS) ===")
    print(df_bins.round(1).to_string(index=False))

    # 9. Gap-Handling Audit (420 Gap-Free vs 42 Gap-Spanning Test Windows)
    gspan = ds["te_gap_span"]
    gap_rows = []
    for grp_lbl, mask in [("Gap-Free Windows (cal span = 14d)", ~gspan), ("Gap-Spanning Windows (cal span > 14d)", gspan)]:
        gap_rows.append({
            "Subset": grp_lbl, "N": int(mask.sum()),
            "PT3_MAE": float(mean_absolute_error(y_te_inv[mask], mean_preds["PatchTST-CM (3x)"][mask])),
            "Ridge_MAE": float(mean_absolute_error(y_te_inv[mask], mean_preds["Ridge (14xF)"][mask])),
            "EqRNN_MAE": float(mean_absolute_error(y_te_inv[mask], mean_preds["Equal-Weight RNN Ens"][mask])),
            "Wkday_MAE": float(mean_absolute_error(y_te_inv[mask], mean_preds["Weekday-Adj Persistence"][mask])),
            "Lag7_MAE": float(mean_absolute_error(y_te_inv[mask], mean_preds["Seasonal Naive (Lag-7)"][mask])),
        })
    df_gap = pd.DataFrame(gap_rows)
    print("\n=== 8. CALENDAR-GAP SENSITIVITY AUDIT (420 CLEAN vs 42 GAP-SPANNING TEST WINDOWS) ===")
    print(df_gap.round(1).to_string(index=False))

    # 10. Eid +/-1-Day Lunar-Sighting Jitter Sensitivity Test (Frozen Seed-42 Trained Models!)
    jitter_rows = []
    for shift in [-1, 0, +1]:
        if shift == 0:
            X_jit = X_te
        else:
            ds_j = prepare_dataset(use_cdd=True, use_eid=True, use_pca=False, tau_override=ds["tau_star"], eid_jitter_test=shift)
            X_jit = ds_j["X_te"]
        X_jit_2d = X_jit.reshape(len(X_jit), -1)
        p_pt_j = np.mean([sy.inverse_transform(predict_torch(m, X_jit)) for m in seed0_pt3_models], axis=0)
        p_rnn_j = np.mean([sy.inverse_transform(predict_torch(m, X_jit)) for m in seed0_rnn_models], axis=0)
        p_rid_j = sy.inverse_transform(ridge_mod.predict(X_jit_2d).reshape(-1, 1))
        jitter_rows.append({
            "Eid_Shift_Days": shift,
            "PT3_All_MAE": float(mean_absolute_error(y_te_inv, p_pt_j)),
            "PT3_Hol_MAE": float(mean_absolute_error(y_te_inv[hol_mask], p_pt_j[hol_mask])),
            "Ridge_All_MAE": float(mean_absolute_error(y_te_inv, p_rid_j)),
            "Ridge_Hol_MAE": float(mean_absolute_error(y_te_inv[hol_mask], p_rid_j[hol_mask])),
            "EqRNN_All_MAE": float(mean_absolute_error(y_te_inv, p_rnn_j)),
            "EqRNN_Hol_MAE": float(mean_absolute_error(y_te_inv[hol_mask], p_rnn_j[hol_mask])),
        })
    df_jitter = pd.DataFrame(jitter_rows)
    print("\n=== 9. EID +/-1-DAY LUNAR-SIGHTING JITTER SENSITIVITY (FROZEN SEED-42 MODELS) ===")
    print(df_jitter.round(1).to_string(index=False))

    # 11. Summer-Inclusive Validation Sensitivity Check (Tune on Full-Year 2024 vs Default Summer-Free Val-ES)
    df_enc_full = ds["df_enc"]
    tr23 = df_enc_full.loc[: "2023-12-31"]
    va24 = df_enc_full.loc["2024-01-01" : "2024-12-31"]  # Includes full Summer 2024!
    sx24 = MinMaxScaler().fit(tr23[ds["cont_cols"]])
    sy24 = MinMaxScaler().fit(tr23[["Max. Demand at eve. peak (Generation end)"]])
    def pack24(sub):
        return np.concatenate([sx24.transform(sub[ds["cont_cols"]]), sub[ds["cat_onehot_cols"]].values.astype(np.float32),
                               sy24.transform(sub[["Max. Demand at eve. peak (Generation end)"]])], axis=1)
    X_tr23, y_tr23 = windowing(pack24(tr23), pack24(tr23)[:, -1], HIST_SIZE)
    X_va24, y_va24 = windowing(pack24(va24), pack24(va24)[:, -1], HIST_SIZE)
    # Tune Ridge alpha on full-year 2024 (summer-inclusive!)
    best_a24, best_m24 = 10.0, float("inf")
    for a in [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]:
        m24 = Ridge(alpha=a).fit(X_tr23.reshape(len(X_tr23), -1), y_tr23.ravel())
        mae24 = mean_absolute_error(y_va24.ravel(), m24.predict(X_va24.reshape(len(X_va24), -1)))
        if mae24 < best_m24: best_m24, best_a24 = mae24, a
    # Evaluate Ridge with alpha_2024 on standard Train -> 2025-26 Test
    p_rid_a24 = sy.inverse_transform(Ridge(alpha=best_a24).fit(X_tr_2d, y_tr.ravel()).predict(X_te_2d).reshape(-1, 1))
    # Tune PatchTST-CM LR on full-year 2024 (summer-inclusive!)
    best_lr24, best_vl24 = 1e-3, float("inf")
    for lr_c in [1e-3, 2.5e-3]:
        _, _, vl24 = train_torch_model(PatchTSTCMModel(n_f), X_tr23, y_tr23, X_va24, y_va24, 42, lr=lr_c)
        if vl24 < best_vl24: best_vl24, best_lr24 = vl24, lr_c
    # Evaluate PatchTST-CM (3x, Seed 42) with lr_2024 on standard Train -> 2025-26 Test
    if best_lr24 == tuned_lrs["PatchTST-CM (Single)"]:
        pt_a24_mae = float(mean_absolute_error(y_te_inv, seed_preds[42]["PatchTST-CM (3x)"]))
    else:
        pt_l24 = []
        for off in [0, 1000, 2000]:
            m24p, _, _ = train_torch_model(PatchTSTCMModel(n_f), X_tr, y_tr, X_va_es, y_va_es, 42 + off, lr=best_lr24)
            pt_l24.append(sy.inverse_transform(predict_torch(m24p, X_te)))
        pt_a24_mae = float(mean_absolute_error(y_te_inv, np.mean(pt_l24, axis=0)))
    df_val_sens = pd.DataFrame([
        {"Model": "Ridge (14xF)", "ValES_Param": f"alpha={ridge_alpha:g}", "ValES_Test_MAE": float(mean_absolute_error(y_te_inv, p_ridge_te)),
         "Year2024_Param": f"alpha={best_a24:g}", "Year2024_Test_MAE": float(mean_absolute_error(y_te_inv, p_rid_a24))},
        {"Model": "PatchTST-CM (3x, S42)", "ValES_Param": f"lr={tuned_lrs['PatchTST-CM (Single)']:g}", "ValES_Test_MAE": float(mean_absolute_error(y_te_inv, seed_preds[42]["PatchTST-CM (3x)"])),
         "Year2024_Param": f"lr={best_lr24:g}", "Year2024_Test_MAE": pt_a24_mae},
    ])
    print("\n=== 10. SUMMER-INCLUSIVE VALIDATION SENSITIVITY (FULL-YEAR 2024 vs JUL-NOV VAL-ES) ===")
    print(df_val_sens.round(2).to_string(index=False))

    # ==========================================================================
    # REGENERATE ZERO-OVERLAP VECTOR PDF FIGURES
    # ==========================================================================
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    # Fig 1: Forecast comparison
    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    ax.plot(ds["te_dates"], y_te_inv, color="#111111", lw=1.35, label="Actual Peak Demand")
    ax.plot(ds["te_dates"], mean_preds["PatchTST-CM (3x)"], color="#0d9488", lw=1.25, label="PatchTST-CM (3×)")
    ax.plot(ds["te_dates"], mean_preds["Ridge (14xF)"], color="#d97706", lw=1.1, ls="--", alpha=0.85, label="Ridge (14×F)")
    ax.plot(ds["te_dates"], mean_preds["Equal-Weight RNN Ens"], color="#1d3557", lw=1.1, ls="-.", alpha=0.85, label="Equal-Weight RNN Ens")
    ax.set_ylabel("Peak Demand (MW)")
    ax.set_title("National Evening Peak Demand — Representative Models (462 Blind Test Days)", pad=28)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=8.2)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig1_forecast_comparison.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 2: Convergence curves
    fig, ax = plt.subplots(figsize=(8.8, 4.7))
    colors = {"DLinear (CI)": "#6d597a", "MLP (14xF)": "#84a98c", "PatchTST-CM (Single)": "#0284c7",
              "PatchTST-RevIN (Single)": "#14b8a6", "BiGRU": "#1d3557", "BiLSTM": "#d97706",
              "BiLSTM-Attention": "#2a9d8f", "Transformer": "#e76f51", "CNN-BiGRU-Transformer": "#9b2226"}
    for name, hlist in val_histories.items():
        max_len = max(len(h) for h in hlist)
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in hlist])
        mu, sd = padded.mean(axis=0), padded.std(axis=0)
        ep = np.arange(1, max_len + 1)
        ax.plot(ep, mu, lw=1.5, color=colors[name], label=name)
        ax.fill_between(ep, mu - sd, mu + sd, color=colors[name], alpha=0.09)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val-ES Loss (Normalized MAE)")
    ax.set_title("Multi-Seed Val-ES Convergence Across Neural Families (Mean ± 1 SD, 5 Seeds)", pad=58)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7.4)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig2_convergence_curves.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 3: Descriptive HLN-DM Forest Plot (Ref = PatchTST-CM 3x)
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    y_pos = np.arange(len(df_dm))
    c_bars = ["#0d9488" if p < 0.01 else ("#2563eb" if p < 0.05 else "#94a3b8") for p in df_dm["Ens5_p_holm"]]
    ax.axvline(0, color="black", ls="--", lw=1.0)
    for i, (_, r) in enumerate(df_dm.iterrows()):
        ax.plot([r["Ens5_CI_lo"], r["Ens5_CI_hi"]], [i, i], color=c_bars[i], lw=2.1)
        ax.plot(r["Ens5_Delta_MW"], i, "o", color=c_bars[i], ms=5.2)
    ax.set_yticks(y_pos); ax.set_yticklabels(df_dm["Comparator"], fontsize=7.9)
    ax.set_xlabel("ΔMAE: PatchTST-CM (3×) − Comparator (MW)  [Negative = PatchTST-CM Lower Error]")
    ax.set_title("Descriptive HLN-Adjusted Diebold–Mariano Tests (Ref: PatchTST-CM 3×; Holm–Bonferroni)", pad=28)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[
        mpatches.Patch(color="#0d9488", label="p_Holm < 0.01"),
        mpatches.Patch(color="#2563eb", label="p_Holm < 0.05"),
        mpatches.Patch(color="#94a3b8", label="Not significant (p_Holm ≥ 0.05)"),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=8.0)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig3_dm_forest.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 4: Architecture-Dependent Ablation (PT3 + Eq-RNN + Ridge) + Regime Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    short_lbl = ["1. Base", "2. +CDD₂₅", "3. +Eid (Full)", "4. +PCA (95%)"]
    xa = np.arange(len(short_lbl)); wa = 0.25
    axes[0].bar(xa - wa, df_abl["PT3_Test_MAE"], yerr=df_abl["PT3_Test_SD"], capsize=3, width=wa, color="#0d9488", label="PatchTST-CM (3×)")
    axes[0].bar(xa, df_abl["EqRNN_Test_MAE"], yerr=df_abl["EqRNN_Test_SD"], capsize=3, width=wa, color="#1d3557", label="Eq-Wt RNN Ens")
    axes[0].bar(xa + wa, df_abl["Ridge_Test_MAE"], width=wa, color="#d97706", label="Ridge (14×F)")
    axes[0].set_xticks(xa); axes[0].set_xticklabels(short_lbl, fontsize=8.3)
    axes[0].set_ylabel("Blind Test MAE (MW)"); axes[0].set_ylim(500, max(df_abl["PT3_Test_MAE"].max(), df_abl["EqRNN_Test_MAE"].max()) + 65)
    axes[0].set_title("(a) Architecture-Dependent Feature & PCA Ablation (5 Seeds)", pad=32)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=7.6)

    cal_df = df_reg[df_reg["Axis"] == "Cal_Regime"].set_index("Regime").loc[["Working Weekday", "Weekend (Fri-Sat)", "Holiday / Eid"]]
    xc = np.arange(len(cal_df)); w = 0.20
    axes[1].bar(xc - 1.5*w, cal_df["Wkday_MAE"], width=w, color="#94a3b8", label="Wkday Persist.")
    axes[1].bar(xc - 0.5*w, cal_df["Ridge_MAE"], width=w, color="#d97706", label="Ridge (14×F)")
    axes[1].bar(xc + 0.5*w, cal_df["EqRNN_MAE"], width=w, color="#1d3557", label="Eq-Wt RNN")
    axes[1].bar(xc + 1.5*w, cal_df["PT3_MAE"], width=w, color="#0d9488", label="PatchTST-CM (3×)")
    axes[1].set_xticks(xc); axes[1].set_xticklabels(["Working\nWeekday", "Weekend\n(Fri–Sat)", "Holiday / Eid\n(n=41)"], fontsize=8.3)
    axes[1].set_ylabel("Blind Test MAE (MW)"); axes[1].set_ylim(0, cal_df[["Wkday_MAE", "Ridge_MAE", "EqRNN_MAE", "PT3_MAE"]].values.max() * 1.16)
    axes[1].set_title("(b) Calendar Regime Error (5-Seed Ensemble Averages)", pad=32)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False, fontsize=7.3)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig4_ablation_regime.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 6: Lagged-Demand (y_{t-1}) Bins (No Realized-Target Bias!)
    fig, ax = plt.subplots(figsize=(9.8, 4.4))
    xb = np.arange(len(df_bins)); wb = 0.16
    ax.bar(xb - 2*wb, df_bins["XGB_MAE"], width=wb, color="#e76f51", label="XGBoost (Level)")
    ax.bar(xb - wb, df_bins["XGBD_MAE"], width=wb, color="#f4a261", label="XGBoost-Diff (Δy)")
    ax.bar(xb, df_bins["Wkday_MAE"], width=wb, color="#94a3b8", label="Wkday Persist.")
    ax.bar(xb + wb, df_bins["Ridge_MAE"], width=wb, color="#d97706", label="Ridge (14×F)")
    ax.bar(xb + 2*wb, df_bins["PT3_MAE"], width=wb, color="#0d9488", label="PatchTST-CM (3×)")
    ax.set_xticks(xb); ax.set_xticklabels([f"{r['Lag1_Demand_Bin']}\n(n={r['N']})" for _, r in df_bins.iterrows()], fontsize=8.5)
    ax.set_ylabel("Blind Test MAE (MW)"); ax.set_ylim(0, df_bins[["XGB_MAE", "XGBD_MAE", "Wkday_MAE", "Ridge_MAE", "PT3_MAE"]].values.max() * 1.15)
    ax.set_title("Blind Test Error Stratified by Known Prior-Day Peak Demand y(t−1) (No Target-Selection Bias)", pad=28)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False, fontsize=7.6)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig6_demand_bins.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 7: Qualitative High-Stress Events
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.3))
    events = [
        ("Eid-ul-Fitr 2025", "2025-03-26", "2025-04-08"),
        ("Eid-ul-Adha 2025", "2025-06-02", "2025-06-15"),
        ("April 2026 Peak Heatwave", "2026-04-12", "2026-04-30"),
    ]
    for ax, (title, s_d, e_d) in zip(axes, events):
        mask = (ds["te_dates"] >= s_d) & (ds["te_dates"] <= e_d)
        d_sub = ds["te_dates"][mask]
        yt_s = y_te_inv.ravel()[mask]
        ypt_s = mean_preds["PatchTST-CM (3x)"].ravel()[mask]
        yr_s = mean_preds["Ridge (14xF)"].ravel()[mask]
        yxd_s = mean_preds["XGBoost-Diff (Δy)"].ravel()[mask]
        ax.plot(d_sub, yt_s, color="#111111", lw=1.7, label="Actual")
        ax.plot(d_sub, ypt_s, color="#0d9488", lw=1.45, label=f"PT-CM(3×) ({mean_absolute_error(yt_s, ypt_s):.0f})")
        ax.plot(d_sub, yr_s, color="#d97706", lw=1.2, ls="-.", label=f"Ridge ({mean_absolute_error(yt_s, yr_s):.0f})")
        ax.plot(d_sub, yxd_s, color="#e76f51", lw=1.1, ls="--", label=f"XGB-Diff ({mean_absolute_error(yt_s, yxd_s):.0f})")
        ax.set_title(title, fontsize=9.3, pad=36)
        ax.set_ylabel("Peak Demand (MW)")
        ax.tick_params(axis="x", rotation=25, labelsize=7.5)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.4)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig7_qualitative_events.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 8: Efficiency Pareto (Zero Overlap!)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    nn_names = ["Ridge (14xF)", "DLinear (CI)", "CNN-BiGRU-Transformer", "PatchTST-CM (Single)",
                "PatchTST-RevIN (CI, 3x)", "BiLSTM-Attention", "Transformer", "MLP (14xF)", "BiGRU",
                "BiLSTM", "PatchTST-CM (3x)", "Equal-Weight RNN Ens"]
    pts_x = [eff_profile[m]["Params"] / 1000.0 for m in nn_names]
    pts_y = [float(df_summary.loc[m, "MAE_mean"]) for m in nn_names]
    cols_p = ["#0d9488" if "PatchTST" in m else ("#d97706" if "Ridge" in m else ("#1d3557" if "RNN" in m else "#64748b")) for m in nn_names]
    sizes_p = [130 if m in ("PatchTST-CM (3x)", "Ridge (14xF)", "Equal-Weight RNN Ens") else 65 for m in nn_names]
    axes[0].scatter(pts_x, pts_y, c=cols_p, s=sizes_p, zorder=4, edgecolors="white")
    offsets = {
        "Ridge (14xF)": (-2, 12), "DLinear (CI)": (4, 10), "CNN-BiGRU-Transformer": (8, -11),
        "PatchTST-CM (Single)": (-46, -15), "PatchTST-RevIN (CI, 3x)": (10, -10),
        "BiLSTM-Attention": (-42, 9), "Transformer": (10, 4), "MLP (14xF)": (10, -4),
        "BiGRU": (-12, 10), "BiLSTM": (10, -2), "PatchTST-CM (3x)": (12, 2), "Equal-Weight RNN Ens": (-128, 8)
    }
    for m, px, py in zip(nn_names, pts_x, pts_y):
        dx, dy = offsets.get(m, (8, 4))
        axes[0].annotate(m, (px, py), xytext=(dx, dy), textcoords="offset points",
                         fontsize=7.0, fontweight="bold" if m in ("PatchTST-CM (3x)", "Ridge (14xF)", "Equal-Weight RNN Ens") else "normal")
    axes[0].set_xlabel("Trainable Parameters (Thousands)"); axes[0].set_ylabel("Blind Test MAE (MW)")
    axes[0].set_title("(a) Accuracy vs. Parameter Count (5-Seed)")
    axes[0].set_xlim(-35, 900); axes[0].set_ylim(545, max(pts_y) + 25); axes[0].grid(alpha=0.25)

    lat_models = ["Ridge (14xF)", "DLinear (CI)", "XGBoost-Diff (Δy)", "MLP (14xF)", "LightGBM",
                  "PatchTST-CM (Single)", "Transformer", "BiLSTM", "BiGRU", "BiLSTM-Attention",
                  "PatchTST-RevIN (CI, 3x)", "PatchTST-CM (3x)", "Equal-Weight RNN Ens"]
    lats = [eff_profile[m]["CPU_ms"] for m in lat_models]
    bar_c = ["#0d9488" if "PatchTST" in m else ("#d97706" if "Ridge" in m else ("#1d3557" if "RNN" in m else "#64748b")) for m in lat_models]
    axes[1].barh(lat_models, lats, color=bar_c, height=0.62)
    for i, v in enumerate(lats): axes[1].text(v + 0.025, i, f"{v:.2f} ms", va="center", fontsize=7.2)
    axes[1].set_xlabel(f"Single-Sample CPU Latency (ms) — {cpu_brand}"); axes[1].set_title("(b) CPU Inference Latency Profile")
    axes[1].set_xlim(0, max(lats) * 1.25); axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig8_efficiency_pareto.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 9: Direct Group Permutation Importance on PatchTST-CM (3x)
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    df_p_plot = df_perm.sort_values("Delta_MAE_Mean_MW", ascending=True)
    ax.barh(df_p_plot["Feature_Group"], df_p_plot["Delta_MAE_Mean_MW"],
            xerr=df_p_plot["Delta_MAE_Std_MW"], capsize=3, color="#0d9488", height=0.56)
    for i, (_, r) in enumerate(df_p_plot.iterrows()):
        x_txt = max(0.0, r["Delta_MAE_Mean_MW"]) + r["Delta_MAE_Std_MW"] + 20
        ax.text(x_txt, i, f"{r['Delta_MAE_Mean_MW']:+.1f} MW (k={int(r['Num_Features'])})", va="center", fontsize=7.8)
    ax.set_xlabel("Increase in Deployed PatchTST-CM (3×) MAE upon Block Permutation (MW)")
    ax.set_title("Direct Group Permutation Importance on PatchTST-CM (3×) (Merged Load Block, 15 Repeats)")
    ax.set_xlim(-25, (df_p_plot["Delta_MAE_Mean_MW"] + df_p_plot["Delta_MAE_Std_MW"]).max() * 1.28)
    ax.axvline(0, color="black", lw=0.8); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig9_shap_attribution.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 11: Conformal Intervals on PatchTST-CM (3x)
    fig, axes = plt.subplots(3, 1, figsize=(10.2, 7.6), sharex=True)
    panels = [
        (axes[0], cp_store_pt3["1. Static Split Conformal"], "#94a3b8",
         f"(a) Static Split Conformal — 90% PICP={cp_store_pt3['1. Static Split Conformal']['PICP_%']:.1f}% (±{cp_store_pt3['1. Static Split Conformal']['PICP_CI_hw']:.1f} pp), Winkler={cp_store_pt3['1. Static Split Conformal']['Winkler_MW']:.0f} MW"),
        (axes[1], cp_store_pt3["4. Plain ACI (No Local Scale)"], "#d97706",
         f"(b) Plain ACI (Best Winkler) — 90% PICP={cp_store_pt3['4. Plain ACI (No Local Scale)']['PICP_%']:.1f}% (±{cp_store_pt3['4. Plain ACI (No Local Scale)']['PICP_CI_hw']:.1f} pp), Winkler={cp_store_pt3['4. Plain ACI (No Local Scale)']['Winkler_MW']:.0f} MW"),
        (axes[2], cp_store_pt3["5. Locally Normalized ACI (LN-ACI)"], "#0d9488",
         f"(c) Locally Normalized ACI (LN-ACI) — 90% PICP={cp_store_pt3['5. Locally Normalized ACI (LN-ACI)']['PICP_%']:.1f}% (±{cp_store_pt3['5. Locally Normalized ACI (LN-ACI)']['PICP_CI_hw']:.1f} pp), Winkler={cp_store_pt3['5. Locally Normalized ACI (LN-ACI)']['Winkler_MW']:.0f} MW"),
    ]
    for ax, obj, colr, ttl in panels:
        ax.fill_between(ds["te_dates"], obj["lo"], obj["hi"], color=colr, alpha=0.28,
                        label=f"90% PI (Mean Width {obj['Mean_Width_MW']:.0f} MW)")
        ax.plot(ds["te_dates"], y_te_inv, color="#111111", lw=1.0, label="Actual Demand")
        ax.plot(ds["te_dates"], mean_preds["PatchTST-CM (3x)"], color="#0f766e", lw=0.95, label="PatchTST-CM (3×)")
        ax.set_ylabel("Peak (MW)")
        ax.set_title(ttl, fontsize=9.0, pad=24)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False, fontsize=7.8)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig11_conformal_intervals.pdf", bbox_inches="tight"); plt.close(fig)

    # Save ALL CSVs & metadata
    df_seeds.to_csv(f"{OUT_DIR}/part1_per_seed_metrics.csv", index=False)
    df_summary.to_csv(f"{OUT_DIR}/part1_summary_metrics.csv")
    df_dm.to_csv(f"{OUT_DIR}/part1_dm_tests.csv", index=False)
    df_mcs.to_csv(f"{OUT_DIR}/part1_mcs.csv", index=False)
    df_cp.to_csv(f"{OUT_DIR}/part1_conformal.csv", index=False)
    df_cp_sea.to_csv(f"{OUT_DIR}/part1_conformal_seasonal.csv", index=False)
    df_abl.to_csv(f"{OUT_DIR}/part1_ablation.csv", index=False)
    df_reg.to_csv(f"{OUT_DIR}/part1_regime_analysis.csv", index=False)
    df_hol_tests.to_csv(f"{OUT_DIR}/part1_regime_tests.csv", index=False)
    df_bins.to_csv(f"{OUT_DIR}/part1_demand_bins.csv", index=False)
    df_gap.to_csv(f"{OUT_DIR}/part1_gap_audit.csv", index=False)
    df_jitter.to_csv(f"{OUT_DIR}/part1_eid_jitter.csv", index=False)
    df_val_sens.to_csv(f"{OUT_DIR}/part1_val_sensitivity.csv", index=False)
    df_perm.to_csv(f"{OUT_DIR}/part1_permutation_importance.csv", index=False)
    pd.DataFrame(eff_profile).T.to_csv(f"{OUT_DIR}/part1_efficiency.csv")

    mean_w = np.mean(learned_weights, axis=0)
    meta_out = {
        "cpu_hardware": cpu_brand, "torch_threads": TORCH_THREADS, "blas_threads": 1,
        "n_clean": ds["n_raw"], "n_features": ds["n_features"], "year_dropped": True,
        "tau_star_C": ds["tau_star"], "tau_slope_detrended_MW_per_C": round(ds["tau_slope"], 2),
        "secular_trend_MW_per_yr": round(ds["trend_mw_per_yr"], 1),
        "ridge_alpha": ridge_alpha, "ridge_diff_alpha": ridged_alpha,
        "tuned_neural_lrs": tuned_lrs,
        "n_train": len(X_tr), "n_val_total": len(ds["X_va"]),
        "n_val_es": len(X_va_es), "n_val_cal": len(X_va_cal), "n_test": len(X_te),
        "seeds": SEEDS, "wkday_persist_mae": round(wkday_mae, 2),
        "pt3_s42_mae": round(float(mean_absolute_error(y_te_inv, seed_preds[42][ref_name])), 2),
        "pt3_5seed_mae": round(float(mean_absolute_error(y_te_inv, mean_preds[ref_name])), 2),
        "learned_weights_per_seed": {str(s): [round(x, 4) for x in w] for s, w in zip(SEEDS, learned_weights)},
        "learned_weights_mean": [round(float(x), 4) for x in mean_w],
        "data_qa": ds["data_qa"],
        "efficiency": eff_profile,
    }
    with open(f"{OUT_DIR}/part1_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nALL v4 BENCHMARK EXPERIMENTS COMPLETED IN {time.perf_counter() - t_start:.1f}s!", flush=True)
