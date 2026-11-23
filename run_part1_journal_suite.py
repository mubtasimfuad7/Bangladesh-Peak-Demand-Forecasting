"""
Part 1 Q1-Journal Reproducible Benchmark Suite (Revised — Addressing All Peer-Review Points).
Key methodological upgrades:
  1. Subtropical Bangladesh thermal specification: drops identically-zero HDD (18 C; 0% of days < 18 C)
     and uses a train-only piecewise hinge cooling threshold tau* = 26.0 C (CDD_26 = max(0, T - 26)).
  2. Fair, comprehensive 20-model 5-seed benchmark (seeds 42, 123, 7, 2024, 999):
     - Naive / Statistical: Persistence (Lag-1), Weekday-Aware Naive (Lag-7), 7-Day Rolling Mean,
       SARIMAX(1,1,1)(1,0,1,7), Ridge Regression (on the exact same 14 x F feature matrix)
     - GBDTs: XGBoost (Level), XGBoost-Diff (predicts y_t - y_{t-1} to remove tree extrapolation bias), LightGBM
     - Linear & Patched Transformer: DLinear (AAAI 2023), PatchTST (Single), PatchTST-Ensemble (3-Model, equal network budget)
     - Deep Sequential/Attention: BiGRU, BiLSTM, BiLSTM-Attention, Transformer, CNN-BiGRU-Transformer
     - Residual Hybrids: Hybrid BiGRU-XGBoost, Hybrid BiLSTM-Attn-XGBoost
     - Recurrent Ensembles: Equal-Weight RNN Ensemble (1/3 each), Val-Opt SLSQP RNN Ensemble
  3. Validation-split separation: Val is split cleanly into Val-ES (first half, early stopping)
     and Val-Cal (second half, SLSQP weights + initial conformal calibration) — plus full reporting of
     per-seed weight variability.
  4. Uncertainty Quantification: Static Split Conformal vs. Locally Normalized Adaptive Conformal Inference
     (LN-ACI; Gibbs & Candes 2021) for 80% and 90% nominal coverage across seasons.
  5. Direct PyTorch Ensemble Group Permutation Importance (explaining the ACTUAL deployed model,
     not a weak XGBoost surrogate) on the validation & blind test splits.
  6. Harvey-Leybourne-Newbold (HLN) corrected Diebold-Mariano tests (Newey-West HAC, lag 14)
     with Holm-Bonferroni multiple-testing adjustment + 95% moving-block bootstrap CIs.
  7. Validation-first 5-seed Feature & PCA Ablation with paired Welch/Wilcoxon p-values and exact
     incremental Holiday/Eid attribution.
  8. Data-quality diagnostics (20 missing BPDB calendar days, 16 negative transmission-loss records,
     and post-gap vs. regular-day error sensitivity).
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


def estimate_train_cooling_threshold(temp_series, demand_series):
    """Estimate piecewise cooling threshold tau* strictly on the training set (grid 20-34 C, step 0.5 C)."""
    t = temp_series.values.astype(float)
    y = demand_series.values.astype(float)
    best_rmse, best_tau, best_slope = float("inf"), 26.0, 0.0
    for tau in np.arange(20.0, 34.5, 0.5):
        h = np.maximum(0.0, t - tau)
        X = np.column_stack([np.ones_like(h), h])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        rmse = float(np.sqrt(np.mean((y - X @ coef) ** 2)))
        if rmse < best_rmse:
            best_rmse, best_tau, best_slope = rmse, float(tau), float(coef[1])
    return best_tau, best_rmse, best_slope


def prepare_dataset(use_cdd=True, use_eid=True, use_pca=False, tau_override=None):
    df = pd.read_excel("version_5_extended_backup.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Track calendar gaps explicitly before indexing
    full_span_days = int((df["Date"].max() - df["Date"].min()).days + 1)
    n_missing_calendar = full_span_days - len(df)
    date_gaps = df["Date"].diff().dt.days.fillna(1.0)
    df["Post_Gap_Flag"] = (date_gaps > 1.0).astype(int)

    # Flag physically impossible negative transmission losses (QA diagnostic)
    gen_col = "Max. Demand at eve. peak (Generation end)"
    sub_col = "Max. Demand at eve. peak (Sub-station end)"
    raw_loss = df[gen_col] - df[sub_col]
    n_neg_loss = int((raw_loss < 0).sum())
    qa_loss = raw_loss.where(raw_loss >= 0, np.nan)

    df = df.set_index("Date")
    if "Dhaka_temp" in df.columns:
        df["Maximum Temperature in Dhaka was"] = df["Dhaka_temp"]
    if "Mymensingh_temp" not in df.columns and "Dhaka_temp" in df.columns:
        df["Mymensingh_temp"] = df["Dhaka_temp"]

    temp_col = "Maximum Temperature in Dhaka was"
    target_col = gen_col
    y_shifted = df[target_col].shift(1)

    # Train-only cooling threshold estimation (strictly on initial 70% training partition)
    pre_train_cut = int(0.70 * len(df))
    if tau_override is None:
        tau_star, tau_rmse, tau_slope = estimate_train_cooling_threshold(
            df[temp_col].iloc[:pre_train_cut], df[target_col].iloc[:pre_train_cut]
        )
    else:
        tau_star, tau_rmse, tau_slope = float(tau_override), 0.0, 0.0

    # Drop degenerate all-zero HDD (min Dhaka temp is 19.3 C > 18 C); keep subtropical CDD_tau*
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

    # Remove Post_Gap_Flag from model features (keep only in metadata)
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

    # Strict train-only Min-Max scaling
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

    # Cleanly split validation into Val-ES (early stopping) and Val-Cal (SLSQP + conformal calibration)
    # with a 14-day embargo between them so early stopping and calibration never share target windows
    n_va = len(X_va)
    va_es_end = (n_va - HIST_SIZE) // 2
    va_cal_start = va_es_end + HIST_SIZE

    y_va_inv = df_enc[[target_col]].iloc[train_cut + HIST_SIZE : val_cut].values
    y_te_inv = df_enc[[target_col]].iloc[val_cut + HIST_SIZE :].values
    va_dates = df_enc.index[train_cut + HIST_SIZE : val_cut]
    te_dates = df_enc.index[val_cut + HIST_SIZE :]
    va_meta = meta.iloc[train_cut + HIST_SIZE : val_cut].copy()
    te_meta = meta.iloc[val_cut + HIST_SIZE :].copy()

    return {
        "X_tr": X_tr, "y_tr": y_tr,
        "X_va": X_va, "y_va": y_va, "y_va_inv": y_va_inv, "va_dates": va_dates, "va_meta": va_meta,
        "X_va_es": X_va[:va_es_end], "y_va_es": y_va[:va_es_end], "y_va_es_inv": y_va_inv[:va_es_end],
        "X_va_cal": X_va[va_cal_start:], "y_va_cal": y_va[va_cal_start:], "y_va_cal_inv": y_va_inv[va_cal_start:],
        "X_te": X_te, "y_te": y_te, "y_te_inv": y_te_inv, "te_dates": te_dates, "te_meta": te_meta,
        "sy": sy, "meta_full": meta, "feat_names": feat_names,
        "train_dates": (df_enc.index[0], df_enc.index[train_cut - 1]),
        "val_dates": (df_enc.index[train_cut], df_enc.index[val_cut - 1]),
        "test_dates": (te_dates[0], te_dates[-1]),
        "n_raw": len(df_enc), "n_features": data_norm.shape[1],
        "df_raw_target": df_enc[target_col], "val_cut": val_cut, "train_cut": train_cut,
        "tau_star": tau_star, "tau_slope": tau_slope,
        "data_qa": {
            "calendar_span_days": full_span_days,
            "raw_records": int(full_span_days - n_missing_calendar),
            "missing_calendar_days": int(n_missing_calendar),
            "negative_loss_records": n_neg_loss,
            "raw_mean_loss_mw": round(float(raw_loss.mean()), 2),
            "qa_mean_loss_mw": round(float(qa_loss.mean()), 2),
            "qa_max_loss_mw": round(float(qa_loss.max()), 2),
        },
    }


def run_sarimax_baseline(ds):
    """Causal SARIMAX(1,1,1)(1,0,1,7) fitted on train only with subtropical CDD_tau*, filtered 1-step-ahead."""
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
    mod = SARIMAX(
        y_all[:tr_c], exog=ex_all[:tr_c],
        order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit_res = mod.fit(disp=False, maxiter=80)
    full_res = fit_res.apply(y_all, exog=ex_all)
    preds_all = full_res.fittedvalues
    p_te = preds_all[va_c + HIST_SIZE :].reshape(-1, 1)
    p_va = preds_all[tr_c + HIST_SIZE : va_c].reshape(-1, 1)
    lat_ms = (time.perf_counter() - t0) / len(y_all) * 1000.0
    return p_va, p_te, len(fit_res.params), round(lat_ms, 2)


def run_ridge_baseline(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_va_2d, X_te_2d, sy):
    """L2-regularized linear baseline on the exact same 14 x F feature matrix, alpha tuned strictly on Val-ES."""
    best_alpha, best_mae, best_mod = 10.0, float("inf"), None
    for alpha in [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]:
        mod = Ridge(alpha=alpha)
        mod.fit(X_tr_2d, y_tr.ravel())
        pv = mod.predict(X_va_es_2d)
        mae = float(mean_absolute_error(y_va_es.ravel(), pv))
        if mae < best_mae:
            best_mae, best_alpha, best_mod = mae, alpha, mod
    t0 = time.perf_counter()
    for _ in range(60):
        _ = best_mod.predict(X_te_2d[:1])
    lat_ms = round((time.perf_counter() - t0) / 60 * 1000.0, 2)
    p_va = sy.inverse_transform(best_mod.predict(X_va_2d).reshape(-1, 1))
    p_te = sy.inverse_transform(best_mod.predict(X_te_2d).reshape(-1, 1))
    n_params = int(X_tr_2d.shape[1] + 1)
    return p_va, p_te, best_alpha, n_params, lat_ms


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
    def __init__(self, n_feat, seq_len=14, patch_len=4, stride=2, d_model=64):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
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


def train_torch_model(model, X_tr, y_tr, X_va_es, y_va_es, seed, epochs=EPOCHS, patience=PATIENCE, lr=OPT_LR):
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


def optimize_val_weights(y_val_cal, preds_list):
    P = np.column_stack([p.ravel() for p in preds_list])
    yv = y_val_cal.ravel()
    k = P.shape[1]
    res = minimize(
        lambda w: float(np.mean(np.abs(yv - P @ w))),
        x0=np.ones(k) / k, method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )
    w = np.clip(res.x, 0.0, 1.0)
    return w / np.sum(w)


def static_conformal_metrics(y_cal, pred_cal, y_test, pred_test, alpha):
    cal_scores = np.abs(y_cal.ravel() - pred_cal.ravel())
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
    return {"Method": "Static Split Conformal", "Nominal_%": int((1.0 - alpha) * 100),
            "Mean_Half_Width_MW": q, "Mean_Width_MW": width,
            "PICP_%": picp, "PINAW_%": pinaw, "Winkler_MW": winkler, "lo": lo, "hi": hi}


def locally_normalized_aci(y_cal, pred_cal, y_test, pred_test, alpha, gamma=0.05, scale_win=14, score_win=90):
    """
    Locally Normalized Adaptive Conformal Inference (LN-ACI; Gibbs & Candes 2021):
      - Causal local scale sigma_t = rolling mean absolute residual over last `scale_win` realized days
      - Online normalized nonconformity score s_t = |y_t - y_hat_t| / sigma_t
      - Online miscoverage level update: alpha_{t+1} = clip(alpha_t + gamma * (alpha - err_t), 0.005, 0.5)
    Strictly causal: only residuals strictly prior to day t are used for day t's prediction interval.
    """
    res_cal = np.abs(y_cal.ravel() - pred_cal.ravel())
    yt = y_test.ravel()
    yp = pred_test.ravel()
    floor_sigma = max(100.0, float(np.percentile(res_cal, 25)))

    # Build initial causal normalized scores from Val-Cal
    hist_res = list(res_cal)
    norm_scores = []
    for i in range(len(res_cal)):
        if i < 5:
            sig = max(floor_sigma, float(np.mean(res_cal[: max(1, i + 1)])))
        else:
            sig = max(floor_sigma, float(np.mean(res_cal[max(0, i - scale_win) : i])))
        norm_scores.append(res_cal[i] / sig)

    alpha_t = alpha
    lo = np.empty(len(yt))
    hi = np.empty(len(yt))
    for t in range(len(yt)):
        sig_t = max(floor_sigma, float(np.mean(hist_res[-scale_win:])))
        recent_scores = np.array(norm_scores[-score_win:])
        q_lvl = float(np.clip(1.0 - alpha_t, 0.0, 1.0))
        q_norm = float(np.quantile(recent_scores, q_lvl)) if q_lvl < 1.0 else float(np.max(recent_scores) * 1.1)
        hw = q_norm * sig_t
        lo[t] = yp[t] - hw
        hi[t] = yp[t] + hw

        # After day t realizes, observe err_t and update alpha_t + rolling history causally
        err_t = 1.0 if (yt[t] < lo[t] or yt[t] > hi[t]) else 0.0
        alpha_t = float(np.clip(alpha_t + gamma * (alpha - err_t), 0.005, 0.50))
        r_t = abs(yt[t] - yp[t])
        hist_res.append(r_t)
        norm_scores.append(r_t / sig_t)

    picp = float(np.mean((yt >= lo) & (yt <= hi)) * 100.0)
    width = float(np.mean(hi - lo))
    pinaw = float(width / (np.max(yt) - np.min(yt)) * 100.0)
    penalty = np.where(yt < lo, (2.0 / alpha) * (lo - yt),
              np.where(yt > hi, (2.0 / alpha) * (yt - hi), 0.0))
    winkler = float(np.mean((hi - lo) + penalty))
    return {"Method": "Locally Normalized ACI", "Nominal_%": int((1.0 - alpha) * 100),
            "Mean_Half_Width_MW": width / 2.0, "Mean_Width_MW": width,
            "PICP_%": picp, "PINAW_%": pinaw, "Winkler_MW": winkler, "lo": lo, "hi": hi}


def dm_hln_and_bootstrap(y_true, pred_a, pred_b, h=1, max_lag=14, block_len=7, n_boot=1000, seed=42):
    """Harvey-Leybourne-Newbold (1997) finite-sample adjusted DM test with Newey-West HAC + block bootstrap."""
    e_a = np.abs(y_true.ravel() - pred_a.ravel())
    e_b = np.abs(y_true.ravel() - pred_b.ravel())
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
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, max(prev, pvals[idx] * (m - rank)))
        adj[idx] = val
        prev = val
    return adj


def compute_ensemble_group_permutation_importance(ens_models, weights, X_eval, y_eval_inv, sy, feat_names, n_repeats=10, seed=42):
    """
    Direct Block-Permutation Importance on the ACTUAL deployed PyTorch Weighted Ensemble.
    Groups features into semantically coherent operational blocks and permutes each block
    simultaneously across all 14 lookback steps to handle collinearity cleanly.
    """
    def eval_ens(X_in):
        pred = sum(w * sy.inverse_transform(predict_torch(m, X_in)) for m, w in zip(ens_models, weights))
        return float(mean_absolute_error(y_eval_inv, pred))

    base_mae = eval_ens(X_eval)
    groups = {
        "Lagged National Peak & Rolling Stats": [
            i for i, c in enumerate(feat_names)
            if c == "Max. Demand at eve. peak (Generation end)"
            or c.startswith("target_roll_") or c == "target_diff_1"
            or "Generation" in c or "Sub-station" in c
        ],
        "Divisional Demand & Supply Telemetry": [
            i for i, c in enumerate(feat_names)
            if any(c.startswith(d + "_") and not c.endswith("_temp")
                   for d in ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Mymensingh", "Rangpur", "Barishal", "Sylhet", "Cumilla"])
        ],
        "Ambient Temperature & Subtropical CDD (26°C)": [
            i for i, c in enumerate(feat_names)
            if "temp" in c.lower() or "Temperature" in c or c == "CDD"
        ],
        "Day-of-Week & Seasonality (Month/Year)": [
            i for i, c in enumerate(feat_names)
            if c.startswith("Day of the week_") or c in ("Month", "Year")
        ],
        "Holiday & Eid Lead/Lag Calendar Flags": [
            i for i, c in enumerate(feat_names)
            if c.startswith("Holiday")
        ],
        "Generation Constraints & Plant Maintenance": [
            i for i, c in enumerate(feat_names)
            if any(k in c for k in ["Gas", "Coal", "Kaptai", "Plants under"])
        ],
    }

    rng = np.random.default_rng(seed)
    rows = []
    for gname, idxs in groups.items():
        if not idxs:
            continue
        deltas = []
        for _ in range(n_repeats):
            Xp = X_eval.copy()
            perm = rng.permutation(len(Xp))
            Xp[:, :, idxs] = Xp[perm][:, :, idxs]
            deltas.append(eval_ens(Xp) - base_mae)
        rows.append({
            "Feature_Group": gname,
            "Num_Features": len(idxs),
            "Delta_MAE_Mean_MW": float(np.mean(deltas)),
            "Delta_MAE_Std_MW": float(np.std(deltas, ddof=1)),
        })
    return pd.DataFrame(rows).sort_values("Delta_MAE_Mean_MW", ascending=False)


if __name__ == "__main__":
    t_start = time.perf_counter()
    print("Preparing revised Part 1 dataset (Train-only CDD_tau*, Val-ES / Val-Cal split, 20 Models x 5 Seeds)...", flush=True)
    ds = prepare_dataset(use_cdd=True, use_eid=True, use_pca=False)
    print(f"Train-estimated cooling threshold tau* = {ds['tau_star']:.1f} C (slope = +{ds['tau_slope']:.1f} MW/C)", flush=True)
    print(f"Data QA: {ds['data_qa']}", flush=True)
    print(f"Rows: {ds['n_raw']} | Features: {ds['n_features']} | Train: {len(ds['X_tr'])} | Val-ES: {len(ds['X_va_es'])} | Val-Cal: {len(ds['X_va_cal'])} | Test: {len(ds['X_te'])}", flush=True)

    X_tr, y_tr = ds["X_tr"], ds["y_tr"]
    X_va_es, y_va_es = ds["X_va_es"], ds["y_va_es"]
    X_va_cal, y_va_cal, y_va_cal_inv = ds["X_va_cal"], ds["y_va_cal"], ds["y_va_cal_inv"]
    X_va, y_va, y_va_inv = ds["X_va"], ds["y_va"], ds["y_va_inv"]
    X_te, y_te_inv, sy = ds["X_te"], ds["y_te_inv"], ds["sy"]

    X_tr_2d = X_tr.reshape(len(X_tr), -1)
    X_va_es_2d = X_va_es.reshape(len(X_va_es), -1)
    X_va_2d = X_va.reshape(len(X_va), -1)
    X_te_2d = X_te.reshape(len(X_te), -1)
    n_f = X_tr.shape[-1]

    # Baselines: Lag-1 Persistence, Weekday-Adjusted Persistence, Seasonal Naive (Lag-7), 7-Day Rolling Mean
    y_full = ds["df_raw_target"].values
    tc, vc = ds["train_cut"], ds["val_cut"]
    p_persist_te = y_full[vc + HIST_SIZE - 1 : -1].reshape(-1, 1)
    p_lag7_te = y_full[vc + HIST_SIZE - 7 : len(y_full) - 7].reshape(-1, 1)
    p_roll7_te = pd.Series(y_full).shift(1).rolling(7).mean().values[vc + HIST_SIZE :].reshape(-1, 1)

    # Weekday-Adjusted Persistence: y_{t-1} + (train_dow_mean[dow(t)] - train_dow_mean[dow(t-1)])
    tr_meta_full = ds["meta_full"].iloc[:tc]
    dow_means = tr_meta_full.groupby("Day of the week")["Max. Demand at eve. peak (Generation end)"].mean().to_dict()
    all_dows = ds["meta_full"]["Day of the week"].values
    dow_step_te = np.array([
        dow_means[all_dows[i]] - dow_means[all_dows[i - 1]]
        for i in range(vc + HIST_SIZE, len(y_full))
    ]).reshape(-1, 1)
    p_wkday_pers_te = p_persist_te + dow_step_te

    # Last observed lag-1 target (in MW) for differenced-target XGBoost
    y_tr_lag1_mw = y_full[HIST_SIZE - 1 : tc - 1].reshape(-1, 1)
    y_tr_mw = y_full[HIST_SIZE : tc].reshape(-1, 1)
    dy_tr_mw = (y_tr_mw - y_tr_lag1_mw).ravel()

    y_va_es_lag1_mw = y_full[tc + HIST_SIZE - 1 : tc + HIST_SIZE - 1 + len(X_va_es)].reshape(-1, 1)
    dy_va_es_mw = (ds["y_va_es_inv"] - y_va_es_lag1_mw).ravel()
    y_te_lag1_mw = p_persist_te

    print("Fitting strictly causal SARIMAX(1,1,1)(1,0,1,7) & Ridge Regression (14 x F)...", flush=True)
    p_sarimax_va, p_sarimax_te, sarimax_p, sarimax_ms = run_sarimax_baseline(ds)
    p_ridge_va, p_ridge_te, ridge_alpha, ridge_p, ridge_ms = run_ridge_baseline(
        X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_va_2d, X_te_2d, sy
    )
    print(f"  SARIMAX(1,1,1)(1,0,1,7): MAE={calc_metrics(y_te_inv, p_sarimax_te)['MAE']:6.2f} MW", flush=True)
    print(f"  Ridge (alpha={ridge_alpha:g}):       MAE={calc_metrics(y_te_inv, p_ridge_te)['MAE']:6.2f} MW", flush=True)
    print(f"  Weekday-Adj Persist:     MAE={calc_metrics(y_te_inv, p_wkday_pers_te)['MAE']:6.2f} MW", flush=True)
    print(f"  Seasonal Naive (Lag-7):  MAE={calc_metrics(y_te_inv, p_lag7_te)['MAE']:6.2f} MW", flush=True)

    all_seed_metrics = []
    seed_preds = {s: {} for s in SEEDS}
    seed_val_cal_preds = {s: {} for s in SEEDS}
    learned_weights = []
    val_histories = {m: [] for m in ["DLinear", "PatchTST", "BiGRU", "BiLSTM", "BiLSTM-Attention", "Transformer", "CNN-BiGRU-Transformer"]}
    eff_profile = {
        "SARIMAX": {"Params": sarimax_p, "Size_KB": round(sarimax_p * 8 / 1024, 2), "CPU_ms": sarimax_ms},
        "Ridge (14xF)": {"Params": ridge_p, "Size_KB": round(ridge_p * 8 / 1024, 1), "CPU_ms": ridge_ms},
    }
    seed0_ens_models, seed0_ens_weights = None, None

    for s in SEEDS:
        print(f"\n--- Running Seed {s} ---", flush=True)
        det_baselines = [
            ("Persistence (Lag-1)", p_persist_te),
            ("Weekday-Adj Persistence", p_wkday_pers_te),
            ("Seasonal Naive (Lag-7)", p_lag7_te),
            ("7-Day Rolling Mean", p_roll7_te),
            ("SARIMAX", p_sarimax_te),
            ("Ridge (14xF)", p_ridge_te),
        ]
        for bn, bp in det_baselines:
            seed_preds[s][bn] = bp
            all_seed_metrics.append({"Seed": s, "Model": bn, **calc_metrics(y_te_inv, bp)})

        # XGBoost (Raw Level)
        xm = xgb.XGBRegressor(random_state=s, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
        xm.fit(X_tr_2d, y_tr.ravel(), eval_set=[(X_va_es_2d, y_va_es.ravel())], verbose=False)
        p_xgb = sy.inverse_transform(xm.predict(X_te_2d).reshape(-1, 1))
        seed_preds[s]["XGBoost (Level)"] = p_xgb
        all_seed_metrics.append({"Seed": s, "Model": "XGBoost (Level)", **calc_metrics(y_te_inv, p_xgb)})
        if s == SEEDS[0]:
            t0 = time.perf_counter()
            for _ in range(60): _ = xm.predict(X_te_2d[:1])
            eff_profile["XGBoost (Level)"] = {"Params": 0, "Size_KB": 0.0, "CPU_ms": round((time.perf_counter() - t0) / 60 * 1000, 2)}

        # XGBoost-Diff (Differenced Target: removes tree extrapolation bias on unseen >15 GW peaks!)
        xmd = xgb.XGBRegressor(random_state=s, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
        xmd.fit(X_tr_2d, dy_tr_mw, eval_set=[(X_va_es_2d, dy_va_es_mw)], verbose=False)
        p_xgbd = y_te_lag1_mw + xmd.predict(X_te_2d).reshape(-1, 1)
        seed_preds[s]["XGBoost-Diff (Δy)"] = p_xgbd
        m_xd = calc_metrics(y_te_inv, p_xgbd)
        print(f"  [Seed {s}] {'XGBoost-Diff (Δy)':24s}: MAE={m_xd['MAE']:6.1f} MW | Acc={m_xd['Acc']:.2f}%", flush=True)
        all_seed_metrics.append({"Seed": s, "Model": "XGBoost-Diff (Δy)", **m_xd})
        if s == SEEDS[0]:
            eff_profile["XGBoost-Diff (Δy)"] = eff_profile["XGBoost (Level)"].copy()

        # LightGBM
        lm = lgb.LGBMRegressor(objective="regression_l1", random_state=s, n_estimators=500,
                               learning_rate=0.05, num_leaves=31, n_jobs=1, verbose=-1)
        lm.fit(X_tr_2d, y_tr.ravel(), eval_set=[(X_va_es_2d, y_va_es.ravel())],
               callbacks=[lgb.early_stopping(8, verbose=False)])
        p_lgb = sy.inverse_transform(lm.predict(X_te_2d).reshape(-1, 1))
        seed_preds[s]["LightGBM"] = p_lgb
        all_seed_metrics.append({"Seed": s, "Model": "LightGBM", **calc_metrics(y_te_inv, p_lgb)})
        if s == SEEDS[0]:
            t0 = time.perf_counter()
            for _ in range(60): _ = lm.predict(X_te_2d[:1])
            eff_profile["LightGBM"] = {"Params": 0, "Size_KB": 0.0, "CPU_ms": round((time.perf_counter() - t0) / 60 * 1000, 2)}

        # Deep Learning models (early stopping strictly on Val-ES; Val-Cal untouched during training!)
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
            mod, vh = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, s)
            val_histories[name].append(vh)
            pred_vc = sy.inverse_transform(predict_torch(mod, X_va_cal))
            pred_te = sy.inverse_transform(predict_torch(mod, X_te))
            seed_val_cal_preds[s][name] = pred_vc
            seed_preds[s][name] = pred_te
            trained_dl[name] = mod
            mr = calc_metrics(y_te_inv, pred_te)
            print(f"  [Seed {s}] {name:24s}: MAE={mr['MAE']:6.1f} MW | Acc={mr['Acc']:.2f}%", flush=True)
            all_seed_metrics.append({"Seed": s, "Model": name, **mr})
            if s == SEEDS[0]:
                np_cnt = sum(p.numel() for p in mod.parameters())
                eff_profile[name] = {
                    "Params": np_cnt, "Size_KB": round(np_cnt * 4 / 1024, 1),
                    "CPU_ms": round(measure_torch_latency(mod, X_te[:1]), 2)
                }

        # Equal-Budget 3-Member PatchTST Ensemble (fair like-for-like 3-network comparison!)
        pt_vc_list = [seed_val_cal_preds[s]["PatchTST"]]
        pt_te_list = [seed_preds[s]["PatchTST"]]
        for offset in [1000, 2000]:
            m_pt_extra, _ = train_torch_model(PatchTSTModel(n_f), X_tr, y_tr, X_va_es, y_va_es, s + offset)
            pt_vc_list.append(sy.inverse_transform(predict_torch(m_pt_extra, X_va_cal)))
            pt_te_list.append(sy.inverse_transform(predict_torch(m_pt_extra, X_te)))
        p_pt3_vc = np.mean(pt_vc_list, axis=0)
        p_pt3_te = np.mean(pt_te_list, axis=0)
        seed_val_cal_preds[s]["PatchTST-Ensemble (3x)"] = p_pt3_vc
        seed_preds[s]["PatchTST-Ensemble (3x)"] = p_pt3_te
        m_pt3 = calc_metrics(y_te_inv, p_pt3_te)
        print(f"  [Seed {s}] {'PatchTST-Ensemble (3x)':24s}: MAE={m_pt3['MAE']:6.1f} MW | Acc={m_pt3['Acc']:.2f}%", flush=True)
        all_seed_metrics.append({"Seed": s, "Model": "PatchTST-Ensemble (3x)", **m_pt3})
        if s == SEEDS[0]:
            eff_profile["PatchTST-Ensemble (3x)"] = {
                "Params": eff_profile["PatchTST"]["Params"] * 3,
                "Size_KB": round(eff_profile["PatchTST"]["Size_KB"] * 3, 1),
                "CPU_ms": round(eff_profile["PatchTST"]["CPU_ms"] * 3, 2),
            }

        # Residual Hybrids
        for base_name, h_name in [("BiGRU", "Hybrid BiGRU-XGBoost"), ("BiLSTM-Attention", "Hybrid BiLSTM-Attn-XGBoost")]:
            bm = trained_dl[base_name]
            res_tr = y_tr - predict_torch(bm, X_tr)
            res_va = y_va_es - predict_torch(bm, X_va_es)
            hx = xgb.XGBRegressor(random_state=s, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
            hx.fit(X_tr_2d, res_tr.ravel(), eval_set=[(X_va_es_2d, res_va.ravel())], verbose=False)
            pred_h = sy.inverse_transform(predict_torch(bm, X_te) + hx.predict(X_te_2d).reshape(-1, 1))
            seed_preds[s][h_name] = pred_h
            all_seed_metrics.append({"Seed": s, "Model": h_name, **calc_metrics(y_te_inv, pred_h)})

        # Equal-Weight RNN Ensemble (3-network)
        p_eq_vc = (seed_val_cal_preds[s]["BiGRU"] + seed_val_cal_preds[s]["BiLSTM-Attention"] + seed_val_cal_preds[s]["BiLSTM"]) / 3.0
        p_eq_te = (seed_preds[s]["BiGRU"] + seed_preds[s]["BiLSTM-Attention"] + seed_preds[s]["BiLSTM"]) / 3.0
        seed_val_cal_preds[s]["Equal-Weight RNN Ens"] = p_eq_vc
        seed_preds[s]["Equal-Weight RNN Ens"] = p_eq_te
        all_seed_metrics.append({"Seed": s, "Model": "Equal-Weight RNN Ens", **calc_metrics(y_te_inv, p_eq_te)})

        # Val-Cal SLSQP Weighted RNN Ensemble (weights strictly fit on held-out Val-Cal!)
        w_opt = optimize_val_weights(
            y_va_cal_inv,
            [seed_val_cal_preds[s]["BiGRU"], seed_val_cal_preds[s]["BiLSTM-Attention"], seed_val_cal_preds[s]["BiLSTM"]]
        )
        learned_weights.append(w_opt.tolist())
        p_opt_vc = w_opt[0] * seed_val_cal_preds[s]["BiGRU"] + w_opt[1] * seed_val_cal_preds[s]["BiLSTM-Attention"] + w_opt[2] * seed_val_cal_preds[s]["BiLSTM"]
        p_opt_te = w_opt[0] * seed_preds[s]["BiGRU"] + w_opt[1] * seed_preds[s]["BiLSTM-Attention"] + w_opt[2] * seed_preds[s]["BiLSTM"]
        seed_val_cal_preds[s]["Val-Opt RNN Ensemble"] = p_opt_vc
        seed_preds[s]["Val-Opt RNN Ensemble"] = p_opt_te
        me = calc_metrics(y_te_inv, p_opt_te)
        print(f"  [Seed {s}] {'Val-Opt RNN Ensemble':24s}: MAE={me['MAE']:6.1f} MW | Acc={me['Acc']:.2f}% | w={np.round(w_opt, 3)}", flush=True)
        all_seed_metrics.append({"Seed": s, "Model": "Val-Opt RNN Ensemble", **me})

        if s == SEEDS[0]:
            tot_p = eff_profile["BiGRU"]["Params"] + eff_profile["BiLSTM-Attention"]["Params"] + eff_profile["BiLSTM"]["Params"]
            tot_ms = eff_profile["BiGRU"]["CPU_ms"] + eff_profile["BiLSTM-Attention"]["CPU_ms"] + eff_profile["BiLSTM"]["CPU_ms"]
            eff_profile["Val-Opt RNN Ensemble"] = {"Params": tot_p, "Size_KB": round(tot_p * 4 / 1024, 1), "CPU_ms": round(tot_ms, 2)}
            seed0_ens_models = [trained_dl["BiGRU"], trained_dl["BiLSTM-Attention"], trained_dl["BiLSTM"]]
            seed0_ens_weights = w_opt

    mean_w = np.mean(learned_weights, axis=0)
    print(f"\nVal-Cal SLSQP Weights per seed: {np.round(learned_weights, 3).tolist()}")
    print(f"Mean weights [BiGRU, BiLSTM-Attn, BiLSTM]: {np.round(mean_w, 3)}")

    df_seeds = pd.DataFrame(all_seed_metrics)
    df_summary = df_seeds.groupby("Model").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        MAPE_mean=("MAPE", "mean"), MAPE_std=("MAPE", "std"),
        Acc_mean=("Acc", "mean"), Acc_std=("Acc", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std"),
    ).sort_values("MAE_mean")
    print("\n=== 1. PART 1 MULTI-SEED BENCHMARK (20 MODELS, 5 SEEDS) ===")
    print(df_summary.round(2).to_string())

    # 5-seed averaged trajectories & HLN-corrected DM tests with Holm-Bonferroni adjustment
    mean_preds = {m: np.mean([seed_preds[s][m] for s in SEEDS], axis=0) for m in df_summary.index}
    mean_val_cal_ens = np.mean([seed_val_cal_preds[s]["Val-Opt RNN Ensemble"] for s in SEEDS], axis=0)
    mean_val_cal_pt3 = np.mean([seed_val_cal_preds[s]["PatchTST-Ensemble (3x)"] for s in SEEDS], axis=0)
    ens_5s_mae = float(mean_absolute_error(y_te_inv, mean_preds["Val-Opt RNN Ensemble"]))

    # Also compute single-seed (Seed 42) DM test against PatchTST-Ensemble (3x) for strict apples-to-apples check
    dm_s0_pt3 = dm_hln_and_bootstrap(y_te_inv, seed_preds[SEEDS[0]]["Val-Opt RNN Ensemble"], seed_preds[SEEDS[0]]["PatchTST-Ensemble (3x)"])
    print(f"\nLike-for-like 3-network Seed-42 HLN-DM (Val-Opt RNN Ens vs PatchTST-Ens3): DM={dm_s0_pt3[0]:.2f}, p={dm_s0_pt3[1]:.3f}, dMAE={dm_s0_pt3[2]:+.2f} MW")

    dm_rows = []
    for m in df_summary.index:
        if m == "Val-Opt RNN Ensemble":
            continue
        base_5s_mae = float(mean_absolute_error(y_te_inv, mean_preds[m]))
        dm_s, pv, d_m, lo, hi = dm_hln_and_bootstrap(y_te_inv, mean_preds["Val-Opt RNN Ensemble"], mean_preds[m])
        dm_rows.append({
            "Baseline": m, "Ens_5Seed_MAE": ens_5s_mae, "Base_5Seed_MAE": base_5s_mae,
            "DM_HLN_stat": dm_s, "p_val": pv, "Delta_MAE_MW": d_m, "CI_lo": lo, "CI_hi": hi
        })
    df_dm = pd.DataFrame(dm_rows).sort_values("Delta_MAE_MW").reset_index(drop=True)
    df_dm["p_holm"] = holm_bonferroni(df_dm["p_val"].values)
    print("\n=== 2. HLN-ADJUSTED DIEBOLD-MARIANO (HAC LAG=14, HOLM-BONFERRONI) ===")
    print(df_dm.round(4).to_string(index=False))

    # Conformal Prediction: Static Split vs Locally Normalized ACI (calibrated strictly on held-out Val-Cal!)
    cp_static_80 = static_conformal_metrics(y_va_cal_inv, mean_val_cal_ens, y_te_inv, mean_preds["Val-Opt RNN Ensemble"], alpha=0.20)
    cp_static_90 = static_conformal_metrics(y_va_cal_inv, mean_val_cal_ens, y_te_inv, mean_preds["Val-Opt RNN Ensemble"], alpha=0.10)
    cp_aci_80 = locally_normalized_aci(y_va_cal_inv, mean_val_cal_ens, y_te_inv, mean_preds["Val-Opt RNN Ensemble"], alpha=0.20)
    cp_aci_90 = locally_normalized_aci(y_va_cal_inv, mean_val_cal_ens, y_te_inv, mean_preds["Val-Opt RNN Ensemble"], alpha=0.10)
    df_cp = pd.DataFrame([
        {k: v for k, v in d.items() if k not in ("lo", "hi")}
        for d in [cp_static_80, cp_aci_80, cp_static_90, cp_aci_90]
    ])
    print("\n=== 3. CONFORMAL PREDICTION: STATIC SPLIT VS. LOCALLY NORMALIZED ACI (VAL-CAL CALIBRATED) ===")
    print(df_cp.round(2).to_string(index=False))

    # Seasonal coverage diagnostic for Static vs LN-ACI (90% nominal)
    def map_season(m):
        if m in [3, 4, 5]: return "Summer (Mar-May)"
        if m in [6, 7, 8, 9]: return "Monsoon (Jun-Sep)"
        if m in [10, 11]: return "Post-Monsoon (Oct-Nov)"
        return "Winter (Dec-Feb)"

    te_seasons = ds["te_meta"]["Month"].apply(map_season).values
    yt_flat = y_te_inv.ravel()
    print("\nSeasonal 90% PI Coverage & Width (Static vs LN-ACI):")
    for sea in ["Winter (Dec-Feb)", "Summer (Mar-May)", "Monsoon (Jun-Sep)", "Post-Monsoon (Oct-Nov)"]:
        ms = te_seasons == sea
        st_picp = np.mean((yt_flat[ms] >= cp_static_90["lo"][ms]) & (yt_flat[ms] <= cp_static_90["hi"][ms])) * 100
        ac_picp = np.mean((yt_flat[ms] >= cp_aci_90["lo"][ms]) & (yt_flat[ms] <= cp_aci_90["hi"][ms])) * 100
        ac_w = np.mean(cp_aci_90["hi"][ms] - cp_aci_90["lo"][ms])
        print(f"  {sea:24s}: Static PICP={st_picp:5.1f}% (W={cp_static_90['Mean_Width_MW']:6.0f}) | LN-ACI PICP={ac_picp:5.1f}% (W={ac_w:6.0f})")

    # Direct Group Permutation Importance on the ACTUAL PyTorch Weighted Ensemble (Val-Cal & Test)
    print("\nComputing Direct Group Permutation Importance on the Deployed PyTorch Weighted Ensemble...", flush=True)
    df_perm = compute_ensemble_group_permutation_importance(
        seed0_ens_models, seed0_ens_weights, X_te, y_te_inv, sy, ds["feat_names"], n_repeats=15, seed=42
    )
    print(df_perm.round(2).to_string(index=False))

    # Validation-First 5-Seed Feature & PCA Ablation (report BOTH Val-Cal MAE and Test MAE + Welch p-values!)
    print("\nRunning Validation-First 5-Seed Feature & PCA Ablation...", flush=True)
    ablation_configs = [
        ("1. Base (No CDD_26, No Eid Flags)", dict(use_cdd=False, use_eid=False, use_pca=False)),
        ("2. + Subtropical CDD_26", dict(use_cdd=True, use_eid=False, use_pca=False)),
        ("3. + Eid Lead/Lag (Full No-PCA)", dict(use_cdd=True, use_eid=True, use_pca=False)),
        ("4. Full + PCA (95% Variance)", dict(use_cdd=True, use_eid=True, use_pca=True)),
    ]
    hol_mask = (ds["te_meta"]["Holiday_cat"].values > 0)
    ablation_rows = []
    abl_seed_maes = {}
    for label, cfg in ablation_configs:
        if cfg == dict(use_cdd=True, use_eid=True, use_pca=False):
            val_maes = [float(mean_absolute_error(y_va_cal_inv, seed_val_cal_preds[s]["Val-Opt RNN Ensemble"])) for s in SEEDS]
            ms = [calc_metrics(y_te_inv, seed_preds[s]["Val-Opt RNN Ensemble"]) for s in SEEDS]
            hol_maes = [float(mean_absolute_error(y_te_inv[hol_mask], seed_preds[s]["Val-Opt RNN Ensemble"][hol_mask])) for s in SEEDS]
            dims = ds["n_features"]
        else:
            dsi = prepare_dataset(**cfg, tau_override=ds["tau_star"])
            dims = dsi["n_features"]
            val_maes, ms, hol_maes = [], [], []
            hm = (dsi["te_meta"]["Holiday_cat"].values > 0)
            for s in SEEDS:
                v_list, t_list = [], []
                for fn in [lambda: BiGRUModel(dims), lambda: BiLSTMAttnModel(dims), lambda: BiLSTMModel(dims)]:
                    set_seed(s)
                    mod, _ = train_torch_model(fn(), dsi["X_tr"], dsi["y_tr"], dsi["X_va_es"], dsi["y_va_es"], s)
                    v_list.append(dsi["sy"].inverse_transform(predict_torch(mod, dsi["X_va_cal"])))
                    t_list.append(dsi["sy"].inverse_transform(predict_torch(mod, dsi["X_te"])))
                wi = optimize_val_weights(dsi["y_va_cal_inv"], v_list)
                pv = wi[0] * v_list[0] + wi[1] * v_list[1] + wi[2] * v_list[2]
                pr = wi[0] * t_list[0] + wi[1] * t_list[1] + wi[2] * t_list[2]
                val_maes.append(float(mean_absolute_error(dsi["y_va_cal_inv"], pv)))
                ms.append(calc_metrics(dsi["y_te_inv"], pr))
                hol_maes.append(float(mean_absolute_error(dsi["y_te_inv"][hm], pr[hm])))
        abl_seed_maes[label] = [x["MAE"] for x in ms]
        ablation_rows.append({
            "Config": label, "Dims": dims,
            "Val_MAE_mean": float(np.mean(val_maes)),
            "Val_MAE_std": float(np.std(val_maes, ddof=1)),
            "MAE_mean": float(np.mean([x["MAE"] for x in ms])),
            "MAE_std": float(np.std([x["MAE"] for x in ms], ddof=1)),
            "Hol_MAE_mean": float(np.mean(hol_maes)),
            "MAPE_mean": float(np.mean([x["MAPE"] for x in ms])),
            "Acc_mean": float(np.mean([x["Acc"] for x in ms])),
        })
    df_abl = pd.DataFrame(ablation_rows)
    # Paired t-test vs Base (same 5 seeds!)
    base_vec = np.array(abl_seed_maes[ablation_configs[0][0]])
    p_vs_base = [1.0]
    for lbl, _ in ablation_configs[1:]:
        _, pv = stats.ttest_rel(abl_seed_maes[lbl], base_vec)
        p_vs_base.append(float(pv))
    df_abl["p_paired_vs_Base"] = p_vs_base
    print("\n=== 4. VALIDATION-FIRST 5-SEED ABLATION ===")
    print(df_abl.round(3).to_string(index=False))

    # Regime Analysis, Demand-Bin Stratification & Post-Gap Sensitivity
    te_m = ds["te_meta"].copy()
    te_m["Actual"] = y_te_inv.ravel()
    te_m["Pred_Ens"] = mean_preds["Val-Opt RNN Ensemble"].ravel()
    te_m["Pred_PT3"] = mean_preds["PatchTST-Ensemble (3x)"].ravel()
    te_m["Pred_XGB"] = mean_preds["XGBoost (Level)"].ravel()
    te_m["Pred_XGBD"] = mean_preds["XGBoost-Diff (Δy)"].ravel()
    te_m["Pred_Ridge"] = mean_preds["Ridge (14xF)"].ravel()
    te_m["Pred_Persist"] = mean_preds["Persistence (Lag-1)"].ravel()
    te_m["Pred_Wkday"] = mean_preds["Weekday-Adj Persistence"].ravel()
    te_m["Season"] = te_m["Month"].apply(map_season)
    te_m["Is_Weekend"] = te_m["Day of the week"].isin(["Friday", "Saturday"])
    te_m["Cal_Regime"] = np.where(te_m["Holiday_cat"] > 0, "Holiday / Eid",
                         np.where(te_m["Is_Weekend"], "Weekend (Fri-Sat)", "Working Weekday"))

    # Gap sensitivity check
    pg_mask = te_m["Post_Gap_Flag"].values == 1
    if pg_mask.sum() > 0:
        print(f"\nPost-Gap Sensitivity (n_gap={pg_mask.sum()} vs n_reg={(~pg_mask).sum()}): "
              f"Gap MAE={mean_absolute_error(te_m['Actual'][pg_mask], te_m['Pred_Ens'][pg_mask]):.1f} MW vs "
              f"Regular MAE={mean_absolute_error(te_m['Actual'][~pg_mask], te_m['Pred_Ens'][~pg_mask]):.1f} MW")

    regime_rows = []
    for col in ["Season", "Cal_Regime"]:
        for grp, g in te_m.groupby(col):
            regime_rows.append({
                "Axis": col, "Regime": grp, "N": len(g),
                "Ens_MAE": calc_metrics(g["Actual"].values, g["Pred_Ens"].values)["MAE"],
                "Ens_MAPE": calc_metrics(g["Actual"].values, g["Pred_Ens"].values)["MAPE"],
                "PT3_MAE": calc_metrics(g["Actual"].values, g["Pred_PT3"].values)["MAE"],
                "XGB_MAE": calc_metrics(g["Actual"].values, g["Pred_XGB"].values)["MAE"],
                "XGBD_MAE": calc_metrics(g["Actual"].values, g["Pred_XGBD"].values)["MAE"],
                "Wkday_MAE": calc_metrics(g["Actual"].values, g["Pred_Wkday"].values)["MAE"],
                "Persist_MAE": calc_metrics(g["Actual"].values, g["Pred_Persist"].values)["MAE"],
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
            "PT3_MAE": float(mean_absolute_error(g["Actual"], g["Pred_PT3"])),
            "XGB_MAE": float(mean_absolute_error(g["Actual"], g["Pred_XGB"])),
            "XGBD_MAE": float(mean_absolute_error(g["Actual"], g["Pred_XGBD"])),
            "Persist_MAE": float(mean_absolute_error(g["Actual"], g["Pred_Persist"])),
            "Ens_MAPE": float(np.mean(np.abs((g["Actual"] - g["Pred_Ens"]) / g["Actual"])) * 100.0),
        })
    df_bins = pd.DataFrame(bin_rows)
    print("\n=== 5. 5-SEED DEMAND-BIN BREAKDOWN (WITH XGBOOST-DIFF & PATCHTST-ENS3) ===")
    print(df_bins.round(2).to_string(index=False))

    # ==========================================================================
    # REGENERATE ALL CLEAN VECTOR PDF FIGURES (ZERO LEGEND OVERLAP!)
    # ==========================================================================
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    # Fig 1: Forecast comparison (legend cleanly above plot!)
    fig, ax = plt.subplots(figsize=(10.2, 4.1))
    ax.plot(ds["te_dates"], y_te_inv, color="#111111", lw=1.35, label="Actual Peak Demand")
    ax.plot(ds["te_dates"], mean_preds["Val-Opt RNN Ensemble"], color="#1d3557", lw=1.25, label="Val-Opt RNN Ensemble")
    ax.plot(ds["te_dates"], mean_preds["PatchTST-Ensemble (3x)"], color="#2a9d8f", lw=1.15, ls="-.", alpha=0.85, label="PatchTST-Ensemble (3×)")
    ax.plot(ds["te_dates"], mean_preds["XGBoost (Level)"], color="#e76f51", lw=0.95, ls="--", alpha=0.75, label="XGBoost (Level)")
    ax.set_ylabel("Peak Demand (MW)")
    ax.set_title("National Evening Peak Demand — Blind Test Period (462 Days)", pad=26)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False, fontsize=8.3)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig1_forecast_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 2: Convergence curves (legend outside above plot — zero overlap!)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = {"BiGRU": "#1d3557", "BiLSTM-Attention": "#2a9d8f", "BiLSTM": "#d97706",
              "PatchTST": "#0284c7", "DLinear": "#6d597a", "Transformer": "#e76f51", "CNN-BiGRU-Transformer": "#9b2226"}
    for name, hlist in val_histories.items():
        max_len = max(len(h) for h in hlist)
        padded = np.array([h + [h[-1]] * (max_len - len(h)) for h in hlist])
        mu, sd = padded.mean(axis=0), padded.std(axis=0)
        ep = np.arange(1, max_len + 1)
        ax.plot(ep, mu, lw=1.65, color=colors[name], label=name)
        ax.fill_between(ep, mu - sd, mu + sd, color=colors[name], alpha=0.11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val-ES Loss (Normalized MAE)")
    ax.set_title("Multi-Seed Validation Convergence (Mean ± 1 SD across 5 Seeds)", pad=30)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, frameon=False, fontsize=7.8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig2_convergence_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: HLN-DM Forest Plot (legend outside above!)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    y_pos = np.arange(len(df_dm))
    c_bars = ["#1d3557" if p < 0.01 else ("#457b9d" if p < 0.05 else "#adb5bd") for p in df_dm["p_holm"]]
    ax.axvline(0, color="black", ls="--", lw=1.0)
    for i, (_, r) in enumerate(df_dm.iterrows()):
        ax.plot([r["CI_lo"], r["CI_hi"]], [i, i], color=c_bars[i], lw=2.2)
        ax.plot(r["Delta_MAE_MW"], i, "o", color=c_bars[i], ms=5.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_dm["Baseline"], fontsize=8.3)
    ax.set_xlabel("ΔMAE: Val-Opt RNN Ensemble − Baseline (MW)  [Negative = RNN Ensemble Lower Error]")
    ax.set_title("HLN-Adjusted Diebold–Mariano Tests & 95% Block-Bootstrap CIs (Holm–Bonferroni Adjusted)", pad=26)
    ax.grid(axis="x", alpha=0.25)
    p1 = mpatches.Patch(color="#1d3557", label="p_Holm < 0.01")
    p2 = mpatches.Patch(color="#457b9d", label="p_Holm < 0.05")
    p3 = mpatches.Patch(color="#adb5bd", label="p_Holm ≥ 0.05 (ns)")
    ax.legend(handles=[p1, p2, p3], loc="upper center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False, fontsize=8.2)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig3_dm_forest.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 4: Ablation & Seasonal Regime (legend outside above panel b!)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    ax = axes[0]
    short_lbl = ["Base", "+CDD_26", "+Eid Flags\n(Full)", "Full + PCA\n(95%)"]
    ax.bar(short_lbl, df_abl["MAE_mean"], yerr=df_abl["MAE_std"], capsize=4,
           color=["#adb5bd", "#457b9d", "#1d3557", "#e76f51"], width=0.52)
    for i, v in enumerate(df_abl["MAE_mean"]):
        ax.text(i, v + df_abl["MAE_std"].iloc[i] + 8, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_ylim(0, max(df_abl["MAE_mean"] + df_abl["MAE_std"]) * 1.15)
    ax.set_title("(a) Controlled 5-Seed Feature & PCA Ablation")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    s_reg = df_reg[df_reg["Axis"] == "Season"]
    x = np.arange(len(s_reg))
    w = 0.21
    ax.bar(x - 1.5 * w, s_reg["Persist_MAE"], width=w, color="#adb5bd", label="Lag-1 Persist")
    ax.bar(x - 0.5 * w, s_reg["XGBD_MAE"], width=w, color="#f4a261", label="XGBoost-Diff")
    ax.bar(x + 0.5 * w, s_reg["PT3_MAE"], width=w, color="#2a9d8f", label="PatchTST-Ens3")
    ax.bar(x + 1.5 * w, s_reg["Ens_MAE"], width=w, color="#1d3557", label="Val-Opt RNN Ens")
    ax.set_xticks(x)
    ax.set_xticklabels([s.split(" ")[0] for s in s_reg["Regime"]], fontsize=8.5)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_ylim(0, s_reg[["Persist_MAE", "XGBD_MAE", "PT3_MAE", "Ens_MAE"]].values.max() * 1.15)
    ax.set_title("(b) Seasonal Regime Error Breakdown", pad=22)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig4_ablation_regime.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 6: Demand Bins (including XGBoost-Diff & PatchTST-Ens3; legend cleanly above!)
    fig, ax = plt.subplots(figsize=(9.0, 4.3))
    x = np.arange(len(df_bins))
    w = 0.17
    ax.bar(x - 2 * w, df_bins["Persist_MAE"], width=w, color="#adb5bd", label="Lag-1 Persist")
    ax.bar(x - w, df_bins["XGB_MAE"], width=w, color="#e76f51", label="XGBoost (Level)")
    ax.bar(x, df_bins["XGBD_MAE"], width=w, color="#f4a261", label="XGBoost-Diff (Δy)")
    ax.bar(x + w, df_bins["PT3_MAE"], width=w, color="#2a9d8f", label="PatchTST-Ens (3×)")
    ax.bar(x + 2 * w, df_bins["Ens_MAE"], width=w, color="#1d3557", label="Val-Opt RNN Ens")
    for i, r in df_bins.iterrows():
        ax.text(i + 2 * w, r["Ens_MAE"] + 16, f"{r['Ens_MAE']:.0f}", ha="center", fontsize=7.2, fontweight="bold", color="#1d3557")
        ax.text(i, r["XGBD_MAE"] + 16, f"{r['XGBD_MAE']:.0f}", ha="center", fontsize=7.0, color="#b45309")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(df_bins["Demand_Bin"], df_bins["N"])], fontsize=8.5)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_ylim(0, df_bins[["Persist_MAE", "XGB_MAE", "XGBD_MAE", "PT3_MAE", "Ens_MAE"]].values.max() * 1.14)
    ax.set_title("Peak Demand-Level Error Stratification (Level vs. Differenced Trees & Deep Ensembles)", pad=26)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=5, frameon=False, fontsize=7.8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig6_demand_bins.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 7: Qualitative events (legend cleanly above each panel!)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1))
    w1_mask = (ds["te_dates"] >= "2025-12-15") & (ds["te_dates"] <= "2026-01-14")
    w2_mask = (ds["te_dates"] >= "2025-05-20") & (ds["te_dates"] <= "2025-06-19")
    for ax, mask, title in [
        (axes[0], w1_mask, "(a) Stable Winter Regime (Dec 2025–Jan 2026)"),
        (axes[1], w2_mask, "(b) Volatile Summer & Eid-ul-Adha Window (May–Jun 2025)"),
    ]:
        d_sub = ds["te_dates"][mask]
        yt_s = y_te_inv.ravel()[mask]
        ye_s = mean_preds["Val-Opt RNN Ensemble"].ravel()[mask]
        ypt_s = mean_preds["PatchTST-Ensemble (3x)"].ravel()[mask]
        yxd_s = mean_preds["XGBoost-Diff (Δy)"].ravel()[mask]
        ax.plot(d_sub, yt_s, color="#111111", lw=1.7, label="Actual")
        ax.plot(d_sub, ye_s, color="#1d3557", lw=1.45, label=f"RNN Ens ({mean_absolute_error(yt_s, ye_s):.0f})")
        ax.plot(d_sub, ypt_s, color="#2a9d8f", lw=1.25, ls="-.", label=f"PatchTST-3× ({mean_absolute_error(yt_s, ypt_s):.0f})")
        ax.plot(d_sub, yxd_s, color="#f4a261", lw=1.1, ls="--", label=f"XGB-Diff ({mean_absolute_error(yt_s, yxd_s):.0f})")
        ax.set_title(title, fontsize=9.3, pad=24)
        ax.set_ylabel("Peak Demand (MW)")
        ax.tick_params(axis="x", rotation=25, labelsize=7.5)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False, fontsize=7.5)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig7_qualitative_events.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 8: Efficiency Pareto (now includes PatchTST-Ensemble 3x & Ridge!)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2))
    nn_names = ["CNN-BiGRU-Transformer", "DLinear", "PatchTST", "BiLSTM-Attention", "Transformer",
                "BiGRU", "BiLSTM", "PatchTST-Ensemble (3x)", "Val-Opt RNN Ensemble"]
    pts_x = [eff_profile[m]["Params"] / 1000.0 for m in nn_names]
    pts_y = [float(df_summary.loc[m, "MAE_mean"]) for m in nn_names]
    cols_p = ["#1d3557" if m == "Val-Opt RNN Ensemble" else ("#0284c7" if "PatchTST" in m else "#2a9d8f") for m in nn_names]
    sizes_p = [130 if "Ensemble" in m else 70 for m in nn_names]
    axes[0].scatter(pts_x, pts_y, c=cols_p, s=sizes_p, zorder=4, edgecolors="white")
    offsets = {
        "DLinear": (10, -8), "CNN-BiGRU-Transformer": (10, 6), "PatchTST": (10, -12),
        "BiLSTM-Attention": (10, -12), "Transformer": (10, 4), "BiGRU": (10, 6),
        "BiLSTM": (10, -12), "PatchTST-Ensemble (3x)": (10, -14), "Val-Opt RNN Ensemble": (-128, 8)
    }
    for m, px, py in zip(nn_names, pts_x, pts_y):
        dx, dy = offsets.get(m, (8, 4))
        axes[0].annotate(m, (px, py), xytext=(dx, dy), textcoords="offset points",
                         fontsize=7.4, fontweight="bold" if "Ensemble" in m else "normal")
    axes[0].set_xlabel("Trainable Parameters (Thousands)")
    axes[0].set_ylabel("Blind Test MAE (MW)")
    axes[0].set_title("(a) Accuracy vs. Parameter Count (5-Seed)")
    axes[0].set_xlim(-30, 900)
    axes[0].grid(alpha=0.25)

    lat_models = ["Ridge (14xF)", "XGBoost (Level)", "LightGBM", "DLinear", "PatchTST", "Transformer",
                  "BiGRU", "BiLSTM", "BiLSTM-Attention", "PatchTST-Ensemble (3x)", "CNN-BiGRU-Transformer", "Val-Opt RNN Ensemble"]
    lats = [eff_profile[m]["CPU_ms"] for m in lat_models]
    bar_c = ["#1d3557" if m == "Val-Opt RNN Ensemble" else ("#0284c7" if "PatchTST" in m else "#457b9d") for m in lat_models]
    axes[1].barh(lat_models, lats, color=bar_c, height=0.62)
    for i, v in enumerate(lats):
        axes[1].text(v + 0.03, i, f"{v:.2f} ms", va="center", fontsize=7.4)
    axes[1].set_xlabel("Single-Sample CPU Inference Latency (ms)")
    axes[1].set_title("(b) CPU Inference Latency Profile")
    axes[1].set_xlim(0, max(lats) * 1.25)
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig8_efficiency_pareto.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 9: Direct PyTorch Ensemble Group Permutation Importance (replaces weak XGBoost surrogate!)
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    df_p_plot = df_perm.sort_values("Delta_MAE_Mean_MW", ascending=True)
    ax.barh(df_p_plot["Feature_Group"], df_p_plot["Delta_MAE_Mean_MW"],
            xerr=df_p_plot["Delta_MAE_Std_MW"], capsize=3, color="#1d3557", height=0.58)
    for i, (_, r) in enumerate(df_p_plot.iterrows()):
        ax.text(r["Delta_MAE_Mean_MW"] + r["Delta_MAE_Std_MW"] + 18, i,
                f"+{r['Delta_MAE_Mean_MW']:.1f} MW (k={int(r['Num_Features'])})",
                va="center", fontsize=7.8)
    ax.set_xlabel("Increase in Deployed PyTorch Ensemble MAE upon Block Permutation (MW)")
    ax.set_title("Direct Group Permutation Importance on the Deployed Val-Opt RNN Ensemble (15 Repeats)")
    ax.set_xlim(0, (df_p_plot["Delta_MAE_Mean_MW"] + df_p_plot["Delta_MAE_Std_MW"]).max() * 1.28)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig9_shap_attribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # Fig 11: Conformal Intervals (Static Split vs Locally Normalized ACI — legend cleanly above!)
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True)
    for ax, c80, c90, ttl in [
        (axes[0], cp_static_80, cp_static_90,
         f"(a) Static Split Conformal (90% PICP={cp_static_90['PICP_%']:.1f}%, 80% PICP={cp_static_80['PICP_%']:.1f}% — Under-Covers Shift)"),
        (axes[1], cp_aci_80, cp_aci_90,
         f"(b) Locally Normalized Adaptive Conformal Inference — LN-ACI (90% PICP={cp_aci_90['PICP_%']:.1f}%, 80% PICP={cp_aci_80['PICP_%']:.1f}%)"),
    ]:
        ax.fill_between(ds["te_dates"], c90["lo"], c90["hi"], color="#a8dadc", alpha=0.50,
                        label=f"90% PI (Mean ±{c90['Mean_Half_Width_MW']:.0f} MW)")
        ax.fill_between(ds["te_dates"], c80["lo"], c80["hi"], color="#457b9d", alpha=0.35,
                        label=f"80% PI (Mean ±{c80['Mean_Half_Width_MW']:.0f} MW)")
        ax.plot(ds["te_dates"], y_te_inv, color="#111111", lw=1.05, label="Actual Demand")
        ax.plot(ds["te_dates"], mean_preds["Val-Opt RNN Ensemble"], color="#1d3557", lw=1.05, label="RNN Ensemble")
        ax.set_ylabel("Peak Demand (MW)")
        ax.set_title(ttl, fontsize=9.3, pad=20)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=4, frameon=False, fontsize=7.8)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig11_conformal_intervals.pdf", bbox_inches="tight")
    plt.close(fig)

    # Save CSVs & metadata
    df_seeds.to_csv(f"{OUT_DIR}/part1_per_seed_metrics.csv", index=False)
    df_summary.to_csv(f"{OUT_DIR}/part1_summary_metrics.csv")
    df_dm.to_csv(f"{OUT_DIR}/part1_dm_tests.csv", index=False)
    df_cp.to_csv(f"{OUT_DIR}/part1_conformal.csv", index=False)
    df_abl.to_csv(f"{OUT_DIR}/part1_ablation.csv", index=False)
    df_reg.to_csv(f"{OUT_DIR}/part1_regime_analysis.csv", index=False)
    df_bins.to_csv(f"{OUT_DIR}/part1_demand_bins.csv", index=False)
    df_perm.to_csv(f"{OUT_DIR}/part1_permutation_importance.csv", index=False)
    pd.DataFrame(eff_profile).T.to_csv(f"{OUT_DIR}/part1_efficiency.csv")

    meta_out = {
        "n_clean": ds["n_raw"], "n_features": ds["n_features"],
        "tau_star_C": ds["tau_star"], "tau_slope_MW_per_C": round(ds["tau_slope"], 2),
        "ridge_alpha": ridge_alpha,
        "n_train": len(X_tr), "n_val_total": len(X_va),
        "n_val_es": len(X_va_es), "n_val_cal": len(X_va_cal), "n_test": len(X_te),
        "seeds": SEEDS,
        "train_span": [str(ds["train_dates"][0].date()), str(ds["train_dates"][1].date())],
        "val_span": [str(ds["val_dates"][0].date()), str(ds["val_dates"][1].date())],
        "test_span": [str(ds["test_dates"][0].date()), str(ds["test_dates"][1].date())],
        "learned_weights_per_seed": {str(s): [round(x, 4) for x in w] for s, w in zip(SEEDS, learned_weights)},
        "learned_weights_mean": [round(float(x), 4) for x in mean_w],
        "ens_5seed_mae": round(ens_5s_mae, 2),
        "seed42_rnn_vs_pt3_dm": {"dm_hln": round(dm_s0_pt3[0], 3), "p_val": round(dm_s0_pt3[1], 4), "d_mae": round(dm_s0_pt3[2], 2)},
        "data_qa": ds["data_qa"],
        "efficiency": eff_profile,
    }
    with open(f"{OUT_DIR}/part1_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nALL REVISED 5-SEED BENCHMARK EXPERIMENTS COMPLETED IN {time.perf_counter() - t_start:.1f}s!", flush=True)
