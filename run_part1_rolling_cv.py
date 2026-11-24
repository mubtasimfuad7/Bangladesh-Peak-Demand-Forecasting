"""
v4 3-Fold Expanding-Window Rolling-Origin CV (ALL 5 SEEDS UNIFORMLY ACROSS ALL 3 FOLDS!) +
Fold-1 Non-Stationarity Controls (PatchTST-CM-Diff Delta-y AND faithful PatchTST-RevIN CI) +
Within-Period Fixed-Test Sensitivity Curve (1.5, 2.5, 3.5, 4.5 yr on fixed 2025-26 test).
"""
import json
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
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

FIG_DIR = "part1_journal_package/figures"
OUT_DIR = "part1_journal_package/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 7, 2024, 999]  # Uniform 5 seeds across ALL folds!
HIST_SIZE = 14
EPOCHS = 45
BATCH_SIZE = 32
PATIENCE = 7
OPT_WD = 1e-5
OPT_DROPOUT = 0.20
TAU_COOL = 25.0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def windowing(X_input, y_input, history_size):
    data, labels = [], []
    for i in range(history_size, len(y_input)):
        data.append(X_input[i - history_size : i, :])
        labels.append(y_input[i])
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)


class PatchTSTCMModel(nn.Module):
    def __init__(self, n_feat, seq_len=14, patch_len=4, stride=2, d_model=64):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.patch_proj = nn.Linear(patch_len * n_feat, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128,
                                         dropout=OPT_DROPOUT, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(OPT_DROPOUT),
                                  nn.Linear(self.n_patches * d_model, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        p = x.unfold(1, self.patch_len, self.stride).reshape(x.size(0), self.n_patches, -1)
        return self.head(self.encoder(self.patch_proj(p) + self.pos_emb))


class PatchTSTRevINModel(nn.Module):
    def __init__(self, seq_len=14, patch_len=4, stride=2, d_model=64, eps=1e-5):
        super().__init__()
        self.patch_len, self.stride, self.eps = patch_len, stride, eps
        self.n_patches = (seq_len - patch_len) // stride + 1
        self.revin_weight = nn.Parameter(torch.ones(1, 1))
        self.revin_bias = nn.Parameter(torch.zeros(1, 1))
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128,
                                         dropout=OPT_DROPOUT, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(OPT_DROPOUT),
                                  nn.Linear(self.n_patches * d_model, 1))
    def forward(self, x):
        y_seq = x[:, :, -1]
        mu = y_seq.mean(dim=1, keepdim=True)
        sd = torch.sqrt(y_seq.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        y_norm = ((y_seq - mu) / sd) * self.revin_weight + self.revin_bias
        patches = y_norm.unfold(dimension=1, size=self.patch_len, step=self.stride)
        out_norm = self.head(self.encoder(self.patch_proj(patches) + self.pos_emb))
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
        w = torch.softmax(self.attn_proj(torch.bmm(h, h.transpose(1, 2)) / math.sqrt(h.size(-1))), dim=-1)
        return self.head(torch.bmm(w, h))


def train_torch_model(model, X_tr, y_tr, X_va_es, y_va_es, seed, lr=2.5e-3):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=OPT_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=4, min_lr=1e-5)
    loss_fn = nn.L1Loss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=BATCH_SIZE, shuffle=True)
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


def fit_ridge_val(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_te_2d, sy):
    best_mae, best_mod = float("inf"), None
    for a in [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]:
        m = Ridge(alpha=a).fit(X_tr_2d, y_tr.ravel())
        mae = mean_absolute_error(y_va_es.ravel(), m.predict(X_va_es_2d))
        if mae < best_mae: best_mae, best_mod = mae, m
    return sy.inverse_transform(best_mod.predict(X_te_2d).reshape(-1, 1))


def load_full_features():
    df = pd.read_excel("version_5_extended_backup.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    if "Dhaka_temp" in df.columns: df["Maximum Temperature in Dhaka was"] = df["Dhaka_temp"]
    if "Mymensingh_temp" not in df.columns and "Dhaka_temp" in df.columns: df["Mymensingh_temp"] = df["Dhaka_temp"]
    if "Year" in df.columns: df = df.drop(columns=["Year"])

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
    meta = df[["Holiday_cat", "Day of the week", target_col, temp_col]].copy()
    df_enc = pd.get_dummies(df, columns=["Day of the week", "Holiday name"], drop_first=True, dtype=float)
    all_cols = df_enc.columns.tolist()
    cont_cols = [c for c in all_cols if c not in cat_cols and c != target_col and not c.startswith("Day of the week_") and not c.startswith("Holiday name_")]
    cat_oh = [c for c in all_cols if c not in cont_cols and c != target_col]
    return df_enc, meta, cont_cols, cat_oh, target_col, temp_col


def eval_split_models(tr_df, va_df, te_df, meta, cont_cols, cat_oh, target_col, temp_col, lrs, run_all=True):
    sx = MinMaxScaler().fit(tr_df[cont_cols])
    sy = MinMaxScaler().fit(tr_df[[target_col]])

    def pack(sub):
        return np.concatenate([sx.transform(sub[cont_cols]), sub[cat_oh].values.astype(np.float32), sy.transform(sub[[target_col]])], axis=1)

    d_tr, d_va, d_te = pack(tr_df), pack(va_df), pack(te_df)
    X_tr, y_tr = windowing(d_tr, d_tr[:, -1], HIST_SIZE)
    X_va, y_va = windowing(d_va, d_va[:, -1], HIST_SIZE)
    X_te, y_te = windowing(d_te, d_te[:, -1], HIST_SIZE)

    n_va = len(X_va)
    va_es_end = max(30, (n_va - HIST_SIZE) // 2)
    X_va_es, y_va_es = X_va[:va_es_end], y_va[:va_es_end]

    y_te_inv = te_df[[target_col]].iloc[HIST_SIZE:].values
    y_pers = te_df[target_col].values[HIST_SIZE - 1 : -1].reshape(-1, 1)

    tr_m = meta.loc[tr_df.index]
    te_m = meta.loc[te_df.index]
    dow_means = tr_m.groupby("Day of the week")[target_col].mean().to_dict()
    te_dows = te_m["Day of the week"].values
    dow_step = np.array([dow_means[te_dows[i]] - dow_means[te_dows[i - 1]] for i in range(HIST_SIZE, len(te_df))]).reshape(-1, 1)
    y_wkday = y_pers + dow_step

    X_tr_2d = X_tr.reshape(len(X_tr), -1)
    X_va_es_2d = X_va_es.reshape(len(X_va_es), -1)
    X_te_2d = X_te.reshape(len(X_te), -1)
    nf = X_tr.shape[-1]

    ex_full = pd.DataFrame(index=meta.index)
    tl1 = meta[temp_col].shift(1).bfill()
    ex_full["CDD_l1"] = np.maximum(0.0, tl1 - TAU_COOL)
    ex_full["Wknd"] = meta["Day of the week"].isin(["Friday", "Saturday"]).astype(float)
    ex_full["Hol"] = (meta["Holiday_cat"] > 0).astype(float)
    sub_m = meta.loc[tr_df.index[0] : te_df.index[-1]]
    ex_m = ex_full.loc[tr_df.index[0] : te_df.index[-1]].values
    y_m = sub_m[target_col].values.astype(float)
    sfit = SARIMAX(y_m[:len(tr_df)], exog=ex_m[:len(tr_df)], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=80)
    p_sar = sfit.apply(y_m, exog=ex_m).fittedvalues[-len(y_te_inv):].reshape(-1, 1)

    p_ridge = fit_ridge_val(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_te_2d, sy)

    out = {
        "N_Train": len(X_tr), "N_Val": len(X_va), "N_Test": len(y_te_inv),
        "Persist_MAE": round(float(mean_absolute_error(y_te_inv, y_pers)), 2),
        "Wkday_MAE": round(float(mean_absolute_error(y_te_inv, y_wkday)), 2),
        "SARIMAX_MAE": round(float(mean_absolute_error(y_te_inv, p_sar)), 2),
        "Ridge_MAE": round(float(mean_absolute_error(y_te_inv, p_ridge)), 2),
    }

    dy_tr = (tr_df[target_col].values[HIST_SIZE:] - tr_df[target_col].values[HIST_SIZE - 1 : -1]).reshape(-1, 1)
    dy_va_es = (va_df[target_col].values[HIST_SIZE : HIST_SIZE + len(X_va_es)] - va_df[target_col].values[HIST_SIZE - 1 : HIST_SIZE - 1 + len(X_va_es)]).reshape(-1, 1)
    s_dy = StandardScaler().fit(dy_tr)
    dy_tr_n = s_dy.transform(dy_tr).astype(np.float32)
    dy_va_n = s_dy.transform(dy_va_es).astype(np.float32)

    if run_all:
        best_m, best_rd = float("inf"), None
        for a in [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0]:
            rd = Ridge(alpha=a).fit(X_tr_2d, dy_tr.ravel())
            mae = mean_absolute_error(dy_va_es.ravel(), rd.predict(X_va_es_2d))
            if mae < best_m: best_m, best_rd = mae, rd
        p_rd = y_pers + best_rd.predict(X_te_2d).reshape(-1, 1)
        out["RidgeDiff_MAE"] = round(float(mean_absolute_error(y_te_inv, p_rd)), 2)

        xmd = xgb.XGBRegressor(random_state=42, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
        xmd.fit(X_tr_2d, dy_tr.ravel(), eval_set=[(X_va_es_2d, dy_va_es.ravel())], verbose=False)
        p_xgbd = y_pers + xmd.predict(X_te_2d).reshape(-1, 1)
        out["XGBDiff_MAE"] = round(float(mean_absolute_error(y_te_inv, p_xgbd)), 2)

    eq_maes, pt3_maes, ptd_maes, rev_maes = [], [], [], []
    for s in SEEDS:
        pt_preds = []
        for off in [0, 1000, 2000]:
            m_pt = train_torch_model(PatchTSTCMModel(nf), X_tr, y_tr, X_va_es, y_va_es, s + off, lr=lrs.get("PatchTST-CM (Single)", 1e-3))
            pt_preds.append(sy.inverse_transform(predict_torch(m_pt, X_te)))
        pt3_maes.append(mean_absolute_error(y_te_inv, np.mean(pt_preds, axis=0)))

        if run_all:
            ptd_preds, rev_preds = [], []
            for off in [0, 1000, 2000]:
                m_pd = train_torch_model(PatchTSTCMModel(nf), X_tr, dy_tr_n, X_va_es, dy_va_n, s + off, lr=lrs.get("PatchTST-CM-Diff (3x)", 1e-3))
                ptd_preds.append(y_pers + s_dy.inverse_transform(predict_torch(m_pd, X_te)))
                m_rv = train_torch_model(PatchTSTRevINModel(HIST_SIZE), X_tr, y_tr, X_va_es, y_va_es, s + off, lr=lrs.get("PatchTST-RevIN (Single)", 1e-3))
                rev_preds.append(sy.inverse_transform(predict_torch(m_rv, X_te)))
            ptd_maes.append(mean_absolute_error(y_te_inv, np.mean(ptd_preds, axis=0)))
            rev_maes.append(mean_absolute_error(y_te_inv, np.mean(rev_preds, axis=0)))

        t_l = []
        for nm, fn in [("BiGRU", lambda: BiGRUModel(nf)), ("BiLSTM-Attention", lambda: BiLSTMAttnModel(nf)), ("BiLSTM", lambda: BiLSTMModel(nf))]:
            m = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, s, lr=lrs.get(nm, 2.5e-3))
            t_l.append(sy.inverse_transform(predict_torch(m, X_te)))
        eq_maes.append(mean_absolute_error(y_te_inv, np.mean(t_l, axis=0)))

    out.update({
        "PatchTST3_MAE": round(float(np.mean(pt3_maes)), 2),
        "PatchTST3_std": round(float(np.std(pt3_maes, ddof=1)), 2),
        "EqRNN_MAE": round(float(np.mean(eq_maes)), 2),
        "EqRNN_std": round(float(np.std(eq_maes, ddof=1)), 2),
    })
    if run_all:
        out.update({
            "PatchTSTDiff_MAE": round(float(np.mean(ptd_maes)), 2),
            "PatchTSTDiff_std": round(float(np.std(ptd_maes, ddof=1)), 2),
            "PatchTSTRevIN_MAE": round(float(np.mean(rev_maes)), 2),
            "PatchTSTRevIN_std": round(float(np.std(rev_maes, ddof=1)), 2),
        })
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    df_enc, meta, cont_cols, cat_oh, target_col, temp_col = load_full_features()
    lrs = json.load(open(f"{OUT_DIR}/part1_meta.json"))["tuned_neural_lrs"] if os.path.exists(f"{OUT_DIR}/part1_meta.json") else {}

    print("=== PART A: 3-FOLD EXPANDING-WINDOW ROLLING CV (ALL 5 SEEDS + DIFF/REVIN CONTROLS) ===", flush=True)
    folds = [
        ("Fold 1 (2023)", "2022-07-01", "2023-01-01", "2023-12-31"),
        ("Fold 2 (2024)", "2023-07-01", "2024-01-01", "2024-12-31"),
        ("Fold 3 (2025-26)", "2024-07-08", "2025-03-09", "2026-06-30"),
    ]
    rows = []
    for fname, val_start, test_start, test_end in folds:
        if fname == "Fold 3 (2025-26)" and os.path.exists(f"{OUT_DIR}/part1_summary_metrics.csv"):
            s5 = pd.read_csv(f"{OUT_DIR}/part1_summary_metrics.csv", index_col=0)
            row = {
                "Fold": fname, "N_Train": 1652, "N_Val": 224, "N_Test": 462,
                "Persist_MAE": round(float(s5.loc["Persistence (Lag-1)", "MAE_mean"]), 2),
                "Wkday_MAE": round(float(s5.loc["Weekday-Adj Persistence", "MAE_mean"]), 2),
                "SARIMAX_MAE": round(float(s5.loc["SARIMAX", "MAE_mean"]), 2),
                "Ridge_MAE": round(float(s5.loc["Ridge (14xF)", "MAE_mean"]), 2),
                "RidgeDiff_MAE": round(float(s5.loc["Ridge-Diff (Δy)", "MAE_mean"]), 2),
                "XGBDiff_MAE": round(float(s5.loc["XGBoost-Diff (Δy)", "MAE_mean"]), 2),
                "PatchTST3_MAE": round(float(s5.loc["PatchTST-CM (3x)", "MAE_mean"]), 2),
                "PatchTST3_std": round(float(s5.loc["PatchTST-CM (3x)", "MAE_std"]), 2),
                "PatchTSTDiff_MAE": round(float(s5.loc["PatchTST-CM-Diff (Δy, 3x)", "MAE_mean"]), 2),
                "PatchTSTDiff_std": round(float(s5.loc["PatchTST-CM-Diff (Δy, 3x)", "MAE_std"]), 2),
                "PatchTSTRevIN_MAE": round(float(s5.loc["PatchTST-RevIN (CI, 3x)", "MAE_mean"]), 2),
                "PatchTSTRevIN_std": round(float(s5.loc["PatchTST-RevIN (CI, 3x)", "MAE_std"]), 2),
                "EqRNN_MAE": round(float(s5.loc["Equal-Weight RNN Ens", "MAE_mean"]), 2),
                "EqRNN_std": round(float(s5.loc["Equal-Weight RNN Ens", "MAE_std"]), 2),
            }
            rows.append(row)
            print(f"  {fname:16s} (5-seed synced): Wkday={row['Wkday_MAE']:.1f} | RidgeDiff={row['RidgeDiff_MAE']:.1f} | PT-CM={row['PatchTST3_MAE']:.1f} | PT-Diff={row['PatchTSTDiff_MAE']:.1f} | PT-RevIN={row['PatchTSTRevIN_MAE']:.1f}", flush=True)
            continue

        tr_df = df_enc.loc[: pd.Timestamp(val_start) - pd.Timedelta(days=1)]
        va_df = df_enc.loc[val_start : pd.Timestamp(test_start) - pd.Timedelta(days=1)]
        te_df = df_enc.loc[test_start : test_end]
        row = {"Fold": fname, **eval_split_models(tr_df, va_df, te_df, meta, cont_cols, cat_oh, target_col, temp_col, lrs, True)}
        rows.append(row)
        print(f"  {fname:16s} (5-seed, N_te={row['N_Test']}): Wkday={row['Wkday_MAE']:.1f} | RidgeDiff={row['RidgeDiff_MAE']:.1f} | PT-CM={row['PatchTST3_MAE']:.1f}±{row['PatchTST3_std']:.1f} | PT-Diff={row['PatchTSTDiff_MAE']:.1f} | PT-RevIN={row['PatchTSTRevIN_MAE']:.1f}", flush=True)

    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(f"{OUT_DIR}/part1_rolling_cv.csv", index=False)
    print("\n=== 3-FOLD ROLLING CV SUMMARY (5 SEEDS UNIFORM) ===")
    print(df_cv.to_string(index=False))
    for col in ["Persist_MAE", "Wkday_MAE", "SARIMAX_MAE", "Ridge_MAE", "RidgeDiff_MAE", "XGBDiff_MAE", "PatchTST3_MAE", "PatchTSTDiff_MAE", "PatchTSTRevIN_MAE", "EqRNN_MAE"]:
        print(f"  3-Fold Mean {col:18s}: {df_cv[col].mean():.2f} MW")

    print("\n=== PART B: WITHIN-PERIOD SENSITIVITY TO TRAILING HISTORY (5 SEEDS, 2025-26 FIXED) ===", flush=True)
    tc_full, vc_full = int(0.70 * len(df_enc)), int(0.80 * len(df_enc))
    va_fixed, te_fixed, tr_full = df_enc.iloc[tc_full:vc_full], df_enc.iloc[vc_full:], df_enc.iloc[:tc_full]
    lc_rows = []
    for yrs, n_days in [(1.5, 548), (2.5, 913), (3.5, 1278), (4.5, len(tr_full))]:
        if n_days == len(tr_full) and os.path.exists(f"{OUT_DIR}/part1_summary_metrics.csv"):
            s5 = pd.read_csv(f"{OUT_DIR}/part1_summary_metrics.csv", index_col=0)
            r_lc = {
                "Train_Years": yrs, "N_Train_Windows": len(tr_full) - HIST_SIZE,
                "SARIMAX_MAE": round(float(s5.loc["SARIMAX", "MAE_mean"]), 2),
                "Ridge_MAE": round(float(s5.loc["Ridge (14xF)", "MAE_mean"]), 2),
                "PatchTST3_MAE": round(float(s5.loc["PatchTST-CM (3x)", "MAE_mean"]), 2),
                "PatchTST3_std": round(float(s5.loc["PatchTST-CM (3x)", "MAE_std"]), 2),
                "EqRNN_MAE": round(float(s5.loc["Equal-Weight RNN Ens", "MAE_mean"]), 2),
                "EqRNN_std": round(float(s5.loc["Equal-Weight RNN Ens", "MAE_std"]), 2),
            }
        else:
            res = eval_split_models(tr_full.iloc[-n_days:], va_fixed, te_fixed, meta, cont_cols, cat_oh, target_col, temp_col, lrs, False)
            r_lc = {
                "Train_Years": yrs, "N_Train_Windows": res["N_Train"],
                "SARIMAX_MAE": res["SARIMAX_MAE"], "Ridge_MAE": res["Ridge_MAE"],
                "PatchTST3_MAE": res["PatchTST3_MAE"], "PatchTST3_std": res["PatchTST3_std"],
                "EqRNN_MAE": res["EqRNN_MAE"], "EqRNN_std": res["EqRNN_std"],
            }
        lc_rows.append(r_lc)
        print(f"  Train={yrs:.1f} yr ({r_lc['N_Train_Windows']:4d} win) | SARIMAX={r_lc['SARIMAX_MAE']:.1f} | Ridge={r_lc['Ridge_MAE']:.1f} | PT-CM={r_lc['PatchTST3_MAE']:.1f}±{r_lc['PatchTST3_std']:.1f} | EqRNN={r_lc['EqRNN_MAE']:.1f}±{r_lc['EqRNN_std']:.1f}", flush=True)

    df_lc = pd.DataFrame(lc_rows)
    df_lc.to_csv(f"{OUT_DIR}/part1_learning_curve.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.7))

    ax = axes[0]; x = np.arange(len(df_cv)); w = 0.12
    series = [
        ("Wkday_MAE", "Wkday Persist", "#94a3b8"),
        ("RidgeDiff_MAE", "Ridge-Diff (Δy)", "#d97706"),
        ("PatchTSTDiff_MAE", "PT-CM-Diff (Δy)", "#2563eb"),
        ("PatchTSTRevIN_MAE", "PT-RevIN (CI)", "#14b8a6"),
        ("EqRNN_MAE", "Eq-Wt RNN Ens", "#1d3557"),
        ("PatchTST3_MAE", "PT-CM (Level)", "#dc2626"),
    ]
    for idx, (col, lbl, colr) in enumerate(series):
        offset = (idx - 2.5) * w
        yerr = (df_cv["EqRNN_std"] if col == "EqRNN_MAE" else
                df_cv["PatchTST3_std"] if col == "PatchTST3_MAE" else
                df_cv["PatchTSTDiff_std"] if col == "PatchTSTDiff_MAE" else
                df_cv["PatchTSTRevIN_std"] if col == "PatchTSTRevIN_MAE" else None)
        ax.bar(x + offset, df_cv[col], yerr=yerr, capsize=2.2, width=w, color=colr, label=lbl)
    ax.set_xticks(x); ax.set_xticklabels([f"{r['Fold']}\n(n={r['N_Test']})" for _, r in df_cv.iterrows()], fontsize=8.3)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_ylim(0, df_cv[["Wkday_MAE", "RidgeDiff_MAE", "PatchTSTDiff_MAE", "PatchTSTRevIN_MAE", "EqRNN_MAE", "PatchTST3_MAE"]].values.max() * 1.18)
    ax.set_title("(a) 3-Fold Rolling CV (5 Seeds; Level vs. Δy & RevIN Controls)", pad=38)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7.5)

    ax = axes[1]; yrs_x = df_lc["Train_Years"].values
    ax.plot(yrs_x, df_lc["SARIMAX_MAE"], marker="s", lw=1.8, ls="--", color="#6d597a", label="SARIMAX")
    ax.plot(yrs_x, df_lc["Ridge_MAE"], marker="^", lw=1.8, ls=":", color="#d97706", label="Ridge (14×F)")
    ax.errorbar(yrs_x, df_lc["EqRNN_MAE"], yerr=df_lc["EqRNN_std"], marker="o", lw=1.9, capsize=3.5, color="#1d3557", label="Eq-Wt RNN Ens")
    ax.errorbar(yrs_x, df_lc["PatchTST3_MAE"], yerr=df_lc["PatchTST3_std"], marker="D", lw=2.1, capsize=3.5, color="#0d9488", label="PatchTST-CM (3×)")
    ax.set_xticks(yrs_x)
    ax.set_xticklabels([f"{y:.1f} yr\n({n}d)" for y, n in zip(df_lc["Train_Years"], df_lc["N_Train_Windows"])], fontsize=8.2)
    ax.set_ylabel("Fixed 2025–26 Test MAE (MW)")
    ax.set_title("(b) Within-Period Sensitivity to Training History (2025–26 Test)", pad=38)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.6)

    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig13_rolling_cv.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"Saved v4 rolling CV & learning curve in {time.perf_counter() - t0:.1f}s!", flush=True)
