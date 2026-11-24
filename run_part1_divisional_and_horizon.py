"""
v4 Same-Split 8-Division & Causal Multi-Horizon Suite (5 Seeds Uniformly!):
  1) 8-Division Benchmark (5 seeds, same 462-day split, Year dropped, detrended tau*=25 C):
     Lag-1 Persistence, Weekday-Adj Persistence, Ridge, Single BiGRU, Equal-Weight RNN Ens, PatchTST-CM (3x)
     + paired HLN-DM p-values (Eq-Wt RNN vs PatchTST-CM 3x and vs Single BiGRU)!
  2) Causal Direct Multi-Horizon (h = 1, 3, 7, 14, 30 days, 5 seeds):
     - Per-horizon Val-ES tuned Direct Ridge (alpha_h logged!)
     - Climatology+Trend (Train-fit AND Train+Val refit!)
     - Fixed Validation-Tuned Convex Blend: y_blend = w_h * Ridge + (1 - w_h) * Clim (w_h in [0, 1] on Val-ES!)
     - Direct PatchTST-CM (3x), Equal-Weight RNN Ens, and Origin Persistence!
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

FIG_DIR = "part1_journal_package/figures"
OUT_DIR = "part1_journal_package/results"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS = [42, 123, 7, 2024, 999]  # Uniform 5 seeds!
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


def windowing_horizon(X_input, y_input, history_size, horizon=1):
    data, labels = [], []
    end = len(y_input) - (horizon - 1)
    for i in range(history_size, end):
        data.append(X_input[i - history_size : i, :])
        labels.append(y_input[i + horizon - 1])
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)


def dm_hln_pval(y_true, pred_a, pred_b, h=1, max_lag=14):
    d = np.abs(y_true.ravel() - pred_a.ravel()) - np.abs(y_true.ravel() - pred_b.ravel())
    T = len(d)
    ml = min(max_lag, max(1, T // 4))
    var_d = np.var(d, ddof=1)
    for k in range(1, ml + 1):
        w = 1.0 - k / (ml + 1.0)
        var_d += 2.0 * w * np.cov(d[k:], d[:-k], ddof=1)[0, 1]
    dm_raw = float(np.mean(d) / np.sqrt(max(var_d, 1e-12) / T))
    hln = dm_raw * math.sqrt((T + 1.0 - 2.0 * h + (h * (h - 1.0)) / T) / T)
    return float(2.0 * stats.t.sf(abs(hln), df=T - 1))


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


def calc_metrics(y_true, y_pred):
    yt, yp = y_true.ravel(), y_pred.ravel()
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / np.clip(yt, 1.0, None))) * 100.0)
    r2 = float(r2_score(yt, yp))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}


def fit_ridge_val(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_te_2d, sy=None):
    best_mae, best_alpha, best_mod = float("inf"), 10.0, None
    for a in [0.1, 1.0, 10.0, 50.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]:
        m = Ridge(alpha=a).fit(X_tr_2d, y_tr.ravel())
        mae = mean_absolute_error(y_va_es.ravel(), m.predict(X_va_es_2d))
        if mae < best_mae: best_mae, best_alpha, best_mod = mae, a, m
    p_va = best_mod.predict(X_va_es_2d).reshape(-1, 1)
    p_te = best_mod.predict(X_te_2d).reshape(-1, 1)
    if sy is not None:
        return sy.inverse_transform(p_va), sy.inverse_transform(p_te), best_alpha
    return p_va, p_te, best_alpha


def fit_climatology_model(dates_tr, y_tr_raw):
    t0 = dates_tr[0]
    def make_design(dates):
        t_yr = ((dates - t0).days.values / 365.25).astype(float)
        doy = dates.dayofyear.values.astype(float)
        cols = [np.ones_like(t_yr), t_yr]
        for k in [1, 2, 3]:
            cols.append(np.sin(2.0 * np.pi * k * doy / 365.25))
            cols.append(np.cos(2.0 * np.pi * k * doy / 365.25))
        dow = dates.dayofweek.values
        for d in range(6): cols.append((dow == d).astype(float))
        return np.column_stack(cols)
    coef, _, _, _ = np.linalg.lstsq(make_design(dates_tr), y_tr_raw.ravel(), rcond=None)
    return lambda dates: (make_design(dates) @ coef).reshape(-1, 1)


def run_divisional_benchmark(df_raw, lrs):
    print("\n=== 1. v4 SAME-SPLIT 8-DIVISION BENCHMARK (5 SEEDS + PAIRED HLN-DM TESTS) ===", flush=True)
    divisions = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Mymensingh", "Rangpur", "Barishal", "Sylhet"]
    rows = []

    for div in divisions:
        df = df_raw.copy()
        if "Year" in df.columns: df = df.drop(columns=["Year"])
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
        dows_all = df["Day of the week"].values
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

        va_es_end = (len(X_va) - HIST_SIZE) // 2
        X_va_es, y_va_es = X_va[:va_es_end], y_va[:va_es_end]

        y_te_inv = df_enc[[target_col]].iloc[va_c + HIST_SIZE :].values
        y_pers = df_enc[target_col].values[va_c + HIST_SIZE - 1 : -1].reshape(-1, 1)

        dow_means = df.iloc[:tr_c].groupby("Day of the week")[target_col].mean().to_dict()
        dow_step = np.array([dow_means[dows_all[i]] - dow_means[dows_all[i - 1]] for i in range(va_c + HIST_SIZE, len(df))]).reshape(-1, 1)
        y_wkday = y_pers + dow_step
        wm = calc_metrics(y_te_inv, y_wkday)
        pm = calc_metrics(y_te_inv, y_pers)

        _, p_ridge, _ = fit_ridge_val(X_tr.reshape(len(X_tr), -1), y_tr, X_va_es.reshape(len(X_va_es), -1), y_va_es, X_te.reshape(len(X_te), -1), sy)
        rm = calc_metrics(y_te_inv, p_ridge)

        nf = X_tr.shape[-1]
        eq_ms, bigru_ms, pt3_ms = [], [], []
        eq_preds, bigru_preds, pt3_preds = [], [], []
        for s in SEEDS:
            pt_l = []
            for off in [0, 1000, 2000]:
                mp = train_torch_model(PatchTSTCMModel(nf), X_tr, y_tr, X_va_es, y_va_es, s + off, lr=lrs.get("PatchTST-CM (Single)", 1e-3))
                pt_l.append(sy.inverse_transform(predict_torch(mp, X_te)))
            p_pt = np.mean(pt_l, axis=0)
            pt3_preds.append(p_pt)
            pt3_ms.append(calc_metrics(y_te_inv, p_pt))

            t_l = []
            for nm, fn in [("BiGRU", lambda: BiGRUModel(nf)), ("BiLSTM-Attention", lambda: BiLSTMAttnModel(nf)), ("BiLSTM", lambda: BiLSTMModel(nf))]:
                m = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, s, lr=lrs.get(nm, 2.5e-3))
                t_l.append(sy.inverse_transform(predict_torch(m, X_te)))
            bigru_preds.append(t_l[0])
            p_eq = np.mean(t_l, axis=0)
            eq_preds.append(p_eq)
            bigru_ms.append(calc_metrics(y_te_inv, t_l[0]))
            eq_ms.append(calc_metrics(y_te_inv, p_eq))

        p_dm_pt = dm_hln_pval(y_te_inv, np.mean(eq_preds, axis=0), np.mean(pt3_preds, axis=0))
        p_dm_bg = dm_hln_pval(y_te_inv, np.mean(eq_preds, axis=0), np.mean(bigru_preds, axis=0))
        p_dm_ri = dm_hln_pval(y_te_inv, np.mean(eq_preds, axis=0), p_ridge)

        rows.append({
            "Division": div,
            "PT3_MAE": float(np.mean([x["MAE"] for x in pt3_ms])), "PT3_MAE_std": float(np.std([x["MAE"] for x in pt3_ms], ddof=1)),
            "PT3_MAPE": float(np.mean([x["MAPE"] for x in pt3_ms])), "PT3_MAPE_std": float(np.std([x["MAPE"] for x in pt3_ms], ddof=1)),
            "EqRNN_MAE": float(np.mean([x["MAE"] for x in eq_ms])), "EqRNN_MAE_std": float(np.std([x["MAE"] for x in eq_ms], ddof=1)),
            "EqRNN_MAPE": float(np.mean([x["MAPE"] for x in eq_ms])), "EqRNN_MAPE_std": float(np.std([x["MAPE"] for x in eq_ms], ddof=1)),
            "Ridge_MAE": rm["MAE"], "Ridge_MAPE": rm["MAPE"],
            "BiGRU_MAE": float(np.mean([x["MAE"] for x in bigru_ms])), "BiGRU_MAPE": float(np.mean([x["MAPE"] for x in bigru_ms])),
            "BiGRU_MAPE_std": float(np.std([x["MAPE"] for x in bigru_ms], ddof=1)),
            "Wkday_MAE": wm["MAE"], "Wkday_MAPE": wm["MAPE"],
            "Persist_MAE": pm["MAE"], "Persist_MAPE": pm["MAPE"],
            "p_DM_EqRNN_vs_PT3": p_dm_pt, "p_DM_EqRNN_vs_BiGRU": p_dm_bg, "p_DM_EqRNN_vs_Ridge": p_dm_ri,
        })
        print(f"  {div:12s}: EqRNN={rows[-1]['EqRNN_MAE']:6.2f}±{rows[-1]['EqRNN_MAE_std']:4.2f} | PT-CM={rows[-1]['PT3_MAE']:6.2f}±{rows[-1]['PT3_MAE_std']:4.2f} (p={p_dm_pt:.3f}) | Ridge={rm['MAE']:6.2f} | BiGRU={rows[-1]['BiGRU_MAE']:6.2f} | Wkday={wm['MAE']:6.2f}", flush=True)

    return pd.DataFrame(rows)


def run_causal_multihorizon_benchmark(df_raw, lrs):
    print("\n=== 2. v4 CAUSAL MULTI-HORIZON (5 SEEDS + FIXED CONVEX BLEND + PER-HORIZON RIDGE ALPHA + REFIT CLIM) ===", flush=True)
    df = df_raw.copy()
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
    clim_tr_fn = fit_climatology_model(df_enc.index[:tr_c], y_raw[:tr_c])
    clim_refit_fn = fit_climatology_model(df_enc.index[:va_c], y_raw[:va_c])
    s1 = pd.read_csv(f"{OUT_DIR}/part1_summary_metrics.csv", index_col=0)

    rows = []
    for h in [1, 3, 7, 14, 30]:
        X_tr, y_tr = windowing_horizon(data_norm[:tr_c], data_norm[:tr_c, -1], HIST_SIZE, h)
        X_va, y_va = windowing_horizon(data_norm[tr_c:va_c], data_norm[tr_c:va_c, -1], HIST_SIZE, h)
        X_te, y_te = windowing_horizon(data_norm[va_c:], data_norm[va_c:, -1], HIST_SIZE, h)

        va_es_end = (len(X_va) - HIST_SIZE) // 2
        X_va_es, y_va_es = X_va[:va_es_end], y_va[:va_es_end]

        dates_va_es_h = df_enc.index[tr_c + HIST_SIZE + h - 1 : tr_c + HIST_SIZE + h - 1 + len(X_va_es)]
        dates_te_h = df_enc.index[va_c + HIST_SIZE + h - 1 :]

        y_va_es_mw = y_raw[tr_c + HIST_SIZE + h - 1 : tr_c + HIST_SIZE + h - 1 + len(X_va_es)].reshape(-1, 1)
        y_te_inv = y_raw[va_c + HIST_SIZE + h - 1 :].reshape(-1, 1)
        y_orig_pers = y_raw[va_c + HIST_SIZE - 1 : len(y_raw) - h].reshape(-1, 1)

        clim_va_es = clim_tr_fn(dates_va_es_h)
        clim_te = clim_tr_fn(dates_te_h)
        clim_refit_te = clim_refit_fn(dates_te_h)

        # Direct Ridge tuned per horizon h on Val-ES!
        X_tr_2d = X_tr.reshape(len(X_tr), -1)
        X_va_es_2d = X_va_es.reshape(len(X_va_es), -1)
        X_te_2d = X_te.reshape(len(X_te), -1)
        p_ridge_va, p_ridge_te, alpha_h = fit_ridge_val(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_te_2d, sy)

        # Fixed Convex Blend: w_h * Direct_Ridge + (1 - w_h) * Climatology+Trend, w_h in [0, 1] chosen on Val-ES!
        best_w, best_b_mae = 0.0, float("inf")
        for w_c in np.linspace(0.0, 1.0, 21):
            pv = w_c * p_ridge_va + (1.0 - w_c) * clim_va_es
            mv = mean_absolute_error(y_va_es_mw, pv)
            if mv < best_b_mae:
                best_b_mae, best_w = mv, float(w_c)
        p_blend_te = best_w * p_ridge_te + (1.0 - best_w) * clim_te

        cm = calc_metrics(y_te_inv, clim_te)
        cm_ref = calc_metrics(y_te_inv, clim_refit_te)
        hm = calc_metrics(y_te_inv, p_blend_te)
        rm = calc_metrics(y_te_inv, p_ridge_te)
        pm = calc_metrics(y_te_inv, y_orig_pers)

        if h == 1:
            pt3_mae = float(s1.loc["PatchTST-CM (3x)", "MAE_mean"])
            pt3_sd = float(s1.loc["PatchTST-CM (3x)", "MAE_std"])
            pt3_mape = float(s1.loc["PatchTST-CM (3x)", "MAPE_mean"])
            eq_mae = float(s1.loc["Equal-Weight RNN Ens", "MAE_mean"])
            eq_sd = float(s1.loc["Equal-Weight RNN Ens", "MAE_std"])
        else:
            nf = X_tr.shape[-1]
            pt3_ms, eq_ms = [], []
            for s in SEEDS:
                pt_l = []
                for off in [0, 1000, 2000]:
                    mp = train_torch_model(PatchTSTCMModel(nf), X_tr, y_tr, X_va_es, y_va_es, s + off, lr=lrs.get("PatchTST-CM (Single)", 1e-3))
                    pt_l.append(sy.inverse_transform(predict_torch(mp, X_te)))
                pt3_ms.append(calc_metrics(y_te_inv, np.mean(pt_l, axis=0)))

                t_l = []
                for nm, fn in [("BiGRU", lambda: BiGRUModel(nf)), ("BiLSTM-Attention", lambda: BiLSTMAttnModel(nf)), ("BiLSTM", lambda: BiLSTMModel(nf))]:
                    m = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, s, lr=lrs.get(nm, 2.5e-3))
                    t_l.append(sy.inverse_transform(predict_torch(m, X_te)))
                eq_ms.append(calc_metrics(y_te_inv, np.mean(t_l, axis=0)))

            pt3_mae = float(np.mean([x["MAE"] for x in pt3_ms]))
            pt3_sd = float(np.std([x["MAE"] for x in pt3_ms], ddof=1))
            pt3_mape = float(np.mean([x["MAPE"] for x in pt3_ms]))
            eq_mae = float(np.mean([x["MAE"] for x in eq_ms]))
            eq_sd = float(np.std([x["MAE"] for x in eq_ms], ddof=1))

        rows.append({
            "Horizon_Days": h, "N_Test": len(y_te_inv),
            "PT3_MAE": pt3_mae, "PT3_SD": pt3_sd, "PT3_MAPE": pt3_mape,
            "EqRNN_MAE": eq_mae, "EqRNN_SD": eq_sd,
            "Ridge_MAE": rm["MAE"], "Ridge_Alpha": alpha_h,
            "Blend_Weight_Ridge": best_w,
            "Clim_Ridge_Blend_MAE": hm["MAE"], "Clim_Ridge_Blend_MAPE": hm["MAPE"],
            "Clim_Trend_MAE": cm["MAE"], "Clim_Trend_MAPE": cm["MAPE"],
            "Clim_Refit_MAE": cm_ref["MAE"],
            "Origin_Persist_MAE": pm["MAE"], "Origin_Persist_MAPE": pm["MAPE"],
        })
        print(f"  h={h:2d}d: PT-CM={pt3_mae:6.1f}±{pt3_sd:4.1f} | EqRNN={eq_mae:6.1f} | Ridge(a={alpha_h:g})={rm['MAE']:6.1f} | Blend(w={best_w:.2f})={hm['MAE']:6.1f} | Clim={cm['MAE']:6.1f} (Refit={cm_ref['MAE']:6.1f}) | Persist={pm['MAE']:6.1f}", flush=True)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    t0 = time.perf_counter()
    lrs = json.load(open(f"{OUT_DIR}/part1_meta.json"))["tuned_neural_lrs"] if os.path.exists(f"{OUT_DIR}/part1_meta.json") else {}
    df_raw = pd.read_excel("version_5_extended_backup.xlsx")
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    df_raw = df_raw.sort_values("Date").set_index("Date")
    if "Dhaka_temp" in df_raw.columns: df_raw["Maximum Temperature in Dhaka was"] = df_raw["Dhaka_temp"]
    if "Mymensingh_temp" not in df_raw.columns and "Dhaka_temp" in df_raw.columns: df_raw["Mymensingh_temp"] = df_raw["Dhaka_temp"]

    df_div = run_divisional_benchmark(df_raw, lrs)
    df_div.to_csv(f"{OUT_DIR}/part1_divisional_pytorch.csv", index=False)

    df_mh = run_causal_multihorizon_benchmark(df_raw, lrs)
    df_mh.to_csv(f"{OUT_DIR}/part1_multihorizon.csv", index=False)

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})

    # Fig 5: Same-Split 8-Division Benchmark (5 Seeds)
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    x = np.arange(len(df_div)); w = 0.16
    ax.bar(x - 2*w, df_div["Wkday_MAPE"], width=w, color="#94a3b8", label="Wkday Persist")
    ax.bar(x - w, df_div["Ridge_MAPE"], width=w, color="#d97706", label="Ridge (Same Split)")
    ax.bar(x, df_div["BiGRU_MAPE"], yerr=df_div["BiGRU_MAPE_std"], capsize=2.5, width=w, color="#64748b", label="Single BiGRU (5-Seed)")
    ax.bar(x + w, df_div["EqRNN_MAPE"], yerr=df_div["EqRNN_MAPE_std"], capsize=2.5, width=w, color="#1d3557", label="Eq-Wt RNN Ens (5-Seed)")
    ax.bar(x + 2*w, df_div["PT3_MAPE"], yerr=df_div["PT3_MAPE_std"], capsize=2.5, width=w, color="#0d9488", label="PatchTST-CM 3× (5-Seed)")
    ax.set_xticks(x); ax.set_xticklabels(df_div["Division"], rotation=15, ha="right")
    ax.set_ylabel("Blind Test MAPE (%)  [Lower is Better]")
    ax.set_title("Same-Split 8-Division Peak Demand Benchmark (462 Blind Test Days, 5 Seeds)", pad=28)
    ax.set_ylim(0, df_div[["Wkday_MAPE", "Ridge_MAPE", "BiGRU_MAPE", "EqRNN_MAPE", "PT3_MAPE"]].values.max() * 1.18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5, frameon=False, fontsize=7.4)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig5_divisional_benchmark.pdf", bbox_inches="tight"); plt.close(fig)

    # Fig 12: Multi-Horizon with Fixed Climatology+Ridge Convex Blend
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))
    h_vals = df_mh["Horizon_Days"].values
    axes[0].errorbar(h_vals, df_mh["PT3_MAE"], yerr=df_mh["PT3_SD"], marker="D", lw=2.0, capsize=3.5, color="#0d9488", label="PatchTST-CM (3×)")
    axes[0].plot(h_vals, df_mh["Ridge_MAE"], marker="^", lw=1.6, ls="-.", color="#d97706", label="Direct Ridge (Tuned α_h)")
    axes[0].plot(h_vals, df_mh["Clim_Ridge_Blend_MAE"], marker="o", lw=2.1, color="#1d3557", label="Clim.+Ridge Convex Blend")
    axes[0].plot(h_vals, df_mh["Clim_Trend_MAE"], marker="s", lw=1.5, ls=":", color="#6d597a", label="Climatology+Trend")
    axes[0].plot(h_vals, df_mh["Origin_Persist_MAE"], marker="x", lw=1.4, ls="--", color="#e76f51", label="Origin Persistence")
    axes[0].set_xticks(h_vals); axes[0].set_xlabel("Direct Forecast Horizon h (Days Ahead)"); axes[0].set_ylabel("Blind Test MAE (MW)")
    axes[0].set_title("(a) Multi-Horizon MAE vs. Climatology & Convex Blend", pad=58)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.4)

    axes[1].plot(h_vals, df_mh["PT3_MAPE"], marker="D", lw=2.0, color="#0d9488", label="PatchTST-CM (3×) MAPE (%)")
    axes[1].plot(h_vals, df_mh["Clim_Ridge_Blend_MAPE"], marker="o", lw=2.1, color="#1d3557", label="Clim.+Ridge Convex Blend (%)")
    axes[1].plot(h_vals, df_mh["Clim_Trend_MAPE"], marker="s", lw=1.5, ls=":", color="#6d597a", label="Climatology+Trend (%)")
    axes[1].plot(h_vals, df_mh["Origin_Persist_MAPE"], marker="x", lw=1.4, ls="--", color="#e76f51", label="Origin Persistence (%)")
    axes[1].set_xticks(h_vals); axes[1].set_xlabel("Direct Forecast Horizon h (Days Ahead)"); axes[1].set_ylabel("Blind Test MAPE (%)")
    axes[1].set_title("(b) Causal Multi-Horizon MAPE (Zero Future Telemetry)", pad=46)
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.4)
    fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig12_multihorizon.pdf", bbox_inches="tight"); plt.close(fig)

    print(f"\nCompleted v4 Divisional + Multi-Horizon Suite in {time.perf_counter() - t0:.1f}s!", flush=True)
