"""
3-Fold Expanding-Window Rolling-Origin CV (2023, 2024, 2025-26) +
Unconfounded Fixed-Test-Set Learning Curve (1.5, 2.5, 3.5, 4.5 yr training history on fixed 2025-26 test set).
Addresses Reviewer Points 1 & 11:
  - Reports 3-fold mean honestly across ALL baselines (Persistence, Lag-7, SARIMAX, Ridge, XGBoost-Diff, PatchTST, PatchTST-Ens3, BiLSTM, Val-Opt RNN Ensemble)
  - Separates training-volume effects from 2023 test-year regime effects via a fixed-test learning curve
  - Uses subtropical train-only CDD_26 (no degenerate HDD) and Val-ES / Val-Cal separation
Saves: part1_rolling_cv.csv, part1_learning_curve.csv, figures/fig13_rolling_cv.pdf
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
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
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


def windowing(X_input, y_input, history_size):
    data, labels = [], []
    for i in range(history_size, len(y_input)):
        data.append(X_input[i - history_size : i, :])
        labels.append(y_input[i])
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32).reshape(-1, 1)


class PatchTSTModel(nn.Module):
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


def train_torch_model(model, X_tr, y_tr, X_va_es, y_va_es, seed):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=OPT_LR, weight_decay=OPT_WD)
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


def optimize_val_weights(y_val, preds_list):
    P = np.column_stack([p.ravel() for p in preds_list])
    yv = y_val.ravel()
    k = P.shape[1]
    res = minimize(lambda w: float(np.mean(np.abs(yv - P @ w))), x0=np.ones(k) / k, method="SLSQP",
                   bounds=[(0.0, 1.0)] * k, constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    w = np.clip(res.x, 0.0, 1.0)
    return w / np.sum(w)


def fit_ridge_val(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_te_2d, sy):
    best_mae, best_mod = float("inf"), None
    for a in [1.0, 10.0, 50.0, 100.0, 300.0, 1000.0]:
        m = Ridge(alpha=a).fit(X_tr_2d, y_tr.ravel())
        mae = mean_absolute_error(y_va_es.ravel(), m.predict(X_va_es_2d))
        if mae < best_mae:
            best_mae, best_mod = mae, m
    return sy.inverse_transform(best_mod.predict(X_te_2d).reshape(-1, 1))


def load_full_features():
    df = pd.read_excel("version_5_extended_backup.xlsx")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    if "Dhaka_temp" in df.columns: df["Maximum Temperature in Dhaka was"] = df["Dhaka_temp"]
    if "Mymensingh_temp" not in df.columns and "Dhaka_temp" in df.columns: df["Mymensingh_temp"] = df["Dhaka_temp"]

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


def eval_split_models(tr_df, va_df, te_df, meta, cont_cols, cat_oh, target_col, temp_col, run_all_baselines=True):
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
    va_cal_start = min(n_va - 20, va_es_end + HIST_SIZE)
    X_va_es, y_va_es = X_va[:va_es_end], y_va[:va_es_end]
    X_va_cal = X_va[va_cal_start:]

    y_va_inv = va_df[[target_col]].iloc[HIST_SIZE:].values
    y_va_es_inv = y_va_inv[:va_es_end]
    y_va_cal_inv = y_va_inv[va_cal_start:]
    y_te_inv = te_df[[target_col]].iloc[HIST_SIZE:].values
    y_pers = te_df[target_col].values[HIST_SIZE - 1 : -1].reshape(-1, 1)
    y_lag7 = te_df[target_col].values[HIST_SIZE - 7 : len(te_df) - 7].reshape(-1, 1)

    X_tr_2d = X_tr.reshape(len(X_tr), -1)
    X_va_es_2d = X_va_es.reshape(len(X_va_es), -1)
    X_te_2d = X_te.reshape(len(X_te), -1)
    nf = X_tr.shape[-1]

    # Causal SARIMAX fitted strictly on tr_df
    ex_full = pd.DataFrame(index=meta.index)
    tl1 = meta[temp_col].shift(1).bfill()
    ex_full["CDD_l1"] = np.maximum(0.0, tl1 - TAU_COOL)
    ex_full["Wknd"] = meta["Day of the week"].isin(["Friday", "Saturday"]).astype(float)
    ex_full["Hol"] = (meta["Holiday_cat"] > 0).astype(float)

    sub_m = meta.loc[tr_df.index[0] : te_df.index[-1]]
    ex_m = ex_full.loc[tr_df.index[0] : te_df.index[-1]].values
    y_m = sub_m[target_col].values.astype(float)
    tr_len = len(tr_df)
    sfit = SARIMAX(y_m[:tr_len], exog=ex_m[:tr_len], order=(1, 1, 1), seasonal_order=(1, 0, 1, 7),
                   enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=80)
    sfull = sfit.apply(y_m, exog=ex_m).fittedvalues
    p_sar = sfull[-len(y_te_inv):].reshape(-1, 1)

    # Ridge on 14 x F
    p_ridge = fit_ridge_val(X_tr_2d, y_tr, X_va_es_2d, y_va_es, X_te_2d, sy)

    out = {
        "N_Train": len(X_tr), "N_Val": len(X_va), "N_Test": len(y_te_inv),
        "Persist_MAE": round(float(mean_absolute_error(y_te_inv, y_pers)), 2),
        "Lag7_MAE": round(float(mean_absolute_error(y_te_inv, y_lag7)), 2),
        "SARIMAX_MAE": round(float(mean_absolute_error(y_te_inv, p_sar)), 2),
        "Ridge_MAE": round(float(mean_absolute_error(y_te_inv, p_ridge)), 2),
    }

    if run_all_baselines:
        dy_tr = (tr_df[target_col].values[HIST_SIZE:] - tr_df[target_col].values[HIST_SIZE - 1 : -1])
        dy_va_es = (va_df[target_col].values[HIST_SIZE : HIST_SIZE + len(X_va_es)] - va_df[target_col].values[HIST_SIZE - 1 : HIST_SIZE - 1 + len(X_va_es)])
        xmd = xgb.XGBRegressor(random_state=42, early_stopping_rounds=8, eval_metric="mae", n_jobs=1)
        xmd.fit(X_tr_2d, dy_tr, eval_set=[(X_va_es_2d, dy_va_es)], verbose=False)
        p_xgbd = y_pers + xmd.predict(X_te_2d).reshape(-1, 1)
        out["XGBDiff_MAE"] = round(float(mean_absolute_error(y_te_inv, p_xgbd)), 2)

        lm = lgb.LGBMRegressor(objective="regression_l1", random_state=42, n_estimators=500, learning_rate=0.05, num_leaves=31, n_jobs=1, verbose=-1)
        lm.fit(X_tr_2d, y_tr.ravel(), eval_set=[(X_va_es_2d, y_va_es.ravel())], callbacks=[lgb.early_stopping(8, verbose=False)])
        p_lgb = sy.inverse_transform(lm.predict(X_te_2d).reshape(-1, 1))
        out["LightGBM_MAE"] = round(float(mean_absolute_error(y_te_inv, p_lgb)), 2)

    ens_maes, ens_mapes, bilstm_maes, ptst_maes, pt3_maes = [], [], [], [], []
    for s in SEEDS:
        pt_preds = []
        for off in [0, 1000, 2000]:
            m_pt = train_torch_model(PatchTSTModel(nf), X_tr, y_tr, X_va_es, y_va_es, s + off)
            pt_preds.append(sy.inverse_transform(predict_torch(m_pt, X_te)))
        ptst_maes.append(mean_absolute_error(y_te_inv, pt_preds[0]))
        pt3_maes.append(mean_absolute_error(y_te_inv, np.mean(pt_preds, axis=0)))

        v_l, t_l = [], []
        for fn in [lambda: BiGRUModel(nf), lambda: BiLSTMAttnModel(nf), lambda: BiLSTMModel(nf)]:
            m = train_torch_model(fn(), X_tr, y_tr, X_va_es, y_va_es, s)
            v_l.append(sy.inverse_transform(predict_torch(m, X_va_cal)))
            t_l.append(sy.inverse_transform(predict_torch(m, X_te)))
        bilstm_maes.append(mean_absolute_error(y_te_inv, t_l[2]))
        w = optimize_val_weights(y_va_cal_inv, v_l)
        pe = w[0] * t_l[0] + w[1] * t_l[1] + w[2] * t_l[2]
        ens_maes.append(mean_absolute_error(y_te_inv, pe))
        ens_mapes.append(float(np.mean(np.abs((y_te_inv - pe) / y_te_inv)) * 100.0))

    out.update({
        "PatchTST_MAE": round(float(np.mean(ptst_maes)), 2),
        "PatchTST3_MAE": round(float(np.mean(pt3_maes)), 2),
        "PatchTST3_std": round(float(np.std(pt3_maes, ddof=1)), 2),
        "BiLSTM_MAE": round(float(np.mean(bilstm_maes)), 2),
        "Ens_MAE_mean": round(float(np.mean(ens_maes)), 2),
        "Ens_MAE_std": round(float(np.std(ens_maes, ddof=1)), 2),
        "Ens_MAPE": round(float(np.mean(ens_mapes)), 2),
        "Ens_Acc": round(100.0 - float(np.mean(ens_mapes)), 2),
    })
    return out


if __name__ == "__main__":
    t0 = time.perf_counter()
    df_enc, meta, cont_cols, cat_oh, target_col, temp_col = load_full_features()

    # =========================================================================
    # PART A: 3-FOLD EXPANDING-WINDOW ROLLING-ORIGIN CV (2023, 2024, 2025-26)
    # =========================================================================
    print("=== PART A: 3-FOLD EXPANDING-WINDOW ROLLING-ORIGIN CV ===", flush=True)
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
                "Lag7_MAE": round(float(s5.loc["Seasonal Naive (Lag-7)", "MAE_mean"]), 2),
                "SARIMAX_MAE": round(float(s5.loc["SARIMAX", "MAE_mean"]), 2),
                "Ridge_MAE": round(float(s5.loc["Ridge (14xF)", "MAE_mean"]), 2),
                "XGBDiff_MAE": round(float(s5.loc["XGBoost-Diff (Δy)", "MAE_mean"]), 2),
                "LightGBM_MAE": round(float(s5.loc["LightGBM", "MAE_mean"]), 2),
                "PatchTST_MAE": round(float(s5.loc["PatchTST", "MAE_mean"]), 2),
                "PatchTST3_MAE": round(float(s5.loc["PatchTST-Ensemble (3x)", "MAE_mean"]), 2),
                "PatchTST3_std": round(float(s5.loc["PatchTST-Ensemble (3x)", "MAE_std"]), 2),
                "BiLSTM_MAE": round(float(s5.loc["BiLSTM", "MAE_mean"]), 2),
                "Ens_MAE_mean": round(float(s5.loc["Val-Opt RNN Ensemble", "MAE_mean"]), 2),
                "Ens_MAE_std": round(float(s5.loc["Val-Opt RNN Ensemble", "MAE_std"]), 2),
                "Ens_MAPE": round(float(s5.loc["Val-Opt RNN Ensemble", "MAPE_mean"]), 2),
                "Ens_Acc": round(float(s5.loc["Val-Opt RNN Ensemble", "Acc_mean"]), 2),
            }
            rows.append(row)
            print(f"  {fname:16s} (synced 5-seed): SARIMAX={row['SARIMAX_MAE']:.1f} | Ridge={row['Ridge_MAE']:.1f} | PT3={row['PatchTST3_MAE']:.1f} | Ens={row['Ens_MAE_mean']:.1f}±{row['Ens_MAE_std']:.1f}", flush=True)
            continue

        tr_df = df_enc.loc[: pd.Timestamp(val_start) - pd.Timedelta(days=1)]
        va_df = df_enc.loc[val_start : pd.Timestamp(test_start) - pd.Timedelta(days=1)]
        te_df = df_enc.loc[test_start : test_end]
        row = {"Fold": fname, **eval_split_models(tr_df, va_df, te_df, meta, cont_cols, cat_oh, target_col, temp_col, True)}
        rows.append(row)
        print(f"  {fname:16s} (N_te={row['N_Test']}): SARIMAX={row['SARIMAX_MAE']:.1f} | Ridge={row['Ridge_MAE']:.1f} | PT3={row['PatchTST3_MAE']:.1f} | Ens={row['Ens_MAE_mean']:.1f}±{row['Ens_MAE_std']:.1f} | Persist={row['Persist_MAE']:.1f}", flush=True)

    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(f"{OUT_DIR}/part1_rolling_cv.csv", index=False)
    print("\n=== 3-FOLD ROLLING CV SUMMARY (EXACT 3-FOLD MEANS) ===")
    print(df_cv.to_string(index=False))
    for col in ["Persist_MAE", "Lag7_MAE", "SARIMAX_MAE", "Ridge_MAE", "XGBDiff_MAE", "LightGBM_MAE", "PatchTST_MAE", "PatchTST3_MAE", "Ens_MAE_mean"]:
        print(f"  3-Fold Mean {col:15s}: {df_cv[col].mean():.2f} MW")

    # =========================================================================
    # PART B: UNCONFOUNDED FIXED-TEST-SET LEARNING CURVE (SAME 2025-26 TEST SET!)
    # =========================================================================
    print("\n=== PART B: UNCONFOUNDED FIXED-TEST LEARNING CURVE (TEST FIXED = 2025-26, n=462) ===", flush=True)
    tc_full = int(0.70 * len(df_enc))
    vc_full = int(0.80 * len(df_enc))
    va_fixed = df_enc.iloc[tc_full:vc_full]
    te_fixed = df_enc.iloc[vc_full:]
    tr_full = df_enc.iloc[:tc_full]

    lc_rows = []
    for yrs, n_days in [(1.5, 548), (2.5, 913), (3.5, 1278), (4.5, len(tr_full))]:
        if n_days == len(tr_full) and os.path.exists(f"{OUT_DIR}/part1_summary_metrics.csv"):
            s5 = pd.read_csv(f"{OUT_DIR}/part1_summary_metrics.csv", index_col=0)
            r_lc = {
                "Train_Years": yrs, "N_Train_Windows": len(tr_full) - HIST_SIZE,
                "SARIMAX_MAE": round(float(s5.loc["SARIMAX", "MAE_mean"]), 2),
                "Ridge_MAE": round(float(s5.loc["Ridge (14xF)", "MAE_mean"]), 2),
                "PatchTST3_MAE": round(float(s5.loc["PatchTST-Ensemble (3x)", "MAE_mean"]), 2),
                "PatchTST3_std": round(float(s5.loc["PatchTST-Ensemble (3x)", "MAE_std"]), 2),
                "Ens_MAE_mean": round(float(s5.loc["Val-Opt RNN Ensemble", "MAE_mean"]), 2),
                "Ens_MAE_std": round(float(s5.loc["Val-Opt RNN Ensemble", "MAE_std"]), 2),
            }
        else:
            tr_sub = tr_full.iloc[-n_days:]
            res = eval_split_models(tr_sub, va_fixed, te_fixed, meta, cont_cols, cat_oh, target_col, temp_col, False)
            r_lc = {
                "Train_Years": yrs, "N_Train_Windows": res["N_Train"],
                "SARIMAX_MAE": res["SARIMAX_MAE"], "Ridge_MAE": res["Ridge_MAE"],
                "PatchTST3_MAE": res["PatchTST3_MAE"], "PatchTST3_std": res["PatchTST3_std"],
                "Ens_MAE_mean": res["Ens_MAE_mean"], "Ens_MAE_std": res["Ens_MAE_std"],
            }
        lc_rows.append(r_lc)
        print(f"  Train={yrs:.1f} yr ({r_lc['N_Train_Windows']:4d} win) | SARIMAX={r_lc['SARIMAX_MAE']:.1f} | Ridge={r_lc['Ridge_MAE']:.1f} | PT3={r_lc['PatchTST3_MAE']:.1f}±{r_lc['PatchTST3_std']:.1f} | RNN-Ens={r_lc['Ens_MAE_mean']:.1f}±{r_lc['Ens_MAE_std']:.1f}", flush=True)

    df_lc = pd.DataFrame(lc_rows)
    df_lc.to_csv(f"{OUT_DIR}/part1_learning_curve.csv", index=False)

    # Combined 2-panel Figure 13: (a) 3-Fold Rolling CV, (b) Fixed-Test Learning Curve (ZERO legend overlap!)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))

    ax = axes[0]
    x = np.arange(len(df_cv))
    w = 0.13
    series = [
        ("Persist_MAE", "Lag-1 Persist", "#adb5bd"),
        ("SARIMAX_MAE", "SARIMAX", "#6d597a"),
        ("Ridge_MAE", "Ridge (14×F)", "#e9c46a"),
        ("XGBDiff_MAE", "XGBoost-Diff", "#f4a261"),
        ("PatchTST3_MAE", "PatchTST-3×", "#2a9d8f"),
        ("Ens_MAE_mean", "Val-Opt RNN Ens", "#1d3557"),
    ]
    for idx, (col, lbl, colr) in enumerate(series):
        offset = (idx - 2.5) * w
        yerr = df_cv["Ens_MAE_std"] if col == "Ens_MAE_mean" else (df_cv["PatchTST3_std"] if col == "PatchTST3_MAE" else None)
        ax.bar(x + offset, df_cv[col], yerr=yerr, capsize=2.5, width=w, color=colr, label=lbl)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Fold']}\n(n={r['N_Test']})" for _, r in df_cv.iterrows()], fontsize=8.3)
    ax.set_ylabel("Blind Test MAE (MW)")
    ax.set_ylim(0, df_cv[["Persist_MAE", "SARIMAX_MAE", "Ridge_MAE", "XGBDiff_MAE", "PatchTST3_MAE", "Ens_MAE_mean"]].values.max() * 1.18)
    ax.set_title("(a) 3-Fold Expanding-Window Rolling CV (2023–2026)", pad=38)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=7.6)

    ax = axes[1]
    yrs_x = df_lc["Train_Years"].values
    ax.plot(yrs_x, df_lc["SARIMAX_MAE"], marker="s", lw=1.8, ls="--", color="#6d597a", label="SARIMAX")
    ax.plot(yrs_x, df_lc["Ridge_MAE"], marker="^", lw=1.8, ls=":", color="#d97706", label="Ridge (14×F)")
    ax.errorbar(yrs_x, df_lc["PatchTST3_MAE"], yerr=df_lc["PatchTST3_std"], marker="D", lw=1.9, capsize=3.5, color="#2a9d8f", label="PatchTST-Ens (3×)")
    ax.errorbar(yrs_x, df_lc["Ens_MAE_mean"], yerr=df_lc["Ens_MAE_std"], marker="o", lw=2.1, capsize=3.5, color="#1d3557", label="Val-Opt RNN Ens")
    ax.set_xticks(yrs_x)
    ax.set_xticklabels([f"{y:.1f} yr\n({n}d)" for y, n in zip(df_lc["Train_Years"], df_lc["N_Train_Windows"])], fontsize=8.2)
    ax.set_ylabel("Fixed 2025–26 Test MAE (MW)")
    ax.set_title("(b) Fixed-Test Learning Curve (Unconfounded Data Volume)", pad=38)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=7.6)

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig13_rolling_cv.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved part1_rolling_cv.csv, part1_learning_curve.csv & fig13_rolling_cv.pdf in {time.perf_counter() - t0:.1f}s!", flush=True)
