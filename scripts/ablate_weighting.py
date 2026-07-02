#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ablate the sample weighting 1/(BaP+0.5) used in training.

Compares weighted vs unweighted models under random k-fold (pooled) and LOSO
(median per station + bias at the clean rural-background sites), to quantify
what the weighting buys. Reuses validate_model's dataset/feature logic.
"""
import os, sys
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
import config  # noqa: E402
import validate_model as vm  # noqa: E402

df = vm.load_dataset().reset_index(drop=True)
vm.FEATURES = vm.resolve_features(df)
FEATURES = vm.FEATURES
print(f"{len(df)} samples, {df['eoi'].nunique()} stations, {len(FEATURES)} features")
RB = ["SK0006R", "SK0004R"]


def fit_predict(tr, te, weighted):
    X, y = tr[FEATURES], np.log1p(tr["bap"])
    w = 1.0 / (tr["bap"] + 0.5) if weighted else None
    params = dict(config.MODEL_PARAMS)
    params["monotone_constraints"] = config.monotone_for(FEATURES)
    m = xgb.XGBRegressor(**params)
    m.fit(X, y, sample_weight=w)
    return np.maximum(0.0, np.expm1(m.predict(te[FEATURES])))


def evaluate(weighted):
    # k-fold (pooled)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.full(len(df), np.nan)
    for tr_i, te_i in kf.split(df):
        oof[te_i] = fit_predict(df.iloc[tr_i], df.iloc[te_i], weighted)
    kf_r2 = vm.pearson_r2(df["bap"], oof)
    kf_rmse = np.sqrt(mean_squared_error(df["bap"], oof))
    kf_mae = mean_absolute_error(df["bap"], oof)

    # LOSO (per-station)
    logo = LeaveOneGroupOut()
    oofl = np.full(len(df), np.nan)
    for tr_i, te_i in logo.split(df, groups=df["eoi"].values):
        oofl[te_i] = fit_predict(df.iloc[tr_i], df.iloc[te_i], weighted)
    st_r2, rb_absmbe = [], []
    for s, g in df.groupby("eoi"):
        idx = g.index
        st_r2.append(vm.pearson_r2(g["bap"], oofl[idx]))
        if s in RB:
            rb_absmbe.append(abs(np.mean(oofl[idx] - g["bap"].values)))
    return {
        "kfold_r2": kf_r2, "kfold_rmse": kf_rmse, "kfold_mae": kf_mae,
        "loso_median_r2": np.nanmedian(st_r2),
        "rb_mean_absMBE": np.mean(rb_absmbe),
    }


rows = []
for w in (True, False):
    r = evaluate(w)
    r["weighting"] = "1/(BaP+0.5)" if w else "none (uniform)"
    rows.append(r)
    print(f"\n[{r['weighting']}]")
    for k, v in r.items():
        if k != "weighting":
            print(f"  {k:16s} {v:.3f}")

out = pd.DataFrame(rows).set_index("weighting")
out.to_csv(os.path.join(config.VALIDATION_DIR, "weighting_ablation.csv"))
print("\n=== SUMMARY ===")
print(out.round(3).to_string())
print(f"\n-> {os.path.join(config.VALIDATION_DIR, 'weighting_ablation.csv')}")
