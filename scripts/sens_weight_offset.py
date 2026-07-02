#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sensitivity of the model to the additive offset c in the sample weight
   w = 1 / (BaP + c).  Production uses c = 0.5.  Also includes the no-weighting
   case for reference.  Reports LOSO median r2, LOSO pooled r2, rural-background
   mean |MBE|, and pooled random k-fold r2 -- the metrics used in the paper's
   weighting ablation (Sec. 2.6 / Table 6)."""
import os, sys, numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneGroupOut

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config
import validate_model as vm  # reuse loader, feature resolver, metrics

RB_STATIONS = {"SK0004R", "SK0006R"}      # the two rural-background sites
OFFSETS = [None, 0.1, 0.25, 0.5, 1.0, 2.0]  # None = uniform weights

def fit_predict(train_df, test_df, feats, offset):
    X_tr, y_tr = train_df[feats], np.log1p(train_df["bap"])
    w = None if offset is None else 1.0 / (train_df["bap"] + offset)
    params = dict(config.MODEL_PARAMS)
    params["monotone_constraints"] = config.monotone_for(feats)
    m = xgb.XGBRegressor(**params)
    m.fit(X_tr, y_tr, sample_weight=w)
    return np.maximum(0.0, np.expm1(m.predict(test_df[feats])))

def main():
    df = vm.load_dataset()
    feats = vm.resolve_features(df)
    print(f"dataset: {len(df)} rows, {df['eoi'].nunique()} stations, {len(feats)} features\n")

    groups = df["eoi"].values
    logo = LeaveOneGroupOut()
    kf = KFold(n_splits=vm.N_SPLITS, shuffle=True, random_state=vm.SEED)

    rows = []
    for off in OFFSETS:
        # --- LOSO ---
        oof = np.full(len(df), np.nan)
        per = []
        for tr_idx, te_idx in logo.split(df, groups=groups):
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            pred = fit_predict(tr, te, feats, off)
            oof[te_idx] = pred
            mm = vm.compute_metrics(te["bap"], pred)
            mm["eoi"] = te["eoi"].iloc[0]
            per.append(mm)
        per = pd.DataFrame(per)
        loso_med = per["R2"].median()
        loso_pool = vm.pearson_r2(df["bap"], oof)
        rb = per[per["eoi"].isin(RB_STATIONS)]
        rb_absmbe = rb["MBE"].abs().mean()

        # --- random k-fold (pooled) ---
        oofk = np.full(len(df), np.nan)
        for tr_idx, te_idx in kf.split(df):
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            oofk[te_idx] = fit_predict(tr, te, feats, off)
        kfold_pool = vm.pearson_r2(df["bap"], oofk)

        label = "none" if off is None else f"{off:.2f}"
        rows.append((label, loso_med, loso_pool, rb_absmbe, kfold_pool))
        print(f"  c={label:>5s}  LOSO med r2={loso_med:.3f}  LOSO pooled r2={loso_pool:.3f}"
              f"  RB |MBE|={rb_absmbe:.3f}  k-fold pooled r2={kfold_pool:.3f}")

    print("\n=== summary (production c = 0.50) ===")
    print(f"{'c':>6s} {'LOSO_med_r2':>12s} {'LOSO_pool_r2':>13s} {'RB_|MBE|':>10s} {'kfold_pool_r2':>14s}")
    for label, lm, lp, rb, kp in rows:
        print(f"{label:>6s} {lm:12.3f} {lp:13.3f} {rb:10.3f} {kp:14.3f}")

if __name__ == "__main__":
    main()
