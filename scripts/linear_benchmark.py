#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear-model benchmark: is the gradient boosting actually buying skill over a
regularized *linear* model on the SAME features and the SAME CV folds?

This answers the standard reviewer question ("why not multiple linear regression?")
and quantitatively substantiates the manuscript's claim that the BaP-driver
relationships are strongly non-linear.

The linear model mirrors the XGBoost setup as closely as a linear model can:
  - same feature set (resolve_features), same target ln(1+BaP) back-transformed,
  - same inverse-concentration sample weighting 1/(BaP+0.5),
  - median imputation + standardization (XGBoost handles NaN natively; Ridge cannot),
  - light L2 regularization (Ridge, alpha=1.0).

Evaluated on the identical schemes as the paper:
  - random k-fold (pooled r², RMSE)         -> compare to §3.1 / Table 2
  - LOSO (median per-station r²)            -> compare to Table 2
  - block-gap 30-day (pooled + median r²/RMSE) -> add a row to Table 3

Metrics use the same r² = squared Pearson and the same compute_metrics() as the
XGBoost validation, so numbers are directly comparable.

Output: runs/<RUN>/output/validation/linear_benchmark.csv  (+ console summary)
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, LeaveOneGroupOut

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
import config                 # noqa: E402
import validate_model as vm   # noqa: E402

BLOCK_DAYS = 30
ALPHA = 1.0


def make_model():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA)),
    ])


def fit_predict(train_df, test_df, feats):
    """Ridge on ln(1+BaP) with 1/(BaP+0.5) weights; back-transform, clip at 0."""
    Xtr, ytr = train_df[feats], np.log1p(train_df["bap"])
    w = 1.0 / (train_df["bap"] + 0.5)
    m = make_model()
    m.fit(Xtr, ytr, ridge__sample_weight=w)
    return np.maximum(0.0, np.expm1(m.predict(test_df[feats])))


def run_kfold(df, feats):
    kf = KFold(n_splits=vm.N_SPLITS, shuffle=True, random_state=vm.SEED)
    oof = np.full(len(df), np.nan)
    for tr, te in kf.split(df):
        oof[te] = fit_predict(df.iloc[tr], df.iloc[te], feats)
    return vm.compute_metrics(df["bap"], oof)


def run_loso(df, feats):
    logo = LeaveOneGroupOut()
    groups = df["eoi"].values
    per = []
    for tr, te in logo.split(df, groups=groups):
        pred = fit_predict(df.iloc[tr], df.iloc[te], feats)
        per.append(vm.compute_metrics(df.iloc[te]["bap"], pred))
    per = pd.DataFrame(per)
    return per["R2"].median(), per["RMSE"].median()


def run_block(df, feats):
    d = df.copy()
    d["date"] = pd.to_datetime(d["datum"])
    origin = d["date"].min()
    d["block"] = ((d["date"] - origin).dt.days // BLOCK_DAYS).astype(int)
    d = d.reset_index(drop=True)
    oof = np.full(len(d), np.nan)
    for b in sorted(d["block"].unique()):
        te = d.index[d["block"] == b]
        tr = d.index[d["block"] != b]
        if len(te) < 3:
            continue
        oof[te] = fit_predict(d.loc[tr], d.loc[te], feats)
    valid = np.isfinite(oof)
    pooled = vm.compute_metrics(d.loc[valid, "bap"], oof[valid])
    st_r2, st_rmse = [], []
    for s, g in d[valid].groupby("eoi"):
        m = vm.compute_metrics(g["bap"], oof[g.index])
        st_r2.append(m["R2"]); st_rmse.append(m["RMSE"])
    return pooled, float(np.nanmedian(st_r2)), float(np.nanmedian(st_rmse))


def main():
    df = vm.load_dataset().reset_index(drop=True)
    feats = vm.resolve_features(df)
    print(f"Linear (Ridge alpha={ALPHA}) benchmark | {len(df)} rows, {len(feats)} features\n")

    kf = run_kfold(df, feats)
    loso_r2, loso_rmse = run_loso(df, feats)
    blk_pooled, blk_med_r2, blk_med_rmse = run_block(df, feats)

    rows = [
        {"scheme": "kfold_pooled",       "r2": kf["R2"],        "rmse": kf["RMSE"]},
        {"scheme": "loso_median",        "r2": loso_r2,         "rmse": loso_rmse},
        {"scheme": "block_pooled",       "r2": blk_pooled["R2"], "rmse": blk_pooled["RMSE"]},
        {"scheme": "block_median_station","r2": blk_med_r2,      "rmse": blk_med_rmse},
    ]
    out = pd.DataFrame(rows)
    out_path = os.path.join(config.VALIDATION_DIR, "linear_benchmark.csv")
    os.makedirs(config.VALIDATION_DIR, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\n✅ linear_benchmark.csv -> {out_path}")
    print("\nCompare to XGBoost: kfold pooled r²=0.69, LOSO median r²=0.80, "
          "block pooled r²=0.69 / median r²=0.76.")


if __name__ == "__main__":
    main()
