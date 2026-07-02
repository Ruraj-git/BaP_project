#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Block-gap cross-validation + baselines (the realistic gap-filling scenario).

Real BaP gaps are contiguous blocks of days, not random days. Here we hold out
whole contiguous time windows (default 30 days) — leave-one-time-block-out — so
that adjacent (autocorrelated) days cannot leak between train and test. For each
held-out block the station's data from OTHER periods stays in training, which is
exactly the operational situation when filling a gap at a monitored station.

On the SAME held-out blocks we score three methods:
  - XGBoost   (the final model, the full resolved feature set via resolve_features)
  - Persistence       (last observed value before the gap, carried forward)
  - Seasonal climatology (per-station calendar-month mean from the training part)

Metrics (original units): r² = squared Pearson, RMSE, MAE, MBE. Pooled + median/station.
Outputs: output/validation/block_*.csv and plots/block_baseline_comparison.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
import validate_model as vm  # reuse load_dataset / resolve_features / fit_predict / compute_metrics

OUT_DIR = vm.OUT_DIR
PLOT_DIR = vm.PLOT_DIR
BLOCK_DAYS = 30
sns.set_theme(style="whitegrid", context="notebook")


def predict_block(df, test_idx, train_idx):
    """Return XGBoost, persistence and climatology predictions for the test rows."""
    train, test = df.loc[train_idx], df.loc[test_idx]

    # --- XGBoost ---
    xgb_pred = vm.fit_predict(train, test)

    # --- Persistence & climatology (per station, from the training part only) ---
    block_start = test["date"].min()
    pers = np.full(len(test), np.nan)
    clim = np.full(len(test), np.nan)
    global_mean = train["bap"].mean()

    for s, gi in test.groupby("eoi").groups.items():
        ts = train[train["eoi"] == s]
        # Persistence: last observed value strictly before the gap window
        before = ts[ts["date"] < block_start]
        pval = before.sort_values("date")["bap"].iloc[-1] if len(before) else (
            ts["bap"].mean() if len(ts) else global_mean)
        # Climatology: per-station mean per calendar month
        month_mean = ts.groupby(ts["date"].dt.month)["bap"].mean()
        st_mean = ts["bap"].mean() if len(ts) else global_mean
        for ridx in gi:
            pos = test.index.get_loc(ridx)
            pers[pos] = pval
            m = test.loc[ridx, "date"].month
            clim[pos] = month_mean.get(m, st_mean)
    return xgb_pred, pers, clim


def main():
    df = vm.load_dataset().reset_index(drop=True)
    vm.FEATURES = vm.resolve_features(df)
    df["date"] = pd.to_datetime(df["datum"])
    origin = df["date"].min()
    df["block"] = ((df["date"] - origin).dt.days // BLOCK_DAYS).astype(int)
    blocks = sorted(df["block"].unique())
    print(f"Block-gap CV: {len(df)} samples, {df['eoi'].nunique()} stations, "
          f"{len(blocks)} blocks of {BLOCK_DAYS} days, {len(vm.FEATURES)} features.")

    oof = {k: np.full(len(df), np.nan) for k in ("xgb", "persistence", "climatology")}
    for b in blocks:
        test_idx = df.index[df["block"] == b]
        train_idx = df.index[df["block"] != b]
        if len(test_idx) < 3:
            continue
        xp, pp, cp = predict_block(df, test_idx, train_idx)
        oof["xgb"][test_idx] = xp
        oof["persistence"][test_idx] = pp
        oof["climatology"][test_idx] = cp
        print(f"  block {b:2d}: n={len(test_idx):4d}  XGB r²="
              f"{vm.compute_metrics(df.loc[test_idx, 'bap'], xp)['R2']:.3f}")

    # --- Pooled + median-per-station metrics for each method ---
    rows, per_station = [], []
    valid = ~np.isnan(oof["xgb"])
    for method, pred in oof.items():
        m = vm.compute_metrics(df.loc[valid, "bap"], pred[valid])
        m["method"] = method
        m["scope"] = "pooled"
        rows.append(m)
        # median per station
        st_r2, st_rmse, st_mbe, st_mae = [], [], [], []
        for s, g in df[valid].groupby("eoi"):
            sm = vm.compute_metrics(g["bap"], pred[g.index])
            st_r2.append(sm["R2"]); st_rmse.append(sm["RMSE"])
            st_mbe.append(sm["MBE"]); st_mae.append(sm["MAE"])
            per_station.append({"method": method, "eoi": s, **sm})
        rows.append({"method": method, "scope": "median_per_station",
                     "n": int(valid.sum()), "R2": np.nanmedian(st_r2),
                     "RMSE": np.nanmedian(st_rmse), "MAE": np.nanmedian(st_mae),
                     "MBE": np.nanmedian(st_mbe)})

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "block_metrics_summary.csv"), index=False)
    pd.DataFrame(per_station).to_csv(os.path.join(OUT_DIR, "block_per_station.csv"), index=False)
    pd.DataFrame({"eoi": df["eoi"], "datum": df["datum"], "bap": df["bap"],
                  **{f"pred_{k}": v for k, v in oof.items()}}).to_csv(
        os.path.join(OUT_DIR, "block_oof_predictions.csv"), index=False)

    # --- Comparison plot (pooled + median r², RMSE) ---
    pooled = summary[summary["scope"] == "pooled"].set_index("method")
    med = summary[summary["scope"] == "median_per_station"].set_index("method")
    order = ["xgb", "climatology", "persistence"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = {"xgb": "XGBoost", "climatology": "Climatology", "persistence": "Persistence"}
    cols = sns.color_palette("tab10", 3)
    for ax, (key, title, src) in zip(axes, [
            ("R2", "Median per-station r²", med),
            ("R2", "Pooled r²", pooled),
            ("RMSE", "Pooled RMSE [ng/m³]", pooled)]):
        vals = [src.loc[m, key] for m in order]
        ax.bar([labels[m] for m in order], vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title)
        if key == "R2":
            ax.set_ylim(min(0, min(vals)) - 0.05, 1.0)
            ax.axhline(0, color="k", lw=0.6)
    fig.suptitle(f"Block-gap CV ({BLOCK_DAYS}-day gaps): model vs. baselines", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOT_DIR, "block_baseline_comparison.png"), dpi=140)
    plt.close(fig)

    print("\n--- BLOCK-GAP CV SUMMARY ---")
    print(summary.set_index(["scope", "method"])[["n", "R2", "RMSE", "MAE", "MBE"]]
          .round(3).to_string())
    print(f"\n✅ Done -> {OUT_DIR} (+ plots/block_baseline_comparison.png)")


if __name__ == "__main__":
    main()
