#!/usr/bin/env python3
"""Linear counterparts of the ladder models E and E+P (spec addendum G) and the flexibility contrasts.

Ridge regression with exactly the tree models' inputs, same target, folds and 500 m data; scored against the
tree models (and PM ridge against PM+I) with the unchanged scoring_core bootstrap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "APR/analysis/clean_rebuild"))
sys.path.insert(0, str(ROOT / "APR/analysis/consolidated_2km"))
from scoring_core import paired, scores  # noqa: E402
from contract_c2km import FEATURES, ORIGIN  # noqa: E402

IND, KEYS = "SK0018A", ["eoi", "datum"]
POLL = ["pm10_mean", "pm25_mean", "no2_mean", "pm10_mean_lag1", "pm25_mean_lag1", "no2_mean_lag1",
        "pm10_mean_3d", "pm25_mean_3d", "no2_mean_3d"]


def design(train, test, feats):
    """Training-only imputation, missing flags, PM2.5 x season interactions, standardisation."""
    def raw(d):
        x = d[feats].astype(float).copy()
        for c in feats:
            if c in POLL:
                x[c] = np.log1p(np.maximum(0, x[c]))
        if any(c.startswith(("pm10", "pm25")) for c in feats):
            x["miss_pm"] = d[["pm10_mean", "pm25_mean"]].isna().all(axis=1).astype(float)
        if any(c.startswith("no2") for c in feats):
            x["miss_no2"] = d["no2_mean"].isna().astype(float)
        return x
    xtr, xte = raw(train), raw(test)
    med = xtr.median()
    xtr, xte = xtr.fillna(med).fillna(0), xte.fillna(med).fillna(0)
    if "pm25_mean" in feats:
        for x in (xtr, xte):
            x["pm25_x_sin"] = x["pm25_mean"] * x["doy_sin"]
            x["pm25_x_cos"] = x["pm25_mean"] * x["doy_cos"]
    sc = StandardScaler().fit(xtr)
    return sc.transform(xtr), sc.transform(xte)


def fit_predict(train, test, feats):
    a, b = design(train, test, feats)
    m = Ridge(alpha=1.0).fit(a, np.log1p(train.bap.to_numpy(float)))
    return np.maximum(0, np.expm1(m.predict(b)))


def annual(x):
    a = x.assign(year=x.datum.dt.year).groupby(["eoi", "year"], as_index=False).agg(
        observed=("observed", "mean"), prediction=("prediction", "mean"), n=("observed", "size"))
    return a[a.n >= 20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    d = pd.read_csv(ROOT / "APR/results/clean_rebuild/r02/inputs/train_ready_permissive_500m.csv", parse_dates=["datum"])
    d = d.sort_values(KEYS).reset_index(drop=True)
    d["block"] = ((d.datum - ORIGIN).dt.days // 30).astype(int)
    preds = []
    for arm in ("E", "E_P"):
        feats = FEATURES[arm]
        for prot, col in (("block30", "block"), ("loso", "eoi")):
            for fold in sorted(d[col].unique()):
                tr, te = d[d[col] != fold], d[d[col] == fold]
                preds.append(pd.DataFrame({"eoi": te.eoi, "datum": te.datum, "observed": te.bap,
                                           "prediction": fit_predict(tr, te, feats), "arm": f"LIN_{arm}",
                                           "protocol": prot, "fold": str(fold)}))
    lin = pd.concat(preds, ignore_index=True)
    trees = pd.read_csv(ROOT / "APR/results/consolidated_500m/assembled/validated_predictions.csv", parse_dates=["datum"])
    trees = trees[(trees.input_variant == "permissive") & (trees.target == "log1p") & (trees.weighting == "uniform")
                  & (trees.augment == "none")]
    base = pd.read_csv(ROOT / "APR/results/clean_rebuild/r02/scores/baseline_predictions.csv", parse_dates=["datum"])
    pmr = base[(base.method == "PM_RIDGE") & (base.protocol == "block30")][KEYS + ["observed", "prediction"]]
    pairs = [("E (trees vs linear)", "block30", trees[(trees.arm == "E") & (trees.protocol == "block30")], lin[(lin.arm == "LIN_E") & (lin.protocol == "block30")]),
             ("E (trees vs linear)", "loso", trees[(trees.arm == "E") & (trees.protocol == "loso")], lin[(lin.arm == "LIN_E") & (lin.protocol == "loso")]),
             ("E+P (trees vs linear)", "block30", trees[(trees.arm == "E_P") & (trees.protocol == "block30")], lin[(lin.arm == "LIN_E_P") & (lin.protocol == "block30")]),
             ("E+P (trees vs linear)", "loso", trees[(trees.arm == "E_P") & (trees.protocol == "loso")], lin[(lin.arm == "LIN_E_P") & (lin.protocol == "loso")]),
             ("PM+I vs PM ridge", "block30", trees[(trees.arm == "PM_I") & (trees.protocol == "block30")], pmr)]
    summ, cons = [], []
    for lab, prot, tx, lx in pairs:
        sel = lambda z: z[(z.datum.dt.year >= 2024) & (z.eoi != IND)][KEYS + ["observed", "prediction"]]
        x, y = sel(tx), sel(lx).merge(sel(tx)[KEYS], on=KEYS, validate="one_to_one")
        assert len(x) == len(y) == 4852, (lab, prot, len(x), len(y))
        for name, f in (("trees", x), ("linear", y)):
            summ.append(dict(comparison=lab, protocol=prot, model=name, estimand="daily", **scores(f.observed, f.prediction)))
            an = annual(f)
            summ.append(dict(comparison=lab, protocol=prot, model=name, estimand="station_year_mean",
                             **scores(an.observed, an.prediction)))
        q = paired(x, y, KEYS, "station_x_block" if prot == "block30" else "station", "RMSE")
        cons.append(dict(comparison=lab, protocol=prot, estimand="daily", **q))
        ax, ay = annual(x), annual(y)
        q = paired(ax, ay.merge(ax[["eoi", "year"]], on=["eoi", "year"]), ["eoi", "year"], "station", "RMSE")
        cons.append(dict(comparison=lab, protocol=prot, estimand="station_year_mean", **q))
    a.output.mkdir(parents=True)
    lin.to_csv(a.output / "linear_predictions.csv", index=False)
    pd.DataFrame(summ).to_csv(a.output / "linear_summary.csv", index=False)
    pd.DataFrame(cons).to_csv(a.output / "linear_contrasts.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(summ)[["comparison", "protocol", "model", "estimand", "RMSE", "agreement_R2"]].round(3).to_string())
    print(pd.DataFrame(cons)[["comparison", "protocol", "estimand", "estimate", "ci_low", "ci_high"]].round(3).to_string())


if __name__ == "__main__":
    main()
