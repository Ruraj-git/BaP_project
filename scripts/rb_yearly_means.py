#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rural-background (RB) annual-mean B[a]P: production vs proxy-free (virtual) mode.

Motivation
----------
The two trained RB stations (SK0004R Stara Lesna, SK0006R Starina) are the only
rural-background anchors in the 21-station training set. This script quantifies
(i) how accurately the production model (M4: proxy + emission, all 21 stations
in training) reproduces their observed annual mean, and (ii) what the model
predicts -- in production and in the honest proxy-free virtual mode V -- at the
five further RB-type cells in the 54-site network that are absent from training.

For the two trained anchors the V column is leave-one-station-out (the station
is excluded), i.e. the honest "what if this site were unmonitored" estimate.
For the five virtual RB cells (no in-period B[a]P) production and V are plain
predictions of the all-21 model.

Annual mean over full calendar years 2024-2025 (matches the network-map
convention; a partial 2023 missing the winter peak would bias the mean low).

Output: runs/<run>/output/validation/rb_yearly_means.csv
Run: GAPFILL_RUN=base OMP_NUM_THREADS=16 python scripts/rb_yearly_means.py
"""
import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
import config  # noqa: E402
import validate_model as vm  # noqa: E402

YEARS = ("2024", "2025")
RB = ["SK0004R", "SK0006R", "SK0002R", "SK0041A", "SK0042A", "SK0406A", "SK0007R"]


def feature_sets(df):
    base = list(vm.BASE_FEATURES)
    g1 = [c for c in vm.GROUP1_FEATURES if c in df.columns]
    g2 = [c for c in vm.GROUP2_FEATURES if c in df.columns]
    cl = [c for c in vm.CONC_LAG_FEATURES if c in df.columns]
    em = [c for c in vm.EMISSION_FEATURES if c in df.columns]
    m4 = base + g1 + g2 + cl + em                                  # production
    v = [c for c in base + g1 + g2 if c not in vm.PROXY_FEATURES
         and c not in vm.TYPOLOGY_FEATURES] + em  # proxy-free + typology-free (true V)
    return m4, v


def train(feats, t):
    params = dict(config.MODEL_PARAMS)
    params["monotone_constraints"] = config.monotone_for(feats)
    m = xgb.XGBRegressor(**params)
    m.fit(t[feats], np.log1p(t["bap"]), sample_weight=1.0 / (t["bap"] + 0.5))
    return m


def predict(m, feats, X):
    return np.maximum(0.0, np.expm1(m.predict(X[feats])))


def main():
    df = vm.load_dataset().reset_index(drop=True)
    M4, V = feature_sets(df)
    print(f"M4 (production) = {len(M4)} features | V (proxy-free) = {len(V)} features")
    m4_all, v_all = train(M4, df), train(V, df)
    trained = set(df["eoi"].unique())
    st = pd.read_csv(config.STATIONS_CSV).set_index("eoi")

    rows = []
    for s in RB:
        gf = pd.read_csv(os.path.join(config.OUTPUT_DIR, f"{s}_filled.csv"))
        gf = gf[gf["datum"].astype(str).str[:4].isin(YEARS)].copy()
        r = {"eoi": s, "name": st.loc[s, "name"].split(",")[0].strip(),
             "altitude": int(st.loc[s, "altitude"]),
             "in_training": s in trained}
        if s in trained:
            d = gf[gf["bap"].notna()].copy()
            r["n_obs"] = len(d)
            r["obs_ym"] = round(d["bap"].mean(), 2)
            r["prod_ym"] = round(predict(m4_all, M4, d).mean(), 2)
            r["prod_MBE"] = round(float(np.mean(predict(m4_all, M4, d) - d["bap"].values)), 2)
            m4l, vl = train(M4, df[df.eoi != s]), train(V, df[df.eoi != s])
            r["loso_ym"] = round(predict(m4l, M4, d).mean(), 2)
            r["V_ym"] = round(predict(vl, V, d).mean(), 2)  # proxy-free, station excluded
        else:
            r["n_obs"] = 0
            r["obs_ym"] = np.nan
            r["prod_ym"] = round(predict(m4_all, M4, gf).mean(), 2)
            r["prod_MBE"] = np.nan
            r["loso_ym"] = np.nan
            r["V_ym"] = round(predict(v_all, V, gf).mean(), 2)
        rows.append(r)

    out = pd.DataFrame(rows)
    path = os.path.join(config.VALIDATION_DIR, "rb_yearly_means.csv")
    out.to_csv(path, index=False)
    print("\n=== RB annual-mean B[a]P (2024-2025, ng/m3; target=1.0) ===")
    print(out.to_string(index=False))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
