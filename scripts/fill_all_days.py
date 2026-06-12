#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Operatívne dopĺňanie BaP pre KAŽDÝ deň (nielen dni s meraním).

Rozdiel oproti validate_model.py:
  - validate_model = ČESTNÁ skúška skill-u (LOSO/k-fold), iba na dňoch s meraním.
  - tento skript   = PRODUKT: model natrénovaný na VŠETKÝCH meraniach sa aplikuje
                     na súvislý denný meteo-rad -> predikcia pre každý deň.

Pre každú stanicu vytvorí súvislý rad s:
  bap_predicted  – model pre každý deň
  bap            – meranie (ak existuje)
  bap_filled     – meranie, inak predikcia
  is_observed    – 1 ak deň mal meranie

Používa rovnaké príznaky ako v5 (group-1 + group-2).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402
from aggregate_daily import create_rich_features, add_temporal_features, get_pollutant_features  # noqa: E402
from validate_model import (BASE_FEATURES, GROUP1_FEATURES, GROUP2_FEATURES,  # noqa: E402
                            CONC_LAG_FEATURES, EMISSION_FEATURES)

TRAIN_READY = config.train_ready("v6")   # locked model = v6 (+bottom-up emission)
STATION_COVARIATES = os.path.join(config.COVARIATES_DIR, "station_covariates.csv")
HOURLY_DIR = config.HOURLY_FEATURES_DIR
OUT_DIR = config.OUTPUT_DIR
IMPORTANCE_CSV = os.path.join(config.VALIDATION_DIR, "feature_importance.csv")


def train_model(df_train, features):
    """Natrénuje XGBoost na všetkých meraniach (log1p + váženie ako v produkcii)."""
    X = df_train[features]
    y = np.log1p(df_train["bap"])
    weights = 1.0 / (df_train["bap"] + 0.5)
    model = xgb.XGBRegressor(**config.MODEL_PARAMS)
    model.fit(X, y, sample_weight=weights)
    return model


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(config.VALIDATION_DIR, exist_ok=True)

    df_train = pd.read_csv(TRAIN_READY).dropna(subset=["bap"]).copy()
    df_train["typ_oblasti_code"] = df_train["typ_oblasti"].str.strip().map(config.AREA_MAP).fillna(0)
    # Kategórie typ_zdroja zafixované z tréningu -> rovnaké kódovanie pri inferencii
    zdroj_cat = df_train["typ_zdroja"].astype("category")
    df_train["typ_zdroja_code"] = zdroj_cat.cat.codes
    zdroj_categories = zdroj_cat.cat.categories

    features = (BASE_FEATURES
                + [c for c in GROUP1_FEATURES if c in df_train.columns]
                + [c for c in GROUP2_FEATURES if c in df_train.columns]
                + [c for c in CONC_LAG_FEATURES if c in df_train.columns]
                + [c for c in EMISSION_FEATURES if c in df_train.columns])

    model = train_model(df_train, features)
    print(f"🎯 Model natrénovaný na {len(df_train)} meraniach, {len(features)} príznakov.")

    # --- Dôležitosť príznakov (gain) z modelu trénovaného na všetkých dátach ---
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    imp = pd.DataFrame({"feature": features})
    imp["gain"] = imp["feature"].map(gain).fillna(0.0)
    imp["weight"] = imp["feature"].map(weight).fillna(0.0)
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(IMPORTANCE_CSV, index=False)
    print(f"📑 Feature importance -> {IMPORTANCE_CSV}")

    stations = pd.read_csv(config.STATIONS_CSV)
    stations["eoi"] = stations["eoi"].astype(str).str.strip()
    covars = pd.read_csv(STATION_COVARIATES)

    obs = pd.read_csv(config.OBSERVATIONS_CSV)
    obs["datum"] = pd.to_datetime(obs["datum"]).dt.date
    obs["eoi"] = obs["eoi"].astype(str).str.strip()

    hourly_files = sorted(glob.glob(os.path.join(HOURLY_DIR, "*_hourly.csv")))
    n_done = 0
    for f in hourly_files:
        eoi = os.path.basename(f).replace("_hourly.csv", "").strip()
        s_meta = stations[stations["eoi"] == eoi]
        if s_meta.empty:
            continue

        # --- Súvislé denné príznaky pre VŠETKY dni ---
        daily = create_rich_features(pd.read_csv(f))
        # Polutanty MUSIA byť pripojené pred add_temporal_features (kvôli lagom koncentrácií)
        poll = get_pollutant_features(eoi)
        if not poll.empty:
            daily = pd.merge(daily, poll, on="datum", how="left")
        for c in ["pm10_mean", "pm25_mean", "no2_mean"]:
            if c not in daily.columns:
                daily[c] = np.nan  # XGBoost zvládne NaN
        daily = add_temporal_features(daily)

        # Statické príznaky
        daily["typ_oblasti_code"] = config.AREA_MAP.get(str(s_meta["typ_oblasti"].values[0]).strip(), 0)
        zd = str(s_meta["typ_zdroja"].values[0]).strip()
        daily["typ_zdroja_code"] = zdroj_categories.get_loc(zd) if zd in zdroj_categories else -1
        daily["altitude"] = s_meta["altitude"].values[0]

        # Group-2 priestorové kovariáty + bottom-up emisie
        cov_row = covars[covars["eoi"] == eoi]
        for c in GROUP2_FEATURES + EMISSION_FEATURES:
            if c in cov_row.columns:
                daily[c] = cov_row[c].values[0] if not cov_row.empty else np.nan

        # --- Predikcia každého dňa ---
        daily["bap_predicted"] = np.maximum(0.0, np.expm1(model.predict(daily[features])))

        # Spojenie s meraniami
        st_obs = obs[obs["eoi"] == eoi][["datum", "bap"]]
        out = pd.merge(daily, st_obs, on="datum", how="left")
        out["is_observed"] = out["bap"].notna().astype(int)
        out["bap_filled"] = out["bap"].fillna(out["bap_predicted"])
        out = out.sort_values("datum")

        out.to_csv(os.path.join(OUT_DIR, f"{eoi}_filled.csv"), index=False)
        n_done += 1

    print(f"✅ Doplnené všetky dni pre {n_done} staníc -> {OUT_DIR}")
    print(f"   (každá stanica = súvislý denný rad {len(daily)} dní)")


if __name__ == "__main__":
    main()
