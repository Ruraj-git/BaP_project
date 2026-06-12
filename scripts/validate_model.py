#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validácia globálneho XGBoost modelu pre dopĺňanie denného BaP.

Implementuje dva nezávislé validačné režimy:
  1) RANDOM K-FOLD  – štandardná k-násobná krížová validácia cez všetky vzorky.
                      (Optimistická: do tréningu aj testu môžu padnúť dni
                       z tej istej stanice -> meria skôr "gap-filling" skill.)
  2) LOSO           – Leave-One-Station-Out. Model nikdy nevidí testovaciu
                      stanicu -> meria skutočný "virtuálna sieť" skill na
                      nemonitorovaných lokalitách.

Metriky sa počítajú v PÔVODNÝCH jednotkách (po expm1 spätnej transformácii):
R2, RMSE, MAE, MBE (bias), n.

Skript je samostatný (nepoužíva config.BASE_PATH, ktorý ukazuje na pôvodný
priečinok) – tréningový dataset číta priamo z tohto repozitára.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- Cesty: odvodené z umiestnenia tohto súboru (repo root) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402  (kvôli MODEL_PARAMS, AREA_MAP)

# Dataset: v3 base, v4 +group-1, v5 +group-2 spatial, v6 +emission/population
# (per-run, resolved via config.RUN; override with TRAIN_READY env var)
TRAIN_READY = os.environ.get("TRAIN_READY", config.TRAIN_READY_CSV)
OUT_DIR = config.VALIDATION_DIR
PLOT_DIR = config.PLOTS_DIR

# Pôvodné príznaky (v3)
BASE_FEATURES = [
    "t_mean", "heating_degree_hours", "vrate_min", "hpbl_min",
    "night_vrate_avg", "night_sshf_min", "total_rain", "ws_max",
    "is_weekend", "month", "pm10_mean", "pm25_mean", "no2_mean",
    "typ_oblasti_code", "typ_zdroja_code", "altitude",
]

# Nové group-1 príznaky (pridané do v4)
GROUP1_FEATURES = [
    "t_range", "wdir_sin", "wdir_cos", "ssrd_total", "sp_mean", "rh_mean",
    "doy_sin", "doy_cos", "heating_season",
    "hpbl_min_3d", "hpbl_min_7d", "vrate_min_3d", "t_mean_3d",
    "t_mean_lag1", "hpbl_min_lag1", "sp_tendency", "stagnation_run",
]

# Nové group-2 priestorové príznaky (pridané vo v5)
GROUP2_FEATURES = [
    "elev_mean", "elev_relief", "tpi_local", "tpi_meso", "tpi_broad", "slope_deg",
    "traffic_load_log", "traffic_hdv_log", "dist_major_road_km",
]

# Lagy reálnych koncentrácií (PM10/PM2.5/NO2) – NIE BaP
CONC_LAG_FEATURES = [
    "pm10_mean_lag1", "pm25_mean_lag1", "no2_mean_lag1",
    "pm10_mean_3d", "pm25_mean_3d", "no2_mean_3d",
]

# Batch-2: bottom-up residential emission + population (true static covariates)
EMISSION_FEATURES = ["emis_bap_log", "emis_pm25_log"]
POP_FEATURES = ["pop_log"]

# Batch-3: wind-conditioned upwind industrial BaP load (time-varying, v7)
DIRECTIONAL_FEATURES = ["wdir_emis_bap_log"]

# Features depending only on co-located measurements (PM/NO2) – absent at virtual sites
PROXY_FEATURES = ["pm10_mean", "pm25_mean", "no2_mean"] + CONC_LAG_FEATURES

# Použijeme len tie príznaky, ktoré v datasete reálne existujú (kompatibilita v3..v6)
def resolve_features(df):
    feats = list(BASE_FEATURES)
    feats += [c for c in GROUP1_FEATURES if c in df.columns]
    feats += [c for c in GROUP2_FEATURES if c in df.columns]
    feats += [c for c in CONC_LAG_FEATURES if c in df.columns]
    feats += [c for c in EMISSION_FEATURES if c in df.columns]
    feats += [c for c in DIRECTIONAL_FEATURES if c in df.columns]
    # population (POP_FEATURES) dropped: redundant with the direct emission term
    return feats

FEATURES = BASE_FEATURES  # default; prepíše sa v main() podľa datasetu

N_SPLITS = 5
SEED = config.MODEL_PARAMS.get("random_state", 42)


def load_dataset():
    """Načíta train_ready a pripraví príznaky rovnako ako tréningový skript."""
    df = pd.read_csv(TRAIN_READY)
    df = df.dropna(subset=["bap"]).copy()

    # Kódovanie typológie – KONZISTENTNE cez celý dataset (rovnako pri všetkých foldoch)
    df["typ_oblasti_code"] = df["typ_oblasti"].str.strip().map(config.AREA_MAP).fillna(0)
    df["typ_zdroja_code"] = df["typ_zdroja"].astype("category").cat.codes
    return df


def pearson_r2(y_true, y_pred):
    """Druhá mocnina Pearsonovho korelačného koeficientu (asociácia, necitlivá na bias/škálu).
    Na rozdiel od koeficientu determinácie (1-SS_res/SS_tot) zostáva v [0,1] aj pri
    posune priemeru – preto sa bias a RMSE reportujú samostatne."""
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    return float(r ** 2)


def compute_metrics(y_true, y_pred):
    """Metriky v pôvodných jednotkách BaP. R2 = druhá mocnina Pearsonovho r."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true)),
        "R2": pearson_r2(y_true, y_pred),  # squared Pearson r
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MBE": float(np.mean(y_pred - y_true)),  # priemerná chyba (bias)
        "obs_mean": float(np.mean(y_true)),
        "pred_mean": float(np.mean(y_pred)),
    }


def fit_predict(train_df, test_df):
    """Natrénuje model na train_df a vráti predikcie (pôvodné jednotky) pre test_df."""
    X_tr, y_tr = train_df[FEATURES], np.log1p(train_df["bap"])
    # Váženie ako v produkčnom skripte: nižšie BaP -> vyššia váha
    weights = 1.0 / (train_df["bap"] + 0.5)

    model = xgb.XGBRegressor(**config.MODEL_PARAMS)
    model.fit(X_tr, y_tr, sample_weight=weights)

    pred = np.maximum(0.0, np.expm1(model.predict(test_df[FEATURES])))
    return pred


def run_kfold(df):
    print(f"\n=== RANDOM {N_SPLITS}-FOLD CV ===")
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.full(len(df), np.nan)  # out-of-fold predikcie
    per_fold = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(df), start=1):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        pred = fit_predict(tr, te)
        oof[te_idx] = pred
        m = compute_metrics(te["bap"], pred)
        m["fold"] = fold
        per_fold.append(m)
        print(f"  fold {fold}: n={m['n']:4d}  R2={m['R2']:.3f}  RMSE={m['RMSE']:.3f}  MAE={m['MAE']:.3f}")

    overall = compute_metrics(df["bap"], oof)
    print(f"  POOLED OOF: R2={overall['R2']:.3f}  RMSE={overall['RMSE']:.3f}  "
          f"MAE={overall['MAE']:.3f}  MBE={overall['MBE']:+.3f}")
    return oof, pd.DataFrame(per_fold), overall


def run_loso(df):
    print("\n=== LEAVE-ONE-STATION-OUT (LOSO) CV ===")
    logo = LeaveOneGroupOut()
    groups = df["eoi"].values
    oof = np.full(len(df), np.nan)
    per_station = []

    for tr_idx, te_idx in logo.split(df, groups=groups):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        eoi = te["eoi"].iloc[0]
        pred = fit_predict(tr, te)
        oof[te_idx] = pred
        m = compute_metrics(te["bap"], pred)
        m["eoi"] = eoi
        m["typ_oblasti"] = te["typ_oblasti"].iloc[0]
        m["typ_zdroja"] = te["typ_zdroja"].iloc[0]
        per_station.append(m)

    per_df = pd.DataFrame(per_station).sort_values("R2", ascending=False)
    for _, r in per_df.iterrows():
        print(f"  {r['eoi']:8s} [{r['typ_oblasti']}{r['typ_zdroja']}] "
              f"n={int(r['n']):4d}  R2={r['R2']:6.3f}  RMSE={r['RMSE']:.3f}  MBE={r['MBE']:+.3f}")

    overall = compute_metrics(df["bap"], oof)
    print(f"  POOLED OOF: R2={overall['R2']:.3f}  RMSE={overall['RMSE']:.3f}  "
          f"MAE={overall['MAE']:.3f}  MBE={overall['MBE']:+.3f}")
    print(f"  MEDIÁN R2 cez stanice: {per_df['R2'].median():.3f}")
    return oof, per_df, overall


def scatter_plot(y_true, y_pred, title, fname):
    mask = ~np.isnan(y_pred)
    yt, yp = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    lim = max(yt.max(), yp.max()) * 1.05
    plt.figure(figsize=(6, 6))
    plt.scatter(yt, yp, s=8, alpha=0.3, edgecolors="none")
    plt.plot([0, lim], [0, lim], "k--", lw=1, label="1:1")
    plt.axhline(1.0, color="gray", ls=":", lw=0.8)
    plt.axvline(1.0, color="gray", ls=":", lw=0.8, label="limit 1 ng/m³")
    r2 = pearson_r2(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    plt.title(f"{title}\nr²(Pearson)={r2:.3f}  RMSE={rmse:.3f}  n={len(yt)}")
    plt.xlabel("Observed BaP [ng/m³]")
    plt.ylabel("Predicted BaP [ng/m³]")
    plt.xlim(0, lim); plt.ylim(0, lim)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=130)
    plt.close()


def breakdown_by_type(df, oof, scheme):
    """Agregované metriky podľa kombinácie typ_oblasti+typ_zdroja."""
    tmp = df.copy()
    tmp["_pred"] = oof
    tmp = tmp.dropna(subset=["_pred"])
    rows = []
    for (lo, zd), g in tmp.groupby(["typ_oblasti", "typ_zdroja"]):
        m = compute_metrics(g["bap"], g["_pred"])
        m.update({"scheme": scheme, "typ_oblasti": lo, "typ_zdroja": zd,
                  "n_stations": g["eoi"].nunique()})
        rows.append(m)
    return pd.DataFrame(rows)


def main():
    global FEATURES
    os.makedirs(PLOT_DIR, exist_ok=True)
    df = load_dataset().reset_index(drop=True)
    FEATURES = resolve_features(df)
    print(f"Dataset: {os.path.basename(TRAIN_READY)} | {len(df)} vzoriek, "
          f"{df['eoi'].nunique()} staníc, {df['datum'].min()}–{df['datum'].max()}")
    n_g1 = len([c for c in GROUP1_FEATURES if c in df.columns])
    n_g2 = len([c for c in GROUP2_FEATURES if c in df.columns])
    n_cl = len([c for c in CONC_LAG_FEATURES if c in df.columns])
    print(f"Features: {len(FEATURES)} (base {len(BASE_FEATURES)} + g1 {n_g1} + g2 {n_g2} + conc-lag {n_cl})")

    kf_oof, kf_folds, kf_overall = run_kfold(df)
    loso_oof, loso_stations, loso_overall = run_loso(df)

    # --- Uloženie metrík ---
    summary = pd.DataFrame([
        {"scheme": "random_kfold", **kf_overall},
        {"scheme": "loso", **loso_overall},
        {"scheme": "loso_median_per_station", "n": len(loso_stations),
         "R2": loso_stations["R2"].median(), "RMSE": loso_stations["RMSE"].median(),
         "MAE": loso_stations["MAE"].median(), "MBE": loso_stations["MBE"].median()},
    ])
    summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
    kf_folds.to_csv(os.path.join(OUT_DIR, "kfold_per_fold.csv"), index=False)
    loso_stations.to_csv(os.path.join(OUT_DIR, "loso_per_station.csv"), index=False)

    bd = pd.concat([
        breakdown_by_type(df, kf_oof, "random_kfold"),
        breakdown_by_type(df, loso_oof, "loso"),
    ], ignore_index=True)
    bd.to_csv(os.path.join(OUT_DIR, "metrics_by_station_type.csv"), index=False)

    # --- Out-of-fold predikcie pre každú vzorku (pre časové rady / grafy) ---
    oof = df[["eoi", "datum", "bap", "typ_oblasti", "typ_zdroja"]].copy()
    oof["pred_kfold"] = kf_oof
    oof["pred_loso"] = loso_oof
    oof.to_csv(os.path.join(OUT_DIR, "oof_predictions.csv"), index=False)

    # --- Grafy ---
    scatter_plot(df["bap"], kf_oof, "Random K-Fold (out-of-fold)", "scatter_kfold.png")
    scatter_plot(df["bap"], loso_oof, "Leave-One-Station-Out", "scatter_loso.png")

    print(f"\n✅ Hotovo. Metriky a grafy: {OUT_DIR}")
    print("\n--- SUMÁR ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
