#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benzo[a]pyrene gap-filling: central path and run configuration.

@author: J. Beňo, D. Štefánik, J. Matejovičová (SHMÚ)
"""

import os

# ===========================================================================
#  RUN CONTROL  --  edit RUN to switch between dataset versions / experiments.
#  PROJECT_ROOT is auto-derived from this file's location, so the project can
#  be moved/copied (e.g. GAPFILL -> GAPFILL_ARTICLE) without editing paths.
# ===========================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = os.path.basename(PROJECT_ROOT)
RUN = os.environ.get("GAPFILL_RUN", "base")   # -> runs/<RUN>/ ;  e.g. "base", "extended"

# --- SHARED INPUTS (not per-run; same across all runs) ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STATIONS_CSV = os.path.join(DATA_DIR, "stations.csv")
METADATA_CSV = os.path.join(DATA_DIR, "other", "nmsko_18.csv")
OBSERVATIONS_CSV = os.path.join(DATA_DIR, "bap_obs.csv")
POLLUTANTS_DIR = os.path.join(DATA_DIR, "pollutants")
COVARIATES_DIR = os.path.join(DATA_DIR, "covariates")   # static grid/station covariates

# --- PER-RUN ARTIFACTS (selected by RUN) ---
RUN_DIR = os.path.join(PROJECT_ROOT, "runs", RUN)
FEATURES_DIR = os.path.join(RUN_DIR, "features")
HOURLY_FEATURES_DIR = os.path.join(FEATURES_DIR, "hourly_extracted")
DAILY_DIR = os.path.join(FEATURES_DIR, "daily_rich_vectors")
OUTPUT_DIR = os.path.join(RUN_DIR, "output", "gap_filled")   # per-station *_filled.csv
VALIDATION_DIR = os.path.join(RUN_DIR, "output", "validation")
PLOTS_DIR = os.path.join(VALIDATION_DIR, "plots")
MODELS_DIR = os.path.join(RUN_DIR, "models")


def train_ready(stage="v6"):
    """Path to a staged training table within the active run.
    v4 = meteo+derived, v5 = +spatial covariates, v6 = +bottom-up emission."""
    return os.path.join(DAILY_DIR, f"train_ready_bap_{stage}.csv")


TRAIN_READY_CSV = train_ready("v6")   # final table (with emission covariates)

# --- GRID CONFIGURATION (ALADIN) ---
PROJ4 = '+proj=lcc +lat_1=48.80182499999999 +lat_2=48.80182499999999 +lat_0=48.80182499999999 +lon_0=18.111565 +x_0=0.0 +y_0=0.0 +a=6371229.0 +b=6371229.0 +units=m +no_defs:'
CUTTER = (200, -86, 125, -138)
XORIG = -501000.0000000002 + (CUTTER[0] * 2000)
YORIG = -373000.0000000017 + (CUTTER[2] * 2000)
DX, DY = 2000, 2000
NX, NY = 215, 110

# --- LOGIKA VÝPOČTU ---
NIGHT_HOURS = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
HEATING_THRESHOLD = 15.5

# --- MODEL HYPERPARAMETERS (XGBOOST) ---
MODEL_PARAMS = {
    'n_estimators': 1200,
    'learning_rate': 0.02,
    'max_depth': 6,
    'reg_alpha': 0.1,
    'reg_lambda': 1.2,
    # n_jobs from env so SLURM allocations can cap threads (default -1 = all cores).
    'n_jobs': int(os.environ.get("GAPFILL_NJOBS", "-1")),
    'random_state': 42
}

# --- PHYSICALLY-MOTIVATED MONOTONE CONSTRAINTS (core_alt set) ---------------
# Direction of the PARTIAL effect (all other features held fixed): +1 = BaP may
# not DECREASE with the feature, -1 = may not INCREASE. Physical priors: more
# residential/road emission and traffic -> not less BaP; farther from major roads
# and higher altitude (above the emission/inversion layer) -> not more BaP.
# Adopted 2026-07-02 after the monotone sweep (fixes the Stara Lesna low-traffic
# extrapolation, LOSO 0.819->0.829, no regime harmed). See experiments/monotone_*.
MONOTONE_DIRS = {
    "traffic_load_log": 1,
    "traffic_hdv_log": 1,
    "dist_major_road_km": -1,
    "emis_bap_log": 1,
    "emis_pm25_log": 1,
    "altitude": -1,
}


# Live toggle (also env GAPFILL_MONOTONE=0). The feature-ablation ladder flips this
# to False to show unconstrained feature value, then back to True for the final
# adopted "M4 + monotone" row; all other scripts keep it on.
MONOTONE_ENABLED = os.environ.get("GAPFILL_MONOTONE", "1") != "0"


def monotone_for(feature_names):
    """monotone_constraints dict restricted to features actually present.

    xgboost raises if a constrained feature is absent from the training columns
    (e.g. ablation runs that drop feature groups), so intersect by name. Pass the
    result as XGBRegressor(monotone_constraints=...) at fit time. Returns None (the
    true no-op) when disabled or when no constrained feature is present -- note an
    EMPTY DICT {} is NOT a no-op with the full params (it changes the fit), so we
    must return None, never {}.
    """
    if not MONOTONE_ENABLED:
        return None
    present = set(feature_names)
    d = {f: dd for f, dd in MONOTONE_DIRS.items() if f in present}
    return d or None

# Váhy pre typy oblastí (R=Rural, S=Suburban, U=Urban)
AREA_MAP = {'R': 0, 'S': 1, 'U': 2}
RURAL_WEIGHT_FACTOR = 2.5

METEO_NC_PATH = "/data/oko/meteo/nc"   # raw ALADIN source (external, shared)

# Nastavenia pre API sťahovanie / extrakciu meteo
POLLUTANTS = ["PM10", "PM25", "NO2"]

# Roky na (do)spracovanie. fetch_pollutants.py aj extract_nc.py spracujú LEN
# tieto roky a výsledok PRIPOJA k existujúcim *_hourly.csv (dedup podľa času,
# najnovšie vyhráva, + chronologické zoradenie).
# Napr. [2023] dotiahne 2023 k už hotovým 2024-2025 bez ich prepisu.
YEARS = [2023]
# Roky s neúplným záznamom (dáta nezačínajú 1. januára). Mesiace pred týmto
# dátumom sa pri sťahovaní preskočia (ALADIN aj merania 2023 sú od 2. júna).
YEAR_START = {2023: "2023-06-02"}

API_BASE_URL = "http://srv-mondo.kol.shmu.sk:8018/observations"
API_SLEEP_TIME = 0.05  # Sekundy medzi požiadavkami (buďme milí k srv-mondo)

# Kombinovaný tag: OBLASŤ + ZDROJ (Hierarchia pre monotónne obmedzenie)
# Čím vyššie číslo, tým vyšší potenciál znečistenia
STATION_TYPE_COMBINED = {
    'RB': 0,  # Rural Background (Kolonické sedlo, Starina)
    'SB': 1,  # Suburban Background
    'UB': 2,  # Urban Background
    'RI': 3,  # Rural Industrial
    'SI': 4,  # Suburban Industrial
    'UI': 5,  # Urban Industrial
    'UT': 6   # Urban Traffic
}
