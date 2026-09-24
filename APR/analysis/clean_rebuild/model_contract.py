"""Frozen feature order and estimator contract for the clean rebuild.

This module deliberately does not import project ``config.py`` or historical
analysis modules: model inputs and constraints must not vary with global state.
"""
from __future__ import annotations

METEO_CALENDAR = [
    "t_mean", "heating_degree_hours", "vrate_min", "hpbl_min", "night_vrate_avg",
    "night_sshf_min", "total_rain", "ws_max", "is_weekend", "month", "t_range",
    "wdir_sin", "wdir_cos", "ssrd_total", "sp_mean", "rh_mean", "doy_sin",
    "doy_cos", "heating_season", "hpbl_min_3d", "hpbl_min_7d", "vrate_min_3d",
    "t_mean_3d", "t_mean_lag1", "hpbl_min_lag1", "sp_tendency", "stagnation_run",
]
STATIC = [
    "elev_mean", "elev_relief", "tpi_local", "tpi_meso", "tpi_broad", "slope_deg",
    "traffic_load_log", "traffic_hdv_log", "dist_major_road_km", "emis_bap_log", "emis_pm25_log",
]
PROXIES = [
    "pm10_mean", "pm25_mean", "no2_mean", "pm10_mean_lag1", "pm25_mean_lag1",
    "no2_mean_lag1", "pm10_mean_3d", "pm25_mean_3d", "no2_mean_3d",
]
PM_XGB = ["pm10_mean", "pm25_mean", "pm10_mean_lag1", "pm25_mean_lag1", "pm10_mean_3d", "pm25_mean_3d", "is_weekend", "month", "doy_sin", "doy_cos", "heating_season"]
G = METEO_CALENDAR + STATIC
G_PLUS_P = G + PROXIES
M_AUX = METEO_CALENDAR + PROXIES + ["typ_oblasti_code", "typ_zdroja_code", "altitude"] + STATIC

FEATURES = {"PM_XGB": PM_XGB, "FULL_ID": M_AUX, "G": G, "G_PLUS_P": G_PLUS_P, "M_AUX": M_AUX}
MONOTONE_DIRECTIONS = {"traffic_load_log": 1, "traffic_hdv_log": 1, "dist_major_road_km": -1, "emis_bap_log": 1, "emis_pm25_log": 1, "altitude": -1}
XGB_PARAMS = {"n_estimators": 1200, "learning_rate": 0.02, "max_depth": 6, "reg_alpha": 0.1, "reg_lambda": 1.2, "objective": "reg:squarederror", "random_state": 42, "n_jobs": 1, "verbosity": 0}

def monotone_tuple(features: list[str]) -> tuple[int, ...]:
    return tuple(MONOTONE_DIRECTIONS.get(feature, 0) for feature in features)
