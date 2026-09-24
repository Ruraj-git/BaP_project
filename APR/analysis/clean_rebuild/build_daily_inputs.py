#!/usr/bin/env python3
"""Create immutable clean-rebuild daily modelling inputs from raw hourly sources.

The input hourly meteorology must have passed validate_station_hourly.py.  Daily
meteorology and proxy lags are calculated on each station's uninterrupted calendar;
B[a]P is joined only after those calculations.  Nothing from a historical training
table is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
START, END = pd.Timestamp("2023-06-02"), pd.Timestamp("2025-12-31")
METEO_FIELDS = ["2t", "10u", "10v", "hpbl", "2r", "sp", "vrate", "ssrd", "sshf", "rain"]
POLLUTANTS = ("pm10", "pm25", "no2")
NIGHT_HOURS = (20, 21, 22, 23, 0, 1, 2, 3, 4, 5)
HEATING_MONTHS = {10, 11, 12, 1, 2, 3, 4}
STATIC = ["elev_mean", "elev_relief", "tpi_local", "tpi_meso", "tpi_broad", "slope_deg",
          "traffic_load_log", "traffic_hdv_log", "dist_major_road_km", "emis_bap_log", "emis_pm25_log"]
AREA_CODE = {"R": 0, "S": 1, "U": 2}
SOURCE_CODE = {"B": 0, "I": 1, "T": 2}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    n = int(frame.duplicated(keys).sum())
    if n:
        raise ValueError(f"{label}: {n} duplicate {keys} keys")


def read_hourly(chunk_dir: Path) -> pd.DataFrame:
    files = sorted(chunk_dir.glob("chunk_*/station_hourly.csv"))
    if len(files) != 48:
        raise ValueError(f"Expected 48 hourly chunks, found {len(files)}")
    h = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    h["time"] = pd.to_datetime(h["time"], errors="raise")
    require_unique(h, ["eoi", "time"], "meteorology")
    if h["time"].min() != START or h["time"].max() != END + pd.Timedelta(hours=23):
        raise ValueError("Unexpected meteorology date range")
    if not np.isfinite(h[METEO_FIELDS].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite meteorology values")
    return h


def daily_meteorology(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.copy()
    h["datum"] = h.time.dt.normalize()
    h["hour"] = h.time.dt.hour
    h["temp_c"] = h["2t"] - 273.15
    h["wind_speed"] = np.hypot(h["10u"], h["10v"])
    rows = []
    for (eoi, datum), g in h.groupby(["eoi", "datum"], sort=True):
        if len(g) != 24 or g.hour.nunique() != 24:
            raise ValueError(f"Incomplete meteorology day: {eoi} {datum:%F}")
        u, v = g["10u"].mean(), g["10v"].mean()
        night = g[g.hour.isin(NIGHT_HOURS)]
        rows.append({
            "eoi": eoi, "datum": datum,
            "t_mean": g.temp_c.mean(), "t_range": g.temp_c.max() - g.temp_c.min(),
            "heating_degree_hours": np.maximum(0.0, 15.5 - g.temp_c).sum(),
            "vrate_min": g.vrate.min(), "hpbl_min": g.hpbl.min(), "total_rain": g.rain.sum(),
            "ws_max": g.wind_speed.max(), "wdir_sin": np.sin(np.arctan2(u, v)),
            "wdir_cos": np.cos(np.arctan2(u, v)), "ssrd_total": g.ssrd.sum(),
            "sp_mean": g.sp.mean(), "rh_mean": g["2r"].mean(),
            "night_vrate_avg": night.vrate.mean(), "night_sshf_min": night.sshf.min(),
        })
    d = pd.DataFrame(rows)
    dt = pd.to_datetime(d.datum)
    d["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    d["month"] = dt.dt.month
    doy = dt.dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["heating_season"] = d.month.isin(HEATING_MONTHS).astype(int)
    return d


def raw_pollutant_daily(eoi: str, pollutant: str) -> pd.DataFrame:
    path = ROOT / "data" / "pollutants" / pollutant / f"{eoi}_hourly.csv"
    if not path.exists():
        return pd.DataFrame(columns=["datum", f"{pollutant}_mean", f"{pollutant}_valid_hours"])
    h = pd.read_csv(path, usecols=["time", "value"])
    h["time"] = pd.to_datetime(h.time, errors="coerce", utc=True)
    h["value"] = pd.to_numeric(h.value, errors="coerce")
    h = h.dropna(subset=["time", "value"]).copy()
    # A duplicate timestamp is permitted only where every retained numeric value agrees.
    conflict = h.groupby("time").value.nunique().gt(1)
    if conflict.any():
        raise ValueError(f"{path}: conflicting duplicated timestamps")
    h = h.drop_duplicates("time")
    h["datum"] = h.time.dt.tz_localize(None).dt.normalize()
    return h.groupby("datum", as_index=False).value.agg(**{f"{pollutant}_mean": "mean", f"{pollutant}_valid_hours": "size"})


def stagnation_count(stagnant):
    return stagnant.groupby((~stagnant).cumsum()).cumsum().astype(int)


def add_proxies_and_temporal(daily: pd.DataFrame, target_stations: list[str], strict: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = pd.date_range(START, END, freq="D")
    parts, coverage = [], []
    for eoi in target_stations:
        base = daily[daily.eoi.eq(eoi)].set_index("datum").reindex(calendar).rename_axis("datum").reset_index()
        base["eoi"] = eoi
        for pol in POLLUTANTS:
            p = raw_pollutant_daily(eoi, pol).set_index("datum")
            base = base.join(p, on="datum")
            col, hours = f"{pol}_mean", f"{pol}_valid_hours"
            if strict:
                base.loc[base[hours].fillna(0) < 18, col] = np.nan
            coverage.append(base[["eoi", "datum", hours]].rename(columns={hours: "valid_hours"}).assign(pollutant=pol, strict_ge18=base[hours].fillna(0).ge(18)))
            base[f"{col}_lag1"] = base[col].shift(1)
            base[f"{col}_3d"] = base[col].rolling(3, min_periods=1).mean()
        for col, window in (("hpbl_min", 3), ("hpbl_min", 7), ("vrate_min", 3), ("t_mean", 3)):
            base[f"{col}_{window}d"] = base[col].rolling(window, min_periods=1).mean()
        base["t_mean_lag1"] = base.t_mean.shift(1)
        base["hpbl_min_lag1"] = base.hpbl_min.shift(1)
        base["sp_tendency"] = base.sp_mean.diff()
        stagnant = (base.ws_max < 2.0) & (base.total_rain < 1.0)
        base["stagnation_run"] = stagnation_count(stagnant)
        parts.append(base)
    return pd.concat(parts, ignore_index=True), pd.concat(coverage, ignore_index=True)


def build_targets() -> pd.DataFrame:
    b = pd.read_csv(ROOT / "data" / "bap_obs.csv", usecols=["datum", "bap", "eoi"])
    b["datum"] = pd.to_datetime(b.datum, errors="raise").dt.normalize()
    b["eoi"] = b.eoi.astype(str).str.strip()
    b["bap"] = pd.to_numeric(b.bap, errors="raise")
    b = b[(b.datum >= START) & (b.datum <= END)].copy()
    require_unique(b, ["eoi", "datum"], "B[a]P")
    if (b.bap < 0).any():
        raise ValueError("Negative B[a]P target")
    return b


def attach_metadata(frame: pd.DataFrame, support: str, stations: pd.DataFrame) -> pd.DataFrame:
    static_file = ROOT / "APR" / "results" / "grid_validation" / "station_500m" / f"station_covariates_{support}.csv"
    s = pd.read_csv(static_file)
    require_unique(s, ["eoi"], f"{support} static")
    if not set(STATIC).issubset(s):
        raise ValueError(f"{static_file}: expected static columns absent")
    if not np.isfinite(s[STATIC].to_numpy(float)).all():
        raise ValueError(f"{static_file}: non-finite static covariates")
    m = frame.merge(stations[["eoi", "typ_oblasti", "typ_zdroja", "altitude"]], on="eoi", how="left", validate="many_to_one")
    m = m.merge(s[["eoi"] + STATIC], on="eoi", how="left", validate="many_to_one")
    m["typ_oblasti"] = m.typ_oblasti.astype(str).str.strip()
    m["typ_zdroja"] = m.typ_zdroja.astype(str).str.strip()
    m["typ_oblasti_code"] = m.typ_oblasti.map(AREA_CODE)
    m["typ_zdroja_code"] = m.typ_zdroja.map(SOURCE_CODE)
    if m[["typ_oblasti", "typ_zdroja", "altitude"] + STATIC].isna().any().any():
        raise ValueError(f"{support}: missing metadata/static covariates")
    if m[["typ_oblasti_code", "typ_zdroja_code"]].isna().any().any():
        raise ValueError(f"{support}: unrecognised typology label")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hourly-chunks", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    hourly = read_hourly(args.hourly_chunks)
    targets_raw = build_targets()
    stations = pd.read_csv(ROOT / "data" / "stations.csv")
    stations["eoi"] = stations.eoi.astype(str).str.strip()
    require_unique(stations, ["eoi"], "station metadata")
    target_stations = sorted(set(targets_raw.eoi).intersection(stations.eoi))
    excluded_no_metadata = targets_raw.loc[~targets_raw.eoi.isin(target_stations)].copy()
    targets = targets_raw.loc[targets_raw.eoi.isin(target_stations)].copy()
    if excluded_no_metadata.eoi.nunique() and set(excluded_no_metadata.eoi) != {"SK0065A"}:
        raise ValueError("Unexpected raw B[a]P stations without metadata")
    daily_met = daily_meteorology(hourly)
    outputs = {}
    args.output.mkdir(parents=True)
    daily_met.to_csv(args.output / "daily_meteorology_all54.csv", index=False)
    for strict, name in ((False, "permissive"), (True, "strict18")):
        d, cov = add_proxies_and_temporal(daily_met, target_stations, strict=strict)
        cov.to_csv(args.output / f"proxy_coverage_{name}.csv", index=False)
        d = d.merge(targets, on=["eoi", "datum"], how="inner", validate="one_to_one")
        require_unique(d, ["eoi", "datum"], f"{name} target input")
        for support in ("500m", "2km_exact"):
            complete = attach_metadata(d, support, stations)
            path = args.output / f"train_ready_{name}_{support}.csv"
            complete.to_csv(path, index=False)
            outputs[path.name] = {"rows": len(complete), "stations": int(complete.eoi.nunique()), "sha256": sha256(path)}
    report = {
        "script_sha256": sha256(Path(__file__)),
        "typology_maps": {"area": AREA_CODE, "source": SOURCE_CODE},
        "source_hashes": {str(path): sha256(path) for path in [ROOT / "data/bap_obs.csv", ROOT / "data/stations.csv"] + sorted((ROOT / "APR/results/grid_validation/station_500m").glob("station_covariates_*.csv"))},
        "source_hourly_chunks": str(args.hourly_chunks), "hourly_station_count": int(hourly.eoi.nunique()),
        "raw_target_rows": int(len(targets_raw)), "target_rows": int(len(targets)), "target_stations": len(target_stations),
        "target_date_min": str(targets.datum.min().date()), "target_date_max": str(targets.datum.max().date()),
        "excluded_no_station_metadata": {
            "stations": sorted(excluded_no_metadata.eoi.unique()), "rows": int(len(excluded_no_metadata)),
            "reason": "No record in data/stations.csv; therefore no audited static covariates."
        },
        "outputs": outputs,
    }
    (args.output / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
