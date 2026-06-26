#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build an authoritative B[a]P observation table from the EEA Air Quality
Download Service (verified E1a dataset), reconciled with the current
`data/bap_obs.csv`.

Why
---
`data/bap_obs.csv` (derived from `data/other/bap_obs.xlsx`) contains a handful of
duplicated / conflicting station-date rows (all in 2023) and diverges from the
officially reported values on ~900 further dates. The EEA *verified* (E1a)
record is the authoritative source the manuscript already cites; it is clean
(one valid + verified value per station-date, no duplicates). This script
rebuilds the observation table from that source.

Merge rule
----------
- For the stations and years that the EEA verified dataset covers
  (BaP = pollutant 5029, currently 2019-2024, 22 Slovak sampling points), use
  the EEA verified value (Validity == 1 and Verification == 1).
- For everything else -- 2025 (not yet verified by EEA), pre-2019, and stations
  the EEA BaP set does not contain (e.g. the trained station SK0078A,
  NoCode-99501) -- keep the current `data/bap_obs.csv` value.

Output: `data/bap_obs_eea.csv` (NEW file; does NOT overwrite the live
`bap_obs.csv`). A `source` column records EEA-verified vs current-pipeline.
Parquet files are cached under `data/eea_bap_cache/` (gitignored with `data/`).

Run: GAPFILL_RUN=base python scripts/build_bap_obs_from_eea.py
     (set EEA_EMAIL to your address; the API requires a syntactically valid one)
"""

import os
import sys
import json
import subprocess

import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402

API_URLS = "https://eeadmz1-downloads-api-appservice.azurewebsites.net/ParquetFile/urls"
BAP_POLLUTANT = "http://dd.eionet.europa.eu/vocabulary/aq/pollutant/5029"  # BaP in PM10
DATASET_VERIFIED = 2          # E1a verified (2013->last reported year)
EMAIL = os.environ.get("EEA_EMAIL", "air-quality-download@example.com")

CACHE = os.path.join(config.DATA_DIR, "eea_bap_cache")
CURRENT = config.OBSERVATIONS_CSV                       # data/bap_obs.csv
OUT = os.path.join(config.DATA_DIR, "bap_obs_eea.csv")


def fetch_url_list():
    # The EEA endpoint streams its response in a way urllib mis-reads (returns
    # only the header); curl reads it correctly, so we shell out.
    body = json.dumps({
        "countries": ["SK"], "cities": [],
        "pollutants": [BAP_POLLUTANT], "dataset": DATASET_VERIFIED,
        "dateTimeStart": "2013-01-01T00:00:00.000Z",
        "dateTimeEnd": "2026-01-01T00:00:00.000Z",
        "aggregationType": "day", "email": EMAIL,
    })
    txt = subprocess.run(
        ["curl", "-sS", "-m", "120", "-X", "POST", API_URLS,
         "-H", "Content-Type: application/json", "-d", body],
        capture_output=True, text=True, check=True).stdout
    urls = [u.strip() for u in txt.splitlines() if u.strip().endswith(".parquet")]
    if not urls:
        raise RuntimeError(f"EEA API returned no parquet URLs:\n{txt[:300]}")
    return urls


def download(urls):
    os.makedirs(CACHE, exist_ok=True)
    paths = []
    for u in urls:
        p = os.path.join(CACHE, os.path.basename(u))
        if not os.path.exists(p):
            subprocess.run(["curl", "-sS", "-m", "120", "-o", p, u], check=True)
        paths.append(p)
    return paths


def load_verified(paths):
    cols = ["Samplingpoint", "Start", "Value", "Validity", "Verification", "AggType"]
    frames = [pd.read_parquet(p, columns=cols) for p in paths]
    e = pd.concat(frames, ignore_index=True)
    e["eoi"] = e["Samplingpoint"].str.extract(r"SPO-(SK\w+?)_05029")
    e["datum"] = pd.to_datetime(e["Start"]).dt.strftime("%Y-%m-%d")
    e["bap"] = pd.to_numeric(e["Value"], errors="coerce").round(4)
    v = e[(e["Validity"] == 1) & (e["Verification"] == 1)].copy()
    # collapse any (rare) multiple sampling points per station-date to the mean
    v = v.groupby(["eoi", "datum"], as_index=False)["bap"].mean()
    v["bap"] = v["bap"].round(4)
    return v


def cached_parquets():
    import glob
    return sorted(glob.glob(os.path.join(CACHE, "*.parquet")))


def main():
    # Cache-first: the EEA endpoint throttles repeated requests, so reuse the
    # local cache when present and only call the API to populate an empty cache.
    paths = cached_parquets()
    if paths:
        print(f"Using cached EEA parquet files: {len(paths)} (in {CACHE})")
    else:
        print(f"EEA email: {EMAIL}  (set EEA_EMAIL to override)")
        urls = fetch_url_list()
        print(f"EEA verified BaP files for SK: {len(urls)}")
        paths = download(urls)
    eea = load_verified(paths)
    eea_years = set(eea["datum"].str[:4])
    eea_stations = set(eea["eoi"])
    print(f"EEA verified: {len(eea)} rows | years {sorted(eea_years)} | "
          f"{len(eea_stations)} stations")

    cur = pd.read_csv(CURRENT)[["eoi", "datum", "bap"]].copy()
    cur["yr"] = cur["datum"].astype(str).str[:4]

    # current rows to KEEP = those EEA does not authoritatively cover
    keep_mask = (~cur["eoi"].isin(eea_stations)) | (~cur["yr"].isin(eea_years))
    cur_keep = cur.loc[keep_mask, ["eoi", "datum", "bap"]].copy()

    eea_out = eea.assign(source="EEA-E1a-verified")
    cur_out = cur_keep.assign(source="current-pipeline")
    new = (pd.concat([eea_out, cur_out], ignore_index=True)
             .drop_duplicates(["eoi", "datum"])
             .sort_values(["eoi", "datum"])
             .reset_index(drop=True))
    new.to_csv(OUT, index=False)

    # ---- impact report ----
    m = cur.merge(eea, on=["eoi", "datum"], suffixes=("_cur", "_eea"), how="inner")
    changed = (m["bap_cur"].round(3) - m["bap_eea"].round(3)).abs() > 0.005
    print("\n=== impact vs current bap_obs.csv ===")
    print(f"  current rows: {len(cur)}  ->  new rows: {len(new)}")
    print(f"  from EEA verified: {len(eea_out)} | retained from current: {len(cur_out)}")
    print(f"  overlapping station-dates: {len(m)}; value corrected (>0.005): {int(changed.sum())}")
    dropped = len(cur) - len(cur_keep) - len(m)
    print(f"  current-only rows (covered stations/years) dropped in favour of EEA: ~{dropped}")
    print(f"\n✅ wrote {OUT}  (live bap_obs.csv untouched)")


if __name__ == "__main__":
    main()
