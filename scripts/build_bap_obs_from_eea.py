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
  (BaP = pollutant 5029, 2019-2024, 23 Slovak sampling points in scope), use
  the EEA verified value (Validity == 1 and Verification == 1). This includes
  the trained station SK0078A (Žarnovica): its BaP sampling point
  SPO-SK0078A_05029_100 IS registered and delivered verified (116 daily rows for
  2024), byte-identical to the national-pipeline values.
- For everything else -- 2025 (not yet verified by EEA), pre-2019, and the
  codeless NoCode-99501 placeholder -- keep the current `data/bap_obs.csv` value.

Completeness guard
------------------
The `/ParquetFile/urls` endpoint is UNRELIABLE: it silently returns a subset of
the registered sampling points (the 2026-06-25 run dropped SK0078A and two
others; on 2026-06-30 it returned nothing at all). To stop a silent drop from
ever corrupting the table again, this script cross-checks the sampling points it
actually loaded against the authoritative EEA `/List` registry (filtered to BaP
= 05029) and ABORTS if any in-scope registered station is missing. EXCLUDE_SPO
holds stations that are EEA-registered but deliberately outside this study's
national dataset (SK0266A Prešov, SK0267A Košice-Štefánikova -- no national BaP
counterpart, not in the trained set) so they don't trip a false alarm.

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

API_BASE = "https://eeadmz1-downloads-api-appservice.azurewebsites.net"
API_URLS = f"{API_BASE}/ParquetFile/urls"
API_LIST = f"{API_BASE}/List"           # registry of sampling points (authoritative)
API_FILE = f"{API_BASE}/ParquetFile"    # direct zip download (fallback)
BAP_POLLUTANT = "http://dd.eionet.europa.eu/vocabulary/aq/pollutant/5029"  # BaP in PM10
BAP_CODE = "05029"            # sampling-point code segment for BaP-in-PM10
DATASET_VERIFIED = 2          # E1a verified (2013->last reported year)
EMAIL = os.environ.get("EEA_EMAIL", "air-quality-download@example.com")

# EEA-registered BaP sampling points that are deliberately OUT of scope for this
# study (no national BaP counterpart in data/bap_obs.csv, not in the trained
# set). Listed here so the completeness guard does not raise a false alarm; any
# OTHER registered station missing from the download is a real error.
EXCLUDE_SPO = {"SK0266A", "SK0267A"}

CACHE = os.path.join(config.DATA_DIR, "eea_bap_cache")
CURRENT = config.OBSERVATIONS_CSV                       # data/bap_obs.csv
OUT = os.path.join(config.DATA_DIR, "bap_obs_eea.csv")


def _request_body():
    return json.dumps({
        "countries": ["SK"], "cities": [],
        "pollutants": [BAP_POLLUTANT], "dataset": DATASET_VERIFIED,
        "dateTimeStart": "2013-01-01T00:00:00.000Z",
        "dateTimeEnd": "2026-01-01T00:00:00.000Z",
        "aggregationType": "day", "email": EMAIL,
    })


def _post(url, body, binary_out=None, timeout="120"):
    # The EEA endpoints stream responses in a way urllib mis-reads (returns only
    # the header); curl reads them correctly, so we shell out.
    cmd = ["curl", "-sS", "-m", timeout, "-X", "POST", url,
           "-H", "Content-Type: application/json", "-d", body]
    if binary_out:
        cmd += ["-o", binary_out]
        subprocess.run(cmd, check=True)
        return None
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def registry_bap_stations():
    """Authoritative set of SK stations with a registered BaP (05029) sampling
    point, from the EEA `/List` endpoint. `/List` ignores the pollutant filter
    and returns every SK sampling point, so we filter client-side on the 05029
    code segment. This is what the download is checked against."""
    txt = _post(API_LIST, _request_body())
    try:
        spos = json.loads(txt)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"EEA /List did not return JSON:\n{txt[:300]}") from exc
    stations = {s.split("SPO-")[1][:7] for s in spos if f"_{BAP_CODE}_" in s}
    if not stations:
        raise RuntimeError(f"EEA /List returned no BaP sampling points:\n{txt[:300]}")
    return stations


def fetch_url_list():
    txt = _post(API_URLS, _request_body())
    urls = [u.strip() for u in txt.splitlines() if u.strip().endswith(".parquet")]
    if not urls:
        raise RuntimeError(f"EEA /ParquetFile/urls returned no parquet URLs:\n{txt[:300]}")
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


def download_via_zip_fallback():
    """Fallback when /ParquetFile/urls returns an empty/short list: pull the
    verified BaP data straight from /ParquetFile (a zip), chunked by year to stay
    under the endpoint's request-size limit, and split it into one full-history
    parquet per sampling point in CACHE (matching the /urls cache layout)."""
    import glob
    import io
    import zipfile
    import pandas as pd

    os.makedirs(CACHE, exist_ok=True)
    body = json.loads(_request_body())
    frames = {}
    for yr in range(2019, 2025):  # verified dataset currently ends 2024
        body["dateTimeStart"] = f"{yr}-01-01T00:00:00.000Z"
        body["dateTimeEnd"] = f"{yr}-12-31T23:59:59.000Z"
        tmp = os.path.join(CACHE, f"_fallback_{yr}.zip")
        _post(API_FILE, json.dumps(body), binary_out=tmp, timeout="180")
        try:
            zf = zipfile.ZipFile(tmp)
        except zipfile.BadZipFile:
            head = open(tmp, "rb").read(200)
            os.remove(tmp)
            raise RuntimeError(f"/ParquetFile fallback ({yr}) not a zip: {head!r}")
        for nm in zf.namelist():
            if f"_{BAP_CODE}_" not in nm:
                continue
            spo = os.path.basename(nm)            # SPO-SKxxxx_05029_100.parquet
            frames.setdefault(spo, []).append(pd.read_parquet(io.BytesIO(zf.read(nm))))
        zf.close()
        os.remove(tmp)
    if not frames:
        raise RuntimeError("/ParquetFile fallback produced no BaP files")
    paths = []
    for spo, parts in frames.items():
        p = os.path.join(CACHE, spo)
        pd.concat(parts, ignore_index=True).to_parquet(p, index=False)
        paths.append(p)
    return sorted(paths)


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


def verify_completeness(loaded_stations):
    """Abort if the download is missing any in-scope registered BaP station.
    This is the guard against the /urls endpoint silently returning a subset
    (see module docstring). Returns the authoritative registry for reporting."""
    registry = registry_bap_stations()
    expected = registry - EXCLUDE_SPO
    missing = expected - loaded_stations
    extra = loaded_stations - registry          # in cache but no longer registered
    print(f"EEA /List registry: {len(registry)} BaP stations "
          f"({len(EXCLUDE_SPO)} excluded by design) -> {len(expected)} in scope; "
          f"loaded {len(loaded_stations)}")
    if missing:
        raise RuntimeError(
            "EEA download is INCOMPLETE -- the following registered BaP stations "
            f"are missing from the loaded data: {sorted(missing)}. The "
            "/ParquetFile/urls endpoint likely returned a partial list. Delete "
            f"{CACHE} and re-run (the script will fall back to /ParquetFile), or "
            "add genuinely out-of-scope stations to EXCLUDE_SPO.")
    if extra:
        print(f"  NOTE: {sorted(extra)} are cached but not in the current "
              "registry (kept; verify they are still intended).")
    return registry


def main():
    # Cache-first: the EEA endpoint throttles repeated requests, so reuse the
    # local cache when present and only call the API to populate an empty cache.
    paths = cached_parquets()
    if paths:
        print(f"Using cached EEA parquet files: {len(paths)} (in {CACHE})")
    else:
        print(f"EEA email: {EMAIL}  (set EEA_EMAIL to override)")
        try:
            urls = fetch_url_list()
            print(f"EEA verified BaP files for SK (via /urls): {len(urls)}")
            paths = download(urls)
        except RuntimeError as exc:
            print(f"  /urls path failed ({exc}); falling back to /ParquetFile zip")
            paths = download_via_zip_fallback()
            print(f"EEA verified BaP files for SK (via fallback): {len(paths)}")
    eea = load_verified(paths)
    eea_years = set(eea["datum"].str[:4])
    eea_stations = set(eea["eoi"])
    # Hard completeness check against the authoritative registry BEFORE writing.
    verify_completeness(eea_stations)
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
