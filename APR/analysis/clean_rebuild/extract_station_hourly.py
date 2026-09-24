#!/usr/bin/env python3
"""Clean-rebuild ALADIN station extraction.

Uses the established supergeo/pygrib environment only for raw GRIB reading.
It writes a new station-hourly chunk, never an existing product.  Accumulations
are de-accumulated within each 00 UTC forecast cycle; instantaneous midnight
uses the previous cycle's lead 24 after valid-time verification.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pygrib

ROOT = Path(__file__).resolve().parents[3]
# Operational ALADIN/SHMU GRIB archive (not public); set ALADIN_GRIB_DIR to its location.
GRIB = Path(os.environ.get("ALADIN_GRIB_DIR", "aladin_grib"))
PIXELS = ROOT / "APR/analysis/final_validation/station_pixels.csv"
ACCUM = ("ssrd", "sshf", "cwp", "lswp", "csf", "lsf")
INSTANT = ("2t", "10u", "10v", "hpbl", "2r", "sp")
RAIN = ("cwp", "lswp", "csf", "lsf")


def path_for(day: datetime, lead: int) -> Path:
    return GRIB / f"{day:%Y/%m/%d}/00/ALA2ECMAQ+{lead:04d}.grb"


def metadata_ok(message, day: datetime, lead: int, accumulated: bool) -> None:
    if int(message["dataDate"]) != int(day.strftime("%Y%m%d")) or int(message["dataTime"]) != 0:
        raise ValueError(f"wrong forecast cycle for {message.shortName}, lead {lead}")
    if accumulated:
        unit = int(message["stepUnits"])
        expected_end = lead if unit == 1 else 60 * lead if unit == 0 else None
        if (str(message["stepType"]) != "accum" or expected_end is None or
                float(message["startStep"]) != 0 or float(message["endStep"]) != expected_end):
            raise ValueError(f"invalid accumulation metadata for {message.shortName}, lead {lead}")


def read_selected(path: Path, names: tuple[str, ...], y: np.ndarray, x: np.ndarray,
                  day: datetime | None = None, lead: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    picked: dict[str, np.ndarray] = {}
    duplicates: dict[str, int] = {}
    grbs = pygrib.open(str(path))
    try:
        for message in grbs:
            name = message.shortName
            if name not in names:
                continue
            if day is not None and lead is not None:
                metadata_ok(message, day, lead, name in ACCUM)
            values = message.values[y, x].astype(float)
            if name in picked:
                duplicates[name] = duplicates.get(name, 1) + 1
                if not np.allclose(values, picked[name], equal_nan=True, atol=0, rtol=0):
                    raise ValueError(f"non-identical duplicate message {name} in {path}")
                continue
            picked[name] = values
    finally:
        grbs.close()
    missing = sorted(set(names) - set(picked))
    if missing:
        raise ValueError(f"missing fields {missing} in {path}")
    return picked, duplicates


def extract_day(day: datetime, pixels: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y, x = pixels.full_y.to_numpy(int), pixels.full_x.to_numpy(int)
    prev = day - timedelta(days=1)
    midnight, duplicates = read_selected(path_for(prev, 24), INSTANT, y, x)
    # Explicit proof that all selected instantaneous fields are valid at this day's midnight.
    check = pygrib.open(str(path_for(prev, 24)))
    try:
        for message in check:
            if message.shortName in INSTANT:
                if int(message["validityDate"]) != int(day.strftime("%Y%m%d")) or int(message["validityTime"]) != 0:
                    raise ValueError(f"lead-24 not valid at midnight for {message.shortName}")
    finally:
        check.close()
    inst = [midnight]
    acc = {name: [] for name in ACCUM}
    for lead in range(1, 25):
        values, dups = read_selected(path_for(day, lead), INSTANT + ACCUM, y, x, day, lead)
        for name, count in dups.items():
            duplicates[name] = duplicates.get(name, 0) + count
        inst.append({name: values[name] for name in INSTANT})
        for name in ACCUM:
            acc[name].append(values[name])
    out = {name: np.vstack([r[name] for r in inst])[:24] for name in INSTANT}
    # A(1)-0 ... A(24)-A(23); every interval remains assigned to its forecast date.
    for name in ACCUM:
        cumulative = np.vstack(acc[name])
        increments = np.diff(np.vstack([np.zeros((1, len(pixels))), cumulative]), axis=0)
        out[name] = increments
    out["rain"] = sum(out[name] for name in RAIN)
    rows = []
    for hour in range(24):
        for j, eoi in enumerate(pixels.eoi):
            instant_row = {name: float(out[name][hour, j]) for name in INSTANT}
            # Native VRATE is absent in 2023.  Match the historical documented
            # definition across all years rather than mixing native/derived fields.
            instant_row["vrate"] = instant_row["hpbl"] * (instant_row["10u"] ** 2 + instant_row["10v"] ** 2) ** 0.5
            rows.append({"eoi": eoi, "time": day + timedelta(hours=hour), **instant_row,
                         "ssrd": float(out["ssrd"][hour, j]), "sshf": float(out["sshf"][hour, j]),
                         "rain": float(out["rain"][hour, j])})
    return pd.DataFrame(rows), {"date": day.strftime("%Y-%m-%d"), "duplicates": duplicates}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True); ap.add_argument("--end", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end)
    if end < start:
        raise ValueError("end precedes start")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    pixels = pd.read_csv(PIXELS)
    pieces, report = [], []
    day = start
    while day <= end:
        piece, info = extract_day(day, pixels)
        pieces.append(piece); report.append(info)
        day += timedelta(days=1)
    result = pd.concat(pieces, ignore_index=True)
    expected = (end - start).days + 1
    if len(result) != expected * 24 * len(pixels) or result.duplicated(["eoi", "time"]).any():
        raise AssertionError("unexpected station-hour coverage")
    output.mkdir(parents=True)
    result.to_csv(output / "station_hourly.csv", index=False)
    (output / "extraction_metadata.json").write_text(json.dumps({
        "start": args.start, "end": args.end, "stations": int(len(pixels)),
        "hours_per_station": int(expected * 24), "instantaneous_midnight": "previous_cycle_lead24",
        "accumulation": "A1_minus_0_through_A24_minus_A23_within_current_cycle",
        "duplicate_messages": report,
    }, indent=2) + "\n")
    print(f"wrote {len(result)} station-hours to {output}")


if __name__ == "__main__":
    main()
