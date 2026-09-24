#!/usr/bin/env python3
"""Freeze feature contracts and deterministic temporal/spatial fold assignments."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost

from model_contract import FEATURES, G, G_PLUS_P, M_AUX, MONOTONE_DIRECTIONS, XGB_PARAMS, monotone_tuple

ROOT = Path(__file__).resolve().parents[3]
INDUSTRIAL, PRIMARY_YEARS, BLOCK_DAYS = "SK0018A", (2024, 2025), 30
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def season_year(dt: pd.Series) -> pd.Series:
    return (dt.dt.year + (dt.dt.month == 12).astype(int)).astype(str) + "_" + dt.dt.month.map(SEASONS)

def complete_season_years(d: pd.DataFrame) -> list[str]:
    labels = []
    available_start, available_end = d.datum.min().normalize(), d.datum.max().normalize()
    for label, g in d.groupby("season_year", sort=True):
        year, season = label.split("_")
        year = int(year)
        bounds = {
            "DJF": (pd.Timestamp(year - 1, 12, 1), pd.Timestamp(year, 2, 28 if year % 4 else 29)),
            "MAM": (pd.Timestamp(year, 3, 1), pd.Timestamp(year, 5, 31)),
            "JJA": (pd.Timestamp(year, 6, 1), pd.Timestamp(year, 8, 31)),
            "SON": (pd.Timestamp(year, 9, 1), pd.Timestamp(year, 11, 30)),
        }[season]
        required = {12, 1, 2} if season == "DJF" else {3,4,5} if season == "MAM" else {6,7,8} if season == "JJA" else {9,10,11}
        if set(g.month.unique()) == required and bounds[0] >= available_start and bounds[1] <= available_end:
            labels.append(label)
    return labels

def task_rows(d: pd.DataFrame) -> list[dict]:
    rows=[]
    def add(arm, protocol, fold, input_variant, support, target="log1p", weighting="uniform"):
        features=FEATURES[arm]
        rows.append({"task_id": len(rows), "arm": arm, "protocol":protocol, "fold":str(fold), "input_variant":input_variant, "static_support":support, "target":target, "weighting":weighting, "n_features":len(features), "feature_order":"|".join(features), "monotone_tuple":"|".join(map(str,monotone_tuple(features)))})
    blocks=sorted(d.block.unique())
    stations=sorted(d.eoi.unique())
    for arm in ("PM_XGB", "FULL_ID", "G", "G_PLUS_P"):
        support="500m" if arm in {"G","G_PLUS_P"} else "2km_exact"
        for fold in blocks: add(arm,"block30",fold,"permissive",support)
    for weighting in ("uniform", "inverse_1_over_bap_plus_0_5"):
        for fold in blocks: add("M_AUX", "block30", fold, "permissive", "2km_exact", weighting=weighting)
        for fold in range(9): add("M_AUX", "dispersed_mod9", fold, "permissive", "2km_exact", weighting=weighting)
    for arm in ("G", "G_PLUS_P"):
        for station in stations: add(arm,"loso",station,"permissive","500m")
    for station in stations: add("G_PLUS_P","loso",station,"strict18","500m")
    for fold in blocks: add("G_PLUS_P","block30",fold,"strict18","500m")
    for arm in ("G_PLUS_P",):
        for fold in blocks: add(arm,"block30",fold,"permissive","500m",target="identity")
        for station in stations: add(arm,"loso",station,"permissive","500m",target="identity")
    for year in PRIMARY_YEARS: 
        for arm in ("G", "G_PLUS_P"): add(arm,"year_out",year,"permissive","500m")
    for label in complete_season_years(d):
        for arm in ("G", "G_PLUS_P"): add(arm,"season_year_out",label,"permissive","500m")
    for season in ("DJF","MAM","JJA","SON"):
        for arm in ("G", "G_PLUS_P"): add(arm,"calendar_season_out",season,"permissive","500m")
    return rows

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--input-2km",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(f"Refusing to overwrite {a.output}")
    d=pd.read_csv(a.input); d.datum=pd.to_datetime(d.datum,errors="raise"); d=d.sort_values(["eoi","datum"]).reset_index(drop=True)
    d2=pd.read_csv(a.input_2km); d2.datum=pd.to_datetime(d2.datum,errors="raise"); d2=d2.sort_values(["eoi","datum"]).reset_index(drop=True)
    if len(d)!=6462 or d.eoi.nunique()!=21 or d.duplicated(["eoi","datum"]).any(): raise AssertionError("input keys/count invalid")
    if not d[["eoi","datum","bap"]].equals(d2[["eoi","datum","bap"]]): raise AssertionError("500m/2km target keys differ")
    d["year"]=d.datum.dt.year; d["month"]=d.datum.dt.month; d["season"]=d.month.map(SEASONS); d["season_year"]=season_year(d.datum); d["block"]=((d.datum-d.datum.min()).dt.days//BLOCK_DAYS).astype(int); d["disp_fold"]=((d.datum-d.datum.min()).dt.days % 9).astype(int); d["loso_station"]=d.eoi
    for name, f in FEATURES.items():
        absent=sorted(set(f)-set(d));
        if absent: raise AssertionError(f"{name} missing features: {absent}")
    if len(G)!=38 or len(G_PLUS_P)!=47 or len(M_AUX)!=50: raise AssertionError("feature-count contract failure")
    if set(G)&set(FEATURES["G_PLUS_P"][38:]): raise AssertionError("G/G+P overlap error")
    p=d.year.isin(PRIMARY_YEARS)
    if (int(p.sum()),int((p & d.eoi.ne(INDUSTRIAL)).sum()),int((p & d.eoi.eq(INDUSTRIAL)).sum())) != (5089,4852,237): raise AssertionError("primary population mismatch")
    for fold in sorted(d.block.unique()):
        train_stations=set(d.loc[d.block.ne(fold), "eoi"])
        test_stations=set(d.loc[d.block.eq(fold), "eoi"])
        if not test_stations.issubset(train_stations):
            raise AssertionError(f"block {fold} would leave a station-indicator level unseen in training")
    tasks=task_rows(d)
    a.output.mkdir(parents=True)
    d[["eoi","datum","bap","year","month","season","season_year","block","disp_fold","loso_station"]].to_csv(a.output/"row_fold_assignments.csv",index=False)
    pd.DataFrame(tasks).to_csv(a.output/"fit_manifest.csv",index=False)
    contract={"inputs":{"500m": {"path":str(a.input),"sha256":sha(a.input)}, "2km_exact": {"path":str(a.input_2km),"sha256":sha(a.input_2km)}},"feature_contract":FEATURES,"monotone_directions":MONOTONE_DIRECTIONS,"resolved_monotone_tuples":{k:list(monotone_tuple(v)) for k,v in FEATURES.items()},"xgboost_parameters":XGB_PARAMS,"target_reference":"log1p(bap)","prediction_reference":"maximum(0, expm1(prediction))","primary":{"years":list(PRIMARY_YEARS),"all":5089,"nonindustrial":4852,"industrial":237},"folds":{"block_origin":str(d.datum.min().date()),"block_days":BLOCK_DAYS,"n_blocks":int(d.block.nunique()),"dispersed_rule":"(datum - 2023-06-02) mod 9","stations":sorted(d.eoi.unique()),"complete_season_years":complete_season_years(d)},"tasks":{"n":len(tasks),"by_protocol":pd.Series([r['protocol'] for r in tasks]).value_counts().to_dict()},"software":{"python":platform.python_version(),"xgboost":xgboost.__version__}}
    contract["input_files"]={}
    for variant in ("permissive", "strict18"):
        for support in ("500m", "2km_exact"):
            path=a.input.parent/f"train_ready_{variant}_{support}.csv"
            variant_df=pd.read_csv(path)
            variant_df.datum=pd.to_datetime(variant_df.datum)
            variant_df=variant_df.sort_values(["eoi","datum"]).reset_index(drop=True)
            if not d[["eoi","datum","bap"]].equals(variant_df[["eoi","datum","bap"]]): raise AssertionError("variant target mismatch")
            contract["input_files"][path.name]={"path":str(path), "sha256":sha(path)}
    contract["initial_prediction_rule"]="unweighted mean of training targets on the fitted scale, explicit base_score"
    contract["station_encoding"]="PM_XGB and FULL_ID: sorted training-station one-hot columns; zero monotonic constraints"
    contract["uncertainty"]={"block30":"independent station and registered-block multinomial counts, product weights", "loso":"station clusters", "reconstruction":"station clusters retaining all years and folds", "seed":20260921, "replicates":2000}
    contract["code_hashes"]={name:sha(Path(__file__).parent/name) for name in ("model_contract.py","fit_worker.py")}
    (a.output/"model_and_fold_contract.json").write_text(json.dumps(contract,indent=2)+"\n")
    print(json.dumps({"tasks":len(tasks),"by_protocol":contract["tasks"]["by_protocol"],"season_year_folds":contract["folds"]["complete_season_years"]},indent=2))
if __name__=="__main__": main()
