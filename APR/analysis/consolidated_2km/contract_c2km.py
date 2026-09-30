#!/usr/bin/env python3
"""Frozen model/fold contract and task manifest for the consolidated 2 km ladder.

Specification: APR/revision/consolidation_2km_spec.md (approved 2026-09-24).
Feature groups and XGBoost settings are imported from the clean-rebuild contract
so they cannot drift; only the model definitions (input-group ladder) are new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import pandas as pd
import xgboost

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "APR/analysis/clean_rebuild"))
from model_contract import (METEO_CALENDAR, MONOTONE_DIRECTIONS, PM_XGB, PROXIES,  # noqa: E402
                            STATIC, XGB_PARAMS, monotone_tuple)
from build_fit_preflight import complete_season_years, season_year  # noqa: E402

CALENDAR = ["is_weekend", "month", "doy_sin", "doy_cos", "heating_season"]
PM6 = [f for f in PROXIES if not f.startswith("no2")]
NO2 = [f for f in PROXIES if f.startswith("no2")]
META = ["typ_oblasti_code", "typ_zdroja_code", "altitude"]
ENV_CAL = METEO_CALENDAR + STATIC  # environmental groups + calendar (order as former G)

FEATURES = {
    "PM": list(PM_XGB),                            # PM + calendar
    "P": PROXIES + CALENDAR,                       # pollutants + calendar
    "E": ENV_CAL,                                  # environmental + calendar
    "E_P": ENV_CAL + PROXIES,                      # principal model
    "E_P_M": METEO_CALENDAR + PROXIES + META + STATIC,
    "E_P_M_I": METEO_CALENDAR + PROXIES + META + STATIC,   # + station indicators in worker
    "PM_I": list(PM_XGB),                          # + station indicators in worker
    "E_PM": ENV_CAL + PM6,                         # programme-matched: no NO2
    "E_NO2": ENV_CAL + NO2,                        # programme-matched: no PM
}
# Reserve decomposition (spec addendum E): E+P with one environmental subgroup removed.
METEO = [f for f in METEO_CALENDAR if f not in CALENDAR]
_GROUPS = {"MET": METEO, "TER": STATIC[:6], "TRAF": STATIC[6:9], "EMIS": STATIC[9:11]}
for _g, _cols in _GROUPS.items():
    FEATURES[f"EP_NO{_g}"] = [f for f in ENV_CAL if f not in _cols] + PROXIES
INDICATOR_ARMS = {"E_P_M_I", "PM_I"}
EXPECTED_COUNTS = {"PM": 11, "P": 14, "E": 38, "E_P": 47, "E_P_M": 50, "E_P_M_I": 50,
                   "PM_I": 11, "E_PM": 44, "E_NO2": 41,
                   "EP_NOMET": 25, "EP_NOTER": 41, "EP_NOTRAF": 44, "EP_NOEMIS": 45}
MASK_SEED = 20260924
ORIGIN = pd.Timestamp("2023-06-02")
PRIMARY_YEARS = (2024, 2025)
INDUSTRIAL = "SK0018A"
SUPPORTS = ("2km_exact", "500m")  # static-covariate support; addendum B adds 500m


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def task_rows(d: pd.DataFrame, support: str = "2km_exact") -> list[dict]:
    rows = []

    def add(arm, protocol, fold, variant="permissive", target="log1p", weighting="uniform", augment="none"):
        f = FEATURES[arm]
        rows.append({"task_id": len(rows), "arm": arm, "protocol": protocol, "fold": str(fold),
                     "input_variant": variant, "static_support": support, "target": target,
                     "weighting": weighting, "augment": augment, "n_features": len(f),
                     "feature_order": "|".join(f), "monotone_tuple": "|".join(map(str, monotone_tuple(f)))})

    blocks = sorted(d.block.unique())
    stations = sorted(d.eoi.unique())
    for arm in ("PM", "P", "E", "E_P", "E_P_M", "E_P_M_I", "PM_I", "E_PM", "E_NO2"):
        for b in blocks:
            add(arm, "block30", b)
    for arm in ("PM", "P", "E", "E_P", "E_P_M", "E_PM", "E_NO2"):
        for s in stations:
            add(arm, "loso", s)
    for k in range(9):
        add("E_P", "dispersed_mod9", k)
    for arm in ("E", "E_P"):
        for y in PRIMARY_YEARS:
            add(arm, "year_out", y)
        for lab in complete_season_years(d):
            add(arm, "season_year_out", lab)
        for s in ("DJF", "MAM", "JJA", "SON"):
            add(arm, "calendar_season_out", s)
    sens = [dict(weighting="inverse_1_over_bap_plus_0_5"), dict(target="identity"),
            dict(variant="strict18"), dict(augment="mask")]
    for kw in sens:
        for b in blocks:
            add("E_P", "block30", b, **kw)
        for s in stations:
            add("E_P", "loso", s, **kw)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=ROOT / "APR/results/clean_rebuild/r02/inputs")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--support", choices=SUPPORTS, default="2km_exact")
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite {a.output}")
    other = [s for s in SUPPORTS if s != a.support][0]
    perm = a.inputs / f"train_ready_permissive_{a.support}.csv"
    strict = a.inputs / f"train_ready_strict18_{a.support}.csv"
    ref500 = a.inputs / f"train_ready_permissive_{other}.csv"
    d = pd.read_csv(perm, parse_dates=["datum"]).sort_values(["eoi", "datum"]).reset_index(drop=True)
    if len(d) != 6462 or d.eoi.nunique() != 21 or d.duplicated(["eoi", "datum"]).any():
        raise AssertionError("input keys/count invalid")
    for other in (strict, ref500):  # identical target records to r02
        o = pd.read_csv(other, parse_dates=["datum"]).sort_values(["eoi", "datum"]).reset_index(drop=True)
        if not d[["eoi", "datum", "bap"]].equals(o[["eoi", "datum", "bap"]]):
            raise AssertionError(f"target records differ: {other.name}")
    for name, f in FEATURES.items():
        if len(f) != EXPECTED_COUNTS[name] or len(set(f)) != len(f):
            raise AssertionError(f"feature count/duplicate failure: {name}")
        absent = sorted(set(f) - set(d))
        if absent:
            raise AssertionError(f"{name} missing features: {absent}")
    d["block"] = ((d.datum - ORIGIN).dt.days // 30).astype(int)
    d["month"] = d.datum.dt.month
    d["season_year"] = season_year(d.datum)
    p = d.datum.dt.year.isin(PRIMARY_YEARS)
    if (int(p.sum()), int((p & d.eoi.ne(INDUSTRIAL)).sum())) != (5089, 4852):
        raise AssertionError("primary population mismatch")
    for b in sorted(d.block.unique()):
        if not set(d.loc[d.block.eq(b), "eoi"]) <= set(d.loc[d.block.ne(b), "eoi"]):
            raise AssertionError(f"block {b}: test station unseen in training")
    tasks = task_rows(d, a.support)
    a.output.mkdir(parents=True)
    pd.DataFrame(tasks).to_csv(a.output / "fit_manifest.csv", index=False)
    by = pd.DataFrame(tasks).groupby(["arm", "protocol"]).size()
    contract = {
        "specification": "APR/revision/consolidation_2km_spec.md", "static_support": a.support,
        "input_files": {x.name: {"path": str(x.relative_to(ROOT)), "sha256": sha(x)} for x in (perm, strict)},
        "feature_contract": FEATURES, "indicator_arms": sorted(INDICATOR_ARMS),
        "monotone_directions": MONOTONE_DIRECTIONS,
        "resolved_monotone_tuples": {k: list(monotone_tuple(v)) for k, v in FEATURES.items()},
        "xgboost_parameters": XGB_PARAMS, "target_reference": "log1p(bap)",
        "prediction_reference": "maximum(0, expm1(prediction))",
        "initial_prediction_rule": "unweighted mean of training targets on the fitted scale, explicit base_score",
        "mask_augmentation": {"seed": MASK_SEED, "groups": {"PM": PM6, "NO2": NO2, "all": PM6 + NO2},
                              "rule": "each training row duplicated once with one group masked, chosen uniformly; test rows unchanged"},
        "folds": {"block_origin": str(ORIGIN.date()), "block_days": 30, "n_blocks": int(d.block.nunique()),
                  "dispersed_rule": "(datum - 2023-06-02) mod 9", "stations": sorted(d.eoi.unique()),
                  "complete_season_years": complete_season_years(d)},
        "tasks": {"n": len(tasks), "by_arm_protocol": {f"{k[0]}|{k[1]}": int(v) for k, v in by.items()}},
        "software": {"python": platform.python_version(), "xgboost": xgboost.__version__},
        "code_hashes": {n: sha(Path(__file__).parent / n) for n in ("contract_c2km.py", "fit_worker_c2km.py")},
    }
    (a.output / "model_and_fold_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(json.dumps({"tasks": len(tasks)}, indent=2))
    print(by.to_string())


if __name__ == "__main__":
    main()
