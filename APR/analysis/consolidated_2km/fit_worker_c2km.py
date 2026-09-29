#!/usr/bin/env python3
"""One manifest-defined XGBoost fit for the consolidated 2 km ladder.

Same estimator, target, weights, initial prediction and holdout rules as
APR/analysis/clean_rebuild/fit_worker.py; additions: station indicators for
arms in the contract's indicator list, and the masking augmentation (§6a).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "APR/analysis/clean_rebuild"))
sys.path.insert(0, str(Path(__file__).parent))
from fit_worker import holdout, key_hash, sha_file  # noqa: E402
from model_contract import XGB_PARAMS, monotone_tuple  # noqa: E402
from contract_c2km import FEATURES, INDICATOR_ARMS, MASK_SEED, NO2, PM6  # noqa: E402


def mask_augment(train: pd.DataFrame) -> pd.DataFrame:
    """Duplicate each training row once with one pollutant group masked (PM, NO2 or all)."""
    rng = np.random.default_rng(MASK_SEED)
    copy = train.copy()
    choice = rng.integers(0, 3, len(copy))
    copy.loc[choice == 0, PM6] = np.nan
    copy.loc[choice == 1, NO2] = np.nan
    copy.loc[choice == 2, PM6 + NO2] = np.nan
    return pd.concat([train, copy], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tasks = pd.read_csv(a.manifest / "fit_manifest.csv")
    contract = json.loads((a.manifest / "model_and_fold_contract.json").read_text())
    if a.task_id not in set(tasks.task_id):
        raise ValueError("task id absent")
    row = tasks.loc[tasks.task_id.eq(a.task_id)].iloc[0]
    if a.dry_run:
        print(row.to_json())
        return
    name = f"train_ready_{row.input_variant}_{row.static_support}.csv"
    inp = ROOT / contract["input_files"][name]["path"]
    if sha_file(inp) != contract["input_files"][name]["sha256"]:
        raise ValueError("frozen input hash mismatch")
    if FEATURES != contract["feature_contract"] or XGB_PARAMS != contract["xgboost_parameters"]:
        raise ValueError("code differs from frozen model contract")
    if "|".join(FEATURES[row.arm]) != row.feature_order:
        raise ValueError("task feature order mismatch")
    d = pd.read_csv(inp)
    d.datum = pd.to_datetime(d.datum)
    d = d.sort_values(["eoi", "datum"]).reset_index(drop=True)
    held = holdout(d, row, d.datum.min())
    if not held.any() or held.all():
        raise ValueError("invalid holdout")
    train, test = d.loc[~held].copy(), d.loc[held].copy()
    excl = str(row.get('train_exclude', '') or '')  # optional training-station exclusion (sensitivity)
    if excl and excl != 'nan':
        train = train[~train.eoi.isin(excl.split('|'))].copy()
    features = list(FEATURES[row.arm])
    if row.arm in INDICATOR_ARMS:
        if set(test.eoi) - set(train.eoi):
            raise ValueError("test station absent from training for station-indicator model")
        names = [f"station__{s}" for s in sorted(train.eoi.unique())]
        for n in names:
            s = n.removeprefix("station__")
            train[n] = (train.eoi == s).astype(int)
            test[n] = (test.eoi == s).astype(int)
        features += names
    if row.augment == "mask":
        train = mask_augment(train)
    elif row.augment != "none":
        raise ValueError(f"unknown augment {row.augment}")
    params = dict(XGB_PARAMS)
    constraint = monotone_tuple(list(FEATURES[row.arm])) + tuple(0 for _ in range(len(features) - len(FEATURES[row.arm])))
    params["monotone_constraints"] = constraint
    y = train.bap.to_numpy(float)
    target = y if row.target == "identity" else np.log1p(y)
    weights = None if row.weighting == "uniform" else 1.0 / (y + 0.5)
    params["base_score"] = float(target.mean())
    model = xgb.XGBRegressor(**params)
    model.fit(train[features], target, sample_weight=weights)
    raw = model.predict(test[features])
    untruncated = raw if row.target == "identity" else np.expm1(raw)
    pred = np.maximum(0, untruncated)
    if not np.isfinite(pred).all():
        raise ValueError("non-finite prediction")
    if a.output is None:
        raise ValueError("output required unless dry-run")
    a.output.mkdir(parents=True, exist_ok=False)
    out = a.output / f"task_{a.task_id:03d}.csv"
    pd.DataFrame({"eoi": test.eoi, "datum": test.datum.dt.strftime("%Y-%m-%d"), "observed": test.bap,
                  "prediction": pred, "prediction_untruncated": untruncated, "prediction_target_scale": raw,
                  "task_id": a.task_id, "arm": row.arm, "protocol": row.protocol, "fold": row.fold,
                  "input_variant": row.input_variant, "target": row.target, "weighting": row.weighting,
                  "augment": row.augment}).to_csv(out, index=False)
    meta = {"task": row.to_dict(), "input": str(inp.relative_to(ROOT)), "input_sha256": sha_file(inp),
            "features": features, "resolved_monotone_tuple": list(constraint),
            "effective_xgboost_parameters": model.get_params(), "train_n": len(train), "test_n": len(test),
            "train_key_hash": key_hash(train), "test_key_hash": key_hash(test), "prediction_sha256": sha_file(out),
            "zero_floor_count": int((untruncated < 0).sum()), "python": platform.python_version(),
            "xgboost": xgb.__version__, "manifest_sha256": sha_file(a.manifest / "fit_manifest.csv"),
            "contract_sha256": sha_file(a.manifest / "model_and_fold_contract.json"),
            "worker_sha256": sha_file(Path(__file__))}
    (a.output / f"task_{a.task_id:03d}.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(json.dumps({"task_id": a.task_id, "arm": row.arm, "protocol": row.protocol,
                      "train_n": len(train), "test_n": len(test)}, indent=2))


if __name__ == "__main__":
    main()
