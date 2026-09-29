#!/usr/bin/env python3
"""Validate all consolidated-ladder fit artifacts (either static support) and assemble prediction rows.

Mirrors APR/analysis/clean_rebuild/assemble_validate.py: every task must exist
exactly once; inputs, training/test keys, estimator settings, initial prediction,
features, constraints, floors, targets and checksums are re-verified.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "APR/analysis/clean_rebuild"))
sys.path.insert(0, str(Path(__file__).parent))
from fit_worker import holdout, key_hash, sha_file  # noqa: E402
from model_contract import monotone_tuple  # noqa: E402
from contract_c2km import FEATURES, INDICATOR_ARMS  # noqa: E402
from fit_worker_c2km import mask_augment  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--fit-glob", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    tasks = pd.read_csv(a.manifest / "fit_manifest.csv")
    contract = json.loads((a.manifest / "model_and_fold_contract.json").read_text())
    worker_sha = sha_file(Path(__file__).parent / "fit_worker_c2km.py")
    found = {}
    for directory in glob.glob(a.fit_glob):
        js, cs = glob.glob(f"{directory}/*.json"), glob.glob(f"{directory}/*.csv")
        if len(js) != 1 or len(cs) != 1:
            raise AssertionError(f"invalid artifact count in {directory}")
        meta = json.loads(Path(js[0]).read_text())
        tid = int(meta["task"]["task_id"])
        if tid in found:
            raise AssertionError(f"duplicate task artifact {tid}")
        found[tid] = (Path(cs[0]), meta)
    expected = set(tasks.task_id.astype(int))
    if set(found) != expected:
        raise AssertionError(f"missing={sorted(expected - set(found))[:10]} extra={sorted(set(found) - expected)[:10]}")
    cache, rows, audit = {}, [], []
    for _, row in tasks.iterrows():
        tid = int(row.task_id)
        csv, meta = found[tid]
        name = f"train_ready_{row.input_variant}_{row.static_support}.csv"
        inp = ROOT / contract["input_files"][name]["path"]
        if name not in cache:
            if sha_file(inp) != contract["input_files"][name]["sha256"]:
                raise AssertionError("input differs from freeze")
            d = pd.read_csv(inp)
            d.datum = pd.to_datetime(d.datum)
            cache[name] = d.sort_values(["eoi", "datum"]).reset_index(drop=True)
        d = cache[name]
        if meta["input"] != str(inp.relative_to(ROOT)) or meta["input_sha256"] != contract["input_files"][name]["sha256"]:
            raise AssertionError(f"task {tid}: input provenance mismatch")
        if meta["task"] != row.to_dict():
            raise AssertionError(f"task {tid}: metadata differs from manifest")
        held = holdout(d, row, d.datum.min())
        train, test = d.loc[~held].copy(), d.loc[held].copy()
        excl = str(row.get('train_exclude', '') or '')  # optional training-station exclusion (sensitivity)
        if excl and excl != 'nan':
            train = train[~train.eoi.isin(excl.split('|'))].copy()
        features = list(FEATURES[row.arm])
        if row.arm in INDICATOR_ARMS:
            features += [f"station__{s}" for s in sorted(train.eoi.unique())]
        if row.augment == "mask":
            train = mask_augment(train)
        if meta["train_key_hash"] != key_hash(train) or meta["test_key_hash"] != key_hash(test):
            raise AssertionError(f"task {tid}: key hash mismatch")
        if meta["prediction_sha256"] != sha_file(csv):
            raise AssertionError(f"task {tid}: prediction checksum mismatch")
        if meta["manifest_sha256"] != sha_file(a.manifest / "fit_manifest.csv") or \
                meta["contract_sha256"] != sha_file(a.manifest / "model_and_fold_contract.json"):
            raise AssertionError("manifest/contract checksum mismatch")
        if meta["worker_sha256"] != worker_sha or worker_sha != contract["code_hashes"]["fit_worker_c2km.py"]:
            raise AssertionError("worker checksum mismatch")
        if any(meta["effective_xgboost_parameters"].get(k) != v for k, v in contract["xgboost_parameters"].items()):
            raise AssertionError("estimator settings mismatch")
        y = train.bap.to_numpy(float)
        fit_target = y if row.target == "identity" else np.log1p(y)
        if not np.isclose(meta["effective_xgboost_parameters"]["base_score"], fit_target.mean(), rtol=1e-12):
            raise AssertionError(f"task {tid}: initial prediction mismatch")
        cons = list(monotone_tuple(list(FEATURES[row.arm]))) + [0] * (len(features) - len(FEATURES[row.arm]))
        if meta["features"] != features or meta["resolved_monotone_tuple"] != cons:
            raise AssertionError(f"task {tid}: feature/constraint mismatch")
        p = pd.read_csv(csv)
        p.datum = pd.to_datetime(p.datum)
        if not np.isfinite(p[["prediction", "prediction_untruncated", "prediction_target_scale"]]).all().all():
            raise AssertionError(f"task {tid}: non-finite prediction")
        if not np.allclose(p.prediction, np.maximum(0, p.prediction_untruncated)) or \
                int((p.prediction_untruncated < 0).sum()) != meta["zero_floor_count"]:
            raise AssertionError(f"task {tid}: floor mismatch")
        ek = test[["eoi", "datum"]].sort_values(["eoi", "datum"]).reset_index(drop=True)
        ak = p[["eoi", "datum"]].sort_values(["eoi", "datum"]).reset_index(drop=True)
        if p.duplicated(["eoi", "datum"]).any() or not ak.equals(ek):
            raise AssertionError(f"task {tid}: held-out key mismatch")
        tgt = test[["eoi", "datum", "bap"]].merge(p[["eoi", "datum", "observed"]], on=["eoi", "datum"], validate="one_to_one")
        if not np.allclose(tgt.bap, tgt.observed, atol=1e-12, rtol=0):
            raise AssertionError(f"task {tid}: target mismatch")
        rows.append(p)
        audit.append({"task_id": tid, "n_test": len(p), "prediction_sha256": meta["prediction_sha256"],
                      "feature_count_effective": len(features)})
    out = pd.concat(rows, ignore_index=True)
    a.output.mkdir(parents=True)
    out.to_csv(a.output / "validated_predictions.csv", index=False)
    pd.DataFrame(audit).to_csv(a.output / "task_audit.csv", index=False)
    report = {"tasks_validated": len(audit), "prediction_rows": len(out),
              "by_arm_protocol": {f"{k[0]}|{k[1]}": int(v) for k, v in out.groupby(["arm", "protocol"]).size().items()}}
    (a.output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"tasks_validated": len(audit), "prediction_rows": len(out)}, indent=2))


if __name__ == "__main__":
    main()
