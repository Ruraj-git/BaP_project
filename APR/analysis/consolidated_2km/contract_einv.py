#!/usr/bin/env python3
"""Manifest for the inverse-weighted environmental model E (spec addendum F): 30-day and LOSO folds,
500 m static support, otherwise reference settings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from contract_c2km import FEATURES, ORIGIN, monotone_tuple

HERE = Path(__file__).parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True, help="500 m manifests directory (contract template)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    contract = json.loads((a.source / "model_and_fold_contract.json").read_text())
    root = HERE.parents[2]
    d = pd.read_csv(root / contract["input_files"]["train_ready_permissive_500m.csv"]["path"], parse_dates=["datum"])
    blocks = sorted(((d.datum - ORIGIN).dt.days // 30).unique())
    stations = sorted(d.eoi.unique())
    f = FEATURES["E"]
    rows = []
    for prot, folds in (("block30", blocks), ("loso", stations)):
        for fold in folds:
            rows.append({"task_id": len(rows), "arm": "E", "protocol": prot, "fold": str(fold),
                         "input_variant": "permissive", "static_support": "500m", "target": "log1p",
                         "weighting": "inverse_1_over_bap_plus_0_5", "augment": "none", "n_features": len(f),
                         "feature_order": "|".join(f), "monotone_tuple": "|".join(map(str, monotone_tuple(f)))})
    contract["feature_contract"] = FEATURES
    contract["tasks"] = {"n": len(rows)}
    contract["sensitivity"] = "inverse-weighted E (spec addendum F)"
    contract["code_hashes"] = {n: sha(HERE / n) for n in ("contract_c2km.py", "fit_worker_c2km.py", "contract_einv.py")}
    a.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(a.output / "fit_manifest.csv", index=False)
    (a.output / "model_and_fold_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(len(rows), "tasks")


if __name__ == "__main__":
    main()
