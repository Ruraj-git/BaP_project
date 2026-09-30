#!/usr/bin/env python3
"""Manifest for the reserve environmental-bundle decomposition (spec addendum E): E+P with one environmental
subgroup removed, 30-day blocks and LOSO folds, 500 m static support, reference settings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from contract_c2km import FEATURES, ORIGIN, monotone_tuple

HERE = Path(__file__).parent
ARMS = ("EP_NOMET", "EP_NOTER", "EP_NOTRAF", "EP_NOEMIS")


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
    ROOT = HERE.parents[2]
    d = pd.read_csv(ROOT / contract["input_files"]["train_ready_permissive_500m.csv"]["path"], parse_dates=["datum"])
    blocks = sorted(((d.datum - ORIGIN).dt.days // 30).unique())
    stations = sorted(d.eoi.unique())
    rows = []
    for arm in ARMS:
        f = FEATURES[arm]
        for prot, folds in (("block30", blocks), ("loso", stations)):
            for fold in folds:
                rows.append({"task_id": len(rows), "arm": arm, "protocol": prot, "fold": str(fold),
                             "input_variant": "permissive", "static_support": "500m", "target": "log1p",
                             "weighting": "uniform", "augment": "none", "n_features": len(f),
                             "feature_order": "|".join(f), "monotone_tuple": "|".join(map(str, monotone_tuple(f)))})
    contract["feature_contract"] = FEATURES
    contract["tasks"] = {"n": len(rows)}
    contract["sensitivity"] = "reserve environmental-bundle decomposition (spec addendum E)"
    contract["code_hashes"] = {n: sha(HERE / n) for n in ("contract_c2km.py", "fit_worker_c2km.py", "contract_decomp.py")}
    a.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(a.output / "fit_manifest.csv", index=False)
    (a.output / "model_and_fold_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(len(rows), "tasks")


if __name__ == "__main__":
    main()
