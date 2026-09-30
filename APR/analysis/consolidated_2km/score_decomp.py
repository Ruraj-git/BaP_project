#!/usr/bin/env python3
"""Score the reserve bundle decomposition: each E+P-minus-group arm minus reference E+P (500 m),
nonindustrial stations, unchanged scoring_core bootstrap."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "APR/analysis/clean_rebuild"))
from scoring_core import paired, scores  # noqa: E402

IND, KEYS = "SK0018A", ["eoi", "datum"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    c = pd.read_csv(a.candidate / "validated_predictions.csv", parse_dates=["datum"])
    r = pd.read_csv(a.reference / "validated_predictions.csv", parse_dates=["datum"])
    r = r[(r.arm == "E_P") & (r.input_variant == "permissive") & (r.target == "log1p") & (r.weighting == "uniform")
          & (r.augment == "none")]
    rows = []
    for arm in sorted(c.arm.unique()):
        for prot in ("block30", "loso"):
            x = c[(c.arm == arm) & (c.protocol == prot) & (c.datum.dt.year >= 2024) & (c.eoi != IND)]
            y = r[(r.protocol == prot) & (r.datum.dt.year >= 2024) & (r.eoi != IND)].merge(x[KEYS], on=KEYS, validate="one_to_one")
            assert len(x) == len(y) == 4852
            q = paired(x, y, KEYS, "station_x_block" if prot == "block30" else "station", "RMSE")
            rows.append(dict(arm=arm, protocol=prot, estimand="daily", rmse_candidate=scores(x.observed, x.prediction)["RMSE"],
                             rmse_reference=scores(y.observed, y.prediction)["RMSE"], **q))
            ax = x.assign(year=x.datum.dt.year).groupby(["eoi", "year"], as_index=False).agg(
                observed=("observed", "mean"), prediction=("prediction", "mean"), n=("observed", "size"))
            ay = y.assign(year=y.datum.dt.year).groupby(["eoi", "year"], as_index=False).agg(
                observed=("observed", "mean"), prediction=("prediction", "mean"), n=("observed", "size"))
            ax, ay = ax[ax.n >= 20], ay[ay.n >= 20]
            q = paired(ax, ay, ["eoi", "year"], "station", "RMSE")
            rows.append(dict(arm=arm, protocol=prot, estimand="station_year_mean",
                             rmse_candidate=scores(ax.observed, ax.prediction)["RMSE"],
                             rmse_reference=scores(ay.observed, ay.prediction)["RMSE"], **q))
    a.output.mkdir(parents=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.output / "decomp_contrasts.csv", index=False)
    print(out[["arm", "protocol", "estimand", "rmse_reference", "rmse_candidate", "estimate", "ci_low", "ci_high"]].round(3).to_string())


if __name__ == "__main__":
    main()
