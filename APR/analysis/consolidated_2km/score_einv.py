#!/usr/bin/env python3
"""Score inverse-weighted E against E (uniform) and E+P (uniform), 20 nonindustrial stations (addendum F)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "APR/analysis/clean_rebuild"))
from scoring_core import paired, scores  # noqa: E402

IND, KEYS = "SK0018A", ["eoi", "datum"]


def annual(x):
    a = x.assign(year=x.datum.dt.year).groupby(["eoi", "year"], as_index=False).agg(
        observed=("observed", "mean"), prediction=("prediction", "mean"), n=("observed", "size"))
    a = a[a.n >= 20].copy()
    a["stratum"] = np.where(a.observed > 1, "higher", "lower")
    return a


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
    r = r[(r.input_variant == "permissive") & (r.target == "log1p") & (r.weighting == "uniform") & (r.augment == "none")]
    rows, summ = [], []
    for prot in ("block30", "loso"):
        sel = lambda d: d[(d.protocol == prot) & (d.datum.dt.year >= 2024) & (d.eoi != IND)]
        x = sel(c)
        refs = {"E": sel(r[r.arm == "E"]), "E_P": sel(r[r.arm == "E_P"])}
        frames = {"E_inv": x, **refs}
        for name, f in frames.items():
            s = scores(f.observed, f.prediction)
            summ.append(dict(model=name, protocol=prot, estimand="daily", **s))
            an = annual(f)
            s = scores(an.observed, an.prediction)
            summ.append(dict(model=name, protocol=prot, estimand="station_year_mean", **s))
            for lev in ("higher", "lower"):
                g = an[an.stratum == lev]
                summ.append(dict(model=name, protocol=prot, estimand=f"station_year_mean_{lev}",
                                 **scores(g.observed, g.prediction)))
        for ref_name, y in refs.items():
            y = y.merge(x[KEYS], on=KEYS, validate="one_to_one")
            assert len(x) == len(y) == 4852
            q = paired(x, y, KEYS, "station_x_block" if prot == "block30" else "station", "RMSE")
            rows.append(dict(contrast=f"E_inv_minus_{ref_name}", protocol=prot, estimand="daily", **q))
            ax, ay = annual(x), annual(y)
            q = paired(ax, ay.merge(ax[["eoi", "year"]], on=["eoi", "year"]), ["eoi", "year"], "station", "RMSE")
            rows.append(dict(contrast=f"E_inv_minus_{ref_name}", protocol=prot, estimand="station_year_mean", **q))
    a.output.mkdir(parents=True)
    pd.DataFrame(summ).to_csv(a.output / "einv_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(a.output / "einv_contrasts.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string())
    print(pd.DataFrame(rows)[["contrast", "protocol", "estimand", "estimate", "ci_low", "ci_high"]].round(3).to_string())


if __name__ == "__main__":
    main()
