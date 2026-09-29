#!/usr/bin/env python3
"""Scores, paired intervals and diagnostics for the consolidated 2 km ladder (spec §7, §6a).

Scoring functions (metrics, pigeonhole/station bootstrap, reconstructed means) are
imported unchanged from APR/analysis/clean_rebuild/scoring_core.py. Simple
comparator predictions (support-independent) are reused from r02.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "APR/analysis/clean_rebuild"))
from scoring_core import paired, reconstruct, scores  # noqa: E402

IND = "SK0018A"
PRIMARY = (2024, 2025)
R02 = ROOT / "APR/results/clean_rebuild/r02"
INV = "inverse_1_over_bap_plus_0_5"


def mid(arm, variant="permissive", target="log1p", weight="uniform", augment="none"):
    return "|".join([arm, target, weight, variant, augment])


REF = mid("E_P")
# (label, candidate, comparator, protocols)
CONTRASTS = [
    ("+NO2", mid("P"), mid("PM"), ("block30", "loso")),
    ("+E", mid("E_P"), mid("P"), ("block30", "loso")),
    ("+P", mid("E_P"), mid("E"), ("block30", "loso")),
    ("+M", mid("E_P_M"), mid("E_P"), ("block30", "loso")),
    ("+I", mid("E_P_M_I"), mid("E_P_M"), ("block30",)),
    ("+I (PM level)", mid("PM_I"), mid("PM"), ("block30",)),
    ("Bundle", mid("E_P_M_I"), mid("PM_I"), ("block30",)),
    ("+NO2 given E", mid("E_P"), mid("E_PM"), ("block30", "loso")),
    ("+PM given E", mid("E_P"), mid("E_NO2"), ("block30", "loso")),
    ("Weights", mid("E_P", weight=INV), REF, ("block30", "loso")),
    ("Target", mid("E_P", target="identity"), REF, ("block30", "loso")),
    ("Strict hours", mid("E_P", variant="strict18"), REF, ("block30", "loso")),
    ("Masking", mid("E_P", augment="mask"), REF, ("block30", "loso")),
    ("Programme-matched", "E_P_MATCHED", REF, ("block30", "loso")),
    ("Support", "ALT", REF, ("block30", "loso")),
    ("Stress", REF, mid("E"), ("year_out", "season_year_out", "calendar_season_out")),
]


def scopes(d):
    yield "all_network", d
    yield "nonindustrial", d[d.eoi.ne(IND)]
    yield "industrial_descriptive", d[d.eoi.eq(IND)]
    for level in ("higher", "lower"):
        yield "nonindustrial_" + level, d[d.eoi.ne(IND) & d.stratum.eq(level)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True, help="train-ready input of this run (pollutant availability, strata)")
    ap.add_argument("--alt-assembled", type=Path, required=True, help="assembled dir of the other-support run")
    ap.add_argument("--alt-label", required=True, help="method label for the other-support E+P, e.g. E_P_2KM")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    ALT = a.alt_label
    CONTRASTS[:] = [(l, ALT if m == "ALT" else m, r, p) for l, m, r, p in CONTRASTS]
    if a.output.exists():
        raise FileExistsError(a.output)
    d = pd.read_csv(a.assembled / "validated_predictions.csv", parse_dates=["datum"])
    d["method"] = d.arm + "|" + d.target + "|" + d.weighting + "|" + d.input_variant + "|" + d.augment
    frame = pd.read_csv(a.dataset, parse_dates=["datum"])
    avail = frame[["eoi", "datum"]].copy()
    avail["pm_avail"] = frame[["pm10_mean", "pm25_mean"]].notna().any(axis=1)
    avail["no2_avail"] = frame["no2_mean"].notna()
    avail["complete"] = frame[["pm10_mean", "pm25_mean", "no2_mean"]].notna().all(axis=1)

    # Programme-matched composite: pick the model matching same-day pollutant availability.
    comp = []
    for prot in ("block30", "loso"):
        base = d[(d.method == REF) & (d.protocol == prot)].merge(avail, on=["eoi", "datum"], validate="one_to_one")
        alt = {k: d[(d.method == mid(k)) & (d.protocol == prot)].set_index(["eoi", "datum"]).prediction
               for k in ("E_PM", "E_NO2", "E")}
        idx = base.set_index(["eoi", "datum"]).index
        pick = np.where(base.pm_avail & base.no2_avail, base.prediction,
                        np.where(base.pm_avail, alt["E_PM"].reindex(idx).to_numpy(),
                                 np.where(base.no2_avail, alt["E_NO2"].reindex(idx).to_numpy(),
                                          alt["E"].reindex(idx).to_numpy())))
        if not np.isfinite(pick).all():
            raise ValueError(f"programme-matched composite incomplete: {prot}")
        c = base[d.columns.drop("method").tolist()].copy()
        c["prediction"] = pick
        c["arm"], c["method"] = "E_P_MATCHED", "E_P_MATCHED"
        comp.append(c)
    # Static-support sensitivity: E+P from the other consolidated run (identical features and folds).
    alt = pd.read_csv(a.alt_assembled / "validated_predictions.csv", parse_dates=["datum"])
    alt = alt[(alt.arm == "E_P") & alt.protocol.isin(["block30", "loso"]) & (alt.input_variant == "permissive")
              & (alt.target == "log1p") & (alt.weighting == "uniform") & (alt.augment == "none")].copy()
    alt["method"], alt["arm"] = ALT, ALT
    d = pd.concat([d, *comp, alt[["eoi", "datum", "observed", "prediction", "arm", "protocol", "fold", "method"]]],
                  ignore_index=True)
    if d.duplicated(["method", "protocol", "eoi", "datum"]).any():
        raise ValueError("duplicated method predictions")
    d["year"] = d.datum.dt.year
    d = d.merge(avail[["eoi", "datum", "complete"]], on=["eoi", "datum"], how="left", validate="many_to_one")

    ref = frame[["eoi", "datum", "bap"]].rename(columns={"bap": "observed"})
    ref["year"] = ref.datum.dt.year
    primary = ref[ref.year.isin(PRIMARY)]
    strata = primary.groupby(["eoi", "year"], as_index=False).agg(sampled_mean=("observed", "mean"), n_sampled=("observed", "size"))
    strata["stratum"] = np.where(strata.sampled_mean > 1, "higher", "lower")

    def label(x):
        return x.merge(strata[["eoi", "year", "stratum"]], on=["eoi", "year"], how="left", validate="many_to_one")

    d = label(d)
    summaries, intervals, influence, station_rows, annual_rows = [], [], [], [], []

    def summarize(t, estimand, protocol, method, support):
        for scope, g in scopes(t):
            if len(g):
                summaries.append(dict(method=method, protocol=protocol, estimand=estimand, support=support, scope=scope,
                                      n=len(g), n_stations=g.eoi.nunique(), **scores(g.observed, g.prediction)))

    dp = d[d.year.isin(PRIMARY)]
    for (method, protocol), g in dp.groupby(["method", "protocol"]):
        summarize(g, "daily", protocol, method, "all_expected_targets")
        for station, st in g.groupby("eoi"):
            station_rows.append(dict(method=method, protocol=protocol, eoi=station, n=len(st), **scores(st.observed, st.prediction)))
        if protocol in ("block30", "loso"):
            an = g.groupby(["eoi", "year"], as_index=False).agg(observed=("observed", "mean"), prediction=("prediction", "mean"), n=("observed", "size"))
            an = an.merge(strata[["eoi", "year", "n_sampled"]], on=["eoi", "year"], validate="one_to_one")
            if not an.n.eq(an.n_sampled).all():
                raise ValueError(f"incomplete station-year support: {method} {protocol}")
            an = label(an[an.n >= 20])
            an["method"], an["protocol"] = method, protocol
            annual_rows.append(an)
            summarize(an, "station_year_mean", protocol, method, "all_sampled_dates")
    annual = pd.concat(annual_rows, ignore_index=True)

    def contrast(x, y, keys, scheme, lab, ma, mb, protocol, estimand, support, influence_ok=False):
        for scope, xx in scopes(x):
            if scope == "industrial_descriptive" or xx.eoi.nunique() < 2:
                continue
            yy = y.merge(xx[keys], on=keys, validate="one_to_one")
            if len(yy) != len(xx):
                raise ValueError(f"paired support differs: {lab} {protocol} {scope}")
            for metric in ("RMSE", "MAE", "bias"):
                try:
                    r = paired(xx, yy, keys, scheme, metric)
                except ValueError as e:  # sparse availability subsets: fall back to station clusters
                    if scheme != "station_x_block" or "empty bootstrap draw" not in str(e):
                        raise
                    r = paired(xx, yy, keys, "station", metric)
                intervals.append(dict(label=lab, method_a=ma, method_b=mb, protocol=protocol, estimand=estimand,
                                      support=support, scope=scope, **r))
            if influence_ok and scope == "nonindustrial":
                for s in sorted(xx.eoi.unique()):
                    ax, by = xx[xx.eoi.ne(s)], yy[yy.eoi.ne(s)]
                    influence.append(dict(label=lab, protocol=protocol, estimand=estimand, excluded_station=s,
                                          difference_RMSE=scores(ax.observed, ax.prediction)["RMSE"] - scores(by.observed, by.prediction)["RMSE"]))

    for lab, ma, mb, prots in CONTRASTS:
        for protocol in prots:
            x, y = dp[(dp.method == ma) & (dp.protocol == protocol)], dp[(dp.method == mb) & (dp.protocol == protocol)]
            if x.empty or y.empty:
                raise ValueError(f"missing predictions for {lab} {protocol}")
            scheme = "station_x_block" if protocol == "block30" else "station"
            contrast(x, y, ["eoi", "datum"], scheme, lab, ma, mb, protocol, "daily", "all_expected_targets",
                     influence_ok=lab in ("+E", "+P"))
            if protocol in ("block30", "loso"):
                ax_, ay_ = annual[(annual.method == ma) & (annual.protocol == protocol)], annual[(annual.method == mb) & (annual.protocol == protocol)]
                contrast(ax_, ay_, ["eoi", "year"], "station", lab, ma, mb, protocol, "station_year_mean", "all_sampled_dates",
                         influence_ok=lab in ("+E", "+P"))
            if lab in ("+E", "+P", "Masking", "Programme-matched") and protocol in ("block30", "loso"):
                for est, sel in (("daily_complete_pollutant_days", x.complete), ("daily_incomplete_pollutant_days", ~x.complete)):
                    xs = x[sel.fillna(False)]
                    if xs.eoi.nunique() >= 2:
                        contrast(xs, y, ["eoi", "datum"], scheme, lab, ma, mb, protocol, est, "pollutant_availability_subset")
                        summarize(xs, est, protocol, ma, "pollutant_availability_subset")
                        summarize(y.merge(xs[["eoi", "datum"]], on=["eoi", "datum"]), est, protocol, mb, "pollutant_availability_subset")

    # Gap tasks: E+P vs reused r02 simple comparators on common bracketable targets
    # (dispersed = pre-specified; 30-day reported for continuity with the current paper).
    base = pd.read_csv(R02 / "scores/baseline_predictions.csv", parse_dates=["datum"])
    recs = []
    for protocol in ("block30", "dispersed_mod9"):
        ep = dp[(dp.method == REF) & (dp.protocol == protocol)][["eoi", "datum", "observed", "prediction", "protocol", "fold"]].copy()
        ep["method"] = REF
        bp = base[base.protocol.eq(protocol)]
        gap = pd.concat([ep, bp[["eoi", "datum", "observed", "prediction", "protocol", "fold", "method"]]], ignore_index=True)
        gap["year"] = gap.datum.dt.year
        gap = label(gap[gap.year.isin(PRIMARY)])
        bracket = gap[gap.method.eq("INTERP_LINEAR") & gap.prediction.notna()][["eoi", "datum"]]
        common = gap.merge(bracket, on=["eoi", "datum"], validate="many_to_one")
        scheme = "station_x_block" if protocol == "block30" else "station"
        for method, g in common.groupby("method"):
            if not np.isfinite(g.prediction).all():
                raise ValueError("nonfinite common-support gap prediction")
            summarize(g, "daily", protocol, method, "common_bracketable")
            if protocol == "dispersed_mod9":
                r = label(reconstruct(primary, g))
                r["method"] = method
                recs.append(r)
                summarize(r, "reconstructed_mean", protocol, method, "common_bracketable_replacements")
        for mb in sorted(set(common.method) - {REF}):
            contrast(common[common.method.eq(REF)], common[common.method.eq(mb)], ["eoi", "datum"], scheme,
                     "Gap vs simple", REF, mb, protocol, "daily", "common_bracketable")
    rec = pd.concat(recs, ignore_index=True)
    for mb in sorted(set(rec.method) - {REF}):
        contrast(rec[rec.method.eq(REF)], rec[rec.method.eq(mb)], ["eoi", "year", "fold"], "station",
                 "Gap vs simple", REF, mb, "dispersed_mod9", "reconstructed_mean", "common_bracketable_replacements")

    a.output.mkdir(parents=True)
    out = {"summary_metrics": pd.DataFrame(summaries), "paired_uncertainty": pd.DataFrame(intervals),
           "station_metrics": pd.DataFrame(station_rows), "station_influence": pd.DataFrame(influence),
           "station_year_means": annual, "observed_strata": strata, "reconstructed_means": rec}
    for k, v in out.items():
        v.to_csv(a.output / f"{k}.csv", index=False)
    (a.output / "scoring_report.json").write_text(json.dumps({k: len(v) for k, v in out.items()}, indent=2) + "\n")
    print(json.dumps({k: len(v) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
