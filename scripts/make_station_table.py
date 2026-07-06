#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Appendix roster of all 54 network stations behind Fig.~\\ref{fig:vnmap}: station
metadata plus the 2024 modelled annual-mean B[a]P, the observed 2024 annual mean
where measured, and the (in-sample) model bias.

For every site we report:
  Station / EoI    identifier and name
  Type             area x source code (UB/UT/SB/ST/SI/RB)
  Alt              station altitude [m]
  M/V              Monitored (>=1 in-period B[a]P observation) or Virtual
  Proxies          which co-located proxies exist in 2024 (PM / NO2 / none)
  Model            2024 annual mean of the pure model prediction (all 366 days;
                   exactly the value mapped in Fig.~5 at every site)
  Obs              2024 annual mean of the B[a]P observations (monitored only)
  Bias             paired mean(model - obs) on observation days (monitored only)

The bias is IN-SAMPLE (the production model is trained on all observations, so
it validates the operational fill's internal consistency, not out-of-sample
skill -- see the per-station LOSO / block-gap MBE in Table~\\ref{tab:perstation}).
Fig.~5 (make_network_map.py) plots the same pure prediction at every site, so
the Model column IS the mapped value, monitored sites included; the operational
filled series (which retains observations where present) differs appreciably
only at the under-predicted industrial site SK0018A.

Outputs (into the active run's validation dir, alongside the figures):
  station_table.csv   machine-readable roster
  station_table.tex   longtable fragment, \\input by both manuscripts
Run: GAPFILL_RUN=base OMP_NUM_THREADS=16 python scripts/make_station_table.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import xgboost as xgb

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
import config  # noqa: E402
import validate_model as vm  # noqa: E402

YEAR = "2024"
TARGET = 1.0
STUDY_START = "2023-06-01"


def latex_escape(s):
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def clean_name(s):
    # names are "<town>, <street>, <note>" with commas as field separators;
    # some fields are empty (-> double commas) and a few carry a "mobil" tag.
    # Normalise internal whitespace, drop blank/"mobil" segments, rejoin.
    parts = [" ".join(p.split()) for p in str(s).split(",")]
    parts = [p for p in parts if p and p.lower() not in ("mobil", "mobile")]
    # strip a stray trailing period on the town field only (e.g. "Vranov nad
    # Topľou."); street/note fields keep legit abbreviations ("nám.", "st.")
    if parts and parts[0].endswith("."):
        parts[0] = parts[0][:-1].rstrip()
    return latex_escape(", ".join(parts))


def main():
    st = pd.read_csv(config.STATIONS_CSV)
    st["eoi"] = st["eoi"].astype(str).str.strip()

    obs = pd.read_csv(config.OBSERVATIONS_CSV)
    obs["eoi"] = obs["eoi"].astype(str).str.strip()
    monitored = set(obs[(obs["datum"] >= STUDY_START) & obs["bap"].notna()]["eoi"].unique())

    # Production model: identical recipe to fill_all_days.py (v6, monotone).
    df = pd.read_csv(config.train_ready("v6")).dropna(subset=["bap"]).copy()
    df["typ_oblasti_code"] = df["typ_oblasti"].str.strip().map(config.AREA_MAP).fillna(0)
    df["typ_zdroja_code"] = df["typ_zdroja"].astype("category").cat.codes
    feats = (list(vm.BASE_FEATURES)
             + [c for c in vm.GROUP1_FEATURES if c in df.columns]
             + [c for c in vm.GROUP2_FEATURES if c in df.columns]
             + [c for c in vm.CONC_LAG_FEATURES if c in df.columns]
             + [c for c in vm.EMISSION_FEATURES if c in df.columns])
    params = dict(config.MODEL_PARAMS)
    params["monotone_constraints"] = config.monotone_for(feats)
    model = xgb.XGBRegressor(**params)
    model.fit(df[feats], np.log1p(df["bap"]), sample_weight=1.0 / (df["bap"] + 0.5))

    def predict(X):
        return np.maximum(0.0, np.expm1(model.predict(X[feats])))

    rows = []
    for f in sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*_filled.csv"))):
        eoi = os.path.basename(f).replace("_filled.csv", "")
        d = pd.read_csv(f)
        d = d[d["datum"].str[:4] == YEAR]
        if d.empty:
            continue
        model_ym = float(predict(d).mean())
        ob = d[d["bap"].notna()]
        n = len(ob)
        obs_ym = float(ob["bap"].mean()) if n else np.nan
        bias = float(np.mean(predict(ob) - ob["bap"].values)) if n else np.nan
        prox = []
        if d["pm25_mean"].notna().any() or d["pm10_mean"].notna().any():
            prox.append("PM")
        if d["no2_mean"].notna().any():
            prox.append(r"NO$_2$")
        rows.append(dict(eoi=eoi, model_ym=model_ym, obs_ym=obs_ym, n_obs=n, bias=bias,
                         proxies="+".join(prox) if prox else "none"))

    m = pd.DataFrame(rows).merge(
        st[["eoi", "name", "lat", "lon", "altitude", "typ_oblasti", "typ_zdroja"]], on="eoi")
    m["MV"] = np.where(m["eoi"].isin(monitored), "M", "V")
    m["type"] = m["typ_oblasti"].str.strip() + m["typ_zdroja"].str.strip()
    m = m.sort_values("model_ym", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(config.VALIDATION_DIR, "station_table.csv")
    m.to_csv(csv_path, index=False)

    # --- LaTeX longtable fragment ---------------------------------------
    n_exc = int((m["model_ym"] > TARGET).sum())
    n_mon = int((m["MV"] == "M").sum())
    lines = []
    lines.append(r"% Auto-generated by scripts/make_station_table.py -- do not edit by hand.")
    lines.append(r"\begingroup\small")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\begin{longtable}{l >{\raggedright\arraybackslash}p{3.1cm} crccrrr}")
    lines.append(r"\caption{All \num{54} network stations behind Fig.~\ref{fig:vnmap}, "
                 r"ordered by modelled \num{2024} annual-mean \bap{}. "
                 r"Type: area (U urban, S suburban, R rural) $\times$ source "
                 r"(B background, T traffic, I industrial). "
                 r"M/V: monitored (\num{" + str(n_mon) + r"} sites with in-period \bap{} "
                 r"observations) or virtual. Proxies lists the co-located "
                 r"\pmt{}/\pmf{}/\nod{} terms present in \num{2024}. "
                 r"Model is the \num{2024} mean of the pure model prediction "
                 r"(all days; at every site identical to the value mapped in "
                 r"Fig.~\ref{fig:vnmap}); "
                 r"Obs is the observed \num{2024} annual mean and Bias the paired "
                 r"mean model$-$obs on observation days (both monitored only). "
                 r"The bias is in-sample -- for held-out per-station skill see "
                 r"Table~\ref{tab:perstation}. Bold marks the "
                 r"\num{" + str(n_exc) + r"} sites above the \num{1}\,\ngm{} target."
                 r"}\label{tab:stations}\\")
    lines.append(r"\toprule")
    header = (r"EoI & Station & Type & Alt & M/V & Proxies & "
              r"Model & Obs & Bias \\")
    unit = (r" & & & [m] & & & [\ngm] & [\ngm] & [\ngm] \\")
    lines.append(header)
    lines.append(unit)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{9}{l}{\small\itshape "
                 r"Table~\ref{tab:stations} continued}\\")
    lines.append(r"\toprule")
    lines.append(header)
    lines.append(unit)
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{9}{r}{\small\itshape continued on next page}\\")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, r in m.iterrows():
        exc = r["model_ym"] > TARGET
        model_s = f"{r['model_ym']:.2f}"
        if exc:
            model_s = r"\textbf{" + model_s + "}"
        obs_s = f"{r['obs_ym']:.2f}" if r["MV"] == "M" and r["n_obs"] > 0 else "--"
        bias_s = f"{r['bias']:+.2f}" if r["MV"] == "M" and r["n_obs"] > 0 else "--"
        lines.append(
            f"{r['eoi']} & {clean_name(r['name'])} & {r['type']} & "
            f"{int(round(r['altitude']))} & {r['MV']} & {r['proxies']} & "
            f"{model_s} & {obs_s} & {bias_s} \\\\")

    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")

    tex_path = os.path.join(config.VALIDATION_DIR, "station_table.tex")
    with open(tex_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"{len(m)} stations | {n_mon} monitored | {n_exc} exceed >{TARGET}")
    print(f"-> {csv_path}")
    print(f"-> {tex_path}")


if __name__ == "__main__":
    main()
