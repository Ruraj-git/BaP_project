#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option-A virtual-network map: 2024 annual-mean modelled B[a]P at all 54
network sites (monitored + virtual), with target-exceedance highlighted.

Reads the filled daily series (output/*_filled.csv) produced by fill_all_days.py,
averages bap_filled over calendar year 2024 per site, and plots the sites on a
lon/lat map. Monitored vs virtual sites use different markers; sites whose
2024 mean exceeds the 1 ng/m3 target value are ringed.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402

TARGET = 1.0  # EU target value for BaP annual mean [ng m^-3]

st = pd.read_csv(config.STATIONS_CSV)
st["eoi"] = st["eoi"].astype(str).str.strip()
# "Monitored" = stations with B[a]P observations within the study period
# (2023-06 onward). Three grid sites were monitored only historically
# (pre-2022) and have no in-period data, so they are unmonitored for our
# period and shown as virtual (predictions only).
STUDY_START = "2023-06-01"
_obs = pd.read_csv(config.OBSERVATIONS_CSV)
_obs["eoi"] = _obs["eoi"].astype(str).str.strip()
_obs = _obs[(_obs["datum"] >= STUDY_START) & _obs["bap"].notna()]
monitored = set(_obs["eoi"].unique())

# The map shows the 2024 annual mean only: 2024 is the most recent year with
# fully verified EEA reference observations (2025 remains pre-release, with
# December instrument outages at part of the network), and the partial 2023
# (missing the Jan-Mar winter peak) would bias the annual mean downward.
# The 2024 meteorological archive lacks 10 January days (source ALADIN gap:
# Jan 1-9 + Jan 31); imputation shifts near-threshold site means by <0.05 and
# would flip only Banska Bystrica SK0214A (0.998; observed 1.01) above target,
# so the mapped exceedance count is conservative.
MEAN_YEARS = ("2024",)
rows = []
for f in sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*_filled.csv"))):
    eoi = os.path.basename(f).replace("_filled.csv", "")
    d = pd.read_csv(f)
    dm = d[d["datum"].str[:4].isin(MEAN_YEARS)]
    rows.append({"eoi": eoi, "mean_bap": dm["bap_filled"].mean(),
                 "ndays": len(dm), "yr0": dm["datum"].min()[:4], "yr1": dm["datum"].max()[:4]})
m = pd.DataFrame(rows).merge(st[["eoi", "name", "lat", "lon", "typ_oblasti", "typ_zdroja"]], on="eoi")
m["monitored"] = m["eoi"].isin(monitored)
yr0, yr1 = m["yr0"].min(), m["yr1"].max()
_period = yr0 if yr0 == yr1 else f"{yr0}–{yr1}"
print(f"{len(m)} sites, {m['monitored'].sum()} monitored / {(~m['monitored']).sum()} virtual; "
      f"mean range {m['mean_bap'].min():.2f}-{m['mean_bap'].max():.2f}; "
      f"exceedances (>{TARGET}): {(m['mean_bap']>TARGET).sum()}")

fig, ax = plt.subplots(figsize=(10, 6))

# Slovakia national border (Natural Earth 10m, lon/lat) for geographic context.
try:
    import cartopy.io.shapereader as shpreader
    import geopandas as gpd
    _shp = shpreader.natural_earth(resolution="10m", category="cultural",
                                   name="admin_0_countries")
    _svk = gpd.read_file(_shp)
    _svk = _svk[_svk["ADMIN"] == "Slovakia"]
    _svk.boundary.plot(ax=ax, edgecolor="0.35", linewidth=1.2, zorder=1)
except Exception as e:  # border is decorative; never let it break the figure
    print(f"[warn] could not draw Slovakia border: {e}")

vmax = 2.0  # cap for contrast; the one industrial outlier (~4.5 ng/m3) saturates the top colour
cmap = plt.get_cmap("YlOrRd")
for mon, marker, lbl in [(True, "o", "monitored"), (False, "^", "virtual")]:
    sub = m[m["monitored"] == mon]
    sc = ax.scatter(sub["lon"], sub["lat"], c=sub["mean_bap"].clip(upper=vmax), cmap=cmap,
                    vmin=0, vmax=vmax, s=120, marker=marker, edgecolors="black",
                    linewidths=0.5, label=lbl, zorder=3)
# ring the target exceedances
exc = m[m["mean_bap"] > TARGET]
ax.scatter(exc["lon"], exc["lat"], s=320, facecolors="none", edgecolors="blue",
           linewidths=1.6, zorder=4, label=f"> {TARGET:.0f} ng m$^{{-3}}$ (target)")

cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02, extend="max")
cb.set_label(f"Annual-mean B[a]P {_period} [ng m$^{{-3}}$]")
ax.set_xlabel("Longitude [°E]")
ax.set_ylabel("Latitude [°N]")
ax.set_title(f"Modelled benzo[a]pyrene across the virtual monitoring network ({_period})")
ax.set_aspect(1.0 / np.cos(np.deg2rad(m["lat"].mean())))
ax.grid(True, ls=":", alpha=0.4)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
fig.tight_layout()
out = os.path.join(config.PLOTS_DIR, "virtual_network_map.png")
fig.savefig(out, dpi=300)
print(f"-> {out}")
