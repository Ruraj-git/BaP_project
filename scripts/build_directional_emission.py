#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directional industrial-emission covariate: wind-conditioned upwind BaP load from
industrial point sources (data/emis/Pointsources.GPKG).

Unlike the static residential term (emis_bap_log), this feature is TIME-VARYING:
a cell/station only "sees" a point source on days when the daily-mean wind blows
from that source toward the location. For location L on a day with wind unit
vector w_hat = (wdir_sin, wdir_cos)  [direction the wind blows TOWARD; from
aggregate_daily.py: wdir = arctan2(u, v) -> sin=u/|w| east, cos=v/|w| north]:

    w_dir_emis(L, day) = Σ_s  BaP_s · exp(-d_s / L0) · max(0,  w_hat · (L - s)/d_s)

i.e. each source's BaP load, distance-attenuated, weighted by how aligned the
wind is with the source->location bearing. Output feature is log1p of that sum.

This is the gridded/operational generalisation of the SK0018A prototype, which
showed mean BaP 3.3 (wind away) -> 7.9 (wind from the U.S. Steel coke oven) and
a partial corr (controlling dispersion+season) of 0.49.

Run as a script: appends `wdir_emis_bap_log` to v6 -> writes v7.
Import as a module: load_sources() + directional_emission() are reused by
fill_all_days.py for the per-station continuous product.
"""

import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402

POINTSOURCE_GPKG = os.path.join(config.DATA_DIR, "emis/Pointsources.GPKG")
LCC = config.PROJ4.rstrip(":")
L0 = 8000.0       # distance-decay length scale [m]
RMAX = 40000.0    # ignore sources beyond this radius [m]
FEATURE = "wdir_emis_bap_log"

_SRC_CACHE = None
_LL2LCC = Transformer.from_crs("EPSG:4326", LCC, always_xy=True)


def load_sources():
    """Return (src_x, src_y, bap) in LCC metres for all BaP>0 point sources."""
    global _SRC_CACHE
    if _SRC_CACHE is None:
        g = gpd.read_file(POINTSOURCE_GPKG)
        bap = pd.to_numeric(g["bap"], errors="coerce").fillna(0.0).values
        m = bap > 0
        tr = Transformer.from_crs(g.crs, LCC, always_xy=True)
        x, y = tr.transform(g.geometry.x.values[m], g.geometry.y.values[m])
        _SRC_CACHE = (np.asarray(x), np.asarray(y), bap[m])
    return _SRC_CACHE


def directional_emission(wdir_sin, wdir_cos, loc_x, loc_y, src=None):
    """Wind-conditioned upwind BaP load (raw, not logged) for one location.

    wdir_sin/wdir_cos: arrays (D,) of the daily wind unit vector (E, N).
    loc_x, loc_y:      scalar LCC coordinates of the location.
    """
    if src is None:
        src = load_sources()
    sx, sy, bap = src
    dx = loc_x - sx
    dy = loc_y - sy
    dist = np.hypot(dx, dy)
    keep = dist < RMAX
    if not keep.any():
        return np.zeros(len(wdir_sin))
    dx, dy, dist, bap = dx[keep], dy[keep], dist[keep], bap[keep]
    ux, uy = dx / dist, dy / dist                 # source -> location unit (E, N)
    decay = bap * np.exp(-dist / L0)              # (M,)
    ws = np.asarray(wdir_sin, dtype=float)
    wc = np.asarray(wdir_cos, dtype=float)
    align = ws[:, None] * ux[None, :] + wc[:, None] * uy[None, :]   # (D, M)
    return (np.maximum(0.0, align) * decay[None, :]).sum(axis=1)


def feature_for_location(wdir_sin, wdir_cos, lon, lat, src=None):
    """log1p directional feature for a single station given lon/lat + daily wind."""
    lx, ly = _LL2LCC.transform(lon, lat)
    return np.log1p(directional_emission(wdir_sin, wdir_cos, lx, ly, src))


def main():
    src = load_sources()
    print(f"🏭 {len(src[2])} BaP point sources, total {src[2].sum():.4f} kg/h")

    df = pd.read_csv(config.train_ready("v6"))
    df["eoi"] = df["eoi"].astype(str).str.strip()
    out = np.full(len(df), np.nan)
    for eoi, g in df.groupby("eoi"):
        lon, lat = g["lon"].iloc[0], g["lat"].iloc[0]
        vals = feature_for_location(g["wdir_sin"].values, g["wdir_cos"].values, lon, lat, src)
        out[df.index.get_indexer(g.index)] = vals
    df[FEATURE] = out

    nz = np.nansum(out > 0)
    print(f"   {FEATURE}: nonzero on {nz}/{len(df)} station-days "
          f"(max={np.nanmax(out):.3f}); by station (mean, top 5):")
    top = (df.assign(**{FEATURE: out}).groupby("eoi")[FEATURE]
           .mean().sort_values(ascending=False).head(5))
    print(top.round(4).to_string())

    out_path = config.train_ready("v7")
    df.to_csv(out_path, index=False)
    print(f"✅ train_ready_bap_v7.csv: {len(df)} rows, {df.shape[1]} cols (+1: {FEATURE})")


if __name__ == "__main__":
    main()
