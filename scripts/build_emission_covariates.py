#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-2 covariates: bottom-up residential BaP emission + population, on the
ALADIN 2 km grid. Appends them to the v5 dataset -> v6, and to the grid /
station covariate tables. Also runs a quick geo-reference sanity check.

Inputs:
  data/emis/SurfaceSources.gpkg  – bottom-up residential emissions, EPSG:3035,
       per-polygon flux (g h-1 m-2) for bap, pm25, ... (sector=Residential)
  data/ghsl/GHS_POP_..._54009_1000_V1_0.tif – GHS-POP 1 km, Mollweide (ESRI:54009)

Output features:
  emis_bap_log   log1p of total residential BaP emission per cell (g h-1)
  emis_pm25_log  log1p of total residential PM2.5 emission per cell
  pop_log        log1p of GHS population per cell
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from pyproj import Transformer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402

XORIG, YORIG = config.XORIG, config.YORIG
DX, DY, NX, NY = config.DX, config.DY, config.NX, config.NY
LCC_PROJ4 = config.PROJ4.rstrip(":")
TE = (XORIG - DX / 2, YORIG - DY / 2, XORIG + (NX - 0.5) * DX, YORIG + (NY - 0.5) * DY)

EMIS_GPKG = os.path.join(config.DATA_DIR, "emis/SurfaceSources.gpkg")
GHSPOP_TIF = os.path.join(config.DATA_DIR, "ghsl/GHS_POP_E2025_GLOBE_R2023A_54009_1000_V1_0.tif")
SPATIAL_DIR = config.COVARIATES_DIR            # static covariates -> shared under data/
TMP_DIR = os.path.join(SPATIAL_DIR, "tmp")

EMISSION_FEATURES = ["emis_bap_log", "emis_pm25_log"]
POP_FEATURES = ["pop_log"]


def build_emission():
    """Sum per-polygon emission (flux x area) into 2 km cells."""
    print("🔥 Bottom-up residential emissions ...")
    gdf = gpd.read_file(EMIS_GPKG)  # EPSG:3035 (equal-area -> area in m^2 is valid)
    area = gdf.geometry.area.values  # m^2
    bap = pd.to_numeric(gdf["bap"], errors="coerce").fillna(0).values * area   # g/h per polygon
    pm25 = pd.to_numeric(gdf["pm25"], errors="coerce").fillna(0).values * area

    pts = gdf.geometry.representative_point()
    tr = Transformer.from_crs(gdf.crs, LCC_PROJ4, always_xy=True)
    x, y = tr.transform(pts.x.values, pts.y.values)
    gi = np.round((x - XORIG) / DX).astype(int)
    gj = np.round((y - YORIG) / DY).astype(int)
    inside = (gi >= 0) & (gi < NX) & (gj >= 0) & (gj < NY)

    bap_grid = np.zeros((NY, NX))
    pm_grid = np.zeros((NY, NX))
    np.add.at(bap_grid, (gj[inside], gi[inside]), bap[inside])
    np.add.at(pm_grid, (gj[inside], gi[inside]), pm25[inside])
    return {"emis_bap_log": np.log1p(bap_grid), "emis_pm25_log": np.log1p(pm_grid)}


def build_population():
    print("👥 GHS-POP population ...")
    os.makedirs(TMP_DIR, exist_ok=True)
    out = os.path.join(TMP_DIR, "pop_2km.tif")
    subprocess.run(["gdalwarp", "-overwrite", "-q",
                    "-t_srs", LCC_PROJ4, "-te", *map(str, TE),
                    "-ts", str(NX), str(NY), "-r", "sum", "-dstnodata", "0",
                    GHSPOP_TIF, out], check=True)
    with rasterio.open(out) as ds:
        arr = ds.read(1).astype(float)  # row 0 = north
    arr[arr < 0] = 0.0
    return {"pop_log": np.log1p(arr[::-1, :])}  # flip to j-orientation


def main():
    layers = {}
    layers.update(build_emission())
    layers.update(build_population())
    feat = list(layers.keys())

    # Update grid covariates (all 215x110 cells)
    grid_path = os.path.join(SPATIAL_DIR, "grid_covariates.csv")
    grid = pd.read_csv(grid_path)
    for k, v in layers.items():
        grid[k] = v[grid["yi"].astype(int).values, grid["xi"].astype(int).values]
    grid.to_csv(grid_path, index=False)

    # Sample at stations + extend station covariates
    st = pd.read_csv(config.STATIONS_CSV)
    st["eoi"] = st["eoi"].astype(str).str.strip()
    for k, v in layers.items():
        st[k] = v[st["y"].astype(int).values, st["x"].astype(int).values]

    cov_path = os.path.join(SPATIAL_DIR, "station_covariates.csv")
    cov = pd.read_csv(cov_path)
    cov = cov.merge(st[["eoi"] + feat], on="eoi", how="left")
    cov.to_csv(cov_path, index=False)

    # Sanity check: emission vs station type / observed BaP
    obs = pd.read_csv(config.OBSERVATIONS_CSV)
    obs["eoi"] = obs["eoi"].astype(str).str.strip()
    omean = obs.groupby("eoi")["bap"].mean()
    chk = st.set_index("eoi")
    chk["bap_obs_mean"] = omean
    chk = chk.dropna(subset=["bap_obs_mean"])
    r = np.corrcoef(chk["emis_bap_log"], chk["bap_obs_mean"])[0, 1]
    print(f"🔎 Sanity: corr(emis_bap_log, mean observed BaP) = {r:.3f} (n={len(chk)})")
    print("   by type (mean emis_bap_log):")
    print(st.assign(typ=st['typ_oblasti']+st['typ_zdroja']).groupby('typ')['emis_bap_log']
          .mean().round(2).to_string())

    # Merge into training table v5 -> v6
    tr = pd.read_csv(config.train_ready("v5"))
    tr = tr.merge(st[["eoi"] + feat], on="eoi", how="left")
    out_path = config.train_ready("v6")
    tr.to_csv(out_path, index=False)
    print(f"✅ train_ready_bap_v6.csv: {len(tr)} rows, {tr.shape[1]} cols (+{len(feat)}: {feat})")


if __name__ == "__main__":
    main()
