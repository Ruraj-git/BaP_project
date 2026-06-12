#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-2 statické priestorové kovariáty na ALADIN 2 km gride (215x110).

Vstupy (lokálne):
  - data/DMR3.5/dmr3_5_10.tif   – 10 m DEM, S-JTSK/Krovák (EPSG:5514)
  - data/roads/TrafficCounts_ATMOPlan.shp – cestné segmenty, EPSG:3035,
    s počtami vozidiel (car/LDV/HDV/bus)

Výstupy:
  - output/spatial/grid_covariates.csv – príznaky pre VŠETKÝCH 215x110 buniek
    (pre budúcu virtuálnu sieť)
  - terénne + dopravné príznaky pridané k staniciam a do train_ready_bap_v5.csv

Geometria gridu (z config.py): uzol (i,j) má LCC súradnicu
    x = XORIG + i*DX ,  y = YORIG + j*DY
Raster vytvorený gdalwarp-om má pixel-centre v uzloch; riadok rastra r = NY-1-j.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from scipy.ndimage import uniform_filter
from scipy.spatial import cKDTree

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402

# --- Geometria cieľového gridu (uzly) ---
XORIG = config.XORIG
YORIG = config.YORIG
DX, DY, NX, NY = config.DX, config.DY, config.NX, config.NY
LCC_PROJ4 = config.PROJ4.rstrip(":")  # odstránenie chybnej dvojbodky na konci

# gdalwarp extent tak, aby pixel-centre boli presne v uzloch
TE = (XORIG - DX / 2, YORIG - DY / 2, XORIG + (NX - 0.5) * DX, YORIG + (NY - 0.5) * DY)

DEM_PATH = os.path.join(config.DATA_DIR, "DMR3.5/dmr3_5_10.tif")
ROADS_PATH = os.path.join(config.DATA_DIR, "roads/TrafficCounts_ATMOPlan.shp")
SPATIAL_DIR = config.COVARIATES_DIR            # static covariates -> shared under data/
TMP_DIR = os.path.join(SPATIAL_DIR, "tmp")

GDALWARP = "gdalwarp"  # v PATH (conda geo env)


def _warp_dem(resample):
    """Resampluje DEM na 2 km LCC grid danou metódou; vráti pole [NY,NX] s j-orientáciou."""
    os.makedirs(TMP_DIR, exist_ok=True)
    out = os.path.join(TMP_DIR, f"dem_{resample}.tif")
    cmd = [GDALWARP, "-overwrite", "-q",
           "-t_srs", LCC_PROJ4,
           "-te", *map(str, TE),
           "-ts", str(NX), str(NY),
           "-r", resample,
           "-dstnodata", "-9999",
           DEM_PATH, out]
    subprocess.run(cmd, check=True)
    with rasterio.open(out) as ds:
        arr = ds.read(1).astype(float)  # riadok 0 = sever (ymax)
    arr[arr == -9999] = np.nan
    # Otočenie na j-orientáciu: index [j, i], j rastie na sever  -> riadok r = NY-1-j
    return arr[::-1, :]


def build_terrain():
    print("⛰️  Terén z DMR 3.5 ...")
    elev = _warp_dem("average")    # priemerná nadm. výška bunky
    elev_min = _warp_dem("min")
    elev_max = _warp_dem("max")

    relief = elev_max - elev_min   # sub-grid členitosť (m)

    # TPI / hĺbka údolia vo viacerých mierkach (uniform_filter na 2 km poli)
    def tpi(field, size):
        # NaN-bezpečný kĺzavý priemer
        valid = np.isfinite(field).astype(float)
        f = np.nan_to_num(field, nan=0.0)
        local_mean = uniform_filter(f, size=size, mode="nearest") / \
            np.maximum(uniform_filter(valid, size=size, mode="nearest"), 1e-6)
        return field - local_mean

    tpi_local = tpi(elev, 3)    # ~6 km okno
    tpi_meso = tpi(elev, 7)     # ~14 km
    tpi_broad = tpi(elev, 15)   # ~30 km

    # Sklon z gradientu 2 km poľa (°)
    gy, gx = np.gradient(np.nan_to_num(elev, nan=np.nanmean(elev)), DY, DX)
    slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))

    return {
        "elev_mean": elev,
        "elev_relief": relief,
        "tpi_local": tpi_local,   # záporné = dno údolia (pasca pre BaP)
        "tpi_meso": tpi_meso,
        "tpi_broad": tpi_broad,
        "slope_deg": slope,
    }


def build_traffic():
    print("🚗 Doprava z TrafficCounts ...")
    gdf = gpd.read_file(ROADS_PATH).to_crs(LCC_PROJ4)

    # Reprezentatívny bod segmentu + emisne vážená záťaž (HDV emituje viac PM/PAH)
    pts = gdf.geometry.representative_point()
    gdf["mx"], gdf["my"] = pts.x.values, pts.y.values
    for c in ["car", "LDV", "HDV", "bus"]:
        gdf[c] = pd.to_numeric(gdf[c], errors="coerce").fillna(0)
    # emisné váhy ~ relatívny príspevok k PM/PAH (HDV >> osobné)
    gdf["load"] = (gdf["car"] + 1.5 * gdf["LDV"] + 8.0 * gdf["HDV"] + 5.0 * gdf["bus"]) \
        * gdf["length_m"]

    # Indexy buniek pre každý segment
    gi = np.round((gdf["mx"].values - XORIG) / DX).astype(int)
    gj = np.round((gdf["my"].values - YORIG) / DY).astype(int)
    inside = (gi >= 0) & (gi < NX) & (gj >= 0) & (gj < NY)

    traf_load = np.zeros((NY, NX))
    traf_hdv = np.zeros((NY, NX))
    np.add.at(traf_load, (gj[inside], gi[inside]), gdf["load"].values[inside])
    np.add.at(traf_hdv, (gj[inside], gi[inside]),
              (gdf["HDV"].values * gdf["length_m"].values)[inside])

    # Vzdialenosť k najbližšej významnej (HDV) ceste – km
    major = gdf[gdf["HDV"] > gdf["HDV"].quantile(0.90)]
    tree = cKDTree(np.c_[major["mx"].values, major["my"].values])
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY))
    cell_x = XORIG + ii * DX
    cell_y = YORIG + jj * DY
    dist, _ = tree.query(np.c_[cell_x.ravel(), cell_y.ravel()])
    dist_major = (dist / 1000.0).reshape(NY, NX)

    # log1p záťaže (silne pravostranné rozdelenie)
    return {
        "traffic_load_log": np.log1p(traf_load),
        "traffic_hdv_log": np.log1p(traf_hdv),
        "dist_major_road_km": dist_major,
    }


def main():
    os.makedirs(SPATIAL_DIR, exist_ok=True)
    layers = {}
    layers.update(build_terrain())
    layers.update(build_traffic())

    feat_names = list(layers.keys())

    # --- Tabuľka pre celý grid (virtuálna sieť) ---
    ii, jj = np.meshgrid(np.arange(NX), np.arange(NY))
    grid = pd.DataFrame({"xi": ii.ravel(), "yi": jj.ravel()})
    for k, v in layers.items():
        grid[k] = v[jj.ravel(), ii.ravel()]
    grid.to_csv(os.path.join(SPATIAL_DIR, "grid_covariates.csv"), index=False)
    print(f"💾 grid_covariates.csv: {len(grid)} buniek, {len(feat_names)} príznakov")

    # --- Sampling na stanice (stations.csv má x,y = uzly gridu) ---
    st = pd.read_csv(config.STATIONS_CSV)
    for k, v in layers.items():
        st[k] = v[st["y"].astype(int).values, st["x"].astype(int).values]

    # Sanity check: DEM výška vs. metadátová altitude
    valid = st[["elev_mean", "altitude"]].dropna()
    r = np.corrcoef(valid["elev_mean"], valid["altitude"])[0, 1]
    print(f"🔎 Kontrola geo-referencie: corr(DEM_elev, metadata_altitude) = {r:.3f} "
          f"(n={len(valid)})")

    st_out = st[["eoi"] + feat_names]
    st_out.to_csv(os.path.join(SPATIAL_DIR, "station_covariates.csv"), index=False)

    # --- Pripojenie k train_ready_v4 -> v5 ---
    tr = pd.read_csv(config.train_ready("v4"))
    tr = pd.merge(tr, st_out, on="eoi", how="left")
    out_path = config.train_ready("v5")
    tr.to_csv(out_path, index=False)
    print(f"✅ train_ready_bap_v5.csv: {len(tr)} riadkov, {tr.shape[1]} stĺpcov "
          f"(+{len(feat_names)} priestorových)")
    print("   Nové príznaky:", feat_names)


if __name__ == "__main__":
    main()
