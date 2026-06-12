#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TPI maps from the high-resolution DMR 3.5 (10 m) — SIDE QUEST, self-contained.

This is intentionally NOT wired into the meteo-model pipeline. It reads the raw
10 m DEM, resamples it to a fine map resolution (default 100 m, native EPSG:5514
S-JTSK/Krovak), and computes the Topographic Position Index (TPI) at several
neighbourhood scales — including a fine ~1 km scale that the 2 km meteo grid
cannot represent. Outputs (GeoTIFF + PNG) go under TPI/output/.

TPI(R) = elev - mean(elev within a circular neighbourhood of radius R)
   negative  -> valley floor / basin (cold-air pooling, B[a]P trap)
   positive  -> ridge / hilltop (well ventilated)
   ~0        -> flat ground or uniform slope

Delete the whole TPI/ folder to remove this side quest; nothing else depends on it.

Run:  python TPI/make_tpi_maps.py
"""

import os
import subprocess

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.signal import fftconvolve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, TwoSlopeNorm, ListedColormap, BoundaryNorm

# --------------------------------------------------------------------------- #
#  Config
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
DEM_10M = os.path.join(HERE, "..", "data", "DMR3.5", "dmr3_5_10.tif")
TMP_DIR = os.path.join(HERE, "tmp")
OUT_DIR = os.path.join(HERE, "output")

MAP_RES_M = 100          # output sampling resolution (10 m native -> 100 m)

# neighbourhood scales: (label, radius_km).  ~1 km is the high-res selling point;
# 6/14/30 km reproduce the model's tpi_local / tpi_meso / tpi_broad.
SCALES = [
    ("fine",  1.0),
    ("local", 6.0),
    ("meso",  14.0),
    ("broad", 30.0),
]

# Weiss landform classification uses a small + large standardized TPI.
WEISS_SMALL_KM = 1.0
WEISS_LARGE_KM = 10.0


# --------------------------------------------------------------------------- #
#  Step 1 — resample DMR 10 m -> MAP_RES_M (native CRS), cached
# --------------------------------------------------------------------------- #
def resample_dem():
    os.makedirs(TMP_DIR, exist_ok=True)
    out = os.path.join(TMP_DIR, f"dem_{MAP_RES_M}m.tif")
    if os.path.exists(out):
        print(f"  (cached) {out}")
        return out
    print(f"  gdalwarp 10 m -> {MAP_RES_M} m (average) ...")
    cmd = ["gdalwarp", "-overwrite", "-q",
           "-tr", str(MAP_RES_M), str(MAP_RES_M),
           "-r", "average",
           "-dstnodata", "-9999",
           "-co", "COMPRESS=DEFLATE", "-co", "TILED=YES",
           DEM_10M, out]
    subprocess.run(cmd, check=True)
    return out


# --------------------------------------------------------------------------- #
#  Step 2 — TPI via NaN-safe circular-window mean (FFT convolution)
# --------------------------------------------------------------------------- #
def disk_kernel(radius_px):
    r = int(round(radius_px))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    k = (x * x + y * y) <= r * r
    return k.astype(np.float32)


def windowed_mean(field, valid, kernel):
    """Mean of `field` over `kernel`, ignoring nodata (valid==0)."""
    num = fftconvolve(np.where(valid, field, 0.0), kernel, mode="same")
    den = fftconvolve(valid.astype(np.float32), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = num / den
    m[den < 0.5] = np.nan
    return m.astype(np.float32)


def compute_tpi(field, valid, radius_km):
    radius_px = (radius_km * 1000.0) / MAP_RES_M
    k = disk_kernel(radius_px)
    tpi = field - windowed_mean(field, valid, k)
    tpi[~valid] = np.nan
    return tpi


# --------------------------------------------------------------------------- #
#  GeoTIFF + PNG writers
# --------------------------------------------------------------------------- #
def write_geotiff(path, arr, profile):
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=-9999,
                compress="deflate", tiled=True)
    out = np.where(np.isfinite(arr), arr, -9999).astype(np.float32)
    with rasterio.open(path, "w", **prof) as ds:
        ds.write(out, 1)


def render_tpi_png(path, tpi, hillshade, title):
    # robust symmetric color limits
    lim = np.nanpercentile(np.abs(tpi), 98)
    lim = max(lim, 1.0)
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=130)
    ax.imshow(hillshade, cmap="gray", alpha=1.0, interpolation="nearest")
    im = ax.imshow(tpi, cmap="RdBu_r", norm=norm, alpha=0.62,
                   interpolation="nearest")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("TPI  (m)   ← valley / basin      ridge / hilltop →")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Weiss 10-class landform classification
# --------------------------------------------------------------------------- #
def standardize(tpi):
    return (tpi - np.nanmean(tpi)) / np.nanstd(tpi)


def weiss_landforms(tpi_small, tpi_large, slope_deg):
    sn = standardize(tpi_small)
    ln = standardize(tpi_large)
    cls = np.full(tpi_small.shape, 0, dtype=np.int16)  # 0 = nodata
    # Weiss (2001) 10-class slope-position scheme
    cls[(sn <= -1) & (ln <= -1)] = 1   # canyons / deeply incised streams
    cls[(sn <= -1) & (ln > -1) & (ln < 1)] = 2   # midslope drainages
    cls[(sn <= -1) & (ln >= 1)] = 3   # upland drainages / headwaters
    cls[(sn > -1) & (sn < 1) & (ln <= -1)] = 4   # U-shaped valleys
    cls[(sn > -1) & (sn < 1) & (ln > -1) & (ln < 1) & (slope_deg <= 5)] = 5   # plains
    cls[(sn > -1) & (sn < 1) & (ln > -1) & (ln < 1) & (slope_deg > 5)] = 6   # open slopes
    cls[(sn > -1) & (sn < 1) & (ln >= 1)] = 7   # upper slopes / mesas
    cls[(sn >= 1) & (ln <= -1)] = 8   # local ridges in valleys
    cls[(sn >= 1) & (ln > -1) & (ln < 1)] = 9   # midslope ridges
    cls[(sn >= 1) & (ln >= 1)] = 10  # mountain tops / high ridges
    return cls


WEISS_LABELS = {
    1: "Canyons / deep streams", 2: "Midslope drainages", 3: "Upland drainages",
    4: "U-shaped valleys", 5: "Plains", 6: "Open slopes",
    7: "Upper slopes / mesas", 8: "Local ridges in valleys",
    9: "Midslope ridges", 10: "Mountain tops / high ridges",
}
WEISS_COLORS = {
    1: "#08306b", 2: "#2171b5", 3: "#6baed6", 4: "#4eb3d3", 5: "#f7f7f7",
    6: "#d9d9a3", 7: "#fdae6b", 8: "#fb6a4a", 9: "#de2d26", 10: "#67000d",
}


def render_weiss_png(path, cls, hillshade):
    keys = list(range(1, 11))
    cmap = ListedColormap([WEISS_COLORS[k] for k in keys])
    norm = BoundaryNorm([0.5] + [k + 0.5 for k in keys], cmap.N)
    masked = np.where(cls >= 1, cls, np.nan)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=130)
    ax.imshow(hillshade, cmap="gray", interpolation="nearest")
    ax.imshow(masked, cmap=cmap, norm=norm, alpha=0.70, interpolation="nearest")
    ax.set_title("Weiss landform classification "
                 f"(TPI {WEISS_SMALL_KM:g} km + {WEISS_LARGE_KM:g} km)", fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=WEISS_COLORS[k]) for k in keys]
    ax.legend(handles, [WEISS_LABELS[k] for k in keys], loc="lower left",
              fontsize=7, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("⛰️  TPI maps from DMR 3.5 (side quest)")

    dem_path = resample_dem()
    with rasterio.open(dem_path) as ds:
        elev = ds.read(1).astype(np.float32)
        profile = ds.profile
        nodata = ds.nodata
    valid = np.isfinite(elev) & (elev != nodata) & (elev > -1000)
    elev = np.where(valid, elev, np.nan).astype(np.float32)
    print(f"  grid {elev.shape}  ({MAP_RES_M} m)   "
          f"elev {np.nanmin(elev):.0f}–{np.nanmax(elev):.0f} m")

    # hillshade backdrop
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(np.nan_to_num(elev, nan=np.nanmean(elev)),
                      vert_exag=8, dx=MAP_RES_M, dy=MAP_RES_M)

    # slope (deg) for the Weiss classification
    gy, gx = np.gradient(np.nan_to_num(elev, nan=np.nanmean(elev)),
                         MAP_RES_M, MAP_RES_M)
    slope = np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))

    tpi_cache = {}
    for label, rkm in SCALES:
        print(f"  TPI {label}  (~{rkm:g} km radius) ...")
        tpi = compute_tpi(elev, valid, rkm)
        tpi_cache[label] = tpi
        tif = os.path.join(OUT_DIR, f"tpi_{label}_{rkm:g}km_{MAP_RES_M}m.tif")
        png = os.path.join(OUT_DIR, f"tpi_{label}_{rkm:g}km_{MAP_RES_M}m.png")
        write_geotiff(tif, tpi, profile)
        render_tpi_png(png, tpi, hs,
                       f"TPI {label} — {rkm:g} km neighbourhood "
                       f"(DMR 3.5 @ {MAP_RES_M} m)")
        print(f"     -> {os.path.basename(tif)}, {os.path.basename(png)}")

    # Weiss landform map
    print("  Weiss landform classification ...")
    t_small = tpi_cache.get("fine") if WEISS_SMALL_KM == 1.0 \
        else compute_tpi(elev, valid, WEISS_SMALL_KM)
    t_large = compute_tpi(elev, valid, WEISS_LARGE_KM)
    cls = weiss_landforms(t_small, t_large, slope)
    cls[~valid] = 0
    write_geotiff(os.path.join(OUT_DIR, "landform_weiss.tif"),
                  cls.astype(np.float32), profile)
    render_weiss_png(os.path.join(OUT_DIR, "landform_weiss.png"), cls, hs)
    print("     -> landform_weiss.tif, landform_weiss.png")

    print(f"✅ done. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
