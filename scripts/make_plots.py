#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots for the BaP gap-filling model (reads outputs of validate_model.py / fill_all_days.py).

Produces (all English):
  1) Per-station time series: observations vs. continuous gap-filled model,
     with skill statistics (LOSO r²/RMSE/MBE/n) in a text box.
     + one combined panel of all monitored stations.
  2) Statistical overviews (seaborn):
     - heatmap: station × {r², RMSE, MBE}
     - heatmap: station type × {r², RMSE, MBE}
     - sorted barplot of r² per station (coloured by type)
  3) Feature importance (gain), coloured by feature group.

NOTE: "r²" = squared Pearson correlation coefficient (see validate_model.py).
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402
from validate_model import (compute_metrics, BASE_FEATURES, GROUP1_FEATURES,  # noqa: E402
                            GROUP2_FEATURES, CONC_LAG_FEATURES)

VAL_DIR = config.VALIDATION_DIR
PLOT_DIR = config.PLOTS_DIR
TS_DIR = os.path.join(PLOT_DIR, "timeseries")
FILLED_DIR = config.OUTPUT_DIR
LIMIT = 1.0  # BaP target value [ng/m³]

sns.set_theme(style="whitegrid", context="notebook")

# --- Feature -> group mapping (for the importance plot) ---
TERRAIN = {"elev_mean", "elev_relief", "tpi_local", "tpi_meso", "tpi_broad",
           "slope_deg", "altitude"}
TRAFFIC = {"traffic_load_log", "traffic_hdv_log", "dist_major_road_km"}
PROXY = {"pm10_mean", "pm25_mean", "no2_mean"}
CALENDAR = {"is_weekend", "month", "doy_sin", "doy_cos", "heating_season"}
TYPOLOGY = {"typ_oblasti_code", "typ_zdroja_code"}
EMISSION = {"emis_bap_log", "emis_pm25_log", "pop_log", "wdir_emis_bap_log"}
BASE_METEO = {"t_mean", "heating_degree_hours", "vrate_min", "hpbl_min",
              "night_vrate_avg", "night_sshf_min", "total_rain", "ws_max"}


def feature_group(f):
    if f in PROXY:
        return "Proxy (same-day PM/NO₂)"
    if f in set(CONC_LAG_FEATURES):
        return "Concentration lag"
    if f in TERRAIN:
        return "Terrain (DEM)"
    if f in TRAFFIC:
        return "Traffic"
    if f in EMISSION:
        return "Emission (bottom-up)"
    if f in CALENDAR:
        return "Calendar/season"
    if f in TYPOLOGY:
        return "Station typology"
    if f in BASE_METEO:
        return "Meteorology (base)"
    return "Meteorology (derived)"


def load():
    st = pd.read_csv(config.STATIONS_CSV)
    st["eoi"] = st["eoi"].astype(str).str.strip()
    names = dict(zip(st["eoi"], st["name"].str.strip().str.rstrip(",")))
    typ = dict(zip(st["eoi"], st["typ_oblasti"].astype(str) + st["typ_zdroja"].astype(str)))

    oof = pd.read_csv(os.path.join(VAL_DIR, "oof_predictions.csv"))
    oof["datum"] = pd.to_datetime(oof["datum"])
    oof["typ"] = oof["typ_oblasti"].astype(str) + oof["typ_zdroja"].astype(str)
    loso_stats = {eoi: compute_metrics(g["bap"], g["pred_loso"])
                  for eoi, g in oof.groupby("eoi")}
    return names, typ, oof, loso_stats


def stats_text(m):
    return (f"LOSO r² = {m['R2']:.2f}\nRMSE = {m['RMSE']:.2f}\n"
            f"MBE = {m['MBE']:+.2f}\nn = {m['n']}")


def plot_one_timeseries(ax, dff, eoi, name, typ, stats):
    """dff = continuous gap-filled series; stats = LOSO metrics or None (virtual station)."""
    dff = dff.sort_values("datum")
    obs = dff[dff["is_observed"] == 1]
    ax.plot(dff["datum"], dff["bap_predicted"], "-", color="#d62728", lw=1.1,
            alpha=0.85, label="Model (all days filled)")
    ax.plot(obs["datum"], obs["bap"], "o", color="black", ms=3.2, label="Observed")
    ax.axhline(LIMIT, color="gray", ls="--", lw=0.8, label=f"Limit {LIMIT:g} ng/m³")
    txt = stats_text(stats) if stats is not None else "virtual station\n(no measurements)"
    ax.text(0.015, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    cover = 100.0 * len(obs) / len(dff)
    ax.set_title(f"{eoi} – {name}  [{typ}]   (observed {cover:.0f} % of days)", fontsize=9)
    ax.set_ylabel("BaP [ng/m³]")


def make_timeseries(names, typ, loso_stats):
    os.makedirs(TS_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(FILLED_DIR, "*_filled.csv")))

    for f in files:
        eoi = os.path.basename(f).replace("_filled.csv", "")
        dff = pd.read_csv(f)
        dff["datum"] = pd.to_datetime(dff["datum"])
        fig, ax = plt.subplots(figsize=(12, 4))
        plot_one_timeseries(ax, dff, eoi, names.get(eoi, eoi), typ.get(eoi, "?"),
                            loso_stats.get(eoi))
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(TS_DIR, f"{eoi}_timeseries.png"), dpi=130)
        plt.close(fig)

    monitored = sorted(loso_stats.keys())
    ncol = 3
    nrow = int(np.ceil(len(monitored) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 6, nrow * 2.4), squeeze=False)
    for k, eoi in enumerate(monitored):
        ax = axes[k // ncol][k % ncol]
        dff = pd.read_csv(os.path.join(FILLED_DIR, f"{eoi}_filled.csv"))
        dff["datum"] = pd.to_datetime(dff["datum"])
        plot_one_timeseries(ax, dff, eoi, names.get(eoi, eoi), typ.get(eoi, "?"),
                            loso_stats.get(eoi))
        ax.tick_params(labelsize=7)
    for k in range(len(monitored), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10)
    fig.suptitle("BaP: continuous gap-filled series + observations (monitored stations)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.02, 1, 0.99])
    fig.savefig(os.path.join(PLOT_DIR, "timeseries_overview.png"), dpi=120)
    plt.close(fig)
    print(f"📈 Time series: {len(files)} stations ({len(monitored)} monitored) "
          f"-> {TS_DIR} + timeseries_overview.png")


def per_station_table(oof, names, typ):
    rows = []
    for eoi, g in oof.groupby("eoi"):
        m = compute_metrics(g["bap"], g["pred_loso"])
        m.update({"eoi": eoi, "name": names.get(eoi, eoi), "typ": typ.get(eoi, "?")})
        rows.append(m)
    return pd.DataFrame(rows).set_index("eoi").sort_values("R2", ascending=False)


def heatmap_stations(df):
    """3-panel heatmap: station × {r², RMSE, MBE}, each with its own colour scale."""
    labels = [f"{i}  {df.loc[i, 'typ']}" for i in df.index]
    specs = [
        ("R2", "RdYlGn", 0, 1, "r² (Pearson)"),
        ("RMSE", "YlOrRd", None, None, "RMSE [ng/m³]"),
        ("MBE", "coolwarm", None, None, "MBE (bias) [ng/m³]"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, max(6, 0.42 * len(df))))
    for k, (ax, (col, cmap, vmin, vmax, title)) in enumerate(zip(axes, specs)):
        if col == "MBE":
            vmax = float(np.nanmax(np.abs(df[col]))); vmin = -vmax
        sns.heatmap(df[[col]].values, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    annot=True, fmt=".2f", cbar=True, cbar_kws={"shrink": 0.5},
                    yticklabels=(labels if k == 0 else False), xticklabels=[title])
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=0)
        ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.suptitle("LOSO performance per station (sorted by r²)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(PLOT_DIR, "heatmap_stations.png"), dpi=140)
    plt.close(fig)
    print("🔥 heatmap_stations.png")


def heatmap_by_type(oof):
    """Heatmap: station type × {r², RMSE, MBE} (aggregated LOSO)."""
    rows = []
    for t, g in oof.groupby("typ"):
        m = compute_metrics(g["bap"], g["pred_loso"])
        m["typ"] = t
        m["n_stations"] = g["eoi"].nunique()
        rows.append(m)
    df = pd.DataFrame(rows).set_index("typ").sort_values("R2", ascending=False)
    mat = df[["R2", "RMSE", "MBE"]]
    fig, ax = plt.subplots(figsize=(6, 0.7 * len(df) + 1.5))
    color = (mat - mat.mean()) / mat.std(ddof=0)  # per-column z-score for colour only
    sns.heatmap(color, ax=ax, cmap="vlag", center=0, annot=mat.round(2), fmt="",
                cbar_kws={"label": "z-score (per column)"},
                yticklabels=[f"{i} (n={df.loc[i, 'n_stations']})" for i in df.index])
    ax.set_title("LOSO performance by station type")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "heatmap_by_type.png"), dpi=140)
    plt.close(fig)
    print("🔥 heatmap_by_type.png")


def barplot_r2(df):
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(df))))
    order = df.sort_values("R2", ascending=True)
    sns.barplot(x=order["R2"], y=order.index, hue=order["typ"], dodge=False,
                palette="tab10", ax=ax)
    ax.axvline(order["R2"].median(), color="k", ls="--", lw=1,
               label=f"median {order['R2'].median():.2f}")
    ax.set_xlabel("r² (Pearson, LOSO)"); ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.legend(title="type", fontsize=8, loc="lower right")
    ax.set_title("LOSO r² per station")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "barplot_r2.png"), dpi=140)
    plt.close(fig)
    print("📊 barplot_r2.png")


def importance_plot(top=None):
    path = os.path.join(VAL_DIR, "feature_importance.csv")
    if not os.path.exists(path):
        print("⚠️  feature_importance.csv not found (run fill_all_days.py first)")
        return
    imp = pd.read_csv(path).sort_values("gain", ascending=False)
    if top:
        imp = imp.head(top)
    imp = imp.iloc[::-1]  # largest on top after barh
    imp["group"] = imp["feature"].map(feature_group)

    groups = ["Proxy (same-day PM/NO₂)", "Concentration lag", "Meteorology (base)",
              "Meteorology (derived)", "Calendar/season", "Terrain (DEM)",
              "Traffic", "Emission (bottom-up)", "Station typology"]
    palette = dict(zip(groups, sns.color_palette("tab10", len(groups))))
    DEFAULT_COLOR = (0.6, 0.6, 0.6)  # any unmapped group -> grey (defensive)
    colors = [palette.get(g, DEFAULT_COLOR) for g in imp["group"]]

    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(imp))))
    ax.barh(imp["feature"], imp["gain"], color=colors)
    ax.set_xlabel("Importance (average gain)")
    ax.set_title("XGBoost feature importance (model trained on all observations)")
    # Legend only for groups actually present
    present = [g for g in groups if g in set(imp["group"])]
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[g]) for g in present]
    ax.legend(handles, present, fontsize=8, loc="lower right", title="Feature group")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "feature_importance.png"), dpi=140)
    plt.close(fig)

    # Aggregated importance per group
    grp = imp.groupby("group")["gain"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(grp.index, grp.values, color=[palette.get(g, DEFAULT_COLOR) for g in grp.index])
    ax.set_xlabel("Total gain")
    ax.set_title("Feature importance aggregated by group")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "feature_importance_by_group.png"), dpi=140)
    plt.close(fig)
    print("📑 feature_importance.png + feature_importance_by_group.png")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    names, typ, oof, loso_stats = load()
    make_timeseries(names, typ, loso_stats)
    df = per_station_table(oof, names, typ)
    df.to_csv(os.path.join(VAL_DIR, "loso_per_station_pearson.csv"))
    heatmap_stations(df)
    heatmap_by_type(oof)
    barplot_r2(df)
    importance_plot()
    print(f"\n✅ Plots done: {PLOT_DIR}")


if __name__ == "__main__":
    main()
