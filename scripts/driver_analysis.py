#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Driver analysis for daily BaP: TreeSHAP (native XGBoost) + the orographic
control / spatial-transferability angle.

Produces:
  - shap_importance.png        mean |SHAP| per feature, coloured by group
  - shap_dependence.png        SHAP dependence for the key physical drivers
                               (valley TPI, boundary layer, heating, ventilation,
                               temperature, season) -> the mechanistic story
SHAP values are on the model's target scale, ln(1+BaP); positive = higher BaP.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
import config  # noqa: E402
import validate_model as vm  # noqa: E402
from make_plots import feature_group  # reuse the feature->group mapping  # noqa: E402

VAL_DIR = config.VALIDATION_DIR
PLOT_DIR = config.PLOTS_DIR
sns.set_theme(style="whitegrid", context="notebook")

DRIVERS = ["tpi_broad", "hpbl_min", "heating_degree_hours", "vrate_min",
           "t_mean", "doy_cos"]
DRIVER_LABELS = {
    "tpi_broad": "Topographic position (valley → ridge) [m]",
    "hpbl_min": "Min. boundary-layer height [m]",
    "heating_degree_hours": "Heating-degree hours",
    "vrate_min": "Min. ventilation rate",
    "t_mean": "Daily mean temperature [°C]",
    "doy_cos": "Season (cos day-of-year; winter→+1)",
}


def fit_and_shap():
    df = vm.load_dataset().reset_index(drop=True)
    feats = vm.resolve_features(df)
    X = df[feats].copy()
    y = np.log1p(df["bap"])
    w = 1.0 / (df["bap"] + 0.5)
    params = dict(config.MODEL_PARAMS)
    params["monotone_constraints"] = config.monotone_for(feats)
    model = xgb.XGBRegressor(**params)
    model.fit(X, y, sample_weight=w)
    dmat = xgb.DMatrix(X, feature_names=feats)
    contribs = model.get_booster().predict(dmat, pred_contribs=True)  # (n, k+1)
    shap = pd.DataFrame(contribs[:, :-1], columns=feats)
    return df, X, shap, feats


def plot_shap_importance(shap, feats, top=22):
    imp = shap.abs().mean().sort_values(ascending=False).head(top).iloc[::-1]
    groups = ["Proxy (same-day PM/NO₂)", "Concentration lag", "Meteorology (base)",
              "Meteorology (derived)", "Calendar/season", "Terrain (DEM)",
              "Traffic", "Emission (bottom-up)", "Station typology"]
    palette = dict(zip(groups, sns.color_palette("tab10", len(groups))))
    colors = [palette[feature_group(f)] for f in imp.index]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(imp))))
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_xlabel("mean |SHAP|  (effect on ln(1+BaP))")
    ax.set_title("SHAP feature importance (TreeSHAP)")
    present = [g for g in groups if g in {feature_group(f) for f in imp.index}]
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[g]) for g in present]
    ax.legend(handles, present, fontsize=8, loc="lower right", title="Feature group")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "shap_importance.png"), dpi=140)
    plt.close(fig)
    print("📑 shap_importance.png")


def _dependence_legend(fig):
    """One shared legend explaining the blue points and the red trend curve."""
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#1f77b4",
               markeredgecolor="none", alpha=0.6, markersize=6,
               label="one station-day (SHAP value)"),
        Line2D([0], [0], color="#d62728", lw=2, label="binned median trend"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=9, frameon=True,
               bbox_to_anchor=(0.995, 0.995))


def plot_shap_dependence(X, shap):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, f in zip(axes.ravel(), DRIVERS):
        x = X[f].values
        s = shap[f].values
        ok = np.isfinite(x) & np.isfinite(s)
        x, s = x[ok], s[ok]
        ax.scatter(x, s, s=6, alpha=0.15, color="#1f77b4", edgecolors="none")
        # binned-median trend (quantile bins)
        try:
            q = pd.qcut(x, 20, duplicates="drop")
            tr = pd.DataFrame({"x": x, "s": s, "q": q}).groupby("q", observed=True)
            ax.plot(tr["x"].median(), tr["s"].median(), "-", color="#d62728", lw=2)
        except ValueError:
            pass
        ax.axhline(0, color="k", lw=0.6)
        # x-axis carries the Table 1 feature identifier; the descriptive
        # meaning + unit goes in the panel title.
        ax.set_xlabel(f, fontsize=10)
        ax.set_title(DRIVER_LABELS.get(f, f), fontsize=9)
        ax.set_ylabel("SHAP (ln(1+BaP))", fontsize=9)
    fig.suptitle("SHAP dependence: physical drivers of daily BaP "
                 "(positive = higher BaP)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _dependence_legend(fig)
    fig.savefig(os.path.join(PLOT_DIR, "shap_dependence.png"), dpi=140)
    plt.close(fig)
    print("📈 shap_dependence.png")


def plot_all_dependence(X, shap, feats):
    """One SHAP dependence plot per feature, written to shap_dependence/."""
    out = os.path.join(PLOT_DIR, "shap_dependence")
    os.makedirs(out, exist_ok=True)
    # rank by importance so filenames sort by influence
    order = shap.abs().mean().sort_values(ascending=False)
    for rank, f in enumerate(order.index, start=1):
        x = X[f].values
        s = shap[f].values
        ok = np.isfinite(x) & np.isfinite(s)
        x, s = x[ok], s[ok]
        if x.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(x, s, s=6, alpha=0.15, color="#1f77b4", edgecolors="none")
        try:
            q = pd.qcut(x, 20, duplicates="drop")
            tr = pd.DataFrame({"x": x, "s": s, "q": q}).groupby("q", observed=True)
            ax.plot(tr["x"].median(), tr["s"].median(), "-", color="#d62728", lw=2)
        except ValueError:
            pass
        ax.axhline(0, color="k", lw=0.6)
        # x-axis carries the Table 1 identifier; description + unit in the title.
        ax.set_xlabel(f, fontsize=10)
        ax.set_ylabel("SHAP (ln(1+BaP))", fontsize=9)
        ax.set_title(f"{DRIVER_LABELS.get(f, f)}\n(mean|SHAP|={order[f]:.4f})",
                     fontsize=9)
        _dependence_legend(fig)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{rank:02d}_{f}.png"), dpi=140)
        plt.close(fig)
    print(f"📂 shap_dependence/  ({len(order)} per-feature plots)")


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    df, X, shap, feats = fit_and_shap()
    shap.abs().mean().sort_values(ascending=False).to_csv(
        os.path.join(VAL_DIR, "shap_importance.csv"), header=["mean_abs_shap"])
    plot_shap_importance(shap, feats)
    plot_shap_dependence(X, shap)
    plot_all_dependence(X, shap, feats)
    print(f"\n✅ Driver analysis done -> {PLOT_DIR}")


if __name__ == "__main__":
    main()
