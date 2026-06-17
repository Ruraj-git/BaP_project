#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the 4-method block-gap comparison (XGBoost, linear ridge, seasonal
climatology, persistence) and draw block_baseline_comparison.png -- without
re-running any model. Reads the two persisted source tables:
  - block_metrics_summary.csv  (from validate_blocks.py: xgb/climatology/persistence)
  - linear_benchmark.csv       (from linear_benchmark.py: the ridge block rows)
and writes the combined table block_baselines_all.csv plus the figure.

Run after: validate_blocks.py  and  linear_benchmark.py.
"""
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
import config  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
BLOCK_DAYS = 30

# short labels in display order (best -> worst)
ORDER = ["XGBoost", "Linear (ridge)", "Climatology", "Persistence"]
ROWKEY = {  # map the CSV's Method strings to short labels
    "XGBoost (this study)": "XGBoost",
    "Linear (Ridge, same features)": "Linear (ridge)",
    "Seasonal climatology": "Climatology",
    "Persistence": "Persistence",
}


def build_combined(val):
    """Assemble the 4-method block-gap table from the two source CSVs."""
    bm = pd.read_csv(os.path.join(val, "block_metrics_summary.csv"))
    lin = pd.read_csv(os.path.join(val, "linear_benchmark.csv")).set_index("scheme")

    def grab(method):
        p = bm[(bm.method == method) & (bm.scope == "pooled")].iloc[0]
        m = bm[(bm.method == method) & (bm.scope == "median_per_station")].iloc[0]
        return p.R2, p.RMSE, m.R2, m.RMSE

    rows = [
        ("XGBoost (this study)", *grab("xgb")),
        ("Linear (Ridge, same features)",
         lin.loc["block_pooled", "r2"], lin.loc["block_pooled", "rmse"],
         lin.loc["block_median_station", "r2"], lin.loc["block_median_station", "rmse"]),
        ("Seasonal climatology", *grab("climatology")),
        ("Persistence", *grab("persistence")),
    ]
    df = pd.DataFrame(rows, columns=["Method", "pool_r2", "pool_rmse", "med_r2", "med_rmse"])
    df.to_csv(os.path.join(val, "block_baselines_all.csv"), index=False)
    return df


def main():
    val = config.VALIDATION_DIR
    df = build_combined(val)
    df["lab"] = df["Method"].map(ROWKEY)
    df = df.set_index("lab").loc[ORDER]

    panels = [
        ("med_r2", "Median per-station r²", "r2"),
        ("pool_r2", "Pooled r²", "r2"),
        ("pool_rmse", "Pooled RMSE [ng/m³]", "rmse"),
    ]
    cols = sns.color_palette("tab10", len(ORDER))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (col, title, kind) in zip(axes, panels):
        vals = df[col].values
        ax.bar(ORDER, vals, color=cols)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=20)
        if kind == "r2":
            ax.set_ylim(min(0, vals.min()) - 0.05, 1.0)
            ax.axhline(0, color="k", lw=0.6)
    fig.suptitle(f"Block-gap CV ({BLOCK_DAYS}-day gaps): model vs. baselines", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(config.PLOTS_DIR, "block_baseline_comparison.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"✅ {out}")


if __name__ == "__main__":
    main()
