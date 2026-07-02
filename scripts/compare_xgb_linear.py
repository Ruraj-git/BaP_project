#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why does XGB beat Ridge? (A) peak (>1 ng) error split, (B) proxy-free
comparison, (C) per-station side-by-side table -> PNG. Block-gap OOF, same
folds/weighting for both models. Production untouched."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
import validate_model as vm
import linear_benchmark as lb
import config
config.MODEL_PARAMS["n_jobs"] = 16
BLOCK_DAYS = 30
OUTDIR = vm.OUT_DIR
PLOTDIR = vm.PLOT_DIR


def block_oof(df, feats, predictor):
    d = df.copy()
    d["date"] = pd.to_datetime(d["datum"])
    o = d["date"].min()
    d["block"] = ((d["date"] - o).dt.days // BLOCK_DAYS).astype(int)
    d = d.reset_index(drop=True)
    oof = np.full(len(d), np.nan)
    for b in sorted(d["block"].unique()):
        te = d.index[d["block"] == b]; tr = d.index[d["block"] != b]
        if len(te) < 3:
            continue
        oof[te] = predictor(d.loc[tr], d.loc[te], feats)
    return oof


def xgb_pred(tr, te, feats):
    vm.FEATURES = feats
    return vm.fit_predict(tr, te)


def lin_pred(tr, te, feats):
    return lb.fit_predict(tr, te, feats)


df = vm.load_dataset().reset_index(drop=True)
full = vm.resolve_features(df)
pfree = [c for c in full if c not in vm.PROXY_FEATURES and c not in vm.TYPOLOGY_FEATURES]
print(f"{len(df)} rows | full={len(full)} feats | proxy-free={len(pfree)} feats")

print("XGB full...");      df["xgb_full"] = block_oof(df, full,  xgb_pred)
print("Ridge full...");    df["lin_full"] = block_oof(df, full,  lin_pred)
print("XGB proxy-free..."); df["xgb_pf"]  = block_oof(df, pfree, xgb_pred)
print("Ridge proxy-free..."); df["lin_pf"] = block_oof(df, pfree, lin_pred)

d = df.dropna(subset=["xgb_full", "lin_full", "xgb_pf", "lin_pf"]).copy()
d["code"] = d["typ_oblasti"].str.strip() + d["typ_zdroja"].str.strip()


def met(g, col):
    return vm.compute_metrics(g["bap"], g[col])


# ---------- (A) peak split ----------
print("\n===== (A) PEAK ANALYSIS: error split by observed BaP =====")
print(f"{'subset':<16}{'n':>6}{'XGB r2':>9}{'Lin r2':>9}"
      f"{'XGB RMSE':>10}{'Lin RMSE':>10}{'XGB MAE':>9}{'Lin MAE':>9}")
peak_rows = []
for label, mask in [("all", d["bap"] > -1),
                    ("<= 1 ng/m3", d["bap"] <= 1.0),
                    ("> 1 ng/m3", d["bap"] > 1.0),
                    ("> 3 ng/m3", d["bap"] > 3.0)]:
    g = d[mask]
    mx, ml = met(g, "xgb_full"), met(g, "lin_full")
    print(f"{label:<16}{len(g):>6}{mx['R2']:>9.3f}{ml['R2']:>9.3f}"
          f"{mx['RMSE']:>10.3f}{ml['RMSE']:>10.3f}{mx['MAE']:>9.3f}{ml['MAE']:>9.3f}")
    peak_rows.append({"subset": label, "n": len(g),
                      "xgb_r2": mx["R2"], "lin_r2": ml["R2"],
                      "xgb_rmse": mx["RMSE"], "lin_rmse": ml["RMSE"],
                      "xgb_mae": mx["MAE"], "lin_mae": ml["MAE"]})
pd.DataFrame(peak_rows).to_csv(os.path.join(OUTDIR, "xgb_vs_linear_peaks.csv"), index=False)

# ---------- (B) proxy-free vs full, pooled + median ----------
print("\n===== (B) PROXY-FREE vs FULL (block-gap) =====")
summ = []
for col, lab in [("xgb_full", "XGB full"), ("lin_full", "Ridge full"),
                 ("xgb_pf", "XGB proxy-free"), ("lin_pf", "Ridge proxy-free")]:
    pooled = met(d, col)
    med = np.nanmedian([met(g, col)["R2"] for _, g in d.groupby("eoi")])
    print(f"  {lab:<18} pooled r2={pooled['R2']:.3f}   median-per-station r2={med:.3f}")
    summ.append({"config": lab, "pooled_r2": pooled["R2"], "pooled_rmse": pooled["RMSE"],
                 "median_station_r2": med})
pd.DataFrame(summ).to_csv(os.path.join(OUTDIR, "xgb_vs_linear_summary.csv"), index=False)

# ---------- (C) per-station table ----------
rows = []
for s, g in d.groupby("eoi"):
    rows.append({
        "Station": s, "Type": g["code"].iloc[0], "n": len(g),
        "XGB": met(g, "xgb_full")["R2"], "Ridge": met(g, "lin_full")["R2"],
        "XGB_pf": met(g, "xgb_pf")["R2"], "Ridge_pf": met(g, "lin_pf")["R2"],
    })
t = pd.DataFrame(rows)
t["d"] = t["XGB"] - t["Ridge"]
t["d_pf"] = t["XGB_pf"] - t["Ridge_pf"]
t = t.sort_values("d").reset_index(drop=True)  # linear-wins float to top
t.to_csv(os.path.join(OUTDIR, "xgb_vs_linear_per_station.csv"), index=False)

med_row = {"Station": "MEDIAN", "Type": "", "n": int(t["n"].sum()),
           "XGB": t["XGB"].median(), "Ridge": t["Ridge"].median(),
           "XGB_pf": t["XGB_pf"].median(), "Ridge_pf": t["Ridge_pf"].median(),
           "d": t["XGB"].median() - t["Ridge"].median(),
           "d_pf": t["XGB_pf"].median() - t["Ridge_pf"].median()}

# ---------- render table image ----------
cols = ["Station", "Type", "n", "XGB", "Ridge", "Δ", "XGB", "Ridge", "Δ"]
disp = []
for _, r in t.iterrows():
    disp.append([r["Station"], r["Type"], int(r["n"]),
                 f"{r['XGB']:.2f}", f"{r['Ridge']:.2f}", f"{r['d']:+.2f}",
                 f"{r['XGB_pf']:.2f}", f"{r['Ridge_pf']:.2f}", f"{r['d_pf']:+.2f}"])
disp.append([med_row["Station"], "", med_row["n"],
             f"{med_row['XGB']:.2f}", f"{med_row['Ridge']:.2f}", f"{med_row['d']:+.2f}",
             f"{med_row['XGB_pf']:.2f}", f"{med_row['Ridge_pf']:.2f}", f"{med_row['d_pf']:+.2f}"])

fig, ax = plt.subplots(figsize=(11, 9))
ax.axis("off")
ax.set_title("Block-gap per-station r²: XGBoost vs Ridge (sorted; linear-wins on top)\n"
             "left block = full features (with PM/NO₂ proxies)   |   "
             "right block = proxy-free (virtual-site / deployment regime)",
             fontsize=12, pad=18)
tab = ax.table(cellText=disp, colLabels=cols, loc="center", cellLoc="center")
tab.auto_set_font_size(False); tab.set_fontsize(9.5); tab.scale(1, 1.45)

# group header band
for j in range(9):
    tab[0, j].set_facecolor("#d9e1f2"); tab[0, j].set_text_props(weight="bold")
ncols = 9
for i in range(len(disp) + 1):
    # subtle separators between the full vs proxy-free blocks
    tab[i, 5].set_edgecolor("#888"); tab[i, 8].set_edgecolor("#888")

# highlight where Ridge >= XGB (Δ <= 0) — green = linear competitive/better
for i, (_, r) in enumerate(t.iterrows(), start=1):
    if r["d"] <= 0:      # full
        for j in (0, 3, 4, 5):
            tab[i, j].set_facecolor("#d7f0d0")
    if r["d_pf"] <= 0:   # proxy-free
        for j in (6, 7, 8):
            tab[i, j].set_facecolor("#d7f0d0")
# median row bold
mrow = len(disp)
for j in range(9):
    tab[mrow, j].set_facecolor("#fff2cc"); tab[mrow, j].set_text_props(weight="bold")

cap = ("Green = Ridge matches or beats XGBoost (Δ = XGB−Ridge ≤ 0). "
       "With proxies the two tie at typical stations; XGBoost's margin concentrates "
       "at high-variance sites and widens once proxies are removed (deployment regime).")
fig.text(0.5, 0.04, cap, ha="center", fontsize=9, wrap=True)
out_png = os.path.join(PLOTDIR, "xgb_vs_linear_per_station.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\n✅ table image -> {out_png}")
print(f"✅ csv -> {os.path.join(OUTDIR, 'xgb_vs_linear_per_station.csv')}")
print(f"✅ peaks csv -> {os.path.join(OUTDIR, 'xgb_vs_linear_peaks.csv')}")
print(f"\nFULL ABS PATH: {os.path.abspath(out_png)}")
