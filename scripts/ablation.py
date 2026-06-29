#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Clean feature ablation on a single dataset (v5), consistent squared-Pearson r².
Reports random k-fold pooled r² and LOSO median per-station r² for nested feature sets."""
import os, sys
import numpy as np
import pandas as pd
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
import validate_model as vm

def main():
    df = vm.load_dataset().reset_index(drop=True)
    base = list(vm.BASE_FEATURES)
    g1 = [c for c in vm.GROUP1_FEATURES if c in df.columns]
    g2 = [c for c in vm.GROUP2_FEATURES if c in df.columns]
    cl = [c for c in vm.CONC_LAG_FEATURES if c in df.columns]
    em = [c for c in vm.EMISSION_FEATURES if c in df.columns]
    pp = [c for c in vm.POP_FEATURES if c in df.columns]
    full = base + g1 + g2 + cl + em + pp
    m4 = base + g1 + g2 + cl + em  # final model (population dropped, see validate_model.resolve_features)
    proxy_free = [c for c in m4 if c not in vm.PROXY_FEATURES]  # virtual-site config = M4 minus proxies
    configs = [
        ("M0: base (meteo+proxy+typology)", base),
        ("M1: + derived meteo (group-1)", base + g1),
        ("M2: + spatial covariates (group-2)", base + g1 + g2),
        ("M3: + concentration lags", base + g1 + g2 + cl),
        ("M4: + bottom-up emission", base + g1 + g2 + cl + em),
        ("M5: + population (full)", full),
        ("V : proxy-free (virtual-site)", proxy_free),
    ]
    rows = []
    for name, feats in configs:
        vm.FEATURES = feats
        _, _, kf = vm.run_kfold(df)
        _, st, _ = vm.run_loso(df)
        rows.append({"model": name, "n_features": len(feats),
                     "kfold_r2": kf["R2"], "kfold_rmse": kf["RMSE"],
                     "loso_median_r2": st["R2"].median()})
        print(f"==> {name}: kfold r²={kf['R2']:.3f}  LOSO median r²={st['R2'].median():.3f}")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(vm.OUT_DIR, "ablation.csv"), index=False)
    print("\n", out.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
