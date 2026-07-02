#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the bottom-up emission help where it should: at rural sites and in the
proxy-free (virtual) regime? Compares LOSO per-station for 4 configs and reports
median r² + bias at the 2 rural-background stations.
(SK0070A/Plášťovce was reclassified rural->suburban; kept as a reference column.)"""
import os, sys
import numpy as np, pandas as pd
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(REPO_ROOT, "scripts"))
import validate_model as vm

RURAL = ["SK0004R", "SK0006R"]   # SK0070A reclassified rural->suburban (kept below as reference)

def loso(df, feats):
    vm.FEATURES = feats
    _, st, ov = vm.run_loso(df)
    return st, ov

def main():
    df = vm.load_dataset().reset_index(drop=True)
    base, g1 = list(vm.BASE_FEATURES), [c for c in vm.GROUP1_FEATURES if c in df.columns]
    g2 = [c for c in vm.GROUP2_FEATURES if c in df.columns]
    cl = [c for c in vm.CONC_LAG_FEATURES if c in df.columns]
    em = [c for c in vm.EMISSION_FEATURES if c in df.columns]
    # Population (POP_FEATURES) was dropped from the final model, so it is
    # excluded here too: the "+ emission" configs isolate the emission effect
    # and match the final M4 model (no population layer).
    m4 = base + g1 + g2 + cl + em                                          # proxy + emission (final M4)
    pf_no = [c for c in base + g1 + g2 if c not in vm.PROXY_FEATURES
             and c not in vm.TYPOLOGY_FEATURES]                            # proxy-free + typology-free, no emission
    pf_em = pf_no + em                                                     # proxy-free + emission

    configs = {
        "proxy, NO emission (M3)": base + g1 + g2 + cl,
        "proxy + emission (M4)": m4,
        "proxy-free, NO emission": pf_no,
        "proxy-free + emission": pf_em,
    }
    rows = []
    for name, feats in configs.items():
        st, _ = loso(df, feats)
        st = st.set_index("eoi")
        rows.append({
            "config": name,
            "LOSO_median_r2": st["R2"].median(),
            "rural_mean_|MBE|": st.loc[RURAL, "MBE"].abs().mean(),
            "SK0004R_MBE": st.loc["SK0004R", "MBE"],
            "SK0006R_MBE": st.loc["SK0006R", "MBE"],
            "SK0070A_MBE": st.loc["SK0070A", "MBE"],
            "rural_median_r2": st.loc[RURAL, "R2"].median(),
        })
        print("done:", name)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(vm.OUT_DIR, "emission_focus.csv"), index=False)
    print("\n", out.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
