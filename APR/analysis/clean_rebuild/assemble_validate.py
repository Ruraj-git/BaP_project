#!/usr/bin/env python3
"""Validate all manifest-defined fit artifacts and assemble prediction rows."""
from __future__ import annotations
import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from fit_worker import key_hash, sha_file
from model_contract import FEATURES, monotone_tuple

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--fit-glob",required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
 if a.output.exists(): raise FileExistsError(f"Refusing to overwrite {a.output}")
 tasks=pd.read_csv(a.manifest/"fit_manifest.csv"); contract=json.loads((a.manifest/"model_and_fold_contract.json").read_text())
 dirs=glob.glob(a.fit_glob); found={}
 for directory in dirs:
  js=glob.glob(str(Path(directory)/"*.json")); cs=glob.glob(str(Path(directory)/"*.csv"))
  if len(js)!=1 or len(cs)!=1: raise AssertionError(f"invalid artifact count in {directory}")
  meta=json.loads(Path(js[0]).read_text()); tid=int(meta["task"]["task_id"])
  if tid in found: raise AssertionError(f"duplicate task artifact {tid}")
  found[tid]=(Path(cs[0]),meta)
 expected=set(tasks.task_id.astype(int))
 if set(found)!=expected: raise AssertionError(f"artifact task mismatch: missing={sorted(expected-set(found))[:5]}, extra={sorted(set(found)-expected)[:5]}")
 cache={}; rows=[]; audit=[]
 for _, row in tasks.iterrows():
  tid=int(row.task_id); csv,meta=found[tid]
  support="2km_exact" if row.static_support=="2km_exact" else "500m"
  inp=Path(contract["inputs"][support]["path"])
  if row.input_variant=="strict18": inp=inp.with_name(inp.name.replace("permissive","strict18"))
  if str(meta["input"])!=str(inp) or meta["input_sha256"]!=sha_file(inp): raise AssertionError(f"task {tid}: input provenance mismatch")
  if str(inp) not in cache:
   d=pd.read_csv(inp); d.datum=pd.to_datetime(d.datum); cache[str(inp)]=d.sort_values(["eoi","datum"]).reset_index(drop=True)
  if sha_file(inp)!=contract["input_files"][inp.name]["sha256"]: raise AssertionError("input differs from freeze")
  d=cache[str(inp)]; assignments=pd.read_csv(a.manifest/"row_fold_assignments.csv",parse_dates=["datum"])
  if not assignments[["eoi","datum"]].equals(d[["eoi","datum"]]): raise AssertionError("fold assignment keys differ")
  column={"block30":"block", "dispersed_mod9":"disp_fold", "loso":"loso_station", "year_out":"year", "calendar_season_out":"season", "season_year_out":"season_year"}[row.protocol]
  held=assignments[column].astype(str).eq(str(row.fold)); train,test=d.loc[~held].copy(),d.loc[held].copy()
  if meta["task"] != row.to_dict(): raise AssertionError(f"task {tid}: metadata differs from manifest")
  if meta["train_key_hash"]!=key_hash(train) or meta["test_key_hash"]!=key_hash(test): raise AssertionError(f"task {tid}: key hash mismatch")
  if meta["prediction_sha256"]!=sha_file(csv): raise AssertionError(f"task {tid}: prediction checksum mismatch")
  if meta["manifest_sha256"]!=sha_file(a.manifest/"fit_manifest.csv") or meta["contract_sha256"]!=sha_file(a.manifest/"model_and_fold_contract.json"): raise AssertionError("manifest checksum mismatch")
  if meta["worker_sha256"]!=contract["code_hashes"]["fit_worker.py"]: raise AssertionError("worker checksum mismatch")
  expected_params=contract["xgboost_parameters"]
  if any(meta["effective_xgboost_parameters"].get(k)!=v for k,v in expected_params.items()): raise AssertionError("estimator settings mismatch")
  fit_target=train.bap.to_numpy(float) if row.target=="identity" else np.log1p(train.bap.to_numpy(float))
  if not np.isclose(meta["effective_xgboost_parameters"]["base_score"],fit_target.mean(),rtol=1e-12): raise AssertionError("initial prediction mismatch")
  p=pd.read_csv(csv); p.datum=pd.to_datetime(p.datum)
  if not np.isfinite(p[["prediction","prediction_untruncated","prediction_target_scale"]]).all().all(): raise AssertionError("non-finite raw prediction")
  if not np.allclose(p.prediction,np.maximum(0,p.prediction_untruncated)): raise AssertionError("floor mismatch")
  if int((p.prediction_untruncated<0).sum())!=meta["zero_floor_count"]: raise AssertionError("floor count mismatch")
  if p.duplicated(["eoi","datum"]).any() or len(p)!=len(test): raise AssertionError(f"task {tid}: invalid prediction keys")
  expected_keys=test[["eoi","datum"]].sort_values(["eoi","datum"]).reset_index(drop=True); actual_keys=p[["eoi","datum"]].sort_values(["eoi","datum"]).reset_index(drop=True)
  if not actual_keys.equals(expected_keys): raise AssertionError(f"task {tid}: held-out key mismatch")
  target=test[["eoi","datum","bap"]].merge(p[["eoi","datum","observed"]],on=["eoi","datum"],validate="one_to_one")
  if not np.allclose(target.bap,target.observed,atol=1e-12,rtol=0) or not np.isfinite(p.prediction).all(): raise AssertionError(f"task {tid}: target/prediction check failed")
  features=list(FEATURES[row.arm])
  if row.arm in {"PM_XGB","FULL_ID"}: features += [f"station__{s}" for s in sorted(train.eoi.unique())]
  if meta["features"]!=features or meta["resolved_monotone_tuple"]!=list(monotone_tuple(list(FEATURES[row.arm]))+tuple(0 for _ in range(len(features)-len(FEATURES[row.arm])))): raise AssertionError(f"task {tid}: feature/constraint mismatch")
  p["task_id"]=tid; p["input_variant"]=row.input_variant; p["static_support"]=row.static_support; p["target_scale"]=row.target; p["weighting"]=row.weighting
  rows.append(p); audit.append({"task_id":tid,"n_test":len(p),"prediction_sha256":meta["prediction_sha256"],"input_sha256":meta["input_sha256"],"feature_count_effective":len(features)})
 out=pd.concat(rows,ignore_index=True); a.output.mkdir(parents=True)
 out.to_csv(a.output/"validated_predictions.csv",index=False); pd.DataFrame(audit).to_csv(a.output/"task_audit.csv",index=False)
 report={"tasks_validated":len(audit),"prediction_rows":len(out),"finite_predictions":bool(np.isfinite(out.prediction).all()),"manifest":str(a.manifest),"fit_glob":a.fit_glob,"by_protocol":out.groupby("protocol").size().to_dict()}
 (a.output/"validation_report.json").write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
if __name__=="__main__": main()
