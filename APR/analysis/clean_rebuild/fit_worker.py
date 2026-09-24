#!/usr/bin/env python3
"""One manifest-defined XGBoost fit for the clean rebuild."""
from __future__ import annotations
import argparse, hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from model_contract import FEATURES, XGB_PARAMS, monotone_tuple

ROOT=Path(__file__).resolve().parents[3]
def sha_file(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def key_hash(d):
 return hashlib.sha256("\n".join((d.eoi.astype(str)+"|"+d.datum.astype(str)).tolist()).encode()).hexdigest()
def holdout(d,row,origin):
 dt=pd.to_datetime(d.datum); protocol=row.protocol; fold=str(row.fold)
 if protocol=="block30": return ((dt-origin).dt.days//30).astype(str).eq(fold)
 if protocol=="loso": return d.eoi.eq(fold)
 if protocol=="dispersed_mod9": return ((dt-origin).dt.days%9).astype(str).eq(fold)
 if protocol=="year_out": return dt.dt.year.astype(str).eq(fold)
 season=dt.dt.month.map({12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"})
 if protocol=="calendar_season_out": return season.eq(fold)
 if protocol=="season_year_out": return ((dt.dt.year+(dt.dt.month==12).astype(int)).astype(str)+"_"+season).eq(fold)
 raise ValueError(f"unknown protocol {protocol}")
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--task-id",type=int); ap.add_argument("--output",type=Path); ap.add_argument("--dry-run",action="store_true"); a=ap.parse_args()
 tasks=pd.read_csv(a.manifest/"fit_manifest.csv"); contract=json.loads((a.manifest/"model_and_fold_contract.json").read_text());
 if a.task_id not in set(tasks.task_id): raise ValueError("task id absent")
 row=tasks.loc[tasks.task_id.eq(a.task_id)].iloc[0]
 if a.dry_run:
  print(row.to_json()); return
 inp=Path(contract["inputs"]["2km_exact" if row.static_support=="2km_exact" else "500m"]["path"])
 if row.input_variant=="strict18": inp=inp.with_name(inp.name.replace("permissive","strict18"))
 if sha_file(inp)!=contract["input_files"][inp.name]["sha256"]: raise ValueError("frozen input hash mismatch")
 if FEATURES != contract["feature_contract"] or XGB_PARAMS != contract["xgboost_parameters"]: raise ValueError("code differs from frozen model contract")
 if "|".join(FEATURES[row.arm])!=row.feature_order: raise ValueError("task feature order mismatch")
 d=pd.read_csv(inp); d.datum=pd.to_datetime(d.datum); d=d.sort_values(["eoi","datum"]).reset_index(drop=True); origin=d.datum.min(); held=holdout(d,row,origin)
 if not held.any() or held.all(): raise ValueError("invalid holdout")
 train,test=d.loc[~held].copy(),d.loc[held].copy(); features=list(FEATURES[row.arm])
 # The station indicators are created from training stations only and are used only
 # for the two flexible temporal block comparators.
 if row.arm in {"PM_XGB","FULL_ID"}:
  names=[f"station__{s}" for s in sorted(train.eoi.unique())]
  for name in names:
   station=name.removeprefix("station__"); train[name]=(train.eoi==station).astype(int); test[name]=(test.eoi==station).astype(int)
  features += names
  if set(test.eoi)-set(train.eoi): raise ValueError("test station absent from training for station-indicator model")
 params=dict(XGB_PARAMS); constraint=monotone_tuple(list(FEATURES[row.arm])) + tuple(0 for _ in range(len(features)-len(FEATURES[row.arm])))
 params["monotone_constraints"]=constraint
 y=train.bap.to_numpy(float); target=y if row.target=="identity" else np.log1p(y)
 weights=None if row.weighting=="uniform" else 1.0/(y+0.5)
 # Explicit training-target mean, identical rule for identity and log1p.
 # Unweighted for both weight recipes so their only difference is fit weights.
 params["base_score"]=float(target.mean())
 model=xgb.XGBRegressor(**params); model.fit(train[features],target,sample_weight=weights)
 raw=model.predict(test[features]); untruncated=raw if row.target=="identity" else np.expm1(raw); pred=np.maximum(0,untruncated)
 if not np.isfinite(pred).all(): raise ValueError("non-finite prediction")
 if a.output is None: raise ValueError("output required unless dry-run")
 a.output.mkdir(parents=True,exist_ok=False)
 out=a.output/f"task_{a.task_id:03d}.csv"; pd.DataFrame({"eoi":test.eoi,"datum":test.datum.dt.strftime("%Y-%m-%d"),"observed":test.bap,"prediction":pred,"prediction_untruncated":untruncated,"prediction_target_scale":raw,"task_id":a.task_id,"arm":row.arm,"protocol":row.protocol,"fold":row.fold}).to_csv(out,index=False)
 meta={"task":row.to_dict(),"input":str(inp),"input_sha256":sha_file(inp),"features":features,"resolved_monotone_tuple":list(constraint),"effective_xgboost_parameters":model.get_params(),"target":row.target,"weighting":row.weighting,"train_n":len(train),"test_n":len(test),"train_key_hash":key_hash(train),"test_key_hash":key_hash(test),"prediction_sha256":sha_file(out),"python":platform.python_version(),"xgboost":xgb.__version__,"thread_caps":{"n_jobs":1}}
 meta.update(booster_config=json.loads(model.get_booster().save_config()), zero_floor_count=int((untruncated<0).sum()), manifest_sha256=sha_file(a.manifest/"fit_manifest.csv"), worker_sha256=sha_file(Path(__file__)), contract_sha256=sha_file(a.manifest/"model_and_fold_contract.json"))
 (a.output/f"task_{a.task_id:03d}.json").write_text(json.dumps(meta,indent=2,default=str)+"\n")
 print(json.dumps({"task_id":a.task_id,"arm":row.arm,"protocol":row.protocol,"train_n":len(train),"test_n":len(test)},indent=2))
if __name__=="__main__": main()
