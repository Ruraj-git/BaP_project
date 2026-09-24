#!/usr/bin/env python3
"""Isolated 30-fit review extension; never edits or imports mutable r02 recipes."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'APR/results/clean_rebuild/r03_review'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def key_hash(frame):
    return hashlib.sha256('\n'.join(frame.eoi.astype(str)+'|'+frame.datum.astype(str)).encode()).hexdigest()

def held_mask(frame, task):
    if task['protocol']=='loso': return frame.eoi.eq(str(task['fold']))
    if task['protocol']=='dispersed_mod9':
        return ((pd.to_datetime(frame.datum)-pd.Timestamp('2023-06-02')).dt.days%9).eq(int(task['fold']))
    raise ValueError('unregistered protocol')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--task-id',type=int,required=True); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    contract=json.loads((RUN/'contract.json').read_text())
    for relative,expected in contract['sources'].items():
        if sha(ROOT/relative)!=expected: raise ValueError('changed frozen source: '+relative)
    tasks=pd.read_csv(RUN/'fit_manifest.csv',dtype={'fold':str})
    if sha(RUN/'fit_manifest.csv')!=contract['manifest_sha256']: raise ValueError('manifest changed')
    selected=tasks[tasks.task_id.eq(a.task_id)]
    if len(selected)!=1: raise ValueError('invalid task id')
    task=selected.iloc[0].to_dict(); recipe=contract['recipes'][task['arm']]
    data=pd.read_csv(ROOT/contract['input']).sort_values(['eoi','datum']).reset_index(drop=True)
    held=held_mask(data,task); train,test=data[~held].copy(),data[held].copy()
    if train.empty or test.empty: raise ValueError('empty split')
    if (len(train),len(test))!=(task['n_train'],task['n_test']): raise ValueError('split count changed')
    if key_hash(train)!=task['train_key_hash'] or key_hash(test)!=task['test_key_hash']: raise ValueError('split keys changed')
    if a.dry_run:
        print(json.dumps({'task':task,'recipe':recipe,'valid':True},indent=2)); return
    out=RUN/'predictions'/f'task_{a.task_id:03d}'
    if out.exists(): raise FileExistsError(out)
    params=dict(contract['xgboost_parameters'])
    target=np.log1p(train.bap.to_numpy(float))
    params.update(base_score=float(target.mean()),monotone_constraints=tuple(recipe['constraints']))
    fit=xgb.XGBRegressor(**params)
    fit.fit(train[recipe['features']],target)
    raw=fit.predict(test[recipe['features']]); pre=np.expm1(raw); pred=np.maximum(0,pre)
    if not np.isfinite(np.column_stack([raw,pre,pred])).all(): raise ValueError('nonfinite predictions')
    predictions=test[['eoi','datum','bap']].rename(columns={'bap':'observed'}).copy()
    predictions['prediction']=pred; predictions['prediction_untruncated']=pre; predictions['prediction_target_scale']=raw
    for key in ['task_id','arm','protocol','fold']: predictions[key]=task[key]
    out.mkdir(parents=True,exist_ok=False)
    predictions.to_csv(out/'predictions.csv',index=False)
    meta={'task':task,'features':recipe['features'],'resolved_monotone_tuple':recipe['constraints'],
          'effective_xgboost_parameters':fit.get_params(),'booster_config':json.loads(fit.get_booster().save_config()),
          'input':contract['input'],'input_sha256':sha(ROOT/contract['input']),
          'contract_sha256':sha(RUN/'contract.json'),'manifest_sha256':sha(RUN/'fit_manifest.csv'),
          'worker_sha256':sha(Path(__file__)),'prediction_sha256':sha(out/'predictions.csv'),
          'train_key_hash':key_hash(train),'test_key_hash':key_hash(test),'n_train':len(train),'n_test':len(test),
          'zero_floor_count':int((pre<0).sum()),'python':platform.python_version(),'xgboost':xgb.__version__}
    (out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'task_id':a.task_id,'arm':task['arm'],'n_test':len(test),'output':str(out)}))

if __name__=='__main__': main()
