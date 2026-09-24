#!/usr/bin/env python3
"""Independent validation/assembly of staged review fits against r02 folds."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from review_fit_worker import ROOT, RUN, sha, key_hash

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tasks',help='Comma-separated task IDs; omitted requires all 30')
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    c=json.loads((RUN/'contract.json').read_text())
    for path,digest in c['sources'].items():
        if sha(ROOT/path)!=digest: raise AssertionError('source changed: '+path)
    if sha(RUN/'fit_manifest.csv')!=c['manifest_sha256']: raise AssertionError('manifest changed')
    tasks=pd.read_csv(RUN/'fit_manifest.csv',dtype={'fold':str})
    wanted=set(map(int,a.tasks.split(','))) if a.tasks else set(range(30))
    if not wanted or not wanted.issubset(set(tasks.task_id)): raise ValueError('invalid task selection')
    d=pd.read_csv(ROOT/c['input']).sort_values(['eoi','datum']).reset_index(drop=True)
    assignments=pd.read_csv(ROOT/c['reused_results']/'manifests/row_fold_assignments.csv')
    if not d[['eoi','datum','bap']].equals(assignments[['eoi','datum','bap']]): raise AssertionError('r02 assignment mismatch')
    rows=[]; audits=[]
    for task in tasks[tasks.task_id.isin(wanted)].to_dict('records'):
        tid=task['task_id']; folder=RUN/'predictions'/f'task_{tid:03d}'
        meta=json.loads((folder/'metadata.json').read_text()); csv=folder/'predictions.csv'
        p=pd.read_csv(csv,dtype={'fold':str})
        recipe=c['recipes'][task['arm']]
        column={'loso':'loso_station','dispersed_mod9':'disp_fold'}[task['protocol']]
        held=assignments[column].astype(str).eq(task['fold'])
        train,test=d[~held],d[held]
        if meta['task']!=task: raise AssertionError(f'{tid}: task metadata mismatch')
        for key,value in [('input',c['input']),('input_sha256',sha(ROOT/c['input'])),
                          ('contract_sha256',sha(RUN/'contract.json')),('manifest_sha256',sha(RUN/'fit_manifest.csv')),
                          ('worker_sha256',c['sources']['APR/analysis/clean_rebuild/review_fit_worker.py']),
                          ('prediction_sha256',sha(csv)),('train_key_hash',key_hash(train)),('test_key_hash',key_hash(test)),
                          ('n_train',len(train)),('n_test',len(test)),('features',recipe['features']),
                          ('resolved_monotone_tuple',recipe['constraints'])]:
            if meta[key]!=value: raise AssertionError(f'{tid}: {key} mismatch')
        params=meta['effective_xgboost_parameters']
        for key,value in c['xgboost_parameters'].items():
            if params[key]!=value: raise AssertionError(f'{tid}: parameter {key}')
        if params['monotone_constraints']!=recipe['constraints']: raise AssertionError('parameter constraint mismatch')
        if not np.isclose(params['base_score'],np.log1p(train.bap).mean(),rtol=1e-12): raise AssertionError('initialisation mismatch')
        learner=meta['booster_config']['learner']
        if int(learner['learner_model_param']['num_feature'])!=len(recipe['features']): raise AssertionError('effective feature count')
        if learner['objective']['name']!='reg:squarederror': raise AssertionError('effective objective')
        if p.duplicated(['eoi','datum']).any() or len(p)!=len(test): raise AssertionError('prediction keys/count')
        if not p[['eoi','datum']].reset_index(drop=True).equals(test[['eoi','datum']].reset_index(drop=True)): raise AssertionError('test support mismatch')
        if not np.allclose(p.observed,test.bap,rtol=0,atol=1e-12): raise AssertionError('target mismatch')
        numeric=p[['observed','prediction','prediction_untruncated','prediction_target_scale']]
        if not np.isfinite(numeric).all().all(): raise AssertionError('nonfinite prediction')
        if not np.allclose(p.prediction_untruncated,np.expm1(p.prediction_target_scale),rtol=2e-6,atol=1e-7): raise AssertionError('backtransform mismatch')
        if not np.allclose(p.prediction,np.maximum(0,p.prediction_untruncated),rtol=0,atol=1e-12): raise AssertionError('floor mismatch')
        if meta['zero_floor_count']!=int(p.prediction_untruncated.lt(0).sum()): raise AssertionError('floor count mismatch')
        for key in ['task_id','arm','protocol','fold']:
            if not p[key].eq(task[key]).all(): raise AssertionError('prediction task label mismatch')
        p['input_variant']='permissive'; p['static_support']=recipe['static_support']; p['target_scale']='log1p'; p['weighting']='uniform'
        rows.append(p); audits.append({'task_id':tid,'arm':task['arm'],'n_train':len(train),'n_test':len(test),'features':len(recipe['features']),'prediction_sha256':sha(csv)})
    predictions=pd.concat(rows,ignore_index=True)
    if predictions.duplicated(['arm','protocol','eoi','datum']).any(): raise AssertionError('duplicate assembled predictions')
    if not a.tasks:
        if len(predictions)!=12924: raise AssertionError('total predictions')
        for arm,g in predictions.groupby('arm'):
            keys=g[['eoi','datum']].sort_values(['eoi','datum']).reset_index(drop=True)
            if not keys.equals(d[['eoi','datum']]): raise AssertionError('incomplete arm support')
    a.output.mkdir(parents=True)
    predictions.to_csv(a.output/'validated_predictions.csv',index=False)
    pd.DataFrame(audits).to_csv(a.output/'task_audit.csv',index=False)
    report={'status':'PASS','tasks':sorted(wanted),'n_tasks':len(audits),'prediction_rows':len(predictions),
            'independent_holdouts':'saved r02 row/fold assignments','validator_sha256':sha(Path(__file__))}
    (a.output/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
