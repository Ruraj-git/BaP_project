#!/usr/bin/env python3
"""Freeze r03 review tasks from r02; small descriptive summaries, no fits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from review_fit_worker import ROOT, RUN, sha, key_hash, held_mask

OLD=ROOT/'APR/results/clean_rebuild/r02'
IND='SK0018A'

def scopes(d):
    return [('all_network',d),('nonindustrial',d[d.eoi.ne(IND)]),('industrial_descriptive',d[d.eoi.eq(IND)])]

def metrics(d):
    err=d.prediction-d.observed
    return dict(n=len(d),n_stations=d.eoi.nunique(),RMSE=float(np.sqrt((err**2).mean())),MAE=float(err.abs().mean()),bias=float(err.mean()))

def describe(d):
    return dict(n=len(d),n_stations=d.eoi.nunique(),mean=d.bap.mean(),median=d.bap.median(),q25=d.bap.quantile(.25),q75=d.bap.quantile(.75),minimum=d.bap.min(),maximum=d.bap.max())

def main():
    if RUN.exists(): raise FileExistsError('Refusing to overwrite '+str(RUN))
    old=json.loads((OLD/'manifests/model_and_fold_contract.json').read_text())
    inp=OLD/'inputs/train_ready_permissive_500m.csv'
    assert sha(inp)==old['input_files'][inp.name]['sha256']
    for name,digest in old['code_hashes'].items(): assert sha(Path(__file__).parent/name)==digest
    data=pd.read_csv(inp).sort_values(['eoi','datum']).reset_index(drop=True)
    folds=pd.read_csv(OLD/'manifests/row_fold_assignments.csv')
    assert data[['eoi','datum','bap']].equals(folds[['eoi','datum','bap']])
    assert len(data)==6462 and data.eoi.nunique()==21 and not data.duplicated(['eoi','datum']).any()
    calendars=['is_weekend','month','doy_sin','doy_cos','heating_season']
    pollutants=old['feature_contract']['G_PLUS_P'][38:]
    assert len(pollutants)==9 and all(x.startswith(('pm10','pm25','no2')) for x in pollutants)
    recipes={'P_CAL':{'features':pollutants+calendars,'constraints':[0]*14,'static_support':'none'},
             'G_PLUS_P':{'features':old['feature_contract']['G_PLUS_P'],'constraints':old['resolved_monotone_tuples']['G_PLUS_P'],'static_support':'500m'}}
    rows=[]
    for arm,protocol,labels in [('P_CAL','loso',sorted(data.eoi.unique())),('G_PLUS_P','dispersed_mod9',range(9))]:
        for label in labels:
            row=dict(task_id=len(rows),arm=arm,protocol=protocol,fold=str(label),n_features=len(recipes[arm]['features']))
            held=held_mask(data,row)
            independent=folds.loso_station.eq(str(label)) if protocol=='loso' else folds.disp_fold.eq(label)
            assert held.equals(independent)
            train,test=data[~held],data[held]
            row.update(n_train=len(train),n_test=len(test),train_key_hash=key_hash(train),test_key_hash=key_hash(test))
            rows.append(row)
    assert len(rows)==30 and rows[0]['arm']=='P_CAL' and rows[21]['arm']=='G_PLUS_P'
    for arm in recipes: assert sum(r['n_test'] for r in rows if r['arm']==arm)==6462
    RUN.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(RUN/'fit_manifest.csv',index=False)
    paths=[inp,OLD/'manifests/model_and_fold_contract.json',OLD/'manifests/row_fold_assignments.csv',
           Path(__file__),Path(__file__).with_name('review_fit_worker.py'),ROOT/'APR/rebuild/R03_REVIEW_RESPONSE_PLAN.md']
    contract={'input':str(inp.relative_to(ROOT)),'sources':{str(p.relative_to(ROOT)):sha(p) for p in paths},
              'manifest_sha256':sha(RUN/'fit_manifest.csv'),'recipes':recipes,'xgboost_parameters':old['xgboost_parameters'],
              'target':'log1p','weighting':'uniform','initialisation':'unweighted mean training target on fitted scale',
              'floor':'max(0,expm1(predicted log target))','seed':42,'uncertainty':old['uncertainty'],
              'reused_results':str(OLD.relative_to(ROOT)),'primary':old['primary']}
    (RUN/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    out=RUN/'no_fit_summaries'; out.mkdir()
    data['year']=pd.to_datetime(data.datum).dt.year
    data['season']=pd.to_datetime(data.datum).dt.month.map({12:'DJF',1:'DJF',2:'DJF',3:'MAM',4:'MAM',5:'MAM',6:'JJA',7:'JJA',8:'JJA',9:'SON',10:'SON',11:'SON'})
    primary=data[data.year.isin([2024,2025])]
    summaries=[]
    for scope,d in scopes(primary):
        summaries.append(dict(scope=scope,group='pooled',label='2024-2025',**describe(d)))
        for group in ['year','season']:
            for label,part in d.groupby(group): summaries.append(dict(scope=scope,group=group,label=str(label),**describe(part)))
    pd.DataFrame(summaries).to_csv(out/'observed_distribution.csv',index=False)
    primary.groupby(['eoi','year']).bap.agg(['size','mean','median','min','max']).reset_index().to_csv(out/'station_year_description.csv',index=False)
    ann=pd.read_csv(OLD/'scores/station_year_means.csv')
    prefix=lambda arm:arm+'|log1p|uniform|permissive|500m'
    g=ann[ann.method.eq(prefix('G'))&ann.protocol.eq('loso')]
    gp=ann[ann.method.eq(prefix('G_PLUS_P'))&ann.protocol.eq('loso')]
    joined=g.merge(gp,on=['eoi','year'],suffixes=('_g','_gp'),validate='one_to_one')
    assert len(joined)==42 and np.allclose(joined.observed_g,joined.observed_gp,atol=1e-12,rtol=0)
    effects=[]
    for scope,d in scopes(joined):
        if scope=='industrial_descriptive': continue
        for excluded in ['none']+sorted(d.eoi.unique()):
            part=d if excluded=='none' else d[d.eoi.ne(excluded)]
            a=float(np.sqrt(((part.prediction_gp-part.observed_g)**2).mean()))
            b=float(np.sqrt(((part.prediction_g-part.observed_g)**2).mean()))
            effects.append(dict(scope=scope,excluded_station=excluded,n_station_years=len(part),n_stations=part.eoi.nunique(),g_plus_p_RMSE=a,g_RMSE=b,difference_RMSE=a-b))
    pd.DataFrame(effects).to_csv(out/'annual_station_influence.csv',index=False)
    preds=pd.read_csv(OLD/'assembled/validated_predictions.csv')
    year=preds[preds.protocol.eq('year_out')&preds.fold.astype(str).eq('2025')&preds.target_scale.eq('log1p')&preds.weighting.eq('uniform')]
    assert len(year)==2*len(primary[primary.year.eq(2025)])
    rows=[]
    for arm,p in year.groupby('arm'):
        for scope,d in scopes(p): rows.append(dict(arm=arm,scope=scope,interpretation='2025 forward-time stress test; development-exposed',**metrics(d)))
    pd.DataFrame(rows).to_csv(out/'year2025_forward_stress.csv',index=False)
    dates=data[['eoi','datum','year']].copy(); dates['datum']=pd.to_datetime(dates.datum)
    dates['days_since_previous_sample']=dates.groupby('eoi').datum.diff().dt.days
    dates[dates.year.isin([2024,2025])].to_csv(out/'observed_sampling_intervals.csv',index=False)
    base=pd.read_csv(OLD/'scores/baseline_predictions.csv')
    brackets=base[base.method.eq('INTERP_LINEAR')&base.protocol.eq('dispersed_mod9')&base.year.isin([2024,2025])]
    brackets[['eoi','datum','fold','bracket_days']].to_csv(out/'dispersed_brackets.csv',index=False)
    rec=pd.read_csv(OLD/'scores/foldwise_reconstruction.csv')
    rec=rec[rec.method.eq('M_AUX|log1p|uniform|permissive|2km_exact')]
    rec[['eoi','year','fold','n_sampled','n_replaced','fraction_replaced']].to_csv(out/'replacement_geometry.csv',index=False)
    assert len(rec)==378 and rec[rec.eoi.ne(IND)].shape[0]==360
    summary_sources=[OLD/'scores/station_year_means.csv',OLD/'assembled/validated_predictions.csv',OLD/'scores/baseline_predictions.csv',OLD/'scores/foldwise_reconstruction.csv']
    (out/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in summary_sources},indent=2)+'\n')
    report={'status':'PREPARED_NOT_SUBMITTED','tasks':30,'P_CAL_loso':21,'G_PLUS_P_dispersed':9,
            'input_rows':len(data),'input_stations':data.eoi.nunique(),'primary_rows':len(primary),
            'all_fold_assignments_match_r02':True,'r02_inputs_and_fit_code_hashes_verified':True,
            'new_fits_run':0,'bootstrap_jobs_run':0,'manuscript_modified':False}
    (RUN/'preflight_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
