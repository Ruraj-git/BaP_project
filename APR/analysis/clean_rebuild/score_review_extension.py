#!/usr/bin/env python3
"""Score the frozen two-comparison extension; no fitting or model selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from review_fit_worker import ROOT, RUN, sha
from scoring_core import scores, paired, reconstruct

OLD=ROOT/'APR/results/clean_rebuild/r02'
IND='SK0018A'

def population(frame, strata=True):
    yield 'all_network',frame
    yield 'nonindustrial',frame[frame.eoi.ne(IND)]
    yield 'industrial_descriptive',frame[frame.eoi.eq(IND)]
    if strata:
        for label in ['higher','lower']:
            yield 'nonindustrial_'+label,frame[frame.eoi.ne(IND)&frame.stratum.eq(label)]

def main():
    out=RUN/'scores'
    if out.exists(): raise FileExistsError(out)
    validation=json.loads((RUN/'assembled/validation_report.json').read_text())
    if validation['status']!='PASS' or validation['n_tasks']!=30: raise ValueError('full validation required')
    frozen=json.loads((RUN/'contract.json').read_text())['sources']
    frozen.update(json.loads((RUN/'no_fit_summaries/source_hashes.json').read_text()))
    for relative,digest in frozen.items():
        if sha(ROOT/relative)!=digest: raise ValueError('changed source: '+relative)
    new=pd.read_csv(RUN/'assembled/validated_predictions.csv',parse_dates=['datum'])
    old=pd.read_csv(OLD/'assembled/validated_predictions.csv',parse_dates=['datum'])
    raw=pd.read_csv(ROOT/json.loads((RUN/'contract.json').read_text())['input'],parse_dates=['datum'])
    reference=raw[['eoi','datum','bap']].rename(columns={'bap':'observed'})
    reference=reference[reference.datum.dt.year.isin([2024,2025])].copy()
    reference['year']=reference.datum.dt.year
    strata=reference.groupby(['eoi','year'],as_index=False).agg(sampled_mean=('observed','mean'),n_sampled=('observed','size'))
    strata['stratum']=np.where(strata.sampled_mean>1,'higher','lower')
    def label(frame):
        frame=frame.copy()
        if 'datum' in frame: frame['year']=frame.datum.dt.year
        return frame.drop(columns=['stratum'],errors='ignore').merge(strata[['eoi','year','stratum']],on=['eoi','year'],validate='many_to_one')
    def primary(frame):
        return label(frame[frame.datum.dt.year.isin([2024,2025])].copy())
    def verify_targets(frame):
        joined=frame.merge(reference[['eoi','datum','observed']],on=['eoi','datum'],how='left',suffixes=('','_ref'),validate='many_to_one')
        if not np.array_equal(joined.observed,joined.observed_ref): raise ValueError('target mismatch')
    summaries=[]; intervals=[]; influence=[]; check_count=0
    def summary(frame,method,protocol,estimand,support,use_strata=True):
        for scope,d in population(frame,use_strata):
            if d.empty: continue
            result=scores(d.observed,d.prediction)
            # Independent point-score arithmetic for every output row.
            error=d.prediction.to_numpy()-d.observed.to_numpy()
            assert np.isclose(result['RMSE'],np.linalg.norm(error)/np.sqrt(len(d)),rtol=1e-12)
            assert np.isclose(result['MAE'],np.sum(np.abs(error))/len(d),rtol=1e-12)
            assert np.isclose(result['bias'],np.sum(error)/len(d),rtol=1e-12,atol=1e-14)
            counts={'n':len(d),'n_stations':d.eoi.nunique()}
            if estimand=='sampled_date_station_year_mean': counts['n_station_years']=len(d)
            if estimand=='foldwise_sampled_mean_recovery':
                counts['n_reconstructions']=len(d); counts['n_station_years']=len(d[['eoi','year']].drop_duplicates())
            summaries.append(dict(method=method,protocol=protocol,estimand=estimand,support=support,period='primary_2024_2025',scope=scope,**counts,**result))
    def contrast(a,b,ma,mb,protocol,estimand,support,keys,use_strata=True):
        for scope,x in population(a,use_strata):
            if scope=='industrial_descriptive' or x.eoi.nunique()<2: continue
            y=b.merge(x[keys],on=keys,validate='one_to_one')
            for metric in ['RMSE','MAE','bias']:
                result=paired(x,y,keys,scheme='station',metric=metric)
                if estimand=='sampled_date_station_year_mean': result['n_station_years']=len(x)
                if estimand=='foldwise_sampled_mean_recovery':
                    result['n_reconstructions']=len(x); result['n_station_years']=len(x[['eoi','year']].drop_duplicates())
                intervals.append(dict(method_a=ma,method_b=mb,protocol=protocol,estimand=estimand,support=support,scope=scope,period='primary_2024_2025',**result))
    # LOSO: new P_CAL, reused exact G/G+P uniform log-target fits only.
    selected=old[old.arm.isin(['G','G_PLUS_P'])&old.protocol.eq('loso')&old.target_scale.eq('log1p')&old.weighting.eq('uniform')&old.input_variant.eq('permissive')]
    loso=primary(pd.concat([selected,new[new.arm.eq('P_CAL')]],ignore_index=True))
    if loso.duplicated(['arm','eoi','datum']).any(): raise ValueError('duplicate LOSO predictions')
    verify_targets(loso)
    if loso.groupby('arm').size().to_dict()!={'G':5089,'G_PLUS_P':5089,'P_CAL':5089}: raise ValueError('LOSO support mismatch')
    annual=[]
    for method,d in loso.groupby('arm'):
        summary(d,method,'loso','daily','all_expected_targets')
        a=d.groupby(['eoi','year'],as_index=False).agg(observed=('observed','mean'),prediction=('prediction','mean'),n_obs=('observed','size'))
        a=a.merge(strata[['eoi','year','n_sampled']],on=['eoi','year'],validate='one_to_one')
        if not a.n_obs.eq(a.n_sampled).all(): raise ValueError('incomplete station-year')
        a=label(a[a.n_obs>=20]); a['method']=method; annual.append(a)
        summary(a,method,'loso','sampled_date_station_year_mean','all_sampled_dates')
    annual=pd.concat(annual,ignore_index=True)
    for comparator in ['P_CAL','G']:
        for frame,col,estimand,keys,support in [(loso,'arm','daily',['eoi','datum'],'all_expected_targets'),(annual,'method','sampled_date_station_year_mean',['eoi','year'],'all_sampled_dates')]:
            a=frame[frame[col].eq('G_PLUS_P')]; b=frame[frame[col].eq(comparator)]
            contrast(a,b,'G_PLUS_P',comparator,'loso',estimand,support,keys)
            if estimand=='sampled_date_station_year_mean':
                for scope,x in population(a,False):
                    if scope=='industrial_descriptive': continue
                    y=b.merge(x[keys],on=keys,validate='one_to_one')
                    for station in sorted(x.eoi.unique()):
                        xs=x[x.eoi.ne(station)]; ys=y[y.eoi.ne(station)]
                        influence.append(dict(method_a='G_PLUS_P',method_b=comparator,scope=scope,excluded_station=station,n_station_years=len(xs),difference_RMSE=scores(xs.observed,xs.prediction)['RMSE']-scores(ys.observed,ys.prediction)['RMSE']))
    # Dispersed: precisely the existing six-method common bracketable support.
    base=pd.read_csv(OLD/'scores/baseline_predictions.csv',parse_dates=['datum'])
    base=base[base.protocol.eq('dispersed_mod9')].copy()
    aux=old[old.arm.eq('M_AUX')&old.protocol.eq('dispersed_mod9')].copy()
    aux['method']=np.where(aux.weighting.eq('uniform'),'M_AUX_UNIFORM','M_AUX_INVERSE')
    gp=new[new.arm.eq('G_PLUS_P')].copy(); gp['method']='G_PLUS_P'
    allgap=primary(pd.concat([base,aux,gp],ignore_index=True)); verify_targets(allgap)
    bracket=allgap[allgap.method.eq('INTERP_LINEAR')&allgap.prediction.notna()][['eoi','datum']]
    gap=allgap.merge(bracket,on=['eoi','datum'],validate='many_to_one')
    if len(bracket)!=5066 or gap.groupby('method').size().ne(5066).any(): raise ValueError('changed common support')
    if gap.method.nunique()!=7 or gap.duplicated(['method','eoi','datum']).any(): raise ValueError('gap method mismatch')
    if not np.isfinite(gap.prediction).all(): raise ValueError('nonfinite gap predictions')
    recovery=[]
    for method,d in gap.groupby('method'):
        summary(d,method,'dispersed_mod9','daily','common_bracketable',False)
        r=label(reconstruct(reference,d)); r['method']=method
        if len(r)!=378 or r[r.eoi.ne(IND)].shape[0]!=360: raise ValueError('reconstruction count')
        # Explicit retained-observation reconstruction independently checks every case.
        for row in r.itertuples():
            observed=reference[reference.eoi.eq(row.eoi)&reference.year.eq(row.year)].set_index('datum').observed.copy()
            hidden=d[d.eoi.eq(row.eoi)&d.year.eq(row.year)&d.fold.astype(str).eq(str(row.fold))]
            observed.loc[hidden.datum]=hidden.prediction.to_numpy()
            if not np.isclose(observed.mean(),row.prediction,rtol=1e-12,atol=1e-12): raise ValueError('explicit reconstruction mismatch')
            check_count+=1
        recovery.append(r)
        summary(r,method,'dispersed_mod9','foldwise_sampled_mean_recovery','common_bracketable_replacements',False)
    recovery=pd.concat(recovery,ignore_index=True)
    for comparator in sorted(set(gap.method)-{'G_PLUS_P'}):
        contrast(gap[gap.method.eq('G_PLUS_P')],gap[gap.method.eq(comparator)],'G_PLUS_P',comparator,'dispersed_mod9','daily','common_bracketable',['eoi','datum'],False)
        contrast(recovery[recovery.method.eq('G_PLUS_P')],recovery[recovery.method.eq(comparator)],'G_PLUS_P',comparator,'dispersed_mod9','foldwise_sampled_mean_recovery','common_bracketable_replacements',['eoi','year','fold'],False)
    out.mkdir()
    outputs={'summary_metrics':pd.DataFrame(summaries),'paired_uncertainty':pd.DataFrame(intervals),'station_year_means':annual,
             'annual_station_influence':pd.DataFrame(influence),'foldwise_reconstruction':recovery,'dispersed_common_predictions':gap,
             'loso_primary_predictions':loso}
    for name,table in outputs.items(): table.to_csv(out/(name+'.csv'),index=False)
    sources=[RUN/'contract.json',RUN/'assembled/validated_predictions.csv',OLD/'assembled/validated_predictions.csv',OLD/'scores/baseline_predictions.csv',Path(__file__),Path(__file__).with_name('scoring_core.py')]
    (out/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in sources},indent=2)+'\n')
    report={'status':'PASS','output_rows':{k:len(v) for k,v in outputs.items()},'explicit_reconstruction_checks':check_count,
            'new_fits_in_scoring':0,'bootstrap_replicates':2000,'seed':20260921,'industrial_only_bootstrap':False,
            'primary_gap_contrast':'G_PLUS_P minus INTERP_LINEAR','primary_loso_contrast':'G_PLUS_P minus P_CAL'}
    (out/'scoring_report.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
