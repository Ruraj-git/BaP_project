#!/usr/bin/env python3
"""Complete corrected-rerun evidence tables, from one explicit result lineage."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scoring_core import scores,paired,reconstruct
from gap_baselines import run as baseline_predictions

IND='SK0018A'
IDENT=['arm','target_scale','weighting','input_variant','static_support']

def method_id(row):
    return '|'.join(str(row[x]) for x in IDENT)

def scopes(d):
    yield 'all_network',d
    yield 'nonindustrial',d[d.eoi.ne(IND)]
    yield 'industrial_descriptive',d[d.eoi.eq(IND)]
    for level in ('higher','lower'):
        yield 'nonindustrial_'+level,d[d.eoi.ne(IND)&d.stratum.eq(level)]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--dataset',type=Path,required=True); a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    d=pd.read_csv(a.input/'validated_predictions.csv',parse_dates=['datum'])
    d['method']=d.apply(method_id,axis=1); d['year']=d.datum.dt.year
    if d.duplicated(['method','protocol','eoi','datum']).any(): raise ValueError('duplicated method predictions')
    frame=pd.read_csv(a.dataset,parse_dates=['datum'])
    reference=frame[['eoi','datum','bap']].rename(columns={'bap':'observed'})
    reference['year']=reference.datum.dt.year
    primary=reference[reference.year.isin([2024,2025])]
    strata=primary.groupby(['eoi','year'],as_index=False).agg(sampled_mean=('observed','mean'),n_sampled=('observed','size'))
    strata['stratum']=np.where(strata.sampled_mean>1,'higher','lower')
    def label(x): return x.merge(strata[['eoi','year','stratum']],on=['eoi','year'],how='left',validate='many_to_one')
    d=label(d); summaries=[]; intervals=[]; station_rows=[]; annual_rows=[]; influence=[]
    def summarize(table,estimand,protocol,method,support,period='primary_2024_2025'):
        for scope,g in scopes(table):
            if len(g): summaries.append(dict(method=method,protocol=protocol,estimand=estimand,support=support,period=period,scope=scope,n=len(g),n_stations=g.eoi.nunique(),**scores(g.observed,g.prediction)))
    for (method,protocol),group in d.groupby(['method','protocol']):
        for period,g in [('all_period',group),('primary_2024_2025',group[group.year.isin([2024,2025])])]:
            summarize(g,'daily',protocol,method,'all_expected_targets',period)
        g=group[group.year.isin([2024,2025])]
        for station,st in g.groupby('eoi'):
            station_rows.append(dict(method=method,protocol=protocol,eoi=station,n=len(st),SSE=float(((st.prediction-st.observed)**2).sum()),**scores(st.observed,st.prediction)))
        # Only these tasks predict every sampled date in each eligible year.
        if protocol in ('block30','loso'):
            annual=g.groupby(['eoi','year'],as_index=False).agg(observed=('observed','mean'),prediction=('prediction','mean'),n=('observed','size'))
            annual=annual.merge(strata[['eoi','year','n_sampled']],on=['eoi','year'],validate='one_to_one')
            if not annual.n.eq(annual.n_sampled).all(): raise ValueError('incomplete annual prediction support')
            annual=label(annual[annual.n>=20]); annual['method']=method; annual['protocol']=protocol; annual_rows.append(annual)
            summarize(annual,'sampled_date_station_year_mean',protocol,method,'all_sampled_dates')
    # Select contrasts by exact identities, not labels that collapse input variants.
    def mid(arm,variant='permissive',target='log1p',weight='uniform'):
        support='500m' if arm.startswith('G') else '2km_exact'
        return '|'.join([arm,target,weight,variant,support])
    comparisons=[(mid('FULL_ID'),mid('PM_XGB')),(mid('G_PLUS_P'),mid('G')),
                 (mid('G_PLUS_P','strict18'),mid('G')),(mid('G_PLUS_P','strict18'),mid('G_PLUS_P')),
                 (mid('M_AUX'),mid('M_AUX',weight='inverse_1_over_bap_plus_0_5')),
                 (mid('G_PLUS_P',target='identity'),mid('G_PLUS_P'))]
    annual=pd.concat(annual_rows,ignore_index=True)
    def contrast(x,y,keys,scheme,method_a,method_b,protocol,estimand,support):
        for scope,xx in scopes(x):
            if scope=='industrial_descriptive' or xx.eoi.nunique()<2: continue
            yy=y.merge(xx[keys],on=keys,validate='one_to_one')
            for metric in ('RMSE','MAE','bias'):
                intervals.append(dict(method_a=method_a,method_b=method_b,protocol=protocol,estimand=estimand,support=support,scope=scope,period='primary_2024_2025',**paired(xx,yy,keys,scheme,metric)))
            if scope=='nonindustrial' and estimand=='daily':
                for station in sorted(xx.eoi.unique()):
                    ax=xx[xx.eoi.ne(station)]; by=yy[yy.eoi.ne(station)]
                    influence.append(dict(method_a=method_a,method_b=method_b,protocol=protocol,excluded_station=station,difference_RMSE=scores(ax.observed,ax.prediction)['RMSE']-scores(by.observed,by.prediction)['RMSE']))
    for ma,mb in comparisons:
        for protocol in ('block30','loso','year_out','season_year_out','calendar_season_out','dispersed_mod9'):
            x=d[(d.method==ma)&(d.protocol==protocol)&d.year.isin([2024,2025])]; y=d[(d.method==mb)&(d.protocol==protocol)&d.year.isin([2024,2025])]
            if x.empty or y.empty: continue
            contrast(x,y,['eoi','datum'],'station_x_block' if protocol=='block30' else 'station',ma,mb,protocol,'daily','all_expected_targets')
            if protocol in ('block30','loso'):
                contrast(annual[(annual.method==ma)&(annual.protocol==protocol)],annual[(annual.method==mb)&(annual.protocol==protocol)],['eoi','year'],'station',ma,mb,protocol,'sampled_date_station_year_mean','all_sampled_dates')
    baseline=baseline_predictions(frame); baseline['year']=baseline.datum.dt.year; baseline=label(baseline)
    gap=d[d.arm.eq('M_AUX')&d.protocol.isin(['block30','dispersed_mod9'])].copy()
    gap=pd.concat([gap,baseline],ignore_index=True)
    recovery=[]; support_rows=[]
    for protocol in ('block30','dispersed_mod9'):
        allgap=gap[gap.protocol.eq(protocol)&gap.year.isin([2024,2025])]
        bracket=allgap[allgap.method.eq('INTERP_LINEAR')&allgap.prediction.notna()][['eoi','datum']]
        common=allgap.merge(bracket,on=['eoi','datum'],validate='many_to_one')
        for method,g in common.groupby('method'):
            if not np.isfinite(g.prediction).all(): raise ValueError('nonfinite common-support gap prediction')
            summarize(g,'daily',protocol,method,'common_bracketable')
            support_rows.append(dict(protocol=protocol,method=method,primary_total=len(primary),common=len(g),excluded=len(primary)-len(g)))
            if protocol=='dispersed_mod9':
                rec=label(reconstruct(primary,g)); rec['method']=method; recovery.append(rec)
                summarize(rec,'foldwise_sampled_mean_recovery',protocol,method,'common_bracketable_replacements')
        ma=mid('M_AUX')
        x=common[common.method.eq(ma)]
        for mb in sorted(set(common.method)-{ma}):
            contrast(x,common[common.method.eq(mb)],['eoi','datum'],'station_x_block' if protocol=='block30' else 'station',ma,mb,protocol,'daily','common_bracketable')
    rec=pd.concat(recovery,ignore_index=True)
    ma=mid('M_AUX')
    for mb in sorted(set(rec.method)-{ma}):
        contrast(rec[rec.method.eq(ma)],rec[rec.method.eq(mb)],['eoi','year','fold'],'station',ma,mb,'dispersed_mod9','foldwise_sampled_mean_recovery','common_bracketable_replacements')
    st=pd.DataFrame(station_rows); st['SSE_share']=st.SSE/st.groupby(['method','protocol']).SSE.transform('sum')
    a.output.mkdir(parents=True)
    outputs={'summary_metrics':pd.DataFrame(summaries),'paired_uncertainty':pd.DataFrame(intervals),'station_metrics':st,'station_influence':pd.DataFrame(influence),'station_year_means':annual,'observed_strata':strata,'baseline_predictions':baseline,'foldwise_reconstruction':rec,'common_support':pd.DataFrame(support_rows)}
    for name,table in outputs.items(): table.to_csv(a.output/(name+'.csv'),index=False)
    floors=d.groupby(['method','protocol']).prediction_untruncated.agg(n='size',negative=lambda x:int((x<0).sum())).reset_index(); floors.to_csv(a.output/'zero_floor_counts.csv',index=False)
    (a.output/'scoring_report.json').write_text(json.dumps({k:len(v) for k,v in outputs.items()},indent=2)+'\n')
    print('Scoring complete:',a.output)

if __name__=='__main__': main()
