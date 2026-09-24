#!/usr/bin/env python3
"""Render the bounded review extension plus unchanged r02 evidence; no fits."""
import hashlib
import json
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import generate_manuscript_r02 as r

ROOT=r.ROOT
RUN=ROOT/'APR/results/clean_rebuild/r03_review'
OUT=ROOT/'APR/manuscript/generated_r03'
S=pd.read_csv(RUN/'scores/summary_metrics.csv')
P=pd.read_csv(RUN/'scores/paired_uncertainty.csv')
SELECTED=[]

def select(table,**filters):
    rows=table
    for k,v in filters.items(): rows=rows[rows[k].eq(v)]
    assert len(rows)==1,(filters,len(rows))
    SELECTED.append({'source':'r03/'+('paired_uncertainty.csv' if table is P else 'summary_metrics.csv'),'filters':filters})
    return rows.iloc[0]

def score(method,protocol='loso',scope='nonindustrial',estimand='daily'):
    return select(S,method=method,protocol=protocol,scope=scope,estimand=estimand)

def pair(comparator='P_CAL',protocol='loso',scope='nonindustrial',estimand='daily'):
    return select(P,method_a='G_PLUS_P',method_b=comparator,protocol=protocol,scope=scope,estimand=estimand,metric='RMSE')

def ci_macros(prefix,q,digits=3):
    for suffix,field in [('', 'estimate'),('Low','ci_low'),('High','ci_high')]: r.macro(prefix+suffix,q[field],digits)

def main():
    # Rerender existing r02 assets into a NEW directory; never alter r02 assets.
    r.OUT=OUT; r.main()
    for manifest in [RUN/'scores/source_hashes.json',RUN/'no_fit_summaries/source_hashes.json']:
        for name,digest in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    annual='sampled_date_station_year_mean'; recovery='foldwise_sampled_mean_recovery'
    ci_macros('EnvDailyDiff',pair())
    ci_macros('EnvMeanDiff',pair(estimand=annual))
    ci_macros('EnvAllDiff',pair(scope='all_network'))
    r.macro('PollutantLoso',score('P_CAL').RMSE)
    r.macro('PollutantMean',score('P_CAL',estimand=annual).RMSE)
    r.macro('EnvReduction',100*(1-score('G_PLUS_P').RMSE/score('P_CAL').RMSE),1)
    r.macro('GDailyCod',score('G').agreement_R2)
    r.macro('GMeanCod',score('G',estimand=annual).agreement_R2)
    r.macro('AllBlockG',r.score(r.G,scope='all_network').RMSE)
    r.macro('AllBlockP',r.score(r.GP,scope='all_network').RMSE)
    for name,method in [('RecoveryGP','G_PLUS_P'),('RecoveryInterp','INTERP_LINEAR')]:
        r.macro(name,score(method,'dispersed_mod9',estimand=recovery).RMSE,4)
    r.macro('DispersedGP',score('G_PLUS_P','dispersed_mod9').RMSE)
    r.macro('DispersedInterp',score('INTERP_LINEAR','dispersed_mod9').RMSE)
    ci_macros('RecoveryDiff',pair('INTERP_LINEAR','dispersed_mod9',estimand=recovery),4)
    ci_macros('DispersedDiff',pair('INTERP_LINEAR','dispersed_mod9'))
    ci_macros('GapMauxDiff',pair('M_AUX_UNIFORM','dispersed_mod9'))
    inf=pd.read_csv(RUN/'scores/annual_station_influence.csv')
    sub=inf[inf.method_b.eq('G')&inf.scope.eq('nonindustrial')]
    r.macro('AnnualOmitMin',-sub.difference_RMSE.max()); r.macro('AnnualOmitMax',-sub.difference_RMSE.min())
    r.macro('AnnualOmitInfluential',-sub.loc[sub.excluded_station.eq('SK0048A'),'difference_RMSE'].item())
    forward=pd.read_csv(RUN/'no_fit_summaries/year2025_forward_stress.csv')
    for method,suffix in [('G','G'),('G_PLUS_P','P')]:
        r.macro('Forward'+suffix,forward.loc[forward.arm.eq(method)&forward.scope.eq('nonindustrial'),'RMSE'].item())
    desc=pd.read_csv(RUN/'no_fit_summaries/observed_distribution.csv')
    rows=[]
    labels=[('all_network','All stations'),('nonindustrial','Nonindustrial'),('industrial_descriptive','Industrial')]
    for scope,label in labels:
        q=desc[desc.scope.eq(scope)&desc.group.eq('pooled')].iloc[0]
        season=lambda s:desc.loc[desc.scope.eq(scope)&desc.group.eq('season')&desc.label.eq(s),'mean'].item()
        rows.append([label,int(q.n),f'{q["mean"]:.3f}',f'{q["median"]:.3f}',f'{q.q25:.3f}--{q.q75:.3f}',f'{season("DJF"):.3f}',f'{season("JJA"):.3f}'])
        if scope=='nonindustrial':
            r.macro('ObservedMean',q['mean']); r.macro('ObservedMedian',q['median'])
    r.table('description_table','Observed B[a]P concentrations in 2024--2025 (ng~m$^{-3}$). Summaries pool sampled days; DJF and JJA denote winter and summer. Quartiles describe the observed distribution, not uncertainty in a mean.','tab:description',['Population','Samples','Mean','Median','Q1--Q3','DJF mean','JJA mean'],rows,'lrrrrrr')
    rows=[]
    for scope,label in labels[:2]:
        a,b=r.score(r.FULL,scope=scope),r.score(r.PM,scope=scope)
        rows.append([label,f'{b.RMSE:.3f}',f'{a.RMSE:.3f}',r.interval(r.pair(r.FULL,r.PM,scope=scope))])
    r.table('flex_table','Matched 30-day flexible-PM comparison, 2024--2025. All stations: 5089 observations; nonindustrial: 4852. Differences are FULL-ID minus PM-XGB with paired 95\\% intervals. Units: ng~m$^{-3}$.','tab:flex',['Population','PM-XGB RMSE','FULL-ID RMSE','Difference [95\\% interval]'],rows,'lrrl')
    rows=[]
    for scope,label in labels[:2]:
        for method,display in [('G','G'),('P_CAL','P+calendar'),('G_PLUS_P','G+P')]:
            d,m=score(method,scope=scope),score(method,scope=scope,estimand=annual)
            rows.append([label,display,f'{d.RMSE:.3f}',f'{d.agreement_R2:.3f}',f'{m.RMSE:.3f}',f'{m.agreement_R2:.3f}'])
    r.table('loso_table','LOSO performance for daily concentrations and station-year means over sampled dates, 2024--2025. Mean scores use 42 station-years (40 nonindustrial). RMSE is in ng~m$^{-3}$; $R^2$ is agreement-based and can be negative.','tab:loso',['Population','Model','Daily RMSE','Daily $R^2$','Mean RMSE','Mean $R^2$'],rows,'llrrrr')
    rows=[]
    methods=[('G+P','G_PLUS_P'),('PM ridge','PM_RIDGE'),('Harmonics','HARMONIC'),('Linear interpolation','INTERP_LINEAR'),('Log-linear interpolation','INTERP_LOGLINEAR')]
    for label,method in methods:
        rows.append([label]+[f'{score(method,"dispersed_mod9",scope,estimand).RMSE:.{4 if estimand==recovery else 3}f}' for estimand in ['daily',recovery] for scope in ['all_network','nonindustrial']])
    r.table('gap_table','Dispersed recovery on identical bracketable targets. Daily scores use 5066 observations (4830 nonindustrial); mean recovery uses 378 station-year/fold reconstructions (360 nonindustrial). Mean recovery replaces only that fold\'s hidden subset. Units: ng~m$^{-3}$.','tab:gap',['Method','Daily: all','Daily: nonind.','Mean: all','Mean: nonind.'],rows,'lrrrr')
    rows=[]
    for scope,label in labels[:2]:
        for est,elabel in [('daily','Daily'),(annual,'Sampled mean')]:
            rows.append([label,elabel,r.interval(pair(scope=scope,estimand=est))])
    r.table('environment_intervals','G+P minus P+calendar LOSO RMSE, with station-cluster 95\\% intervals. Units: ng~m$^{-3}$.','tab:envinterval',['Population','Quantity','Difference [95\\% interval]'],rows,'lll')
    rows=[]
    for scope,label in labels[:2]:
        for method,display in [('G_PLUS_P','G+P'),('M_AUX_UNIFORM','M-AUX uniform'),('M_AUX_INVERSE','M-AUX inverse')]:
            rows.append([label,display,f'{score(method,"dispersed_mod9",scope).RMSE:.3f}',f'{score(method,"dispersed_mod9",scope,recovery).RMSE:.4f}'])
    r.table('auxiliary_gap_table','Supporting dispersed-gap configurations on the same bracketable targets (ng~m$^{-3}$). M-AUX uses different metadata and spatial support; its contrast with G+P is not an isolated feature effect.','tab:auxgap',['Population','Model','Daily RMSE','Mean-recovery RMSE'],rows,'llrr')
    # Information forest: the same four registered contrasts, both populations.
    fig,ax=plt.subplots(figsize=(8,3.6))
    definitions=[('Richer inputs beyond PM: 30-day',lambda scope:r.pair(r.FULL,r.PM,scope=scope)),('Pollutants beyond G: 30-day',lambda scope:r.pair(r.GP,r.G,scope=scope)),('Pollutants beyond G: LOSO',lambda scope:pair('G',scope=scope)),('Environment beyond P: LOSO',lambda scope:pair(scope=scope))]
    for scope,color,shift,display in [('all_network','#555555',-.10,'All 21 stations'),('nonindustrial','#0072B2',.10,'Nonindustrial 20')]:
        for y,(_,fun) in enumerate(definitions):
            q=fun(scope); val=-q.estimate; lo=-q.ci_high; hi=-q.ci_low
            ax.errorbar(val,y+shift,xerr=[[val-lo],[hi-val]],fmt='o',capsize=3,color=color,label=display if y==0 else None)
    ax.set_yticks(range(4),[d[0] for d in definitions]); ax.invert_yaxis(); ax.axvline(0,color='grey',ls=':',lw=1)
    ax.set_xlabel('Reduction in daily RMSE (ng m$^{-3}$), 95% interval'); ax.legend(loc='lower right',fontsize=9)
    fig.tight_layout(); fig.savefig(OUT/'information_benefits.pdf'); plt.close(fig)
    # Larger separate-year station plots; all sites retained, industrial marked.
    ann=pd.read_csv(RUN/'scores/station_year_means.csv')
    obs=ann[ann.method.eq('G')]
    order=obs.groupby('eoi')['observed'].mean().sort_values().index.tolist()
    fig,axes=plt.subplots(2,1,figsize=(9,6.8),sharex=True,sharey=True)
    for year,ax in zip([2024,2025],axes):
        for method,label,color,marker,offset in [('G','G','#E69F00','^',-.15),('G_PLUS_P','G+P','#0072B2','o',.15),('Observed','Observed','#222222','D',0)]:
            d=(obs if method=='Observed' else ann[ann.method.eq(method)])
            d=d[d.year.eq(year)].set_index('eoi').loc[order]
            y=d.observed if method=='Observed' else d.prediction
            ax.scatter(np.arange(21)+offset,y,color=color,marker=marker,s=35,label=label,zorder=3)
        ax.set_title(str(year),loc='left'); ax.set_ylabel('Sampled-date mean (ng m$^{-3}$)'); ax.grid(axis='y',alpha=.22)
        ax.axvspan(order.index('SK0018A')-.45,order.index('SK0018A')+.45,color='#eeeeee',zorder=0)
    axes[0].legend(ncol=3,loc='upper left',fontsize=11)
    axes[1].set_xticks(range(21),[s+('*' if s=='SK0018A' else '') for s in order],rotation=55,ha='right',fontsize=9)
    axes[1].set_xlabel('Stations ordered by observed concentration; * industrial')
    fig.tight_layout(); fig.savefig(OUT/'station_means.pdf'); plt.close(fig)
    # Current network metadata, not historical candidate sites or concentrations.
    stations=pd.read_csv(ROOT/'data/stations.csv')
    coords=stations.set_index('eoi').loc[sorted(order)].reset_index()
    assert len(coords)==21 and coords.eoi.is_unique and coords[['lat','lon']].notna().all().all()
    assert set(coords.loc[coords.typ_zdroja.eq('I'),'eoi'])=={'SK0018A'}
    coords['map_number']=range(1,22)
    boundary_path=ROOT/'APR/data/boundaries/raw/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp'
    countries=gpd.read_file(boundary_path); boundary=countries[countries.ADM0_A3.eq('SVK')].to_crs(4326)
    assert len(boundary)==1
    fig,ax=plt.subplots(figsize=(8,3.8)); boundary.plot(ax=ax,facecolor='#f4f4f4',edgecolor='#555555',linewidth=.8)
    for code,marker,color,label in [('B','o','#0072B2','Background'),('T','s','#009E73','Traffic'),('I','^','#D55E00','Industrial')]:
        d=coords[coords.typ_zdroja.eq(code)]; ax.scatter(d.lon,d.lat,s=45,marker=marker,c=color,edgecolors='white',linewidth=.5,label=f'{label} (n={len(d)})',zorder=3)
    offsets={'SK0008A':(-10,-12),'SK0002A':(10,-7),'SK0048A':(-22,2),'SK0061A':(2,12),'SK0076A':(7,-12),'SK0214A':(5,-13),'SK0263A':(-12,6),'SK0078A':(5,-12),'SK0071A':(-12,6)}
    for d in coords.itertuples():
        leader=dict(arrowstyle='-',lw=.4,color='#666666') if d.eoi in ['SK0002A','SK0048A','SK0061A','SK0076A'] else None
        ax.annotate(str(d.map_number),(d.lon,d.lat),xytext=offsets.get(d.eoi,(5,5)),textcoords='offset points',fontsize=8,color='#222222',arrowprops=leader)
    ax.set_aspect(1/np.cos(np.deg2rad(coords.lat.mean()))); ax.set_xlabel('Longitude (°E)'); ax.set_ylabel('Latitude (°N)')
    ax.legend(loc='upper left',fontsize=9,frameon=False); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(OUT/'network.pdf'); plt.close(fig)
    coords.to_csv(OUT/'mapped_stations.csv',index=False)
    rows=[]
    for d in coords.itertuples():
        name=d.name.strip(' ,').replace('&',r'\&').replace('_',r'\_')
        rows.append([d.map_number,d.eoi,name,d.typ_zdroja])
    r.table('station_key','Station identifiers for the network map. Source classes are background (B), traffic (T) and industrial (I). Only the 21 evaluated sites are shown.','tab:stationkey',['Map no.','Station','Name','Class'],rows,'rlp{0.52\\textwidth}c')
    geom=pd.read_csv(RUN/'scores/foldwise_reconstruction.csv')
    geom=geom[geom.method.eq('G_PLUS_P')]
    brackets=pd.read_csv(RUN/'scores/dispersed_common_predictions.csv')
    # Interpolation rows carry bracket metadata; tree rows share the keys but
    # intentionally have no interpolation-specific bracket column values.
    brackets=brackets[brackets.method.eq('INTERP_LINEAR')]
    assert len(brackets)==5066 and len(geom)==378
    assert geom.n_replaced.sum()==len(brackets) and brackets.bracket_days.notna().all()
    rows=[]
    for fold,g in geom.groupby('fold'):
        b=brackets[brackets.fold.eq(fold)]
        rows.append([int(fold),len(b),f'{b.bracket_days.median():.1f}',f'{100*g.fraction_replaced.median():.1f}',f'{100*g.fraction_replaced.min():.1f}--{100*g.fraction_replaced.max():.1f}'])
    r.table('geometry_table','Dispersed withholding geometry on common bracketable support, all 21 stations in 2024--2025. Each fold contributes 42 station-year reconstructions. Fractions are calculated separately for each station-year, using its complete observed sampled-date count as denominator. Brackets use retained observations only.','tab:geometry',['Fold','Scored targets','Median bracket (days)','Median replaced (\\%)','Range (\\%)'],rows,'rrrrr')
    (OUT/'numbers.tex').write_text('% Generated from r02 plus the bounded r03 extension.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in r.MACROS.items()))
    pd.DataFrame(r.AUDIT).to_csv(OUT/'number_registry.csv',index=False)
    (OUT/'selected_rows.json').write_text(json.dumps(r.SELECTED+SELECTED,indent=2)+'\n')
    hashes=json.loads((OUT/'source_registry.json').read_text())
    paths=[RUN/'scores/summary_metrics.csv',RUN/'scores/paired_uncertainty.csv',RUN/'scores/station_year_means.csv',RUN/'scores/annual_station_influence.csv',RUN/'no_fit_summaries/observed_distribution.csv',RUN/'no_fit_summaries/year2025_forward_stress.csv',ROOT/'data/stations.csv',Path(__file__)]+[boundary_path.with_suffix(s) for s in ['.shp','.dbf','.shx','.prj','.cpg']]
    hashes.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    for filename in ['foldwise_reconstruction.csv','dispersed_common_predictions.csv']:
        p=RUN/'scores'/filename
        hashes[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'source_registry.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Rendered review-extension manuscript assets:',OUT)

if __name__=='__main__': main()
