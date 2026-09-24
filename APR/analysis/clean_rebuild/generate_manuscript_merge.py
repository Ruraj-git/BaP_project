"""Merged-paper reporting only. Preserve r02/r03/v2; no fits or bootstraps."""
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import generate_manuscript_r03 as g
import review_reading_assets

ROOT=g.ROOT
OUT=ROOT/'APR/manuscript/generated_merge'
V2=ROOT/'APR/manuscript_v2/generated_v2'
R2=ROOT/'APR/results/clean_rebuild/r02'
R3=ROOT/'APR/results/clean_rebuild/r03_review'

def main():
    g.OUT=OUT
    g.main()  # rerender existing evidence into a separate directory
    sources=[V2/'section3_numbers.json',V2/'network_relief.pdf',V2/'station_table.tex',
             ROOT/'APR/analysis/manuscript_v2/section21_assets.py',
             ROOT/'APR/analysis/manuscript_v2/section3_assets.py',__file__]
    source_values=json.loads((V2/'section3_numbers.json').read_text())
    extra={k:v for k,v in source_values.items() if k!='detail'}
    # Independently check imported point values against frozen predictions.
    p=pd.read_csv(R2/'assembled/validated_predictions.csv',usecols=['eoi','datum','observed','prediction','arm','protocol','input_variant','target_scale','weighting'])
    p=p[p.datum.str[:4].isin(['2024','2025'])&p.input_variant.eq('permissive')&p.weighting.eq('uniform')]
    def pick(arm,protocol,scale='log1p'):
        x=p[p.arm.eq(arm)&p.protocol.eq(protocol)&p.target_scale.eq(scale)]
        assert len(x)==5089 and not x.duplicated(['eoi','datum']).any()
        return x
    def rmse(x): return np.sqrt(np.mean((x.prediction-x.observed)**2))
    def verify(name,value,digits=3):
        assert extra[name]==f'{value:.{digits}f}',(name,extra[name],value)
    gp=pick('G_PLUS_P','loso'); gl=pick('G','loso')
    verify('AllLosoG',rmse(gl)); verify('AllLosoP',rmse(gp))
    for name,frame in [('IndLosoG',gl),('IndBlockG',pick('G','block30')),('IndBlockP',pick('G_PLUS_P','block30'))]:
        verify(name,rmse(frame[frame.eoi.eq('SK0018A')]))
    new=pd.read_csv(R3/'scores/loso_primary_predictions.csv')
    pc=new[new.arm.eq('P_CAL')&new.eoi.eq('SK0018A')]
    verify('IndLosoPcal',rmse(pc))
    verify('IndObsMean',gp.loc[gp.eoi.eq('SK0018A'),'observed'].mean())
    se=(gp.prediction-gp.observed)**2
    verify('IndSELosoGP',100*se[gp.eoi.eq('SK0018A')].sum()/se.sum(),0)
    # Long-gap comparisons reuse v2's already computed paired intervals; do not
    # repeat resampling during a reporting-only merge. Verify point estimates.
    baseline=pd.read_csv(R2/'scores/baseline_predictions.csv')
    baseline=baseline[baseline.protocol.eq('block30')&baseline.datum.str[:4].isin(['2024','2025'])]
    daily=pick('G_PLUS_P','block30')
    for scope in ['Non','All']:
        lin=baseline[baseline.method.eq('INTERP_LINEAR')&baseline.prediction.notna()]
        if scope=='Non': lin=lin[lin.eoi.ne('SK0018A')]
        keys=lin[['eoi','datum']]
        a=daily.merge(keys,on=['eoi','datum'],validate='one_to_one')
        ridge=baseline[baseline.method.eq('PM_RIDGE')].merge(keys,on=['eoi','datum'],validate='one_to_one')
        assert len(a)==len(lin)==len(ridge)
        for suffix,frame in [('GP',a),('Lin',lin),('Ridge',ridge)]: verify('Bracket'+scope+suffix,rmse(frame))
        for suffix,frame in [('Lin',lin),('Ridge',ridge)]: verify('Bracket'+scope+'Diff'+suffix,rmse(a)-rmse(frame))
        if scope=='Non': longgap=a
    for protocol,label in [('block30','Block'),('loso','Loso')]:
        ident=pick('G_PLUS_P',protocol,'identity'); ref=pick('G_PLUS_P',protocol)
        verify('Ident'+label+'Diff',rmse(ident[ident.eoi.ne('SK0018A')])-rmse(ref[ref.eoi.ne('SK0018A')]))
    # Verify intervals already present in the canonical score tables.
    canon=pd.read_csv(R2/'scores/paired_uncertainty.csv')
    for label in ['AllLosoPoll','IdentBlock','IdentLoso']:
        if label=='AllLosoPoll':
            c=g.P[(g.P.method_a=='G_PLUS_P')&(g.P.method_b=='G')&(g.P.protocol=='loso')&(g.P.scope=='all_network')&(g.P.estimand=='daily')&(g.P.metric=='RMSE')]
        else:
            c=canon[canon.method_a.str.contains('G_PLUS_P.*identity',regex=True)&canon.method_b.str.contains('G_PLUS_P.*log1p',regex=True)&canon.protocol.eq('block30' if label=='IdentBlock' else 'loso')&canon.scope.eq('nonindustrial')&canon.estimand.eq('daily')&canon.metric.eq('RMSE')]
        assert len(c)==1,(label,len(c))
        for suffix,col in [('Diff','estimate'),('Low','ci_low'),('High','ci_high')]:
            key=label+('Diff'+suffix if label=='AllLosoPoll' and suffix!='Diff' else suffix)
            verify(key,c.iloc[0][col])
    for name,method,est in [('NLosoPDailyCod','G_PLUS_P','daily'),('NLosoPMeanCod','G_PLUS_P','sampled_date_station_year_mean')]:
        extra[name]=f'{g.score(method,estimand=est).agreement_R2:.2f}'
    (OUT/'additional_numbers.tex').write_text('% Imported v2 intervals; independently checked point values. No new resampling.\n'+''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in extra.items()))
    (OUT/'additional_number_registry.json').write_text(json.dumps(extra,indent=2)+'\n')
    # Two years remain separate; add the new P+calendar comparator.
    ann=pd.read_csv(R3/'scores/station_year_means.csv')
    obs=ann[ann.method.eq('G')]
    order=obs.groupby('eoi')['observed'].mean().sort_values().index.tolist()
    fig,axes=plt.subplots(2,1,figsize=(9.5,7.2),sharex=True,sharey=True)
    for year,ax in zip([2024,2025],axes):
        for method,color,marker,offset,label in [('G','#E69F00','^',-.25,'G'),('P_CAL','#CC79A7','s',-.08,'P+calendar'),('G_PLUS_P','#0072B2','o',.10,'G+P'),('Observed','#222222','D',.25,'Observed')]:
            x=obs if method=='Observed' else ann[ann.method.eq(method)]
            x=x[x.year.eq(year)].set_index('eoi').loc[order]
            ax.scatter(np.arange(len(order))+offset,x.observed if method=='Observed' else x.prediction,c=color,marker=marker,s=30,label=label,zorder=3)
        ax.set_title(str(year),loc='left'); ax.set_ylabel('Sampled-date mean (ng m$^{-3}$)')
        ax.axvspan(order.index('SK0018A')-.45,order.index('SK0018A')+.45,color='#eeeeee',zorder=0)
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=4,loc='upper left',fontsize=10)
    axes[1].set_xticks(range(len(order)),[e+('*' if e=='SK0018A' else '') for e in order],rotation=55,ha='right',fontsize=9)
    axes[1].set_xlabel('Stations ordered by observed concentration; * industrial')
    fig.tight_layout(); fig.savefig(OUT/'station_means.pdf'); plt.close(fig)
    # Time-series rendering reuses already validated 30-day/LOSO predictions.
    fig,axes=plt.subplots(2,1,figsize=(9,5.6),sharex=True)
    for (station,title),ax in zip([('SK0025A','Jelšava: higher concentration'),('SK0048A','Bratislava, Jeséniova: lower concentration')],axes):
        o=daily[daily.eoi.eq(station)].sort_values('datum')
        ax.scatter(pd.to_datetime(o.datum),o.observed,s=10,c='#222222',label='Observed',zorder=4)
        for frame,label,color,ls in [(daily,'G+P, 30-day','#56B4E9','-'),(gp,'G+P, LOSO','#0072B2','-'),(gl,'G, LOSO','#E69F00','--')]:
            a=frame[frame.eoi.eq(station)].copy(); a['datum']=pd.to_datetime(a.datum); a=a.sort_values('datum')
            pad=a.loc[a.datum.diff().dt.days.gt(7)].copy(); pad['datum']-=pd.Timedelta(days=1); pad['prediction']=np.nan
            a=pd.concat([a,pad]).sort_values('datum')
            ax.plot(a.datum,a.prediction,c=color,ls=ls,lw=1,label=label)
        ax.set_title(title,loc='left',fontsize=10); ax.set_ylabel('B[a]P (ng m$^{-3}$)'); ax.grid(axis='y',alpha=.2)
    axes[0].legend(ncol=4,fontsize=8.5,loc='upper center'); fig.tight_layout(); fig.savefig(OUT/'timeseries.pdf'); plt.close(fig)
    # Reuse the existing hillshade asset, without rereading the large raw DEM.
    (OUT/'network_relief.pdf').write_bytes((V2/'network_relief.pdf').read_bytes())
    # Import cleaned names only; coordinates and map order remain r03 metadata.
    stationtext=(V2/'station_table.tex').read_text()
    names={m.group(1):m.group(2) for m in re.finditer(r'\d+ & (SK\w+) & ([^&]+) & [BTI] &',stationtext)}
    mapped=pd.read_csv(OUT/'mapped_stations.csv'); assert set(names)==set(mapped.eoi)
    rows=[[q.map_number,q.eoi,names[q.eoi].strip(),q.typ_zdroja] for q in mapped.itertuples()]
    g.r.table('station_key','Station identifiers for the main-text network map. Classes: background (B), traffic (T), industrial (I).','tab:stationkey',['Map no.','Station','Name','Class'],rows,'rlp{0.52\\textwidth}c')
    # Harmonise labels without changing numerical scores.
    for name in ['loso_table','gap_table','strata_table','auxiliary_gap_table','environment_intervals']:
        path=OUT/(name+'.tex'); text=path.read_text()
        text=text.replace('station-year means over sampled dates','predicted sampled-date means').replace('Sampled mean','Predicted mean').replace('Sampled-date mean','Predicted sampled-date mean').replace('mean recovery uses','reconstructed-mean scoring uses').replace('Mean recovery replaces','Each reconstructed mean replaces').replace('Mean-recovery RMSE','Reconstructed-mean RMSE')
        if name=='loso_table': text=text.replace('Mean scores use','Predicted-mean scores use')
        if name=='gap_table': text=text.replace('Units: ng~m$^{-3}$.','All entries are RMSE in ng~m$^{-3}$.')
        path.write_text(text)
    t=(OUT/'longgap_table.tex').read_text(); err=longgap.prediction-longgap.observed
    row=f'G+P & {len(longgap)} & {rmse(longgap):.3f} & {abs(err).mean():.3f} & {err.mean():.3f} '+r'\\'
    t=t.replace('\\midrule\n','\\midrule\n'+row+'\n',1)
    (OUT/'longgap_table.tex').write_text(t)
    # Store pooled-bias diagnostic as descriptive evidence only.
    static=pd.read_csv(R2/'inputs/train_ready_permissive_500m.csv').groupby('eoi')[['tpi_local','tpi_meso','tpi_broad']].first()
    bias=(gp.assign(error=gp.prediction-gp.observed).groupby('eoi').error.mean())
    gbias=(gl.assign(error=gl.prediction-gl.observed).groupby('eoi').error.mean())
    diagnostic=static.join(gbias.rename('G_bias')).join(bias.rename('GP_bias')).drop('SK0018A')
    assert set(diagnostic.nlargest(3,'tpi_local').index)=={'SK0048A','SK0076A','SK0045A'}
    assert set(diagnostic.nlargest(3,'G_bias').index)=={'SK0048A','SK0076A','SK0045A'}
    diagnostic.to_csv(OUT/'terrain_context.csv')
    sources += review_reading_assets.main(OUT, g.r.table, daily, gp, gl)
    hashes=json.loads((OUT/'source_registry.json').read_text())
    sources += [R2/'assembled/validated_predictions.csv',R2/'scores/baseline_predictions.csv',R2/'scores/paired_uncertainty.csv',R3/'scores/loso_primary_predictions.csv']
    for pth in sources:
        from pathlib import Path
        pth=Path(pth); hashes[str(pth.relative_to(ROOT))]=hashlib.sha256(pth.read_bytes()).hexdigest()
    (OUT/'source_registry.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('Merged assets rendered; no fitting or resampling:',OUT)

if __name__=='__main__': main()
