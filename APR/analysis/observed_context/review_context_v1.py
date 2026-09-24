"""Checked descriptive context for manuscript v1 only. No fits or v2 writes."""
import hashlib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/'APR/results/clean_rebuild/r02/inputs/train_ready_permissive_500m.csv'
OLD=ROOT/'APR/results/observed_context'
OUT=OLD/'reviewed_v1'
META=ROOT/'APR/manuscript/generated_r03/mapped_stations.csv'
COLORS={'B':('#0072B2','o','Background'),'T':('#009E73','s','Traffic'),
        'I':('#D55E00','^','Industrial (1 site)')}

def main():
    d=pd.read_csv(SRC,parse_dates=['datum'])
    d=d[d.datum.dt.year.isin([2024,2025])].copy()
    assert len(d)==5089 and not d.duplicated(['eoi','datum']).any()
    h=d[np.isfinite(d.pm10_mean)&d.pm10_mean.gt(0)].copy()
    assert len(h)==4755 and h.eoi.nunique()==20
    assert set(d.eoi)-set(h.eoi)=={'SK0006R'}
    assert h.t_mean.notna().all() and h.bap.gt(0).all()
    h['ratio']=1000*h.bap/h.pm10_mean
    h['season']=np.select([h.datum.dt.month.isin([12,1,2]),h.datum.dt.month.isin([6,7,8])],['DJF','JJA'],default='other')
    h['cls']=h.typ_zdroja
    seasonal=h.groupby(['cls','season']).agg(n=('ratio','size'),ratio_median=('ratio','median'),pm10_mean=('pm10_mean','mean')).reset_index()
    winter=h[h.season.eq('DJF')].groupby(['eoi','cls']).agg(n=('ratio','size'),ratio_median=('ratio','median'),ratio_q25=('ratio',lambda x:x.quantile(.25)),ratio_q75=('ratio',lambda x:x.quantile(.75))).reset_index().sort_values('ratio_median')
    checks=0
    for name,new,keys,fields in [('seasonal_by_class',seasonal,['cls','season'],['n','ratio_median','pm10_mean']),('winter_ratio_by_station',winter,['eoi','cls'],['n','ratio_median','ratio_q25','ratio_q75'])]:
        old=pd.read_csv(OLD/(name+'.csv')).set_index(keys).sort_index()
        n=new.set_index(keys).sort_index()
        assert old.index.equals(n.index)
        for field in fields:
            assert np.allclose(n[field],old[field],rtol=0,atol=1e-10),field
            checks+=len(n)
    h['bin']=pd.cut(h.t_mean,[-20,-5,0,5,10,15,20,35])
    assert h.bin.notna().all()
    bins=h.groupby(['cls','bin'],observed=True).agg(n=('ratio','size'),n_stations=('eoi','nunique'),temperature_median=('t_mean','median'),ratio_median=('ratio','median'),ratio_q25=('ratio',lambda x:x.quantile(.25)),ratio_q75=('ratio',lambda x:x.quantile(.75)),fraction_le006=('bap',lambda x:(x<=.06).mean())).reset_index()
    bg=seasonal[seasonal.cls.eq('B')].set_index('season')
    niw=h[h.cls.ne('I')&h.season.eq('DJF')]
    cold=niw[niw.t_mean.le(-5)]; warm=niw[niw.t_mean.gt(5)]
    summer=h[h.cls.ne('I')&h.season.eq('JJA')]
    macros={'CtxCount':str(len(h)),'CtxWinter':f'{bg.loc["DJF","ratio_median"]:.0f}',
            'CtxSummer':f'{bg.loc["JJA","ratio_median"]:.0f}',
            'CtxWinterPM':f'{bg.loc["DJF","pm10_mean"]:.1f}','CtxSummerPM':f'{bg.loc["JJA","pm10_mean"]:.1f}',
            'CtxCold':f'{cold.ratio.median():.0f}','CtxWarm':f'{warm.ratio.median():.0f}',
            'CtxColdN':str(len(cold)),'CtxWarmN':str(len(warm)),
            'CtxLowPercent':f'{100*(summer.bap<=.06).mean():.0f}',
            'CtxMinWinter':f'{winter.loc[winter.cls.ne("I"),"ratio_median"].min():.0f}',
            'CtxMaxWinter':f'{winter.loc[winter.cls.ne("I"),"ratio_median"].max():.0f}'}
    OUT.mkdir(exist_ok=True)
    seasonal.to_csv(OUT/'seasonal_by_class.csv',index=False)
    winter.to_csv(OUT/'winter_ratio_by_station.csv',index=False)
    bins.to_csv(OUT/'temperature_bins_by_class.csv',index=False)
    (OUT/'numbers.tex').write_text('% Descriptive context, reviewed v1; generated, no fits.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    fig,(a,b)=plt.subplots(1,2,figsize=(11,5.5),gridspec_kw={'width_ratios':[1,1.1]})
    for code,(color,marker,label) in COLORS.items():
        t=bins[bins.cls.eq(code)&bins.n.ge(10)]
        if code!='I': a.fill_between(t.temperature_median,t.ratio_q25,t.ratio_q75,color=color,alpha=.12,lw=0)
        a.plot(t.temperature_median,t.ratio_median,color=color,marker=marker,ls='--' if code=='I' else '-',label=label)
    a.set_yscale('log'); a.set_ylim(1,350)
    a.set_yticks([1,3,10,30,100,300],['1','3','10','30','100','300']); a.minorticks_off()
    a.set_xlabel('Daily temperature (°C), median within bin')
    a.set_ylabel('B[a]P / PM$_{10}$ (ng mg$^{-1}$), median')
    a.set_title('(a) Observed temperature pattern',loc='left',fontsize=11)
    a.legend(frameon=False,loc='lower left',fontsize=9)
    stations=pd.read_csv(META).set_index('eoi')
    w=winter.assign(y=np.arange(len(winter)))
    for code,(color,marker,label) in COLORS.items():
        g=w[w.cls.eq(code)]
        b.errorbar(g.ratio_median,g.y,xerr=[g.ratio_median-g.ratio_q25,g.ratio_q75-g.ratio_median],fmt=marker,color=color,ms=6,elinewidth=1)
    labels=[stations.loc[e,'name'].split(',')[0].strip()+' ('+e+')' for e in w.eoi]
    b.set_yticks(w.y,labels,fontsize=8)
    b.set_xlabel('Winter B[a]P / PM$_{10}$ (ng mg$^{-1}$)')
    b.set_title('(b) Winter station medians and IQRs',loc='left',fontsize=11)
    a.grid(axis='y',alpha=.25); b.grid(axis='x',alpha=.25)
    for ax in (a,b):
        for side in ['top','right']: ax.spines[side].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT/'figure_bap_pm_context.pdf'); fig.savefig(OUT/'figure_bap_pm_context.png',dpi=180); plt.close(fig)
    paths=[SRC,META,Path(__file__),OLD/'seasonal_by_class.csv',OLD/'winter_ratio_by_station.csv']
    report={'status':'PASS','checked_published_cells':checks,'n_primary':len(d),'n_ratio':len(h),'n_stations':20,
            'macros':macros,'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.csv','.tex','.pdf','.png']},
            'limitations':['Pooled descriptive medians, not causal or equal-station estimates','0.06 is a reporting threshold, not an assigned record-level detection limit','Calendar-day PM10 and filter windows are not verified aligned','No fits or bootstrap; no changes to manuscript_v2 or original outputs']}
    (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['sources','outputs']},indent=2))

if __name__=='__main__': main()
