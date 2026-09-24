#!/usr/bin/env python3
"""Render manuscript assets only: no fits, resampling, or historical results."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'APR/results/clean_rebuild/r02'
OUT = ROOT / 'APR/manuscript/generated_r02'
S = pd.read_csv(SRC / 'scores/summary_metrics.csv')
P = pd.read_csv(SRC / 'scores/paired_uncertainty.csv')
PERIOD = 'primary_2024_2025'
MACROS = {}
AUDIT = []
SELECTED = []

def mid(arm, variant='permissive', target='log1p', weight='uniform'):
    return '|'.join([arm, target, weight, variant, '500m' if arm.startswith('G') else '2km_exact'])

G, GP, FULL, PM, MA = map(mid, ['G', 'G_PLUS_P', 'FULL_ID', 'PM_XGB', 'M_AUX'])
INV = mid('M_AUX', weight='inverse_1_over_bap_plus_0_5')

def select(table, **filters):
    rows = table
    for key, value in filters.items():
        rows = rows[rows[key].eq(value)]
    assert len(rows) == 1, (filters, len(rows))
    SELECTED.append({'source': 'paired_uncertainty.csv' if table is P else 'summary_metrics.csv', 'filters': filters})
    return rows.iloc[0]

def score(method, protocol='block30', scope='nonindustrial', estimand='daily', support='all_expected_targets'):
    return select(S, method=method, protocol=protocol, scope=scope, estimand=estimand, support=support, period=PERIOD)

def pair(a, b, protocol='block30', scope='nonindustrial', estimand='daily', support='all_expected_targets'):
    return select(P, method_a=a, method_b=b, protocol=protocol, scope=scope, estimand=estimand, support=support, period=PERIOD, metric='RMSE')

def macro(name, value, digits=3):
    rendered = f'{value:.{digits}f}' if isinstance(value, (float, int)) else str(value)
    assert name not in MACROS
    MACROS[name] = rendered
    AUDIT.append(dict(macro=name, raw=str(value), rendered=rendered))

def interval(row):
    return f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"

def table(name, caption, label, header, rows, layout):
    text = '\\begin{table}[htbp]\n\\centering\\small\n\\caption{' + caption + '}\n\\label{' + label + '}\n'
    text += '\\begin{tabular}{' + layout + '}\n\\toprule\n'
    text += ' & '.join(header) + r' \\' + '\n\\midrule\n'
    text += '\n'.join(' & '.join(map(str, row)) + r' \\' for row in rows)
    text += '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    (OUT / (name + '.tex')).write_text(text)

def main():
    OUT.mkdir(exist_ok=True)
    data = pd.read_csv(SRC / 'inputs/train_ready_permissive_500m.csv')
    macro('CObsCount', len(data), 0)
    macro('CPrimaryCount', int(pd.to_datetime(data.datum).dt.year.isin([2024, 2025]).sum()), 0)
    macro('CNonCount', int(score(GP)['n']), 0)
    macro('CIndustrialCount', int(score(GP, scope='industrial_descriptive')['n']), 0)
    macro('CStart', '2 June 2023'); macro('CEnd', '30 December 2025')
    for name, mask in [('LowMinimum', data.bap.eq(data.bap.min())), ('LowExact', data.bap.le(3/55)), ('LowRounded', data.bap.le(.06))]:
        macro(name, int(mask.sum()), 0); macro(name + 'Percent', 100 * mask.mean(), 1)
    macro('NFlexReduction', 100 * (1-score(FULL).RMSE/score(PM).RMSE), 1)
    for name, method in [('NFlexPM', PM), ('NFlexFull', FULL)]: macro(name, score(method).RMSE)
    q = pair(FULL, PM)
    for suffix, field in [('', 'estimate'), ('Low', 'ci_low'), ('High', 'ci_high')]: macro('NFlexDiff'+suffix, q[field])
    for prefix, protocol in [('NBlock', 'block30'), ('NLoso', 'loso')]:
        for suffix, method in [('G', G), ('P', GP)]:
            macro(prefix+suffix, score(method, protocol).RMSE)
            macro(prefix+'Mean'+suffix, score(method, protocol, estimand='sampled_date_station_year_mean', support='all_sampled_dates').RMSE)
    q=pair(GP,G,'loso',estimand='sampled_date_station_year_mean',support='all_sampled_dates')
    for suffix,field in [('', 'estimate'),('Low','ci_low'),('High','ci_high')]: macro('AnnualDiff'+suffix,q[field])
    influence=pd.read_csv(SRC/'scores/station_influence.csv')
    for name,protocol in [('Temporal','block30'),('Spatial','loso')]:
        x=influence[influence.method_a.eq(GP)&influence.method_b.eq(G)&influence.protocol.eq(protocol)]
        assert len(x)==20 and x.difference_RMSE.lt(0).all()
        macro(name+'OmitMin',-x.difference_RMSE.max()); macro(name+'OmitMax',-x.difference_RMSE.min())
        macro(name+'Strict',score(mid('G_PLUS_P','strict18'),protocol).RMSE)
    for level in ['higher','lower']:
        for label,method in [('G',G),('P',GP)]:
            q=score(method,'loso',scope='nonindustrial_'+level,estimand='sampled_date_station_year_mean',support='all_sampled_dates')
            macro(level.capitalize()+label+'RMSE',q.RMSE)
            if label=='P': macro(level.capitalize()+'Bias',q.bias)
    q=score(GP,'loso',scope='industrial_descriptive')
    macro('IndustrialRMSE',q.RMSE); macro('IndustrialBias',q.bias)
    q=pair(mid('G_PLUS_P',target='identity'),GP,'loso',estimand='sampled_date_station_year_mean',support='all_sampled_dates')
    for suffix,field in [('', 'estimate'),('Low','ci_low'),('High','ci_high')]: macro('IdentityMeanDiff'+suffix,q[field])
    for name,method in [('G',G),('P',GP)]:
        vals=[score(method,p).RMSE for p in ['year_out','season_year_out','calendar_season_out']]
        macro('Robust'+name+'Min',min(vals)); macro('Robust'+name+'Max',max(vals))
    rows = []
    for label, method in [('PM-XGB', PM), ('FULL-ID', FULL)]:
        q=score(method); rows.append([label, f'{q.RMSE:.3f}', f'{q.MAE:.3f}', f'{q.bias:.3f}'])
    table('flex_table', 'Flexible-PM comparison: 4852 observations at 20 nonindustrial stations, 2024--2025, under matched uniform-weight 30-day withholding. Errors are in ng~m$^{-3}$.', 'tab:flex', ['Model', 'RMSE', 'MAE', 'Bias'], rows, 'lrrr')
    rows=[]
    for label, protocol in [('30-day gap', 'block30'), ('Unseen station', 'loso')]:
        for what, estimand, support in [('Daily', 'daily', 'all_expected_targets'), ('Sampled-date mean', 'sampled_date_station_year_mean', 'all_sampled_dates')]:
            a,b=score(G,protocol,estimand=estimand,support=support),score(GP,protocol,estimand=estimand,support=support)
            rows.append([label,what,f'{a.RMSE:.3f}',f'{b.RMSE:.3f}',interval(pair(GP,G,protocol,estimand=estimand,support=support))])
    table('controlled_table', 'Controlled G/G+P comparison at nonindustrial stations. Daily scores use 4852 observations; mean scores use 40 station-years. Differences are G+P minus G, with paired 95\\% intervals; negative values favour pollutant inputs. Units: ng~m$^{-3}$.', 'tab:controlled', ['Task','Quantity','G RMSE','G+P RMSE','Difference [95\\% interval]'],rows,'llrrl')
    methods=[('M-AUX, uniform',MA),('M-AUX, inverse',INV),('PM ridge','PM_RIDGE'),('Harmonics','HARMONIC'),('Linear interpolation','INTERP_LINEAR'),('Log-linear interpolation','INTERP_LOGLINEAR')]
    rows=[]
    for label,method in methods:
        daily=score(method,'dispersed_mod9',support='common_bracketable')
        rec=score(method,'dispersed_mod9',estimand='foldwise_sampled_mean_recovery',support='common_bracketable_replacements')
        rows.append([label,f'{daily.RMSE:.3f}',f'{daily.MAE:.3f}',f'{rec.RMSE:.4f}'])
    gap_n=int(score(MA,'dispersed_mod9',support='common_bracketable')['n'])
    table('gap_table',f'Dispersed-gap recovery at nonindustrial sites on {gap_n} identical bracketable targets. Mean recovery replaces only the withheld subset of an otherwise measured record (360 station-year/fold reconstructions). Units: ng~m$^{{-3}}$.','tab:gap',['Method','Daily RMSE','Daily MAE','Mean-recovery RMSE'],rows,'lrrr')
    rec=pd.read_csv(SRC/'scores/foldwise_reconstruction.csv')
    macro('MeanHiddenPercent', 100*rec[rec.method.eq(MA)].fraction_replaced.median(), 2)
    for name,method in [('MeanNonUniform',MA),('MeanNonInterp','INTERP_LINEAR')]: macro(name,score(method,'dispersed_mod9',estimand='foldwise_sampled_mean_recovery',support='common_bracketable_replacements').RMSE,4)
    q=pair(MA,'INTERP_LINEAR','dispersed_mod9',estimand='foldwise_sampled_mean_recovery',support='common_bracketable_replacements')
    for suffix,field in [('', 'estimate'),('Low','ci_low'),('High','ci_high')]: macro('MeanDiff'+suffix,q[field],4)
    rows=[]
    for label,protocol in [('Year out','year_out'),('Season-year out','season_year_out'),('Calendar season out','calendar_season_out')]:
        a,b=score(G,protocol),score(GP,protocol)
        rows.append([label,str(int(a['n'])),f'{a.RMSE:.3f}',f'{b.RMSE:.3f}',interval(pair(GP,G,protocol))])
    table('robustness_table','Temporal stress tests at nonindustrial stations, 2024--2025. Differences are G+P minus G. Season-year scoring excludes incomplete seasons. Station-cluster intervals do not quantify uncertainty over different years or seasons. Units: ng~m$^{-3}$.','tab:robustness',['Withholding','n','G RMSE','G+P RMSE','Difference [95\\% interval]'],rows,'lrrrl')
    rows=[]
    for label,a,b,protocol in [('Strict-hour G+P vs G',mid('G_PLUS_P','strict18'),G,'block30'),('Strict-hour G+P vs G',mid('G_PLUS_P','strict18'),G,'loso'),('Strict vs standard G+P',mid('G_PLUS_P','strict18'),GP,'block30'),('Strict vs standard G+P',mid('G_PLUS_P','strict18'),GP,'loso'),('Uniform vs inverse M-AUX',MA,INV,'block30'),('Original vs log target G+P',mid('G_PLUS_P',target='identity'),GP,'block30'),('Original vs log target G+P',mid('G_PLUS_P',target='identity'),GP,'loso')]:
        rows.append([label,'30-day' if protocol=='block30' else 'LOSO',interval(pair(a,b,protocol))])
    table('sensitivity_table','Nonindustrial daily RMSE sensitivity contrasts, first configuration minus second, with paired 95\\% intervals (ng~m$^{-3}$). Strict-hour inputs require at least 18 valid hours per pollutant day before temporal summaries are recomputed.','tab:sensitivity',['Contrast','Task','Difference [95\\% interval]'],rows,'lll')
    rows=[]
    for scope,label in [('all_network','All 21'),('nonindustrial','Nonindustrial 20'),('industrial_descriptive','Industrial 1')]:
        for protocol,task in [('block30','30-day'),('loso','LOSO')]:
            for method,model in [(G,'G'),(GP,'G+P')]:
                q=score(method,protocol,scope); rows.append([label,task,model,int(q['n']),f'{q.RMSE:.3f}',f'{q.MAE:.3f}',f'{q.bias:.3f}'])
    table('population_table','Daily scores by population, 2024--2025. The industrial subset is one station, SK0018A, and is descriptive. Units: ng~m$^{-3}$.','tab:populations',['Population','Task','Model','n','RMSE','MAE','Bias'],rows,'lllrrrr')
    rows=[]
    for label,method in methods:
        q=score(method,'block30',support='common_bracketable')
        rows.append([label,int(q['n']),f'{q.RMSE:.3f}',f'{q.MAE:.3f}',f'{q.bias:.3f}'])
    table('longgap_table','Thirty-day-gap comparators at nonindustrial stations on common bracketable dates, 2024--2025. All model baselines are fitted inside each withholding fold. Units: ng~m$^{-3}$.','tab:longgap',['Method','n','RMSE','MAE','Bias'],rows,'lrrrr')
    rows=[]
    for label,scope in [('Higher ($>1$)','nonindustrial_higher'),('Lower ($\\leq1$)','nonindustrial_lower')]:
        for model,method in [('G',G),('G+P',GP)]:
            q=score(method,'loso',scope,estimand='sampled_date_station_year_mean',support='all_sampled_dates')
            rows.append([label,model,int(q['n']),f'{q.RMSE:.3f}',f'{q.MAE:.3f}',f'{q.bias:.3f}'])
    table('strata_table','Descriptive LOSO sampled-date mean errors by observed concentration stratum (ng~m$^{-3}$), with 20 nonindustrial station-years in each stratum.','tab:strata',['Stratum','Model','Station-years','RMSE','MAE','Bias'],rows,'llrrrr')
    # Exact corrected intervals, drawn without resampling or model fitting.
    fig,ax=plt.subplots(figsize=(7.5,2.8))
    for y,(a,b,protocol) in enumerate([(FULL,PM,'block30'),(GP,G,'block30'),(GP,G,'loso')]):
        q=pair(a,b,protocol); val=-q.estimate; low=-q.ci_high; high=-q.ci_low
        ax.errorbar(val,y,xerr=[[val-low],[high-val]],fmt='o',color='#176b91',capsize=4)
    ax.set_yticks(range(3),['Richer inputs beyond PM: 30-day','Pollutants beyond G: 30-day','Pollutants beyond G: LOSO'])
    ax.invert_yaxis(); ax.axvline(0,color='grey',lw=.8); ax.set_xlabel('Reduction in daily RMSE (ng m$^{-3}$), with 95% interval')
    fig.tight_layout(); fig.savefig(OUT/'information_benefits.pdf'); plt.close(fig)
    (OUT/'numbers.tex').write_text('% Generated exclusively from clean-rebuild r02.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in MACROS.items()))
    pd.DataFrame(AUDIT).to_csv(OUT/'number_registry.csv',index=False)
    (OUT/'selected_rows.json').write_text(json.dumps(SELECTED,indent=2)+'\n')
    sources=[SRC/'scores'/f for f in ['summary_metrics.csv','paired_uncertainty.csv','foldwise_reconstruction.csv','station_influence.csv','station_year_loso.pdf']]+[SRC/'inputs/train_ready_permissive_500m.csv',Path(__file__)]
    (OUT/'source_registry.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
    print('Rendered r02 manuscript assets:',OUT)

if __name__=='__main__': main()
