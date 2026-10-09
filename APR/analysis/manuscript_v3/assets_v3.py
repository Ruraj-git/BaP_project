"""Manuscript_v3 assets from the consolidated input ladder (reporting only; no fits, no new resampling).

Primary: 500 m static support (APR/results/consolidated_500m/scores).
Support sensitivity: 2 km run (APR/results/consolidated_2km/scores_v2).
Writes APR/manuscript_v3/generated_v3/:
  numbers.tex / numbers.json   ladder scores and contrast macros
  ladder_table.tex             main-text ladder results (30-day, LOSO)
  contrast_table.tex           supplementary: all input-group contrasts with intervals
  sensitivity_table.tex        supplementary: fitting/input-handling/support sensitivities
  population_table, robustness_table, strata_table, longgap_table, gap_table,
  dispersed_intervals (.tex)   regenerated with ladder names
  information_benefits.pdf, station_means.pdf, timeseries.pdf
"""
import json
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

APR = Path(__file__).resolve().parents[2]
S5 = APR / 'results/consolidated_500m/scores'
S2 = APR / 'results/consolidated_2km/scores_v2'
A5 = APR / 'results/consolidated_500m/assembled/validated_predictions.csv'
A2 = APR / 'results/consolidated_2km/assembled/validated_predictions.csv'
R02B = APR / 'results/clean_rebuild/r02/scores/baseline_predictions.csv'
INPUTS = APR / 'results/clean_rebuild/r02/inputs'
sys.path.insert(0, str(APR / 'analysis/clean_rebuild'))
from scoring_core import paired  # noqa: E402  (same bootstrap as the scoring runs)
OUT = APR / 'manuscript_v3/generated_v3'
IND = 'SK0018A'
KEYS = ['eoi', 'datum']

MID = lambda arm, target='log1p', w='uniform', v='permissive', aug='none': '|'.join([arm, target, w, v, aug])
NAME = {'PM': 'PM', 'P': 'P', 'E': 'E', 'E_P': 'E+P', 'E_P_M': 'E+P+M', 'E_P_M_I': 'E+P+M+I',
        'PM_I': 'PM+I', 'E_PM': 'E+PM', 'E_NO2': 'E+NO$_2$'}
CODE = {'PM': 'Pm', 'P': 'Pol', 'E': 'Env', 'E_P': 'EnvPol', 'E_P_M': 'EnvPolMeta', 'E_P_M_I': 'EnvPolMetaInd',
        'PM_I': 'PmInd', 'E_PM': 'EnvPm', 'E_NO2': 'EnvNo'}
LADDER = ['PM', 'P', 'E', 'E_P', 'E_P_M', 'E_P_M_I', 'PM_I']
BASE = {'HARMONIC': 'Harmonics', 'INTERP_LINEAR': 'Linear interpolation',
        'INTERP_LOGLINEAR': 'Log-linear interpolation', 'PM_RIDGE': 'PM ridge'}
# (macro code, score label, candidate arm, comparator arm, display label, tasks)
INPUT_CONTRASTS = [
    ('AddNo', '+NO2', 'P', 'PM', '+NO$_2$: P vs PM', ('block30', 'loso')),
    ('AddE', '+E', 'E_P', 'P', '+E: E+P vs P', ('block30', 'loso')),
    ('AddP', '+P', 'E_P', 'E', '+P: E+P vs E', ('block30', 'loso')),
    ('AddM', '+M', 'E_P_M', 'E_P', '+M: E+P+M vs E+P', ('block30', 'loso')),
    ('AddI', '+I', 'E_P_M_I', 'E_P_M', '+I: E+P+M+I vs E+P+M', ('block30',)),
    ('AddIPm', '+I (PM level)', 'PM_I', 'PM', '+I: PM+I vs PM', ('block30',)),
    ('Bundle', 'Bundle', 'E_P_M_I', 'PM_I', 'Bundle: E+P+M+I vs PM+I', ('block30',)),
    ('NoGivenE', '+NO2 given E', 'E_P', 'E_PM', '+NO$_2$ given E: E+P vs E+PM', ('block30', 'loso')),
    ('PmGivenE', '+PM given E', 'E_P', 'E_NO2', '+PM given E: E+P vs E+NO$_2$', ('block30', 'loso')),
]
SENS = [  # (macro code, score label, display)
    ('Weights', 'Weights', 'Inverse vs uniform weights'),
    ('Target', 'Target', 'Untransformed vs log target'),
    ('Strict', 'Strict hours', '$\\geq$18 valid hours vs all hours'),
    ('Mask', 'Masking', 'Masking augmentation vs none'),
    ('Matched', 'Programme-matched', 'Programme-matched vs E+P'),
    ('Support', 'Support', '2~km vs 500~m static cells'),
]
TASK = {'block30': '30-day', 'loso': 'LOSO'}
TCODE = {'block30': 'Block', 'loso': 'Loso'}
SCOPE = {'nonindustrial': 'Non', 'all_network': 'All'}


def mn(x, d=3):
    t = f'{x:.{d}f}'
    if float(t) == 0:
        t = t.lstrip('-')  # no signed zero
    return t.replace('-', '\\mn{}')


def tri(r, d=3):
    """Estimate, lower and upper limit as three aligned cells; supported differences in equal-width bold."""
    parts = [mn(r.estimate, d), mn(r.ci_low, d), mn(r.ci_high, d)]
    if r.ci_low > 0 or r.ci_high < 0:
        parts = [f'\\nb{{{x}}}' for x in parts]
    return ' & '.join(parts)


TRI = 'r@{\\ [}r@{,\\ }r@{]\\hspace{1.4em}}'   # inner column group (space after bracket)
TRI_END = 'r@{\\ [}r@{,\\ }r@{]}'                 # last column group
TRI_NA = '\\multicolumn{3}{c}{--}'


def iv(r, d=3):
    return f'{mn(r.estimate, d)} [{mn(r.ci_low, d)}, {mn(r.ci_high, d)}]'


def one(df, **kw):
    x = df
    for k, v in kw.items():
        x = x[x[k] == v]
    assert len(x) == 1, (kw, len(x))
    return x.iloc[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(S5 / 'summary_metrics.csv')
    P = pd.read_csv(S5 / 'paired_uncertainty.csv')
    S2k = pd.read_csv(S2 / 'summary_metrics.csv')
    P2k = pd.read_csv(S2 / 'paired_uncertainty.csv')
    P = P[P.metric == 'RMSE']
    P2k = P2k[P2k.metric == 'RMSE']
    num = {}

    def summ(src, method, protocol, scope, est='daily'):
        sup = {'daily': 'all_expected_targets', 'station_year_mean': 'all_sampled_dates'}.get(est, 'pollutant_availability_subset')
        x = src[(src.method == method) & (src.protocol == protocol) & (src.estimand == est) & (src.support == sup) & (src.scope == scope)]
        return one(x.drop_duplicates(), method=method)  # availability subsets are summarised once per contrast

    def pair(src, label, protocol, scope, est='daily', method_b=None):
        x = src[(src.label == label) & (src.protocol == protocol) & (src.scope == scope) & (src.estimand == est)]
        if method_b:
            x = x[x.method_b == method_b]
        assert len(x) == 1, (label, protocol, scope, est, len(x))
        return x.iloc[0]

    # ---- ladder scores
    for arm, code in CODE.items():
        for prot in ('block30', 'loso'):
            if prot == 'loso' and arm in ('E_P_M_I', 'PM_I'):
                continue
            for scope, sc in SCOPE.items():
                d = summ(S, MID(arm), prot, scope)
                m = summ(S, MID(arm), prot, scope, 'station_year_mean')
                num[f'L{code}{TCODE[prot]}{sc}Rmse'] = f'{d.RMSE:.3f}'
                num[f'L{code}{TCODE[prot]}{sc}Rsq'] = mn(d.agreement_R2, 2)
                num[f'L{code}{TCODE[prot]}{sc}MeanRmse'] = f'{m.RMSE:.3f}'
                num[f'L{code}{TCODE[prot]}{sc}MeanRsq'] = mn(m.agreement_R2, 2)

    # ---- input-group contrasts
    def put(prefix, r, comp_rmse=None):
        num[prefix] = mn(r.estimate)
        num[prefix + 'Low'] = mn(r.ci_low)
        num[prefix + 'High'] = mn(r.ci_high)
        num[prefix + 'Sup'] = 'supported' if (r.ci_low > 0 or r.ci_high < 0) else 'unresolved'
        if comp_rmse is not None:
            num[prefix + 'Pct'] = f'{-100 * r.estimate / comp_rmse:.0f}'

    for code, lab, a, b, _, prots in INPUT_CONTRASTS:
        for prot in prots:
            for scope, sc in SCOPE.items():
                for est, ec in (('daily', ''), ('station_year_mean', 'Mean')):
                    r = pair(P, lab, prot, scope, est)
                    assert r.method_a == MID(a) and r.method_b == MID(b)
                    put(f'C{code}{TCODE[prot]}{sc}{ec}', r, summ(S, MID(b), prot, scope, est).RMSE)
    for code, lab, _ in SENS:
        for prot in ('block30', 'loso'):
            for scope, sc in SCOPE.items():
                for est, ec in (('daily', ''), ('station_year_mean', 'Mean')):
                    put(f'S{code}{TCODE[prot]}{sc}{ec}', pair(P, lab, prot, scope, est))
    for code, lab in (('Mask', 'Masking'), ('Matched', 'Programme-matched'), ('AddE', '+E'), ('AddP', '+P')):
        for prot in ('block30', 'loso'):
            for est, ec in (('daily_incomplete_pollutant_days', 'Incomplete'), ('daily_complete_pollutant_days', 'Complete')):
                put(f'X{code}{TCODE[prot]}Non{ec}', pair(P, lab, prot, 'nonindustrial', est))
    inc = summ(S, MID('E_P'), 'loso', 'nonindustrial', 'daily_incomplete_pollutant_days')
    num['IncompleteLosoNonN'] = str(int(inc.n))
    num['IncompleteLosoNonStations'] = str(int(inc.n_stations))
    for prot in ('block30', 'loso'):
        for scope, lev in (('nonindustrial_higher', 'Higher'), ('nonindustrial_lower', 'Lower')):
            for est, ec in (('daily', ''), ('station_year_mean', 'Mean')):
                put(f'SWeights{TCODE[prot]}{lev}{ec}', pair(P, 'Weights', prot, scope, est))

    # ---- support sensitivity: the +E and +P contrasts evaluated at 2 km
    for code, lab in (('AddE', '+E'), ('AddP', '+P'), ('NoGivenE', '+NO2 given E')):
        for prot in ('block30', 'loso'):
            for est, ec in (('daily', ''), ('station_year_mean', 'Mean')):
                r = pair(P2k, lab, prot, 'nonindustrial', est)
                put(f'K{code}{TCODE[prot]}Non{ec}', r)
    for prot in ('block30', 'loso'):
        for scope, sc in SCOPE.items():
            num[f'KEnvPol{TCODE[prot]}{sc}Rmse'] = f"{summ(S2k, MID('E_P'), prot, scope).RMSE:.3f}"
            num[f'KEnv{TCODE[prot]}{sc}Rmse'] = f"{summ(S2k, MID('E'), prot, scope).RMSE:.3f}"
    st = pd.read_csv(S5 / 'station_metrics.csv')
    st = st[(st.protocol == 'loso') & (st.eoi != IND)]
    w = st.pivot_table(index='eoi', columns='method', values='RMSE')
    num['SupportBetterStations'] = str(int((w[MID('E_P')] < w['E_P_2KM']).sum()))
    num['AddEBetterStations'] = str(int((w[MID('E_P')] < w[MID('P')]).sum()))
    num['AddPBetterStations'] = str(int((w[MID('E_P')] < w[MID('E')]).sum()))

    inf = pd.read_csv(S5 / 'station_influence.csv')
    for lab, code in (('+E', 'AddE'), ('+P', 'AddP')):
        for prot in ('block30', 'loso'):
            for est, ec in (('daily', ''), ('station_year_mean', 'Mean')):
                x = -inf[(inf.label == lab) & (inf.protocol == prot) & (inf.estimand == est)].difference_RMSE
                num[f'Infl{code}{TCODE[prot]}{ec}Min'] = f'{x.min():.3f}'
                num[f'Infl{code}{TCODE[prot]}{ec}Max'] = f'{x.max():.3f}'

    ym5 = pd.read_csv(S5 / 'station_year_means.csv')
    fr = pd.read_csv(INPUTS / 'train_ready_permissive_500m.csv', parse_dates=['datum'])
    obs_mean = fr[fr.datum.dt.year.isin([2024, 2025])].groupby('eoi')['bap'].mean()  # as main-text Table 2
    rows = []
    for e in obs_mean.drop(IND).sort_values().index:
        a5, a2 = w.loc[e, MID('E_P')], w.loc[e, 'E_P_2KM']
        rows.append(f'{e} & {obs_mean[e]:.2f} & {w.loc[e, MID("P")]:.3f} & {a5:.3f} & {a2:.3f} & {mn(a2 - a5)} \\\\')
    (OUT / 'support_station_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Withheld-station (LOSO) daily RMSE of E+P with 500~m and '
        '2~km static cells at each nonindustrial station, 2024--2025, with P for reference. Observed mean over all '
        '2024--2025 sampled dates, as in main-text Table~2; stations are ordered by it. Positive differences favour 500~m. The largest difference, at SK0071A, reflects a valley-bottom station whose '
        '2~km cell places it above its surroundings. Units: ng~m$^{-3}$.}\n'
        '\\label{tab:supportstations}\n\\begin{tabular}{lrrrrr}\n\\toprule\n'
        'Station & Observed mean & P & E+P, 500~m & E+P, 2~km & 2~km minus 500~m \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # Supplementary Fig. S1: per-station support comparison as a dumbbell plot (replaces the former table)
    order_s = obs_mean.drop(IND).sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for yy, e in enumerate(order_s):
        a5, a2, pp_ = w.loc[e, MID('E_P')], w.loc[e, 'E_P_2KM'], w.loc[e, MID('P')]
        ax.plot([a5, a2], [yy, yy], color='#bbbbbb', lw=2, zorder=1)
        ax.scatter(a5, yy, color='#0072B2', s=36, zorder=3, label='E+P, 500 m cells' if yy == 0 else None)
        ax.scatter(a2, yy, color='#D55E00', marker='s', s=32, zorder=3, label='E+P, 2 km cells' if yy == 0 else None)
        ax.scatter(pp_, yy, color='#555555', marker='|', s=90, zorder=2, label='P (no static inputs)' if yy == 0 else None)
    ax.set_yticks(range(len(order_s)), [f'{e} ({obs_mean[e]:.2f})' for e in order_s], fontsize=8.5)
    ax.set_xlabel('Withheld-station (LOSO) daily RMSE, 2024–2025 (ng m$^{-3}$)')
    ax.set_ylabel('Station (observed mean, ng m$^{-3}$)')
    ax.grid(axis='x', color='#eeeeee', lw=.6)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(loc='lower right', fontsize=8.5, frameon=True, edgecolor='#999999', fancybox=False)
    fig.tight_layout()
    fig.savefig(OUT / 'support_stations.pdf')
    fig.savefig(OUT / 'support_stations.png', dpi=200)
    plt.close(fig)

    # ---- single-station influence on the 2 km +E contrast (review item 1)
    inf2 = pd.read_csv(S2 / 'station_influence.csv')
    x = inf2[(inf2.label == '+E') & (inf2.protocol == 'loso') & (inf2.estimand == 'daily')].set_index('excluded_station')
    assert x.difference_RMSE.idxmin() == 'SK0071A'

    def loso_pred(path):
        q = pd.read_csv(path, parse_dates=['datum'])
        q = q[(q.input_variant == 'permissive') & (q.target == 'log1p') & (q.weighting == 'uniform')
              & (q.augment == 'none') & (q.protocol == 'loso') & (q.datum.dt.year >= 2024) & (q.eoi != IND)]
        return q
    for path, tag in ((A2, 'K'), (A5, 'F')):
        q = loso_pred(path)
        q = q[q.eoi != 'SK0071A']
        put(f'{tag}AddELosoNonNoOsc', pd.Series(paired(q[q.arm == 'E_P'], q[q.arm == 'P'], KEYS, 'station', 'RMSE')))
    cols = ['tpi_local', 'elev_mean']
    t5 = pd.read_csv(INPUTS / 'train_ready_permissive_500m.csv').groupby('eoi')[cols].first()
    t2 = pd.read_csv(INPUTS / 'train_ready_permissive_2km_exact.csv').groupby('eoi')[cols].first()
    assert t5.tpi_local.idxmin() == 'SK0071A' and t2.tpi_local.idxmax() == 'SK0071A'
    num['OscTpiFive'] = mn(t5.loc['SK0071A', 'tpi_local'], 0)
    num['OscTpiTwo'] = '+' + mn(t2.loc['SK0071A', 'tpi_local'], 0)
    num['OscElevDiff'] = f"{t2.loc['SK0071A', 'elev_mean'] - t5.loc['SK0071A', 'elev_mean']:.0f}"
    # ---- E+PM vs PM station-year means at withheld stations (review item 2)
    yl = ym5[(ym5.protocol == 'loso') & (ym5.eoi != IND)]
    put('XEnvPmVsPmLosoNonMean', pd.Series(paired(yl[yl.method == MID('E_PM')], yl[yl.method == MID('PM')],
                                                  ['eoi', 'year'], 'station', 'RMSE')))
    num['IndLosoPm'] = f"{summ(S, MID('PM'), 'loso', 'industrial_descriptive').RMSE:.3f}"
    ind = {arm: summ(S, MID(arm), 'loso', 'industrial_descriptive').RMSE
           for arm in ('PM', 'P', 'E', 'E_P', 'E_P_M', 'E_PM', 'E_NO2')}
    lo_arm, hi_arm = min(ind, key=ind.get), max(ind, key=ind.get)
    assert (lo_arm, hi_arm) == ('PM', 'E_P')
    num['IndLosoMin'], num['IndLosoMax'] = f'{ind[lo_arm]:.3f}', f'{ind[hi_arm]:.3f}'
    num['IndLosoEnv'] = f"{ind['E']:.3f}"

    # ---- stress tests (E+P vs E)
    for prot, tc in (('year_out', 'Year'), ('season_year_out', 'SeasonYear'), ('calendar_season_out', 'CalSeason')):
        put(f'Stress{tc}', pair(P, 'Stress', prot, 'nonindustrial'))

    (OUT / 'numbers.tex').write_text('% Generated by APR/analysis/manuscript_v3/assets_v3.py\n' +
                                     ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
    (OUT / 'numbers.json').write_text(json.dumps(num, indent=1) + '\n')

    # ---- main ladder table
    rows = []
    for arm in LADDER + ['E_PM', 'E_NO2']:
        if arm == 'E_PM':
            rows.append('\\midrule\n\\multicolumn{9}{@{}l}{\\emph{Programme-matched variants of E+P}} \\\\')
        c = [NAME[arm]]
        for prot in ('block30', 'loso'):
            if prot == 'loso' and arm in ('E_P_M_I', 'PM_I'):
                c += ['--'] * 5
                continue
            n = summ(S, MID(arm), prot, 'nonindustrial')
            a = summ(S, MID(arm), prot, 'all_network')
            c += [f'{n.RMSE:.3f}', mn(n.agreement_R2, 2), f'{a.RMSE:.3f}']
            if prot == 'loso':
                m = summ(S, MID(arm), prot, 'nonindustrial', 'station_year_mean')
                c += [f'{m.RMSE:.3f}', mn(m.agreement_R2, 2)]
        if arm == 'E_P':
            c = [f'\\nb{{{x}}}' for x in c]
        rows.append(' & '.join(c) + ' \\\\')
    (OUT / 'ladder_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n'
        '\\caption{Input-ladder results, 2024--2025, for daily concentrations and (LOSO) station-year means over sampled dates. '
        'Non.: 20 nonindustrial stations (4852 daily values; 40 station-years); All: 21 stations (5089). '
        'RMSE in ng~m$^{-3}$; $R^2$ is the coefficient of determination. Models with station indicators (I) are not '
        'applicable to withheld stations. E+P, the principal model, is in bold. Paired contrasts are shown in '
        'Fig.~\\ref{fig:benefits}, with values in Supplementary Table~S4.}\n\\label{tab:ladder}\n'
        '\\begin{tabular}{@{}lrrrrrrrr@{}}\n\\toprule\n'
        ' & \\multicolumn{3}{c}{30-day gaps, daily} & \\multicolumn{3}{c}{LOSO, daily} & \\multicolumn{2}{c}{LOSO, station-year mean} \\\\\n'
        '\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(l){8-9}\n'
        'Model & RMSE non. & $R^2$ non. & RMSE all & RMSE non. & $R^2$ non. & RMSE all & RMSE non. & $R^2$ non. \\\\\n'
        '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- supplementary contrast table
    rows = []
    for code, lab, a, b, disp, prots in INPUT_CONTRASTS:
        for prot in prots:
            c = [disp, TASK[prot]]
            for scope, est in (('nonindustrial', 'daily'), ('all_network', 'daily'), ('nonindustrial', 'station_year_mean')):
                c.append(tri(pair(P, lab, prot, scope, est)))
            rows.append(' & '.join(c) + ' \\\\')
    (OUT / 'contrast_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}\n'
        '\\caption{Input-group RMSE contrasts, 2024--2025, candidate minus comparator with paired 95\\% bootstrap '
        'intervals (ng~m$^{-3}$; main-text Section~2.6); negative values favour the candidate; supported differences are in bold. Daily 30-day intervals resample stations and blocks; '
        'all others resample stations. Model RMSEs are in main-text Table~4. Static support 500~m.}\n'
        f'\\label{{tab:contrasts}}\n\\begin{{tabular}}{{@{{}}ll{TRI}{TRI}{TRI_END}@{{}}}}\n\\toprule\n'
        ' & & \\multicolumn{6}{c}{Daily} & \\multicolumn{3}{c}{Station-year mean} \\\\\n'
        '\\cmidrule(lr){3-8}\\cmidrule(l){9-11}\n'
        'Contrast & Task & \\multicolumn{3}{c}{Nonindustrial} & \\multicolumn{3}{c}{All stations} & '
        '\\multicolumn{3}{c}{Nonindustrial} \\\\\n'
        '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- main-text contrast table (nonindustrial; bold = supported)
    def cell(lab, prot, est):
        x = P[(P.label == lab) & (P.protocol == prot) & (P.scope == 'nonindustrial') & (P.estimand == est)]
        return TRI_NA if x.empty else tri(x.iloc[0])
    main_rows = [('+E (E+P vs P)', '+E'), ('+P (E+P vs E)', '+P'), ('+NO$_2$ (P vs PM)', '+NO2'),
                 ('+NO$_2$ given E (E+P vs E+PM)', '+NO2 given E'), ('+PM given E (E+P vs E+NO$_2$)', '+PM given E'),
                 ('+M (E+P+M vs E+P)', '+M'), ('+I (E+P+M+I vs E+P+M)', '+I'), ('+I (PM+I vs PM)', '+I (PM level)'),
                 ('Bundle (E+P+M+I vs PM+I)', 'Bundle')]
    rows = [' & '.join([d, cell(l, 'block30', 'daily'), cell(l, 'loso', 'daily'), cell(l, 'loso', 'station_year_mean')]) + ' \\\\'
            for d, l in main_rows]
    (OUT / 'main_contrast_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Input-group contrasts at the 20 nonindustrial stations, 2024--2025: RMSE difference, candidate minus '
        'comparator, with paired 95\\% bootstrap intervals (ng~m$^{-3}$; Section~\\ref{sec:scoring}). Negative values '
        'favour the candidate; supported '
        'differences (interval excludes zero) are in bold. Models with station indicators are not applicable to withheld '
        'stations. Full-network contrasts are given in Supplementary Section S4.}\n\\label{tab:maincontrasts}\n'
        f'\\begin{{tabular}}{{@{{}}l{TRI}{TRI}{TRI_END}@{{}}}}\n\\toprule\n'
        ' & \\multicolumn{3}{c}{Task 1 (30-day gaps)} & \\multicolumn{6}{c}{Task 2 (withheld stations)} \\\\\n'
        '\\cmidrule(lr){2-4}\\cmidrule(l){5-10}\n'
        'Contrast & \\multicolumn{3}{c}{Daily} & \\multicolumn{3}{c}{Daily} & '
        '\\multicolumn{3}{c}{Station-year mean} \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- supplementary sensitivity table
    rows = []
    for code, lab, disp in SENS:
        if lab == 'Weights':
            continue  # weighting contrasts are in the main-text weighting table
        c = [disp]
        for prot in ('block30', 'loso'):
            c.append(tri(pair(P, lab, prot, 'nonindustrial', 'daily')))
        c.append(tri(pair(P, lab, 'loso', 'nonindustrial', 'station_year_mean')))
        rows.append(' & '.join(c) + ' \\\\')
        if lab == 'Weights':
            for scope, lev in (('nonindustrial_higher', '\\quad station-years $>1$~ng~m$^{-3}$'),
                               ('nonindustrial_lower', '\\quad station-years $\\leq1$~ng~m$^{-3}$')):
                rows.append(' & '.join([lev] + [tri(pair(P, lab, p_, scope, 'daily')) for p_ in ('block30', 'loso')]
                                       + [tri(pair(P, lab, 'loso', scope, 'station_year_mean'))]) + ' \\\\')
        if lab in ('Masking', 'Programme-matched'):
            rows.append(' & '.join(['\\quad incomplete-pollutant days'] +
                                   [tri(pair(P, lab, p_, 'nonindustrial', 'daily_incomplete_pollutant_days')) for p_ in ('block30', 'loso')]
                                   + [TRI_NA]) + ' \\\\')
    rows.append('\\midrule\n\\multicolumn{10}{@{}l}{\\emph{Temporal stress tests, daily, E+P minus E}} \\\\')
    for prot_, disp_ in (('year_out', 'Year out'), ('season_year_out', 'Season-year out'),
                         ('calendar_season_out', 'Calendar season out')):
        e_, ep_ = summ(S, MID('E'), prot_, 'nonindustrial'), summ(S, MID('E_P'), prot_, 'nonindustrial')
        rows.append(f'{disp_} ($n={int(ep_.n)}$) & {tri(pair(P, "Stress", prot_, "nonindustrial"))} & '
                    f'\\multicolumn{{6}}{{l}}{{(E {e_.RMSE:.3f}, E+P {ep_.RMSE:.3f})}} \\\\')
    (OUT / 'sensitivity_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Sensitivity contrasts for the principal model E+P at nonindustrial stations, variant minus '
        'reference with paired 95\\% bootstrap intervals (ng~m$^{-3}$; main-text Section~2.6); positive values mean the '
        'variant is worse; supported differences are in bold. '
        'Reference: uniform weights, log target, all available pollutant hours, native missing-value handling, '
        '500~m static cells. Programme-matched predictions use E+P, E+PM, E+NO$_2$ or E according to the '
        'pollutant groups available on each target day, and equal E+P on complete-pollutant days. Stress tests withhold each primary year, each complete '
        'season-year (incomplete boundary seasons excluded) or each calendar season across all years; their intervals '
        'do not quantify variability over other years or seasons. The weighting block gives E+P and E with inverse '
        'minus uniform weights (levels in main-text Table~5); its bias rows are differences in station-year-mean bias '
        '(prediction minus observation), where negative values mean lower predictions.}\n'
        f'\\label{{tab:sensitivity}}\n\\begin{{tabular}}{{@{{}}l{TRI}{TRI}{TRI_END}@{{}}}}\n\\toprule\n'
        ' & \\multicolumn{3}{c}{30-day gaps} & \\multicolumn{6}{c}{Withheld stations (LOSO)} \\\\\n'
        '\\cmidrule(lr){2-4}\\cmidrule(l){5-10}\n'
        'Variant & \\multicolumn{3}{c}{Daily} & \\multicolumn{3}{c}{Daily} & '
        '\\multicolumn{3}{c}{Station-year mean} \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- population table (E, E+P)
    rows = []
    for scope, disp, sc in (('all_network', 'All 21', 'all_network'), ('nonindustrial', 'Nonindustrial 20', 'nonindustrial'),
                            ('industrial_descriptive', 'Industrial 1', 'industrial_descriptive')):
        for prot in ('block30', 'loso'):
            for arm in ('E', 'P', 'E_P'):
                r = summ(S, MID(arm), prot, sc)
                rows.append(f'{disp} & {TASK[prot]} & {NAME[arm]} & {int(r.n)} & {r.RMSE:.3f} & {r.MAE:.3f} & {mn(r.bias)} \\\\')
    (OUT / 'population_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Daily scores by population, 2024--2025. The industrial '
        'subset is one station, SK0018A, and is descriptive. Units: ng~m$^{-3}$.}\n\\label{tab:populations}\n'
        '\\begin{tabular}{lllrrrr}\n\\toprule\nPopulation & Task & Model & n & RMSE & MAE & Bias \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- robustness table
    rows = []
    for prot, disp in (('year_out', 'Year out'), ('season_year_out', 'Season-year out'), ('calendar_season_out', 'Calendar season out')):
        e, ep = summ(S, MID('E'), prot, 'nonindustrial'), summ(S, MID('E_P'), prot, 'nonindustrial')
        r = pair(P, 'Stress', prot, 'nonindustrial')
        rows.append(f'{disp} & {int(ep.n)} & {e.RMSE:.3f} & {ep.RMSE:.3f} & {iv(r)} \\\\')
    (OUT / 'robustness_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Temporal stress tests at nonindustrial stations, '
        '2024--2025. Differences are E+P minus E. Season-year scoring excludes incomplete seasons. Station-cluster '
        'intervals do not quantify uncertainty over different years or seasons. Units: ng~m$^{-3}$.}\n'
        '\\label{tab:robustness}\n\\begin{tabular}{lrrrl}\n\\toprule\n'
        'Withholding & n & E RMSE & E+P RMSE & Difference [95\\% interval] \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- strata table (LOSO station-year means)
    ym = pd.read_csv(S5 / 'station_year_means.csv')
    ym = ym[(ym.protocol == 'loso') & (ym.eoi != IND)]
    rows = []
    for lev, disp in (('higher', 'Higher ($>1$)'), ('lower', 'Lower ($\\leq1$)')):
        for arm in ('E', 'P', 'E_P'):
            x = ym[(ym.method == MID(arm)) & (ym.stratum == lev)]
            e = x.prediction - x.observed
            rows.append(f'{disp} & {NAME[arm]} & {len(x)} & {np.sqrt((e**2).mean()):.3f} & {e.abs().mean():.3f} & {mn(e.mean())} \\\\')
            num_key = f'Strata{CODE[arm]}{lev.capitalize()}Bias'
            num[num_key] = mn(e.mean())
    (OUT / 'strata_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Descriptive LOSO station-year-mean errors by observed '
        'concentration stratum (ng~m$^{-3}$), with 20 nonindustrial station-years in each stratum.}\n\\label{tab:strata}\n'
        '\\begin{tabular}{llrrrr}\n\\toprule\nStratum & Model & Station-years & RMSE & MAE & Bias \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    (OUT / 'numbers.tex').write_text('% Generated by APR/analysis/manuscript_v3/assets_v3.py\n' +
                                     ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
    (OUT / 'numbers.json').write_text(json.dumps(num, indent=1) + '\n')

    # ---- gap-task tables (E+P vs simple comparators, common bracketable)
    methods = [MID('E_P')] + list(BASE)
    disp = {MID('E_P'): 'E+P', **BASE}
    order = [MID('E_P'), 'PM_RIDGE', 'HARMONIC', 'INTERP_LINEAR']  # log-linear interpolation dropped (near-identical to linear)
    rows = []
    for m in order:
        r = summ_b = one(S, method=m, protocol='block30', estimand='daily', support='common_bracketable', scope='nonindustrial')
        rows.append(f'{disp[m]} & {int(r.n)} & {r.RMSE:.3f} & {r.MAE:.3f} & {mn(r.bias)} \\\\')
    (OUT / 'longgap_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\small\n\\caption{Thirty-day-gap comparators at nonindustrial stations on '
        'common bracketable dates, 2024--2025. E+P uses its 30-day out-of-fold predictions; model comparators are '
        'fitted inside each withholding fold. Units: ng~m$^{-3}$.}\n\\label{tab:longgap}\n'
        '\\begin{tabular}{lrrrr}\n\\toprule\nMethod & n & RMSE & MAE & Bias \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    fr_ = pd.read_csv(INPUTS / 'train_ready_permissive_500m.csv', parse_dates=['datum'])
    prim_ = fr_[fr_.datum.dt.year.isin([2024, 2025])]
    pop_mean = {'all_network': prim_.bap.mean(), 'nonindustrial': prim_[prim_.eoi != IND].bap.mean()}
    rows = []
    for m in order:
        c = [disp[m]]
        for prot_, est, sup, d in (('block30', 'daily', 'common_bracketable', 3),
                                   ('dispersed_mod9', 'daily', 'common_bracketable', 3),
                                   ('dispersed_mod9', 'reconstructed_mean', 'common_bracketable_replacements', 3)):
            for scope in ('all_network', 'nonindustrial'):
                r_ = one(S, method=m, protocol=prot_, estimand=est, support=sup, scope=scope).RMSE
                if est == 'reconstructed_mean':  # station-year mean after filling: RMSE and share of the observed mean
                    c.append(f"{r_:.{d}f} ({100 * r_ / pop_mean[scope]:.1f}\\%)")
                else:
                    c.append(f"{r_:.{d}f}")
        rows.append(' & '.join(c) + ' \\\\')
    nd = one(S, method=MID('E_P'), protocol='dispersed_mod9', estimand='daily', support='common_bracketable', scope='all_network').n
    nn = one(S, method=MID('E_P'), protocol='dispersed_mod9', estimand='daily', support='common_bracketable', scope='nonindustrial').n
    bd = one(S, method=MID('E_P'), protocol='block30', estimand='daily', support='common_bracketable', scope='all_network').n
    bn = one(S, method=MID('E_P'), protocol='block30', estimand='daily', support='common_bracketable', scope='nonindustrial').n
    (OUT / 'gap_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{E+P and simple comparators '
        'on identical bracketable targets, 2024--2025: RMSE (ng~m$^{-3}$). 30-day gaps: '
        f'{int(bd)} daily values ({int(bn)} nonindustrial); dispersed gaps: {int(nd)} ({int(nn)} nonindustrial). '
        'The reconstructed station-year mean (Section~\\ref{sec:validation}; 378 station-year/fold reconstructions, '
        '360 nonindustrial) combines retained observations with filled values over all sampled dates; the percentage '
        'relates its RMSE to the observed mean concentration. Model comparators are fitted inside each fold.}\n\\label{tab:gap}\n'
        '\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n'
        ' & \\multicolumn{2}{c}{30-day gaps} & \\multicolumn{2}{c}{Dispersed gaps} & '
        '\\multicolumn{2}{c}{Dispersed gaps} \\\\\n'
        ' & \\multicolumn{2}{c}{daily values} & \\multicolumn{2}{c}{daily values} & '
        '\\multicolumn{2}{c}{reconstructed station-year mean} \\\\\n'
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\nMethod & All & Nonind. & All & Nonind. & All & Nonind. \\\\\n'
        '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    rows = []
    for scope, sd in (('all_network', 'All stations'), ('nonindustrial', 'Nonindustrial')):
        for m in order[1:]:
            c = [sd, BASE[m],
                 tri(pair(P, 'Gap vs simple', 'block30', scope, 'daily', method_b=m), 3),
                 tri(pair(P, 'Gap vs simple', 'dispersed_mod9', scope, 'daily', method_b=m), 3),
                 tri(pair(P, 'Gap vs simple', 'dispersed_mod9', scope, 'reconstructed_mean', method_b=m), 4)]
            rows.append(' & '.join(c) + ' \\\\')
    (OUT / 'dispersed_intervals.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}\n\\caption{E+P minus each '
        'simple comparator on identical bracketable targets, RMSE difference with paired 95\\% bootstrap intervals '
        '(ng~m$^{-3}$; main-text Section~2.6; station~$\\times$~block resampling for 30-day gaps, station clusters '
        'otherwise). Negative differences favour E+P; supported differences are in bold. RMSEs are in main-text '
        f'Table~6. Targets: 30-day gaps {int(bd)} ({int(bn)} nonindustrial); dispersed gaps {int(nd)} ({int(nn)}); '
        'reconstructed means 378 station-year/fold cases (360).}\n\\label{tab:dispersedintervals}\n'
        f'\\begin{{tabular}}{{@{{}}ll{TRI}{TRI}{TRI_END}@{{}}}}\n\\toprule\n'
        ' & & \\multicolumn{3}{c}{30-day gaps} & \\multicolumn{6}{c}{Dispersed gaps} \\\\\n'
        '\\cmidrule(lr){3-5}\\cmidrule(l){6-11}\n'
        'Population & Comparator & \\multicolumn{3}{c}{Daily} & \\multicolumn{3}{c}{Daily} & '
        '\\multicolumn{3}{c}{Reconstructed mean} \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- reconstructed-mean error as a share of the nonindustrial mean (Results 3.4)
    non_mean = pop_mean['nonindustrial']
    for m, k in ((MID('E_P'), 'RecoveryPctEP'), ('INTERP_LINEAR', 'RecoveryPctLin')):
        r = one(S, method=m, protocol='dispersed_mod9', estimand='reconstructed_mean',
                support='common_bracketable_replacements', scope='nonindustrial').RMSE
        num[k] = f'{100 * r / non_mean:.0f}'

    # ---- Fig. 3: input-group contrasts, three panels; lower section: drop-one decomposition of E (nonindustrial)
    fig_rows = [('+NO$_2$ (P vs PM)', '+NO2'), ('+E (E+P vs P)', '+E'), ('+P (E+P vs E)', '+P'),
                ('+NO$_2$ given E (E+P vs E+PM)', '+NO2 given E'), ('+PM given E (E+P vs E+NO$_2$)', '+PM given E'),
                ('+M (E+P+M vs E+P)', '+M'), ('+I (E+P+M+I vs E+P+M)', '+I'),
                ('+I (PM+I vs PM)', '+I (PM level)'), ('Bundle (E+P+M+I vs PM+I)', 'Bundle')]
    dec_rows = [('Meteorology', 'EP_NOMET'), ('Terrain', 'EP_NOTER'), ('Traffic', 'EP_NOTRAF'),
                ('Residential emissions', 'EP_NOEMIS')]
    dec3 = pd.read_csv(APR / 'results/consolidated_500m_decomp/scores/decomp_contrasts.csv')
    dec3 = dec3[dec3.metric == 'RMSE']
    panels = [('block30', 'daily', '(a) 30-day gaps\ndaily values'), ('loso', 'daily', '(b) Withheld stations\ndaily values'),
              ('loso', 'station_year_mean', '(c) Withheld stations\nstation-year means')]
    gap = 1.0  # vertical space for the section heading
    ypos_dec = [len(fig_rows) + gap + k for k in range(len(dec_rows))]
    flex_rows = [('PM+I vs PM ridge', 'PM+I vs PM ridge'), ('E vs ridge E', 'E (trees vs linear)'),
                 ('E+P vs ridge E+P', 'E+P (trees vs linear)')]
    lcon = pd.read_csv(APR / 'results/consolidated_500m_linear/linear_contrasts.csv')
    lcon = lcon[lcon.metric == 'RMSE']
    ypos_flex = [ypos_dec[-1] + 1 + gap + k for k in range(len(flex_rows))]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 7.4), sharey=True, sharex=True)
    for ax, (prot, est, title) in zip(axes, panels):
        for y, (disp_, lab) in enumerate(fig_rows):
            for scope, color, shift, name in (('all_network', '#555555', -.14, 'All 21 stations'),
                                              ('nonindustrial', '#0072B2', .14, 'Nonindustrial 20')):
                x = P[(P.label == lab) & (P.protocol == prot) & (P.scope == scope) & (P.estimand == est)]
                if x.empty:
                    continue
                r = x.iloc[0]
                v, lo_, hi_ = -r.estimate, -r.ci_high, -r.ci_low
                ax.errorbar(v, y + shift, xerr=[[v - lo_], [hi_ - v]], fmt='o', ms=4, capsize=2.2, color=color,
                            label=name if y == 1 else None)
        # drop-one: increase in RMSE when the subgroup is removed = gain from adding it back (positive = helps)
        for y, (disp_, arm) in zip(ypos_dec, dec_rows):
            r = one(dec3, arm=arm, protocol=prot, estimand=est)
            ax.errorbar(r.estimate, y, xerr=[[r.estimate - r.ci_low], [r.ci_high - r.estimate]], fmt='o', ms=4,
                        capsize=2.2, color='#0072B2')
        # flexibility: RMSE reduction from trees relative to ridge with identical inputs (positive = trees better)
        for y, (disp_, comp) in zip(ypos_flex, flex_rows):
            x = lcon[(lcon.comparison == comp) & (lcon.protocol == prot) & (lcon.estimand == est)]
            if x.empty:
                continue
            r = x.iloc[0]
            v, lo_, hi_ = -r.estimate, -r.ci_high, -r.ci_low
            ax.errorbar(v, y, xerr=[[v - lo_], [hi_ - v]], fmt='o', ms=4, capsize=2.2, color='#0072B2')
        ax.axhline(ypos_dec[-1] + .5 + gap / 2, color='#999999', lw=.8)
        ax.axhline(len(fig_rows) - .5 + gap / 2, color='#999999', lw=.8)
        ax.axvline(0, color='grey', ls=':', lw=1)
        ax.set_title(title, fontsize=9.5, loc='left')
        ax.set_xlabel('Reduction in RMSE (ng m$^{-3}$)', fontsize=9)
        ax.grid(axis='x', color='#eeeeee', lw=.6)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
    ticks = list(range(len(fig_rows))) + ypos_dec + ypos_flex
    axes[0].set_yticks(ticks, [r[0] for r in fig_rows] + [r[0] for r in dec_rows] + [r[0] for r in flex_rows], fontsize=8.5)
    axes[0].text(-0.02, ypos_dec[-1] + .5 + gap * .75, 'Trees vs ridge regression, same inputs', transform=axes[0].get_yaxis_transform(),
                 ha='right', va='center', fontsize=8.5, style='italic')
    axes[0].text(-0.02, len(fig_rows) - .5 + gap * .75, 'E+P vs E+P without subgroup', transform=axes[0].get_yaxis_transform(),
                 ha='right', va='center', fontsize=8.5, style='italic')
    axes[0].invert_yaxis()
    # place the legend in panel (c)'s empty indicator rows (+I, PM-level +I, Bundle do not apply to withheld stations)
    y0, y1 = axes[2].get_ylim()
    frac = (y0 - 7.0) / (y0 - y1)  # inverted axis: fraction from bottom at row index 7
    axes[2].legend(loc='center right', bbox_to_anchor=(1.0, frac), fontsize=8, frameon=True, title='Legend',
                   title_fontsize=8, edgecolor='#999999', fancybox=False)
    fig.tight_layout()
    fig.savefig(OUT / 'information_benefits.pdf')
    fig.savefig(OUT / 'information_benefits.png', dpi=200)
    plt.close(fig)

    # ---- Fig. 4 station means and Fig. S1 time series
    p = pd.read_csv(A5, parse_dates=['datum'])
    p = p[(p.input_variant == 'permissive') & (p.target == 'log1p') & (p.weighting == 'uniform') & (p.augment == 'none')]
    pick = lambda arm, prot: p[(p.arm == arm) & (p.protocol == prot) & (p.datum.dt.year >= 2024)][KEYS + ['observed', 'prediction']]
    COL = {'Observed': '#222222', 'E': '#E69F00', 'P': '#CC79A7', 'E+P': '#0072B2'}
    lo = {a: pick(a, 'loso') for a in ('E', 'P', 'E_P')}
    ann = pd.read_csv(S5 / 'station_year_means.csv')
    ann = ann[ann.protocol == 'loso']
    obs = ann[ann.method == MID('E_P')]
    order_st = obs.groupby('eoi')['observed'].mean().sort_values().index.tolist()
    ev5 = pd.read_csv(APR / 'results/consolidated_500m_einv/assembled/validated_predictions.csv', parse_dates=['datum'])
    ev5 = ev5[(ev5.protocol == 'loso') & ev5.datum.dt.year.isin([2024, 2025])]
    einv_ann = ev5.assign(year=ev5.datum.dt.year).groupby(['eoi', 'year'], as_index=False).agg(prediction=('prediction', 'mean'))
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True, sharey=True)
    for year, ax in zip((2024, 2025), axes):
        for arm, color, marker, off, lab_ in (('E', COL['E'], '^', -.30, 'E'),
                                              ('E_inv', '#D55E00', 'v', -.15, 'E (inverse weights)'),
                                              ('P', COL['P'], 's', 0.0, 'P'), ('E_P', COL['E+P'], 'o', .15, 'E+P'),
                                              ('Observed', COL['Observed'], 'D', .30, 'Observed')):
            x = obs if arm == 'Observed' else (einv_ann if arm == 'E_inv' else ann[ann.method == MID(arm)])
            x = x[x.year == year].set_index('eoi').reindex(order_st)
            ax.scatter(np.arange(len(order_st)) + off, x.observed if arm == 'Observed' else x.prediction, c=color,
                       marker=marker, s=26, label=lab_, zorder=3)
        ax.set_title(str(year), loc='left')
        ax.set_ylabel('Station-year mean (ng m$^{-3}$)')
        k = order_st.index(IND)
        ax.axvspan(k - .45, k + .45, color='#eeeeee', zorder=0)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(ncol=5, loc='upper left', fontsize=9)
    axes[1].set_xticks(range(len(order_st)), [e + ('*' if e == IND else '') for e in order_st], rotation=55, ha='right', fontsize=9)
    axes[1].set_xlabel('Stations ordered by observed concentration; * industrial')
    fig.tight_layout()
    fig.savefig(OUT / 'station_means.pdf')
    fig.savefig(OUT / 'station_means.png', dpi=200)
    plt.close(fig)

    def broken(df):
        df = df.sort_values('datum')
        g = df.datum.diff().dt.days > 7
        pad = df.loc[g].assign(prediction=np.nan, datum=df.loc[g, 'datum'] - pd.Timedelta(days=1))
        return pd.concat([df, pad]).sort_values('datum')

    b30 = pick('E_P', 'block30')
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    for ax, (s_, title) in zip(axes, (('SK0025A', 'Jelšava (SK0025A), background, high concentration'),
                                      ('SK0048A', 'Bratislava, Jeséniova (SK0048A), background, low concentration'))):
        o = b30[b30.eoi == s_].sort_values('datum')
        ax.scatter(o.datum, o.observed, s=9, color=COL['Observed'], label='Observed', zorder=4)
        q = broken(o)
        ax.plot(q.datum, q.prediction, color='#56B4E9', lw=1.1, label='E+P, 30-day gaps', zorder=3)
        q = broken(lo['E_P'][lo['E_P'].eoi == s_])
        ax.plot(q.datum, q.prediction, color=COL['E+P'], lw=1.1, label='E+P, LOSO', zorder=3)
        q = broken(lo['E'][lo['E'].eoi == s_])
        ax.plot(q.datum, q.prediction, color=COL['E'], lw=1.0, ls='--', label='E, LOSO', zorder=2)
        if s_ == 'SK0048A':  # E overpredicts this clean site strongly; truncate and report
            e_ = lo['E'][lo['E'].eoi == s_].prediction
            num['TsCapDays'] = str(int((e_ > 5).sum()))
            num['TsCapMax'] = f'{e_.max():.1f}'
            ax.set_ylim(-0.2, 5)
        ax.set_title(title, loc='left', fontsize=9.5)
        ax.set_ylabel('B[a]P (ng m$^{-3}$)')
        ax.grid(axis='y', color='#e5e5e5', lw=.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8.5, ncol=4, loc='upper center')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.tight_layout()
    fig.savefig(OUT / 'timeseries.pdf')
    fig.savefig(OUT / 'timeseries.png', dpi=200)
    plt.close(fig)
    # ---- Fig. 5: Task 1 demonstration on one withheld network-wide winter block
    # Rule: block with the highest nonindustrial network mean B[a]P in 2024-2025; stations nearest the
    # 25th, 50th and 75th percentiles of nonindustrial station means, plus the highest-mean station.
    fb = pd.read_csv(INPUTS / 'train_ready_permissive_500m.csv', parse_dates=['datum'])
    fb['block'] = ((fb.datum - pd.Timestamp('2023-06-02')).dt.days // 30)
    pr_ = fb[fb.datum.dt.year.isin([2024, 2025]) & (fb.eoi != IND)]
    blk = int(pr_.groupby('block').bap.mean().idxmax())
    smean = pr_.groupby('eoi').bap.mean()
    picks = [(smean - smean.quantile(q)).abs().idxmin() for q in (.25, .5, .75)] + [smean.idxmax()]
    assert len(set(picks)) == 4
    b0, b1 = fb.loc[fb.block == blk, 'datum'].min(), fb.loc[fb.block == blk, 'datum'].max()
    num['DemoBlockStart'], num['DemoBlockEnd'] = b0.strftime('%-d %B %Y'), b1.strftime('%-d %B %Y')
    base_ = pd.read_csv(R02B, parse_dates=['datum'])
    lin = base_[(base_.protocol == 'block30') & (base_.method == 'INTERP_LINEAR')]
    epb = p[(p.protocol == 'block30') & (p.arm == 'E_P')]
    names = {'SK0076A': 'Rovinka', 'SK0214A': 'Banská Bystrica, Štefánikovo nábr.', 'SK0008A': 'Ružomberok',
             'SK0025A': 'Jelšava', 'SK0002A': 'Bratislava, Trnavské mýto', 'SK0050A': 'Prievidza'}
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True)
    w0, w1 = b0 - pd.Timedelta(days=21), b1 + pd.Timedelta(days=21)
    for ax, st_ in zip(axes.ravel(), picks):
        o = fb[(fb.eoi == st_) & fb.datum.between(w0, w1)]
        inb = o.block == blk
        ax.axvspan(b0 - pd.Timedelta(hours=12), b1 + pd.Timedelta(hours=12), color='#eeeeee', zorder=0)
        ax.scatter(o.datum[~inb], o.bap[~inb], s=14, color='#888888', zorder=3, label='Observed, retained')
        ax.scatter(o.datum[inb], o.bap[inb], s=18, color='#222222', zorder=4, label='Observed, withheld')
        e = epb[(epb.eoi == st_) & epb.datum.between(b0, b1)].sort_values('datum')
        ax.plot(e.datum, e.prediction, color='#0072B2', lw=1.4, marker='o', ms=3, zorder=5, label='E+P (held out)')
        pp = p[(p.protocol == 'block30') & (p.arm == 'P') & (p.eoi == st_) & p.datum.between(b0, b1)].sort_values('datum')
        ax.plot(pp.datum, pp.prediction, color='#E69F00', lw=1.2, ls='--', marker='s', ms=2.5, zorder=4, label='P (held out)')
        ax.set_title(f'{names.get(st_, st_)} ({st_}), mean {smean[st_]:.2f} ng m$^{{-3}}$', loc='left', fontsize=9)
        ax.grid(axis='y', color='#e5e5e5', lw=.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('B[a]P (ng m$^{-3}$)')
    for ax in axes[1]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axes[0, 0].legend(frameon=False, fontsize=7.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'gap_demo.pdf')
    fig.savefig(OUT / 'gap_demo.png', dpi=200)
    plt.close(fig)
    num['DemoStations'] = ', '.join(f'{names.get(x, x)} ({x})' for x in picks)

    # ---- environmental-group drop-one decomposition (spec addendum E; Results 3.2-3.3, Table S6)
    dec = pd.read_csv(APR / 'results/consolidated_500m_decomp/scores/decomp_contrasts.csv')
    dec = dec[dec.metric == 'RMSE']
    gname = {'EP_NOMET': ('Met', 'Meteorology (22)'), 'EP_NOTER': ('Ter', 'Terrain (6)'),
             'EP_NOTRAF': ('Traf', 'Traffic (3)'), 'EP_NOEMIS': ('Emis', 'Residential emissions (2)')}
    drows = []
    for arm in ('EP_NOEMIS', 'EP_NOMET', 'EP_NOTER', 'EP_NOTRAF'):
        code, disp_ = gname[arm]
        cells = [disp_]
        for prot in ('block30', 'loso'):
            for est, ec in (('daily', ''), ('station_year_mean', 'Mean')):
                r = one(dec, arm=arm, protocol=prot, estimand=est)
                put(f'D{code}{TCODE[prot]}{ec}', r)
                num[f'D{code}{TCODE[prot]}{ec}Rmse'] = f'{r.rmse_candidate:.3f}'
                cells.append(tri(r))
        drows.append(' & '.join(cells) + ' \\\\')
    (OUT / 'decomp_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}\n'
        '\\caption{Drop-one decomposition of the environmental group at the 20 nonindustrial stations: E+P with one '
        'environmental subgroup removed minus E+P, RMSE difference with paired 95\\% bootstrap intervals (ng~m$^{-3}$; '
        'main-text Section~2.6). Positive '
        'values mean that removing the subgroup increased error; supported differences are in bold. Correlated subgroups can '
        'substitute for one another, so a small effect does not show that a subgroup is uninformative.}\n'
        f'\\label{{tab:decomp}}\n\\begin{{tabular}}{{@{{}}l{TRI}{TRI}{TRI}{TRI_END}@{{}}}}\n\\toprule\n'
        ' & \\multicolumn{6}{c}{30-day gaps} & \\multicolumn{6}{c}{Withheld station (LOSO)} \\\\\n'
        '\\cmidrule(lr){2-7}\\cmidrule(l){8-13}\nSubgroup removed & \\multicolumn{3}{c}{Daily} & '
        '\\multicolumn{3}{c}{Station-year mean} & \\multicolumn{3}{c}{Daily} & \\multicolumn{3}{c}{Station-year mean} \\\\\n'
        '\\midrule\n' + '\n'.join(drows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- inverse-weighted E (spec addendum F): macros, Table S3 block, supplementary figure
    EV = APR / 'results/consolidated_500m_einv'
    es = pd.read_csv(EV / 'scores/einv_summary.csv')
    ec = pd.read_csv(EV / 'scores/einv_contrasts.csv')
    ec = ec[ec.metric == 'RMSE']
    g_ = lambda m, pr, est: one(es, model=m, protocol=pr, estimand=est)
    num['EInvLosoMeanRsq'] = mn(g_('E_inv', 'loso', 'station_year_mean').agreement_R2, 2)
    num['EInvLosoMeanRmse'] = f"{g_('E_inv', 'loso', 'station_year_mean').RMSE:.3f}"
    num['EInvLosoRmse'] = f"{g_('E_inv', 'loso', 'daily').RMSE:.3f}"
    num['EInvLosoHigherBias'] = mn(g_('E_inv', 'loso', 'station_year_mean_higher').bias, 2)
    num['EUniLosoHigherBias'] = mn(g_('E', 'loso', 'station_year_mean_higher').bias, 2)
    num['EInvBlockMeanRmse'] = f"{g_('E_inv', 'block30', 'station_year_mean').RMSE:.3f}"
    num['EUniBlockMeanRmse'] = f"{g_('E', 'block30', 'station_year_mean').RMSE:.3f}"
    for con, code in (('E_inv_minus_E', 'EInvVsE'), ('E_inv_minus_E_P', 'EInvVsEP')):
        for pr in ('block30', 'loso'):
            for est, ecd in (('daily', ''), ('station_year_mean', 'Mean')):
                put(f'{code}{TCODE[pr]}{ecd}', one(ec, contrast=con, protocol=pr, estimand=est))
    einv_rows = ['\\midrule\n\\multicolumn{10}{@{}l}{\\emph{Environmental model E with inverse weights}} \\\\']
    for con, disp_ in (('E_inv_minus_E', 'E inverse vs E uniform'), ('E_inv_minus_E_P', 'E inverse vs E+P uniform')):
        cells = [disp_] + [tri(one(ec, contrast=con, protocol=pr, estimand=est))
                           for pr, est in (('block30', 'daily'), ('loso', 'daily'), ('loso', 'station_year_mean'))]
        einv_rows.append(' & '.join(cells) + ' \\\\')

    # ---- weighting choice at withheld stations (main-text table): uniform vs inverse weights for E+P and E
    ymw = pd.read_csv(S5 / 'station_year_means.csv')
    ymw = ymw[(ymw.protocol == 'loso') & (ymw.eoi != IND)]
    WINV = 'inverse_1_over_bap_plus_0_5'
    sy = {('E_P', 'uni'): ymw[ymw.method == MID('E_P')], ('E_P', 'inv'): ymw[ymw.method == MID('E_P', w=WINV)],
          ('E', 'uni'): ymw[ymw.method == MID('E')]}
    evd = pd.read_csv(EV / 'assembled/validated_predictions.csv', parse_dates=['datum'])
    evd = evd[(evd.protocol == 'loso') & evd.datum.dt.year.isin([2024, 2025]) & (evd.eoi != IND)]
    evw = evd.assign(year=evd.datum.dt.year).groupby(['eoi', 'year'], as_index=False).agg(prediction=('prediction', 'mean'))
    a5w = pd.read_csv(A5, parse_dates=['datum'])
    a5w = a5w[(a5w.protocol == 'loso') & a5w.datum.dt.year.isin([2024, 2025]) & (a5w.eoi != IND) & (a5w.target == 'log1p')
              & (a5w.augment == 'none') & (a5w.input_variant == 'permissive')]
    dly = {('E_P', 'uni'): a5w[(a5w.arm == 'E_P') & (a5w.weighting == 'uniform')],
           ('E_P', 'inv'): a5w[(a5w.arm == 'E_P') & (a5w.weighting == WINV)],
           ('E', 'uni'): a5w[(a5w.arm == 'E') & (a5w.weighting == 'uniform')], ('E', 'inv'): evd}
    dly = {k: v[KEYS + ['observed', 'prediction']].sort_values(KEYS).reset_index(drop=True) for k, v in dly.items()}
    dstat = {}
    for k, v in dly.items():
        assert len(v) == 4852 and v[KEYS].equals(dly[('E', 'uni')][KEYS]), k
        e = v.prediction - v.observed
        dstat[k] = dict(rmse=np.sqrt((e ** 2).mean()), bias=e.mean(),
                        r2=1 - (e ** 2).sum() / ((v.observed - v.observed.mean()) ** 2).sum())
    ref = sy[('E', 'uni')][['eoi', 'year', 'observed', 'stratum']]
    sy[('E', 'inv')] = ref.merge(evw, on=['eoi', 'year'], how='left', validate='1:1')
    for k, v in sy.items():
        assert len(v) == 40 and v.prediction.notna().all() and (v.stratum.value_counts() == 20).all(), k
        assert v.set_index(['eoi', 'year']).observed.sort_index().equals(ref.set_index(['eoi', 'year']).observed.sort_index())
    for k, ref_rmse in {('E_P', 'uni'): summ(S, MID('E_P'), 'loso', 'nonindustrial').RMSE,
                        ('E_P', 'inv'): summ(S, MID('E_P', w=WINV), 'loso', 'nonindustrial').RMSE,
                        ('E', 'uni'): summ(S, MID('E'), 'loso', 'nonindustrial').RMSE,
                        ('E', 'inv'): g_('E_inv', 'loso', 'daily').RMSE}.items():
        assert abs(dstat[k]['rmse'] - ref_rmse) < 1e-9, k  # daily scores agree with the scoring runs
    WCODE = {'E_P': 'EnvPol', 'E': 'Env'}

    def sy_stats(v):
        e = v.prediction - v.observed
        r2 = 1 - (e ** 2).sum() / ((v.observed - v.observed.mean()) ** 2).sum()
        return dict(rmse=np.sqrt((e ** 2).mean()), r2=r2, hi=e[v.stratum == 'higher'].mean(),
                    lo=e[v.stratum == 'lower'].mean())

    sgn = lambda x: ('+' if float(f'{x:.2f}') > 0 else '') + mn(x, 2)
    sup_ = lambda r: bool(r.ci_low > 0 or r.ci_high < 0)
    bold = lambda t, flag: f'\\nb{{{t}}}' if flag else t
    wrows, s3w = [], []
    for arm in ('E_P', 'E'):
        st_ = {k: sy_stats(sy[(arm, k)]) for k in ('uni', 'inv')}
        a_, b_ = sy[(arm, 'inv')], sy[(arm, 'uni')]
        if arm == 'E_P':
            dd, dm = pair(P, 'Weights', 'loso', 'nonindustrial', 'daily'), pair(P, 'Weights', 'loso', 'nonindustrial', 'station_year_mean')
        else:
            dd, dm = one(ec, contrast='E_inv_minus_E', protocol='loso', estimand='daily'), one(ec, contrast='E_inv_minus_E', protocol='loso', estimand='station_year_mean')
        put(f'W{WCODE[arm]}DiffDaily', dd)
        put(f'W{WCODE[arm]}DiffMean', dm)
        dbias = pd.Series(paired(dly[(arm, 'inv')], dly[(arm, 'uni')], KEYS, 'station', 'bias'))
        put(f'W{WCODE[arm]}DiffDailyBias', dbias)
        flag = {'daily': sup_(dd), 'rmse': sup_(dm), 'dbias': sup_(dbias)}
        for lev, q in (('higher', 'hi'), ('lower', 'lo')):
            r = pd.Series(paired(a_[a_.stratum == lev], b_[b_.stratum == lev], ['eoi', 'year'], 'station', 'bias'))
            put(f'W{WCODE[arm]}Diff{q.capitalize()}', r)
            flag[q] = sup_(r)
            flag['r_' + q] = r
        for k, lab in (('uni', 'uniform'), ('inv', 'inverse')):
            s_ = st_[k]
            fl = flag if k == 'inv' else {q: False for q in ('daily', 'rmse', 'dbias', 'hi', 'lo')}  # bold marks supported inverse-minus-uniform differences
            d_ = dstat[(arm, k)]
            cells = [bold(f'{d_["rmse"]:.3f}', fl['daily']), mn(d_['r2'], 2), bold(sgn(d_['bias']), fl['dbias']),
                     bold(f'{s_["rmse"]:.3f}', fl['rmse']), mn(s_['r2'], 2), bold(sgn(s_['hi']), fl['hi']), bold(sgn(s_['lo']), fl['lo'])]
            wrows.append(f'{NAME[arm]} & {lab} & ' + ' & '.join(cells) + ' \\\\')
            for q in ('rmse', 'r2', 'hi', 'lo'):
                num[f'W{WCODE[arm]}{k.capitalize()}{dict(rmse="Rmse", r2="Rsq", hi="Hi", lo="Lo")[q]}'] = (f'{s_[q]:.3f}' if q == 'rmse' else
                                                                         mn(s_[q], 2) if q == 'r2' else sgn(s_[q]))
            num[f'W{WCODE[arm]}{k.capitalize()}Daily'] = f'{d_["rmse"]:.3f}'
            num[f'W{WCODE[arm]}{k.capitalize()}DailyRsq'] = mn(d_['r2'], 2)
            num[f'W{WCODE[arm]}{k.capitalize()}DailyBias'] = sgn(d_['bias'])
        if arm == 'E_P':
            wrows.append('\\midrule')
        # Table S3 weighting block: inverse minus uniform with paired intervals (levels in main-text Table 5)
        db = pair(P, 'Weights', 'block30', 'nonindustrial', 'daily') if arm == 'E_P' else \
            one(ec, contrast='E_inv_minus_E', protocol='block30', estimand='daily')
        s3w.append(f'{NAME[arm]} & {tri(db)} & {tri(dd)} & {tri(dm)} \\\\')
        s3w.append(f'\\quad bias, all days & {TRI_NA} & {tri(dbias)} & {TRI_NA} \\\\')
        for q, lev in (('hi', '$>1$'), ('lo', '$\\leq1$')):
            s3w.append(f'\\quad bias, station-years {lev}~ng~m$^{{-3}}$ & {TRI_NA} & {TRI_NA} & {tri(flag["r_" + q])} \\\\')
    t_ = (OUT / 'sensitivity_table.tex').read_text()
    t_ = t_.replace('\\midrule\n\\multicolumn{10}{@{}l}{\\emph{Temporal stress tests',
                    '\\midrule\n\\multicolumn{10}{@{}l}{\\emph{Inverse minus uniform weights}} \\\\\n' + '\n'.join(s3w)
                    + '\n\\midrule\n\\multicolumn{10}{@{}l}{\\emph{Temporal stress tests', 1)
    assert 'Inverse minus uniform weights' in t_
    (OUT / 'sensitivity_table.tex').write_text(t_)
    (OUT / 'weighting_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Uniform and inverse weights, $1/(\\mathrm{B[a]P}+0.5)$, at withheld stations (LOSO), 20 nonindustrial '
        'stations, 2024--2025 (ng~m$^{-3}$); 4852 daily values and 40 station-years. Bias is prediction minus observation; '
        'station-year-mean bias is averaged over the 20 station-years above and the 20 at or below 1~ng~m$^{-3}$. Inverse-weight values in '
        'bold differ from uniform weights with a paired 95\\% bootstrap interval excluding zero (Section~\\ref{sec:scoring}); '
        'the differences and their intervals are in Supplementary Table~S3.}\n'
        '\\label{tab:weighting}\n\\begin{tabular}{@{}llrrrrrrr@{}}\n\\toprule\n'
        ' & & \\multicolumn{3}{c}{Daily} & \\multicolumn{4}{c}{Station-year mean} \\\\\n\\cmidrule(lr){3-5}\\cmidrule(l){6-9}\n'
        'Model & Weights & RMSE & $R^2$ & Bias & RMSE & $R^2$ & Bias $>1$ & Bias $\\leq1$ \\\\\n\\midrule\n'
        + '\n'.join(wrows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- model flexibility: trees vs ridge with identical inputs (spec addendum G; Results 3.5, Table S6)
    LB = APR / 'results/consolidated_500m_linear'
    ls_ = pd.read_csv(LB / 'linear_summary.csv')
    lc_ = pd.read_csv(LB / 'linear_contrasts.csv')
    lc_ = lc_[lc_.metric == 'RMSE']
    comps = [('PM+I vs PM ridge', 'Pm', 'PM+I vs PM ridge'),
             ('E (trees vs linear)', 'Env', 'E'),
             ('E+P (trees vs linear)', 'EnvPol', 'E+P')]
    lrows = []
    for comp, code, disp_ in comps:
        for prot in ('block30', 'loso'):
            sub = ls_[(ls_.comparison == comp) & (ls_.protocol == prot)]
            if sub.empty:
                continue
            g2 = lambda m, est: one(sub, model=m, estimand=est)
            num[f'Lin{code}{TCODE[prot]}Rmse'] = f"{g2('linear', 'daily').RMSE:.3f}"
            num[f'Lin{code}{TCODE[prot]}MeanRsq'] = mn(g2('linear', 'station_year_mean').agreement_R2, 2)
            rd = one(lc_, comparison=comp, protocol=prot, estimand='daily')
            rm = one(lc_, comparison=comp, protocol=prot, estimand='station_year_mean')
            put(f'Flex{code}{TCODE[prot]}', rd, g2('linear', 'daily').RMSE)
            put(f'Flex{code}{TCODE[prot]}Mean', rm)
            lrows.append(' & '.join([disp_, TASK[prot], f"{g2('trees', 'daily').RMSE:.3f}", f"{g2('linear', 'daily').RMSE:.3f}",
                                     tri(rd), mn(g2('trees', 'station_year_mean').agreement_R2, 2),
                                     mn(g2('linear', 'station_year_mean').agreement_R2, 2), tri(rm)]) + ' \\\\')
    (OUT / 'linear_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{2pt}\n'
        '\\caption{Model flexibility at the 20 nonindustrial stations: gradient-boosted trees against ridge regression with '
        'identical inputs, target and folds, 2024--2025. RMSE in ng~m$^{-3}$; differences (in daily and station-year-mean '
        'RMSE) are trees minus ridge with paired '
        '95\\% bootstrap intervals (main-text Section~2.6); supported differences are in bold. At the PM level, the '
        'linear comparator is PM ridge (PM, seasonal terms and station indicators).}\n\\label{tab:linear}\n'
        f'\\begin{{tabular}}{{@{{}}llrr{TRI}rr{TRI_END}@{{}}}}\n\\toprule\n'
        ' & & \\multicolumn{5}{c}{Daily} & \\multicolumn{5}{c}{Station-year mean} \\\\\n'
        '\\cmidrule(lr){3-7}\\cmidrule(l){8-12}\n'
        'Inputs & Task & Trees & Ridge & \\multicolumn{3}{c}{Difference} & $R^2$ trees & $R^2$ ridge & '
        '\\multicolumn{3}{c}{Difference} \\\\\n\\midrule\n' + '\n'.join(lrows) +
        '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- Rovinka (SK0076A) 2025 LOSO station-year mean: native E+P vs programme-matched (Discussion 4.1)
    frm = pd.read_csv(INPUTS / 'train_ready_permissive_500m.csv', parse_dates=['datum'])
    frm['pm_ok'] = frm[['pm10_mean', 'pm25_mean']].notna().any(axis=1)
    frm['no2_ok'] = frm.no2_mean.notna()
    rv = p[(p.protocol == 'loso') & (p.eoi == 'SK0076A') & (p.datum.dt.year == 2025)]
    rv = rv.pivot_table(index='datum', columns='arm', values='prediction').join(
        frm[frm.eoi == 'SK0076A'].set_index('datum')[['pm_ok', 'no2_ok']])
    matched = np.where(rv.pm_ok & rv.no2_ok, rv.E_P, np.where(rv.pm_ok, rv.E_PM, np.where(rv.no2_ok, rv.E_NO2, rv.E)))
    num['RovinkaNative'] = f'{rv.E_P.mean():.2f}'
    num['RovinkaMatched'] = f'{matched.mean():.2f}'
    # incomplete-pollutant days: share from SK0006R and SK0076A (Discussion 4.4)
    prim = frm[frm.datum.dt.year.isin([2024, 2025]) & (frm.eoi != IND)]
    incm = ~prim[['pm10_mean', 'pm25_mean', 'no2_mean']].notna().all(axis=1)
    assert int(incm.sum()) == int(num['IncompleteLosoNonN'])
    num['IncompleteTwoStations'] = str(int((incm & prim.eoi.isin(['SK0006R', 'SK0076A'])).sum()))
    # sampling intervals and dispersed-fold coverage (Methods 2.5, Supplementary S5), all 21 stations, 2024-2025
    smp = frm[frm.datum.dt.year.isin([2024, 2025])].sort_values(['eoi', 'datum'])
    gap_d = smp.groupby('eoi').datum.diff().dt.days.dropna()
    for k, lab in ((1, 'One'), (2, 'Two'), (3, 'Three')):
        num[f'Interval{lab}Pct'] = f'{100 * gap_d.eq(k).mean():.0f}'
    num['IntervalLongerPct'] = f'{100 * gap_d.gt(3).mean():.0f}'
    fold_ = (smp.datum - frm.datum.min()).dt.days % 9
    nf = smp.assign(f=fold_, y=smp.datum.dt.year).groupby(['eoi', 'y']).f.nunique()
    num['AllFoldsStationYears'] = str(int(nf.eq(9).sum()))
    num['StationYearsTotal'] = str(len(nf))
    num['MinFoldsStationYear'] = str(int(nf.min()))

    (OUT / 'numbers.tex').write_text('% Generated by APR/analysis/manuscript_v3/assets_v3.py\n' +
                                     ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
    (OUT / 'numbers.json').write_text(json.dumps(num, indent=1) + '\n')
    print(len(num), 'macros written')


if __name__ == '__main__':
    main()
