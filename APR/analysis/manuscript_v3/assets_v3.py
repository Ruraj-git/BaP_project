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
    num['IndLosoMin'], num['IndLosoMax'] = f'{ind[lo_arm]:.2f}', f'{ind[hi_arm]:.2f}'
    num['IndLosoEnv'] = f"{ind['E']:.2f}"

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
            c = [f'\\textbf{{{x}}}' for x in c]
        rows.append(' & '.join(c) + ' \\\\')
    (OUT / 'ladder_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n'
        '\\caption{Input-ladder results, 2024--2025, for daily concentrations and (LOSO) station-year means over sampled dates. '
        'Non.: 20 nonindustrial stations (4852 daily values; 40 station-years); All: 21 stations (5089). '
        'RMSE in ng~m$^{-3}$; $R^2$ is the coefficient of determination. Models with station indicators (I) are not '
        'applicable to withheld stations. E+P, the principal model, is in bold. Paired contrasts are in '
        'Table~\\ref{tab:maincontrasts} and Fig.~\\ref{fig:benefits}.}\n\\label{tab:ladder}\n'
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
                c.append(iv(pair(P, lab, prot, scope, est)))
            rows.append(' & '.join(c) + ' \\\\')
    (OUT / 'contrast_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Input-group RMSE contrasts, 2024--2025, candidate minus comparator with paired 95\\% intervals '
        '(ng~m$^{-3}$); negative values favour the candidate. Daily 30-day intervals resample stations and blocks; '
        'all others resample stations. Model RMSEs are in main-text Table~5. Static support 500~m.}\n'
        '\\label{tab:contrasts}\n\\begin{tabular}{@{}llccc@{}}\n\\toprule\n'
        'Contrast & Task & Daily, nonindustrial & Daily, all stations & Station-year mean, nonind. \\\\\n'
        '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- main-text contrast table (nonindustrial; bold = supported)
    def cell(lab, prot, est):
        x = P[(P.label == lab) & (P.protocol == prot) & (P.scope == 'nonindustrial') & (P.estimand == est)]
        if x.empty:
            return '--'
        r = x.iloc[0]
        e = mn(r.estimate)
        if r.ci_low > 0 or r.ci_high < 0:
            e = f'\\textbf{{{e}}}'
        return f'{e} [{mn(r.ci_low)}, {mn(r.ci_high)}]'
    main_rows = [('+E (E+P vs P)', '+E'), ('+P (E+P vs E)', '+P'), ('+NO$_2$ (P vs PM)', '+NO2'),
                 ('+NO$_2$ given E (E+P vs E+PM)', '+NO2 given E'), ('+PM given E (E+P vs E+NO$_2$)', '+PM given E'),
                 ('+M (E+P+M vs E+P)', '+M'), ('+I (E+P+M+I vs E+P+M)', '+I'), ('+I at PM level (PM+I vs PM)', '+I (PM level)'),
                 ('Bundle (E+P+M+I vs PM+I)', 'Bundle')]
    rows = [' & '.join([d, cell(l, 'block30', 'daily'), cell(l, 'loso', 'daily'), cell(l, 'loso', 'station_year_mean')]) + ' \\\\'
            for d, l in main_rows]
    (OUT / 'main_contrast_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Input-group contrasts at the 20 nonindustrial stations, 2024--2025: RMSE difference, candidate minus '
        'comparator, with paired 95\\% intervals (ng~m$^{-3}$). Negative values favour the candidate; bold marks '
        'supported differences (interval excludes zero). Models with station indicators are not applicable to withheld '
        'stations. Full-network contrasts are given in Supplementary Section S4.}\n\\label{tab:maincontrasts}\n'
        '\\begin{tabular}{@{}lccc@{}}\n\\toprule\n'
        'Contrast & Task 1: daily & Task 2: daily & Task 2: station-year mean \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- supplementary sensitivity table
    rows = []
    for code, lab, disp in SENS:
        c = [disp]
        for prot in ('block30', 'loso'):
            c.append(iv(pair(P, lab, prot, 'nonindustrial', 'daily')))
        c.append(iv(pair(P, lab, 'loso', 'nonindustrial', 'station_year_mean')))
        rows.append(' & '.join(c) + ' \\\\')
        if lab == 'Weights':
            for scope, lev in (('nonindustrial_higher', '\\quad station-years $>1$~ng~m$^{-3}$'),
                               ('nonindustrial_lower', '\\quad station-years $\\leq1$~ng~m$^{-3}$')):
                rows.append(' & '.join([lev] + [iv(pair(P, lab, p_, scope, 'daily')) for p_ in ('block30', 'loso')]
                                       + [iv(pair(P, lab, 'loso', scope, 'station_year_mean'))]) + ' \\\\')
        if lab in ('Masking', 'Programme-matched'):
            rows.append(' & '.join(['\\quad incomplete-pollutant days'] +
                                   [iv(pair(P, lab, p_, 'nonindustrial', 'daily_incomplete_pollutant_days')) for p_ in ('block30', 'loso')]
                                   + ['--']) + ' \\\\')
    rows.append('\\midrule\n\\multicolumn{4}{@{}l}{\\emph{Temporal stress tests, daily, E+P minus E}} \\\\')
    for prot_, disp_ in (('year_out', 'Year out'), ('season_year_out', 'Season-year out'),
                         ('calendar_season_out', 'Calendar season out')):
        e_, ep_ = summ(S, MID('E'), prot_, 'nonindustrial'), summ(S, MID('E_P'), prot_, 'nonindustrial')
        rows.append(f'{disp_} ($n={int(ep_.n)}$) & \\multicolumn{{3}}{{l}}{{{iv(pair(P, "Stress", prot_, "nonindustrial"))} '
                    f'(E {e_.RMSE:.3f}, E+P {ep_.RMSE:.3f})}} \\\\')
    (OUT / 'sensitivity_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Sensitivity contrasts for the principal model E+P at nonindustrial stations, variant minus '
        'reference with paired 95\\% intervals (ng~m$^{-3}$); positive values mean the variant is worse. '
        'Reference: uniform weights, log target, all available pollutant hours, native missing-value handling, '
        '500~m static cells. Programme-matched predictions use E+P, E+PM, E+NO$_2$ or E according to the '
        'pollutant groups available on each target day, and equal E+P on complete-pollutant days. Stress tests withhold each primary year, each complete '
        'season-year (incomplete boundary seasons excluded) or each calendar season across all years; their intervals '
        'do not quantify variability over other years or seasons.}\n'
        '\\label{tab:sensitivity}\n\\begin{tabular}{@{}lccc@{}}\n\\toprule\n'
        'Variant & 30-day, daily & LOSO, daily & LOSO, station-year mean \\\\\n\\midrule\n'
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
    order = [MID('E_P'), 'PM_RIDGE', 'HARMONIC', 'INTERP_LINEAR', 'INTERP_LOGLINEAR']
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
    rows = []
    for m in order:
        c = [disp[m]]
        for prot_, est, sup, d in (('block30', 'daily', 'common_bracketable', 3),
                                   ('dispersed_mod9', 'daily', 'common_bracketable', 3),
                                   ('dispersed_mod9', 'reconstructed_mean', 'common_bracketable_replacements', 4)):
            for scope in ('all_network', 'nonindustrial'):
                c.append(f"{one(S, method=m, protocol=prot_, estimand=est, support=sup, scope=scope).RMSE:.{d}f}")
        rows.append(' & '.join(c) + ' \\\\')
    nd = one(S, method=MID('E_P'), protocol='dispersed_mod9', estimand='daily', support='common_bracketable', scope='all_network').n
    nn = one(S, method=MID('E_P'), protocol='dispersed_mod9', estimand='daily', support='common_bracketable', scope='nonindustrial').n
    bd = one(S, method=MID('E_P'), protocol='block30', estimand='daily', support='common_bracketable', scope='all_network').n
    bn = one(S, method=MID('E_P'), protocol='block30', estimand='daily', support='common_bracketable', scope='nonindustrial').n
    (OUT / 'gap_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\caption{E+P and simple comparators '
        'on identical bracketable targets, 2024--2025 (RMSE, ng~m$^{-3}$). 30-day gaps: '
        f'{int(bd)} daily values ({int(bn)} nonindustrial); dispersed gaps: {int(nd)} ({int(nn)} nonindustrial). '
        'Reconstructed means use 378 station-year/fold reconstructions (360 nonindustrial), each replacing only '
        'that fold\'s bracketable withheld dates. Model comparators are fitted inside each fold.}\n\\label{tab:gap}\n'
        '\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n'
        ' & \\multicolumn{2}{c}{30-day gaps, daily} & \\multicolumn{2}{c}{Dispersed, daily} & '
        '\\multicolumn{2}{c}{Dispersed, reconstructed mean} \\\\\n'
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\nMethod & All & Nonind. & All & Nonind. & All & Nonind. \\\\\n'
        '\\midrule\n' + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')
    rows = []
    for scope, sd in (('all_network', 'All stations'), ('nonindustrial', 'Nonindustrial')):
        for m in order[1:]:
            c = [sd, BASE[m]]
            for est in ('daily', 'reconstructed_mean'):
                r = pair(P, 'Gap vs simple', 'dispersed_mod9', scope, est, method_b=m)
                c.append(iv(r, 4))
            rows.append(' & '.join(c) + ' \\\\')
    (OUT / 'dispersed_intervals.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n\\caption{Complete '
        'dispersed-gap RMSE contrasts: E+P minus each comparator, with paired station-cluster 95\\% intervals '
        f'(ng~m$^{{-3}}$). Negative differences favour E+P. Daily scores use {int(nd)} dates ({int(nn)} nonindustrial); '
        'reconstructed means use 378 station-year/fold cases (360 nonindustrial). Targets are identical within each '
        'contrast.}\n\\label{tab:dispersedintervals}\n\\begin{tabular}{llll}\n\\toprule\n'
        'Population & Comparator & Daily & Reconstructed mean \\\\\n\\midrule\n'
        + '\n'.join(rows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    # ---- reconstructed-mean error as a share of the nonindustrial mean (Results 3.4)
    fr_ = pd.read_csv(INPUTS / 'train_ready_permissive_500m.csv', parse_dates=['datum'])
    non_mean = fr_[fr_.datum.dt.year.isin([2024, 2025]) & (fr_.eoi != IND)].bap.mean()
    for m, k in ((MID('E_P'), 'RecoveryPctEP'), ('INTERP_LINEAR', 'RecoveryPctLin')):
        r = one(S, method=m, protocol='dispersed_mod9', estimand='reconstructed_mean',
                support='common_bracketable_replacements', scope='nonindustrial').RMSE
        num[k] = f'{100 * r / non_mean:.0f}'

    # ---- Fig. 3: input-group contrasts
    fig_rows = [('+NO$_2$ (P vs PM)', '+NO2'), ('+E (E+P vs P)', '+E'), ('+P (E+P vs E)', '+P'),
                ('+M (E+P+M vs E+P)', '+M'), ('+I (E+P+M+I vs E+P+M)', '+I'), ('Bundle (E+P+M+I vs PM+I)', 'Bundle')]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, prot in zip(axes, ('block30', 'loso')):
        for y, (disp_, lab) in enumerate(fig_rows):
            for scope, color, shift, name in (('all_network', '#555555', -.13, 'All 21 stations'),
                                              ('nonindustrial', '#0072B2', .13, 'Nonindustrial 20')):
                x = P[(P.label == lab) & (P.protocol == prot) & (P.scope == scope) & (P.estimand == 'daily')]
                if x.empty:
                    continue
                r = x.iloc[0]
                v, lo, hi = -r.estimate, -r.ci_high, -r.ci_low
                ax.errorbar(v, y + shift, xerr=[[v - lo], [hi - v]], fmt='o', ms=4.5, capsize=2.5, color=color,
                            label=name if y == 1 else None)
        ax.axvline(0, color='grey', ls=':', lw=1)
        ax.set_title('30-day gaps' if prot == 'block30' else 'Withheld station (LOSO)', fontsize=10)
        ax.set_xlabel('Reduction in daily RMSE (ng m$^{-3}$)')
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
    axes[0].set_yticks(range(len(fig_rows)), [r[0] for r in fig_rows])
    axes[0].invert_yaxis()
    axes[1].legend(loc='lower right', fontsize=8.5, frameon=False)
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
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True, sharey=True)
    for year, ax in zip((2024, 2025), axes):
        for arm, color, marker, off in (('E', COL['E'], '^', -.25), ('P', COL['P'], 's', -.08),
                                        ('E_P', COL['E+P'], 'o', .10), ('Observed', COL['Observed'], 'D', .25)):
            x = obs if arm == 'Observed' else ann[ann.method == MID(arm)]
            x = x[x.year == year].set_index('eoi').loc[order_st]
            ax.scatter(np.arange(len(order_st)) + off, x.observed if arm == 'Observed' else x.prediction, c=color,
                       marker=marker, s=30, label=NAME.get(arm, arm), zorder=3)
        ax.set_title(str(year), loc='left')
        ax.set_ylabel('Station-year mean (ng m$^{-3}$)')
        k = order_st.index(IND)
        ax.axvspan(k - .45, k + .45, color='#eeeeee', zorder=0)
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(ncol=4, loc='upper left', fontsize=10)
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
        li = lin[(lin.eoi == st_) & lin.datum.between(b0, b1)].dropna(subset=['prediction']).sort_values('datum')
        ax.plot(li.datum, li.prediction, color='#E69F00', lw=1.2, ls='--', zorder=4, label='Linear interpolation')
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
                e = mn(r.estimate)
                if r.ci_low > 0 or r.ci_high < 0:
                    e = f'\\textbf{{{e}}}'
                cells.append(f'{e} [{mn(r.ci_low)}, {mn(r.ci_high)}]')
        drows.append(' & '.join(cells) + ' \\\\')
    (OUT / 'decomp_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Drop-one decomposition of the environmental group at the 20 nonindustrial stations: E+P with one '
        'environmental subgroup removed minus E+P, RMSE difference with paired 95\\% intervals (ng~m$^{-3}$). Positive '
        'values mean that removing the subgroup increased error; bold marks supported differences. Correlated subgroups can '
        'substitute for one another, so a small effect does not show that a subgroup is uninformative.}\n'
        '\\label{tab:decomp}\n\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n'
        ' & \\multicolumn{2}{c}{30-day gaps} & \\multicolumn{2}{c}{Withheld station (LOSO)} \\\\\n'
        '\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\nSubgroup removed & Daily & Station-year mean & Daily & Station-year mean \\\\\n'
        '\\midrule\n' + '\n'.join(drows) + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

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

    (OUT / 'numbers.tex').write_text('% Generated by APR/analysis/manuscript_v3/assets_v3.py\n' +
                                     ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in num.items()))
    (OUT / 'numbers.json').write_text(json.dumps(num, indent=1) + '\n')
    print(len(num), 'macros written')


if __name__ == '__main__':
    main()
