"""Matched-contrast figure and supplementary table for manuscript_v2 (reporting only).

Reads saved summary metrics and paired intervals (r02, r03_review); no fits, no
new resampling. Writes to APR/manuscript_v2/generated_v2/:
  information_benefits.pdf/.png  Fig. 3, rows ordered by comparison (i)-(iii)
  contrast_table.tex             Supplementary table: both RMSEs + paired interval
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

APR = Path(__file__).resolve().parents[2]
R2 = APR / 'results/clean_rebuild/r02/scores'
R3 = APR / 'results/clean_rebuild/r03_review/scores'
OUT = APR / 'manuscript_v2/generated_v2'
TAG = '|log1p|uniform|permissive|'
PERIOD, SUPPORT = 'primary_2024_2025', 'all_expected_targets'
SCOPES = [('all_network', 'All stations'), ('nonindustrial', 'Nonindustrial')]

# (label, task, candidate, comparator, source): candidate adds the tested inputs.
CONTRASTS = [
    ('(i) Richer bundle beyond PM', '30-day', 'FULL-ID', 'PM-XGB',
     ('r02', 'FULL_ID' + TAG + '2km_exact', 'PM_XGB' + TAG + '2km_exact', 'block30')),
    ('(ii) Environment beyond pollutants', 'LOSO', 'G+P', 'P+calendar',
     ('r03', 'G_PLUS_P', 'P_CAL', 'loso')),
    ('(iii) Pollutants beyond environment', '30-day', 'G+P', 'G',
     ('r02', 'G_PLUS_P' + TAG + '500m', 'G' + TAG + '500m', 'block30')),
    ('(iii) Pollutants beyond environment', 'LOSO', 'G+P', 'G',
     ('r03', 'G_PLUS_P', 'G', 'loso')),
]


def mn(x, d=3):
    """Format with a true minus sign (\\mn macro defined in the manuscript preamble)."""
    return f'{x:.{d}f}'.replace('-', '\\mn{}')


def one(df, **kw):
    x = df
    for k, v in kw.items():
        x = x[x[k] == v]
    assert len(x) == 1, (kw, len(x))
    return x.iloc[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src = {'r02': (pd.read_csv(R2 / 'summary_metrics.csv'), pd.read_csv(R2 / 'paired_uncertainty.csv')),
           'r03': (pd.read_csv(R3 / 'summary_metrics.csv'), pd.read_csv(R3 / 'paired_uncertainty.csv'))}
    rows, plot = [], []
    quantities = [('daily', SUPPORT, 'Daily'),
                  ('sampled_date_station_year_mean', 'all_sampled_dates', 'Station-year mean')]
    for est, support, qname in quantities:
        for label, task, cand, comp, (s, a, b, prot) in CONTRASTS:
            if est != 'daily' and prot != 'loso':
                continue  # station-year-mean contrasts are reported for the LOSO comparisons
            S, P = src[s]
            for scope, sname in SCOPES:
                common = dict(protocol=prot, estimand=est, support=support, period=PERIOD, scope=scope)
                ra = one(S, method=a, **common).RMSE
                rb = one(S, method=b, **common).RMSE
                q = one(P, method_a=a, method_b=b, metric='RMSE', **common)
                assert abs((ra - rb) - q.estimate) < 1e-9
                rows.append([f"{label.split(')')[0]}) {cand} vs {comp}", task, qname, sname,
                             f'{rb:.3f}', f'{ra:.3f}', f'{mn(q.estimate)} [{mn(q.ci_low)}, {mn(q.ci_high)}]'])
                if est == 'daily':
                    plot.append((f'{label}: {task}', scope, -q.estimate, -q.ci_high, -q.ci_low))

    body = '\n'.join(' & '.join(r) + r' \\' for r in rows)
    (OUT / 'contrast_table.tex').write_text(
        '\\begin{table}[htbp]\n\\centering\\scriptsize\\setlength{\\tabcolsep}{3pt}\n'
        '\\caption{Matched RMSE contrasts, 2024--2025: daily contrasts shown in Fig.~3 of the main text and station-year-mean contrasts for the LOSO comparisons. '
        'Comparisons are numbered as in the main text; each candidate adds the tested inputs to its '
        'comparator. Differences are candidate minus comparator with paired 95\\% intervals; negative '
        'values favour the candidate. Units: ng~m$^{-3}$.}\n\\label{tab:contrasts}\n'
        '\\begin{tabular}{llllrrl}\n\\toprule\n'
        'Comparison & Task & Quantity & Population & \\shortstack{Comparator\\\\RMSE} & \\shortstack{Candidate\\\\RMSE} & Difference [95\\% interval] \\\\\n'
        '\\midrule\n' + body + '\n\\bottomrule\n\\end{tabular}\n\\end{table}\n')

    labels = list(dict.fromkeys(p[0] for p in plot))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for scope, color, shift, display in [('all_network', '#555555', -.12, 'All 21 stations'),
                                         ('nonindustrial', '#0072B2', .12, 'Nonindustrial 20')]:
        for y, lab in enumerate(labels):
            _, _, val, lo, hi = next(p for p in plot if p[0] == lab and p[1] == scope)
            ax.errorbar(val, y + shift, xerr=[[val - lo], [hi - val]], fmt='o', capsize=3, color=color,
                        label=display if y == 0 else None)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color='grey', ls=':', lw=1)
    ax.set_xlabel('Reduction in daily RMSE (ng m$^{-3}$), 95% interval')
    ax.legend(loc='upper right', fontsize=9, frameon=False)
    for s_ in ('top', 'right'):
        ax.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / 'information_benefits.pdf')
    fig.savefig(OUT / 'information_benefits.png', dpi=200)
    plt.close(fig)
    for r in rows:
        print(r)


if __name__ == '__main__':
    main()
