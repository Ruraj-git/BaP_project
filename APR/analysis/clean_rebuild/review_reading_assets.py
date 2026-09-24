"""Small reporting checks and displays for the September reading corrections.

No fits or resampling. DEM reads use its stored 1:32 overview, never full resolution.
Called by generate_manuscript_merge.py after its standard assets have been rendered.
"""
import hashlib
import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt, binary_fill_holes

ROOT = Path(__file__).resolve().parents[3]
R2 = ROOT / 'APR/results/clean_rebuild/r02'


def availability(out, table):
    source = R2 / 'inputs/train_ready_permissive_500m.csv'
    data = pd.read_csv(source)
    data = data[data.datum.str[:4].isin(['2024', '2025'])].copy()
    assert len(data) == 5089 and not data.duplicated(['eoi', 'datum']).any()
    preds = pd.read_csv(ROOT / 'APR/results/clean_rebuild/r03_review/scores/loso_primary_predictions.csv')
    matched = data.merge(preds[preds.arm.eq('G_PLUS_P')][['eoi', 'datum', 'observed']],
                         on=['eoi', 'datum'], validate='one_to_one')
    assert len(matched) == len(data) and np.array_equal(matched.bap, matched.observed)
    features = [p + '_mean' + suffix for p in ['pm10', 'pm25', 'no2'] for suffix in ['', '_lag1', '_3d']]
    detail = data[['eoi', 'datum', *features]].copy()
    detail[features] = np.isfinite(detail[features])
    summary = detail.groupby('eoi')[features].sum().add_suffix('_n')
    summary.insert(0, 'n_scored', detail.groupby('eoi').size())
    for f in features:
        summary[f + '_pct'] = 100 * summary[f + '_n'] / summary.n_scored
    assert summary.loc['SK0006R', [f + '_n' for f in features[:6]]].eq(0).all()
    assert summary.loc['SK0018A', [f + '_n' for f in features[6:]]].eq(0).all()
    summary.to_csv(out / 'pollutant_availability.csv')
    detail.to_csv(out / 'pollutant_availability_dates.csv', index=False)
    rows = [[st, int(r.n_scored), *[f'{r[f + "_mean_pct"]:.1f}' for f in ['pm10', 'pm25', 'no2']]]
            for st, r in summary.iterrows()]
    table('pollutant_availability',
          'Availability of same-day pollutant means on scored B[a]P dates, 2024--2025 '
          '(5089 observations). Percentages use all scored dates at each station as '
          'the denominator. Availability means a finite daily predictor, not adequate '
          'hourly completeness; the primary input has no minimum-hour requirement.',
          'tab:availability', ['Station', 'Scored dates', r'PM$_{10}$ (\%)',
          r'PM$_{2.5}$ (\%)', r'NO$_2$ (\%)'], rows, 'lrrrr')
    # Verify the same availability contract for the 2 km PM benchmark.
    other = pd.read_csv(R2 / 'inputs/train_ready_permissive_2km_exact.csv')
    other = other[other.datum.str[:4].isin(['2024', '2025'])]
    joined = data.merge(other[['eoi', 'datum', *features]], on=['eoi', 'datum'],
                        validate='one_to_one', suffixes=('_a', '_b'))
    assert len(joined) == len(data)
    for f in features:
        assert np.allclose(joined[f+'_a'], joined[f+'_b'], equal_nan=True)
    # Compact main-text station key, outcome context and actual input availability.
    names = {m.group(1): m.group(2).strip() for m in re.finditer(
        r'\d+ & (SK\w+) & ([^&]+) & [BTI]', (out/'station_key.tex').read_text())}
    mapped = pd.read_csv(out/'mapped_stations.csv').sort_values('map_number')
    observed = data.groupby('eoi').bap.mean()
    overview = mapped[['map_number','eoi','typ_zdroja']].set_index('eoi').join(summary)
    overview['name'] = [names[e] for e in overview.index]
    overview['observed_mean'] = observed
    assert len(overview)==21 and overview.n_scored.sum()==5089
    overview.to_csv(out/'station_overview.csv')
    rows = [[int(r.map_number), e, r['name'], r.typ_zdroja, int(r.n_scored),
             f'{r.observed_mean:.2f}', *[f'{r[f+"_mean_pct"]:.1f}' for f in ['pm10','pm25','no2']]]
            for e,r in overview.iterrows()]
    table('station_overview',
          'Station key and scored-date context, 2024--2025. Numbers identify '
          'sites in Fig.~\\ref{fig:network}. Classes: background (B), traffic (T), '
          'industrial (I). Mean is observed B[a]P over sampled dates (ng~m$^{-3}$); '
          'the final three columns are percentages of scored dates with a finite '
          'same-day pollutant mean, not percentages satisfying an hourly-completeness threshold.',
          'tab:stationoverview', ['No.','Code','Station','Class','$n$','Mean',
          r'PM$_{10}$',r'PM$_{2.5}$',r'NO$_2$'], rows, r'rlL{0.24\textwidth}crrrrr')
    path=out/'station_overview.tex'
    path.write_text(path.read_text().replace(r'\centering\small',
        r'\centering\footnotesize\setlength{\tabcolsep}{4pt}'))
    return source


def dispersed_intervals(out, table):
    source=ROOT/'APR/results/clean_rebuild/r03_review/scores/paired_uncertainty.csv'
    p=pd.read_csv(source)
    methods=[('PM_RIDGE','PM ridge'),('HARMONIC','Harmonics'),
             ('INTERP_LINEAR','Linear interpolation'),('INTERP_LOGLINEAR','Log-linear interpolation')]
    p=p[p.method_a.eq('G_PLUS_P') & p.protocol.eq('dispersed_mod9') &
        p.metric.eq('RMSE') & p.method_b.isin([m for m,_ in methods])].copy()
    assert len(p)==16 and p.direction.eq('a_minus_b').all()
    assert p.period.eq('primary_2024_2025').all() and p.resampling.eq('station').all()
    assert p[p.scope.eq('nonindustrial')].ci_high.lt(0).all()
    unresolved=p[(p.ci_low<=0)&(p.ci_high>=0)]
    assert len(unresolved)==1 and unresolved.iloc[0].method_b=='HARMONIC'
    assert unresolved.iloc[0].scope=='all_network' and unresolved.iloc[0].estimand=='foldwise_sampled_mean_recovery'
    p.to_csv(out/'dispersed_intervals.csv',index=False)
    rows=[]
    for scope,label in [('all_network','All stations'),('nonindustrial','Nonindustrial')]:
        for method,name in methods:
            row=[label,name]
            for estimand in ['daily','foldwise_sampled_mean_recovery']:
                x=p[p.scope.eq(scope)&p.method_b.eq(method)&p.estimand.eq(estimand)]
                assert len(x)==1
                q=x.iloc[0]
                row.append(f'{q.estimate:.4f} [{q.ci_low:.4f}, {q.ci_high:.4f}]')
            rows.append(row)
    table('dispersed_intervals',
          'Complete dispersed-gap RMSE contrasts: G+P minus each comparator, '
          'with paired station-cluster 95\\% intervals (ng~m$^{-3}$). Negative '
          'differences favour G+P. Daily scores use 5066 dates (4830 nonindustrial); '
          'reconstructed means use 378 station-year/fold cases (360 nonindustrial). '
          'Targets are identical within each contrast. Only the full-network '
          'reconstructed-mean contrast with harmonics includes zero.',
          'tab:dispersedintervals', ['Population','Comparator','Daily','Reconstructed mean'], rows, 'llll')
    path=out/'dispersed_intervals.tex'
    path.write_text(path.read_text().replace(r'\centering\small',
        r'\centering\footnotesize\setlength{\tabcolsep}{4pt}'))
    return source


def relief(out):
    dempath = ROOT / 'data/DMR3.5/dmr3_5_10.tif'
    with rasterio.open(dempath) as r:
        assert 32 in r.overviews(1)
        dem = r.read(1, out_shape=(r.height // 32, r.width // 32), masked=True).astype(float)
        ext = [r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top]
        dx = (r.bounds.right-r.bounds.left)/dem.shape[1]
        dy = (r.bounds.top-r.bounds.bottom)/dem.shape[0]
        crs = r.crs
    mask = np.ma.getmaskarray(dem) | ~np.isfinite(dem.data) | (dem.data < -100) | (dem.data > 3000)
    # Fill only for gradient calculation, then restore the exact common mask.
    # This avoids NaN propagation into valid edge pixels; it invents no plotted heights.
    nearest = distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = dem.data[tuple(nearest)]
    shade = LightSource(azdeg=315, altdeg=40).hillshade(filled, vert_exag=4, dx=dx, dy=dy)
    assert np.isfinite(shade[~mask]).all()
    holes = binary_fill_holes(~mask) & mask
    mapped = pd.read_csv(out / 'mapped_stations.csv')
    pts = gpd.GeoDataFrame(mapped, geometry=gpd.points_from_xy(mapped.lon, mapped.lat), crs=4326).to_crs(crs)
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.imshow(np.ma.array(dem.data, mask=mask), extent=ext, cmap='Greys',
              vmin=-600, vmax=2600, interpolation='nearest', zorder=0)
    ax.imshow(np.ma.array(shade, mask=mask), extent=ext, cmap='gray',
              alpha=.35, interpolation='nearest', zorder=1)
    # Any true internal coverage void remains visibly distinct, not interpolated.
    if holes.any():
        ax.imshow(np.ma.array(np.ones(mask.shape), mask=~holes), extent=ext,
                  cmap='Greys', vmin=0, vmax=4, interpolation='nearest', zorder=2)
    styles = {'B': ('Background', '#0072B2', 'o'), 'T': ('Traffic', '#009E73', 's'),
              'I': ('Industrial', '#D55E00', '^')}
    for code, (label, color, marker) in styles.items():
        s = pts[pts.typ_zdroja.eq(code)]
        ax.scatter(s.geometry.x, s.geometry.y, s=46, marker=marker, color=color,
                   edgecolors='white', linewidth=.7, label=f'{label} (n={len(s)})', zorder=4)
    offsets = {'SK0002A': (25, -5), 'SK0048A': (-25, 17), 'SK0061A': (8, 26),
               'SK0076A': (25, -24), 'SK0214A': (15, -15), 'SK0263A': (-20, 13),
               'SK0008A': (-10, -16), 'SK0078A': (-12, -14), 'SK0071A': (-10, 12)}
    for row in pts.itertuples():
        ax.annotate(str(row.map_number), (row.geometry.x, row.geometry.y),
                    xytext=offsets.get(row.eoi, (6, 6)), textcoords='offset points',
                    ha='center', fontsize=9, zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#555555', lw=.5),
                    bbox=dict(boxstyle='round,pad=.12', fc='white', ec='none', alpha=.85))
    cities = gpd.GeoSeries(gpd.points_from_xy([17.107, 21.258], [48.148, 48.716]), crs=4326).to_crs(crs)
    for name, xy, offset in zip(['Bratislava', 'Košice'], cities, [(-3, -40), (9, 0)]):
        ax.annotate(name, (xy.x, xy.y), xytext=offset, textcoords='offset points',
                    fontsize=9, fontstyle='italic', zorder=5)
    x0, y0 = ext[0]+300000, ext[2]+18000
    ax.plot([x0, x0+50000], [y0, y0], color='#222222', lw=2)
    ax.text(x0+25000, y0+5000, '50 km', ha='center', fontsize=9)
    ax.set_xlim(ext[0]-12000, ext[1]+10000); ax.set_ylim(ext[2]-14000, ext[3]+5000)
    ax.set_aspect('equal'); ax.axis('off')
    ax.legend(loc='upper left', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out / 'network_relief.pdf', dpi=250)
    fig.savefig(out / 'network_relief.png', dpi=160)
    plt.close(fig)
    metadata = {'source': str(dempath.relative_to(ROOT)), 'overview_factor': 32,
                'shape': list(dem.shape), 'file_size': dempath.stat().st_size,
                'overview_sha256': hashlib.sha256(dem.data.tobytes()+mask.tobytes()).hexdigest(),
                'valid_pixels': int((~mask).sum()), 'internal_void_pixels': int(holes.sum()),
                'mask_rule': 'one common DEM/shade mask; no independent boundary or shadow',
                'void_handling': 'no plotted infilling; internal voids neutral grey'}
    (out / 'map_render_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')


def timeseries(out, daily, gp, gl):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey='row')
    for i, (station, title) in enumerate([('SK0025A', 'Jelšava (SK0025A)'),
                                       ('SK0048A', 'Bratislava, Jeséniova (SK0048A)')]):
        obs = daily[daily.eoi.eq(station)].sort_values('datum')
        for j, arms in enumerate([[(daily, 'G+P', '#0072B2', '-')],
                                 [(gl, 'G', '#E69F00', '--'), (gp, 'G+P', '#0072B2', '-')]]):
            ax = axes[i, j]
            ax.scatter(pd.to_datetime(obs.datum), obs.observed, s=12, facecolors='none',
                       edgecolors='#222222', linewidths=.6, label='Observed', zorder=4)
            for frame, label, color, ls in arms:
                a = frame[frame.eoi.eq(station)].copy()
                a['datum'] = pd.to_datetime(a.datum); a = a.sort_values('datum')
                pad = a.loc[a.datum.diff().dt.days.gt(7)].copy()
                pad['datum'] -= pd.Timedelta(days=1); pad['prediction'] = np.nan
                a = pd.concat([a, pad]).sort_values('datum')
                ax.plot(a.datum, a.prediction, label=label, color=color, ls=ls, lw=1)
            ax.set_title(title+' — '+('30-day gaps' if j==0 else 'LOSO'), fontsize=10)
            ax.grid(axis='y', alpha=.2)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.legend(loc='upper right', fontsize=8, framealpha=.9)
        axes[i, 0].set_ylabel('B[a]P (ng m$^{-3}$)')
    fig.tight_layout()
    fig.savefig(out / 'timeseries.pdf'); fig.savefig(out / 'timeseries.png', dpi=140)
    plt.close(fig)


def main(out, table, daily, gp, gl):
    source = availability(out, table)
    intervals = dispersed_intervals(out, table)
    relief(out)
    timeseries(out, daily, gp, gl)
    return [Path(__file__), source, intervals, R2 / 'inputs/train_ready_permissive_2km_exact.csv']
