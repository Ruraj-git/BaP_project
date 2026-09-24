"""Fig. 1 for manuscript_v2: station network coloured by area type, shaped by source type.

Reporting only (no fits). Rendering follows review_reading_assets.relief()
(same 1:32 DEM overview, common DEM/shade mask, label offsets, scale bar),
changing only the station symbology:
  marker shape  = source type (circle background, square traffic, triangle industrial)
  marker colour = area type, one-hue blue ordinal ramp (urban dark, suburban
                  medium, rural light; dataviz ordinal check passes on white),
                  with a dark outline so light fills stay visible on grey relief.
Writes APR/manuscript_v2/generated_v2/network_typology.{pdf,png} and a count check.
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource
from matplotlib.lines import Line2D
from scipy.ndimage import distance_transform_edt, binary_fill_holes

ROOT = Path(__file__).resolve().parents[3]
V2 = ROOT / 'APR/manuscript_v2'
OUT = V2 / 'generated_v2'
MAPPED = V2 / 'generated_merge/mapped_stations.csv'
DEM = ROOT / 'data/DMR3.5/dmr3_5_10.tif'

AREA = {'U': ('urban', '#104281'), 'S': ('suburban', '#2a78d6'), 'R': ('rural', '#86b6ef')}
SOURCE = {'B': ('background', 'o'), 'T': ('traffic', 's'), 'I': ('industrial', '^')}
ORDER = ['RB', 'SB', 'UB', 'UT', 'SI']
EXPECTED = {'RB': 2, 'SB': 6, 'UB': 5, 'UT': 7, 'SI': 1}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    st = pd.read_csv(MAPPED)
    st['cls'] = st.typ_oblasti + st.typ_zdroja
    counts = st.cls.value_counts().to_dict()
    assert counts == EXPECTED, counts

    with rasterio.open(DEM) as r:
        assert 32 in r.overviews(1)
        dem = r.read(1, out_shape=(r.height // 32, r.width // 32), masked=True).astype(float)
        ext = [r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top]
        dx = (r.bounds.right - r.bounds.left) / dem.shape[1]
        dy = (r.bounds.top - r.bounds.bottom) / dem.shape[0]
        crs = r.crs
    mask = np.ma.getmaskarray(dem) | ~np.isfinite(dem.data) | (dem.data < -100) | (dem.data > 3000)
    nearest = distance_transform_edt(mask, return_distances=False, return_indices=True)
    shade = LightSource(azdeg=315, altdeg=40).hillshade(dem.data[tuple(nearest)], vert_exag=4, dx=dx, dy=dy)
    holes = binary_fill_holes(~mask) & mask
    pts = gpd.GeoDataFrame(st, geometry=gpd.points_from_xy(st.lon, st.lat), crs=4326).to_crs(crs)

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.imshow(np.ma.array(dem.data, mask=mask), extent=ext, cmap='Greys',
              vmin=-600, vmax=2600, interpolation='nearest', zorder=0)
    ax.imshow(np.ma.array(shade, mask=mask), extent=ext, cmap='gray',
              alpha=.35, interpolation='nearest', zorder=1)
    if holes.any():
        ax.imshow(np.ma.array(np.ones(mask.shape), mask=~holes), extent=ext,
                  cmap='Greys', vmin=0, vmax=4, interpolation='nearest', zorder=2)
    handles = []
    for cls in ORDER:
        area, color = AREA[cls[0]]
        source, marker = SOURCE[cls[1]]
        s = pts[pts.cls.eq(cls)]
        ax.scatter(s.geometry.x, s.geometry.y, s=70, marker=marker, color=color,
                   edgecolors='#1a1a1a', linewidth=.8, zorder=4)
        handles.append(Line2D([], [], ls='', marker=marker, ms=9.5, mfc=color, mec='#1a1a1a', mew=.8,
                              label=f'{cls} (n = {len(s)})'))
    offsets = {'SK0002A': (25, -5), 'SK0048A': (-25, 17), 'SK0061A': (8, 26),
               'SK0076A': (25, -24), 'SK0214A': (15, -15), 'SK0263A': (-20, 13),
               'SK0008A': (-10, -16), 'SK0078A': (-12, -14), 'SK0071A': (-10, 12)}
    for row in pts.itertuples():
        ax.annotate(str(row.map_number), (row.geometry.x, row.geometry.y),
                    xytext=offsets.get(row.eoi, (6, 6)), textcoords='offset points',
                    ha='center', fontsize=10, zorder=6,
                    arrowprops=dict(arrowstyle='-', color='#555555', lw=.5),
                    bbox=dict(boxstyle='round,pad=.12', fc='white', ec='none', alpha=.85))
    cities = gpd.GeoSeries(gpd.points_from_xy([17.107, 21.258], [48.148, 48.716]), crs=4326).to_crs(crs)
    for name, xy, offset in zip(['Bratislava', 'Košice'], cities, [(-45, -40), (9, 0)]):
        ax.annotate(name, (xy.x, xy.y), xytext=offset, textcoords='offset points',
                    fontsize=10.5, fontstyle='italic', zorder=5)
    x0, y0 = ext[0] + 300000, ext[2] + 18000
    ax.plot([x0, x0 + 50000], [y0, y0], color='#222222', lw=2)
    ax.text(x0 + 25000, y0 + 5000, '50 km', ha='center', fontsize=10.5)
    ax.set_xlim(ext[0] - 12000, ext[1] + 10000); ax.set_ylim(ext[2] - 14000, ext[3] + 5000)
    ax.set_aspect('equal'); ax.axis('off')
    ax.legend(handles=handles, loc='upper left', fontsize=11, frameon=False, borderaxespad=0,
              title='Area and source type', title_fontsize=11, alignment='left', handletextpad=.3)
    fig.tight_layout()
    fig.savefig(OUT / 'network_typology.pdf', dpi=250)
    fig.savefig(OUT / 'network_typology.png', dpi=160)
    plt.close(fig)
    (OUT / 'network_typology.json').write_text(json.dumps(
        {'counts': counts, 'area_colours': {k: v[1] for k, v in AREA.items()},
         'source_markers': {k: v[1] for k, v in SOURCE.items()}}, indent=2) + '\n')
    print(counts)


if __name__ == '__main__':
    main()
