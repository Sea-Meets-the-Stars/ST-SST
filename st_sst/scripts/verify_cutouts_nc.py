""" Verify the full LLC4320_v2 Gulf Stream cutout file (Implement prompt #5).

Reads the concatenated file and

1. reuses `st_sst.scripts.check_cutout_file` for the layout checks (dims, table
   variables, int64 UIDs, chunking / compression, credit) and adds: no NaN in
   ``sst`` (streamed in 1024-cutout blocks), unique UIDs, ``lat_ne > lat_sw``,
   ``lon_ne > lon_sw``, resampling ratio ``min(nj, ni) / npix >= 1``;
2. checks the time span: the distinct snapshot times equal
   ``snapshot_hours()`` for the file's stride (default 9/10 h, 1,001 snapshots);
3. tabulates and plots the UTC-hour (24 bins) and month (13 bins) histograms,
   with flatness statistics (max/min ratio, chi-square against the expectation);
4. per-snapshot yield (min/max/mean/median/std), plotted against time;
5. distributions of the per-cutout statistics, extents and ``min_depth``;
6. a 2D histogram of cutout centres (spatial coverage);
7. example cutouts: ``--n-examples`` seeded random cutouts spanning the year, and
   the extremes (highest grad_p90, lowest std, warmest / coldest mean) as
   demeaned anomalies.

Figures go to ``--outdir`` with names ``{prefix}*.png``. Prints a report and
exits non-zero if any check fails. Only the per-cutout table is loaded; ``sst``
is streamed, and only the example cutouts are read in full.

Usage (repository root)::

    conda run -n ocean14 python -m st_sst.scripts.verify_cutouts_nc
"""

import argparse
import math
import os
import sys
import time as _time

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats as sstats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402
from matplotlib.ticker import MaxNLocator             # noqa: E402

from st_sst.extract import llc_v2_cutouts as cut      # noqa: E402
from st_sst.scripts import check_cutout_file          # noqa: E402

FULL_NAME = 'LLC4320v2_GulfStream_SST_128km_64px.nc'
SST_CMAP = 'RdYlBu_r'                                 # as in the pilot figures
ANOM_CMAP = 'RdBu_r'
DIST_COLS = cut.STAT_KEYS + ('km_w', 'km_h', 'nj', 'ni', 'min_depth')
UNITS = {'mean': 'degC', 'std': 'degC', 'min': 'degC', 'max': 'degC', 'p10': 'degC',
         'p90': 'degC', 'grad_mean': 'degC/km', 'grad_p90': 'degC/km', 'grad_max': 'degC/km',
         'km_w': 'km', 'km_h': 'km', 'nj': 'cells', 'ni': 'cells', 'min_depth': 'm'}
YIELD_TOL = 0.03          # per-hour / per-month mean yield within 3% of the overall mean


def default_file() -> str:
    root = os.environ.get('OS_OGCM', '/home/xavier/Oceanography/data/OGCM/')
    return os.path.join(root, 'LLC_v2', 'cutouts', FULL_NAME)


def parser(options=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--file', default=default_file(), help='concatenated cutout file')
    p.add_argument('--outdir', default='reports/figures', help='figure directory')
    p.add_argument('--prefix', default='full_', help='figure file-name prefix')
    p.add_argument('--n-examples', type=int, default=12, help='random example cutouts')
    p.add_argument('--seed', type=int, default=2023, help='seed for the example cutouts')
    return p.parse_args() if options is None else p.parse_args(options)


# ---------------------------------------------------------------------------
# Streaming pass over sst
# ---------------------------------------------------------------------------

def stream_sst(path: str, block: int = 1024):
    """Non-finite count and global min / max of ``sst``, read *block* cutouts at a time."""
    import netCDF4
    n_bad, vmin, vmax = 0, np.inf, -np.inf
    with netCDF4.Dataset(path) as nc:
        v = nc.variables['sst']
        v.set_auto_mask(False)
        for k in range(0, v.shape[0], block):
            a = v[k:k + block]
            fin = np.isfinite(a)
            n_bad += int(a.size - fin.sum())
            if fin.any():
                vmin = min(vmin, float(a[fin].min()))
                vmax = max(vmax, float(a[fin].max()))
    return n_bad, vmin, vmax


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def flatness(counts, expected=None) -> dict:
    """max/min ratio and chi-square of *counts* against *expected* (uniform if None)."""
    counts = np.asarray(counts, dtype=float)
    if expected is None:
        expected = np.full_like(counts, counts.mean())
    expected = np.asarray(expected, float) * counts.sum() / np.sum(expected)
    chi2, p = sstats.chisquare(counts, expected)
    return {'min': counts.min(), 'max': counts.max(), 'mean': counts.mean(),
            'max/min': counts.max() / counts.min(), 'chi2': chi2, 'dof': counts.size - 1,
            'p': p}


def fmt_flat(name, f) -> str:
    return (f"  {name}: min {f['min']:.0f}, max {f['max']:.0f}, mean {f['mean']:.1f}, "
            f"max/min {f['max/min']:.3f}, chi2 {f['chi2']:.1f} (dof {f['dof']}, p {f['p']:.3g})")


def time_tables(times: pd.DatetimeIndex, per: pd.Series):
    """Per-UTC-hour and per-month tables of snapshots, cutouts and mean yield."""
    snaps = per.index
    hour = pd.DataFrame({
        'snapshots': np.bincount(snaps.hour, minlength=24),
        'cutouts': np.bincount(times.hour, minlength=24)}, index=pd.Index(range(24), name='hour'))
    hour['yield'] = hour['cutouts'] / hour['snapshots']
    months = pd.period_range('2023-01', '2024-01', freq='M')
    sm = pd.Series(1, index=snaps.to_period('M')).groupby(level=0).sum()
    cm = pd.Series(1, index=times.to_period('M')).groupby(level=0).sum()
    month = pd.DataFrame({'snapshots': sm.reindex(months, fill_value=0),
                          'cutouts': cm.reindex(months, fill_value=0)})
    month.index.name = 'month'
    month['days'] = [p.days_in_month for p in months]
    month['cutouts/day'] = month['cutouts'] / month['days']
    month['yield'] = month['cutouts'] / month['snapshots']
    return hour, month


def dist_table(tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in DIST_COLS:
        x = tab[c].values.astype(float)
        q = np.percentile(x, [0, 5, 50, 95, 100])
        rows.append({'quantity': c, 'units': UNITS[c], 'min': q[0], 'p5': q[1], 'median': q[2],
                     'p95': q[3], 'max': q[4], 'mean': x.mean()})
    return pd.DataFrame(rows).set_index('quantity')


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_time_histograms(hour, month, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.bar(hour.index, hour['cutouts'], color='#1f77b4', width=0.85)
    ax.axhline(hour['cutouts'].mean(), color='k', ls='--', lw=1, label='mean')
    ax.set_xlabel('UTC hour of snapshot')
    ax.set_ylabel('cutouts')
    ax.set_xticks(range(0, 24, 3))
    ax.set_ylim(0, hour['cutouts'].max() * 1.15)
    ax2 = ax.twinx()
    ax2.plot(hour.index, hour['snapshots'], 'o', color='#d62728', ms=4, label='snapshots')
    ax2.set_ylabel('snapshots', color='#d62728')
    ax2.set_ylim(0, hour['snapshots'].max() * 1.15)
    ax.set_title(f"UTC hour: {hour['cutouts'].min()}-{hour['cutouts'].max()} cutouts, "
                 f"{hour['snapshots'].min()}-{hour['snapshots'].max()} snapshots per hour",
                 fontsize=10)
    ax.legend(loc='lower left', fontsize=8)
    ax2.legend(loc='lower right', fontsize=8)

    ax = axes[1]
    lab = [str(p) for p in month.index]
    x = np.arange(len(month))
    ax.bar(x, month['cutouts/day'], color='#2ca02c', width=0.85)
    ax.axhline(month['cutouts/day'].mean(), color='k', ls='--', lw=1, label='mean')
    ax.set_xticks(x)
    ax.set_xticklabels(lab, rotation=60, fontsize=8)
    ax.set_ylabel('cutouts per day')
    ax.set_ylim(0, month['cutouts/day'].max() * 1.15)
    ax2 = ax.twinx()
    ax2.plot(x, month['yield'], 's', color='#9467bd', ms=4, label='mean yield / snapshot')
    ax2.set_ylabel('mean tiles per snapshot', color='#9467bd')
    ax2.set_ylim(0, month['yield'].max() * 1.15)
    ax.set_title(f"Month: {month['cutouts/day'].min():.0f}-{month['cutouts/day'].max():.0f} "
                 f"cutouts/day, yield {month['yield'].min():.1f}-{month['yield'].max():.1f}",
                 fontsize=10)
    ax.legend(loc='lower left', fontsize=8)
    ax2.legend(loc='lower right', fontsize=8)
    fig.suptitle('Full dataset: time sampling (9/10 h stride from 2023-01-01 01:00 UTC)',
                 fontsize=11)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def plot_yield(per: pd.Series, outfile):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True,
                             gridspec_kw={'width_ratios': [2.2, 1]})
    ax = axes[0]
    ax.plot(per.index, per.values, '.', ms=3, color='#1f77b4', alpha=0.6, label='snapshot')
    roll = per.rolling(37, center=True, min_periods=10).mean()      # ~2 weeks
    ax.plot(per.index, roll.values, '-', color='k', lw=1.5, label='running mean (37 snapshots)')
    ax.set_xlabel('snapshot time (UTC)')
    ax.set_ylabel('cutouts (tiles) per snapshot')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_title(f"{per.size} snapshots, {per.sum():,} cutouts", fontsize=10)
    ax = axes[1]
    bins = np.arange(per.min() - 0.5, per.max() + 1.5)
    ax.hist(per.values, bins=bins, color='#1f77b4')
    ax.axvline(per.mean(), color='k', ls='--', lw=1, label=f"mean {per.mean():.1f}")
    ax.axvline(204.0, color='#d62728', ls=':', lw=1.5, label='pilot 204.0')
    ax.set_xlabel('cutouts per snapshot')
    ax.set_ylabel('snapshots')
    ax.legend(fontsize=8)
    ax.set_title(f"{per.min()}-{per.max()}, median {per.median():.0f}, std {per.std():.1f}",
                 fontsize=10)
    fig.suptitle('Full dataset: per-snapshot yield', fontsize=11)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def plot_stat_histograms(tab, outfile):
    ncol = 5
    nrow = int(math.ceil(len(DIST_COLS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow),
                             constrained_layout=True)
    for ax, c in zip(axes.flat, DIST_COLS):
        x = tab[c].values.astype(float)
        if c in ('nj', 'ni'):
            bins = np.arange(x.min() - 0.5, x.max() + 1.5)
        elif c == 'min_depth':
            bins = np.logspace(np.log10(max(x.min(), 1)), np.log10(x.max()), 50)
            ax.set_xscale('log')
        else:
            bins = 60
        ax.hist(x, bins=bins, color='#1f77b4')
        ax.axvline(np.median(x), color='k', ls='--', lw=1)
        if c.startswith('grad'):
            ax.set_yscale('log')
        ax.set_xlabel(f"{c} ({UNITS[c]})")
        ax.set_ylabel('cutouts')
        ax.set_title(f"{c}: {x.min():.3g} - {x.max():.3g}, median {np.median(x):.3g}",
                     fontsize=9)
    for ax in list(axes.flat)[len(DIST_COLS):]:
        ax.axis('off')
    fig.suptitle(f"Full dataset: per-cutout statistics of the 64 x 64 images, realized extents "
                 f"and min depth ({len(tab):,} cutouts; dashed = median; gradients log-y)",
                 fontsize=11)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


INTERIOR = (33., 41., -65., -52.)   # open ocean, away from the coast, shelf and Bermuda


def centre_uniformity(tab, region=INTERIOR) -> dict:
    """Uniformity of cutout centres in an open-ocean *region*.

    Large scale: counts in 1 and 2 deg bins, coefficient of variation (CV) vs the
    Poisson CV. Sub-tile scale: centres per native row ``ci = i0 + ni//2`` (along
    latitude on face 10) and column ``cj = j0 + nj//2`` (along longitude), 5 cells
    trimmed at each end of the region.
    """
    r = tab[tab['lat'].between(region[0], region[1]) & tab['lon'].between(region[2], region[3])]
    out = {'region': region, 'n': len(r)}
    for b in (1.0, 2.0):
        H = np.histogram2d(r['lat'], r['lon'], [np.arange(region[0], region[1] + 1e-9, b),
                                                np.arange(region[2], region[3] + 1e-9, b)])[0]
        out[f'{b:g}deg'] = {'bins': H.size, 'cv': H.std() / H.mean(),
                            'poisson_cv': 1 / np.sqrt(H.mean()), 'max/min': H.max() / H.min()}
    for name, c in (('ci', r['i0'] + r['ni'] // 2), ('cj', r['j0'] + r['nj'] // 2)):
        h = c.value_counts().reindex(range(int(c.min()), int(c.max()) + 1), fill_value=0)
        h = h.iloc[5:-5]
        out[name] = {'hist': h, 'cells': h.size, 'zero': int((h == 0).sum()), 'mean': h.mean(),
                     'cv': h.std() / h.mean(), 'poisson_cv': 1 / np.sqrt(h.mean()),
                     'min': int(h.min()), 'max': int(h.max())}
    return out


def plot_coverage(tab, attrs, uni, outfile):
    box = [attrs[k] for k in ('box_lat_min', 'box_lat_max', 'box_lon_min', 'box_lon_max')]
    lon, lat = tab['lon'].values, tab['lat'].values
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1.25, 1]})
    reg = uni['region']
    for ax, step in ((axes[0, 0], 1.0), (axes[0, 1], 0.25)):
        lon_e = np.arange(box[2] - 0.5, box[3] + 0.5 + 1e-9, step)
        lat_e = np.arange(box[0] - 0.5, box[1] + 0.5 + 1e-9, step)
        if step == 1.0:
            Z = np.histogram2d(lon, lat, bins=[lon_e, lat_e])[0]
            Z = np.where(Z > 0, Z, np.nan)
            cmap, lab = 'viridis', 'cutout centres per 1 x 1 deg'
            u = uni['1deg']
            title = (f"Cutout centres per 1 deg bin; interior (green box) CV {u['cv']:.3f} "
                     f"(Poisson {u['poisson_cv']:.3f})")
        else:
            Z = sstats.binned_statistic_2d(lon, lat, tab['mean'].values, 'mean',
                                           bins=[lon_e, lat_e])[0]
            cmap, lab = SST_CMAP, 'mean of cutout mean SST (degC)'
            title = 'Mean cutout SST by centre, 0.25 deg bins (annual)'
        pm = ax.pcolormesh(lon_e, lat_e, Z.T, cmap=cmap, shading='flat', rasterized=True)
        ax.add_patch(Rectangle((box[2], box[0]), box[3] - box[2], box[1] - box[0],
                               fill=False, lw=1.5, ec='m', ls='--'))
        ax.add_patch(Rectangle((reg[2], reg[0]), reg[3] - reg[2], reg[1] - reg[0],
                               fill=False, lw=1.2, ec='lime'))
        ax.set_xlim(box[2] - 0.5, box[3] + 0.5)
        ax.set_ylim(box[0] - 0.5, box[1] + 0.5)
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(box[:2]))))
        ax.set_xlabel('longitude (deg E)')
        ax.set_ylabel('latitude (deg N)')
        ax.set_title(title, fontsize=10)
        fig.colorbar(pm, ax=ax, label=lab, shrink=0.85)
    for ax, key, what in ((axes[1, 0], 'ci', 'native row ci = i0 + ni//2 (along latitude; '
                                               'i increases southward)'),
                          (axes[1, 1], 'cj', 'native column cj = j0 + nj//2 (along longitude)')):
        u = uni[key]
        h = u['hist']
        ax.bar(h.index, h.values, width=1.0, color='#1f77b4')
        ax.axhline(u['mean'], color='k', ls='--', lw=1, label=f"mean {u['mean']:.0f}")
        ax.set_xlabel(what)
        ax.set_ylabel('cutout centres (interior)')
        ax.set_title(f"{key}: {u['cells']} cells, {u['zero']} with no centre, CV {u['cv']:.2f} "
                     f"(Poisson {u['poisson_cv']:.2f}), range {u['min']}-{u['max']}", fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle('Full dataset: spatial coverage (box dashed; white = no centre: land or within '
                 '~64 km of it)\nBottom: centres per native cell in the interior (green box), '
                 'showing the tile-scale modulation of the tiling', fontsize=11)
    fig.savefig(outfile, dpi=120)
    plt.close(fig)


def read_cutouts(ds, idx):
    """``sst`` of the cutouts *idx* (read one by one; only these are decompressed)."""
    return {int(k): ds['sst'].isel(cutout=int(k)).values for k in idx}


def draw_cutout(fig, ax, img, row, anomaly=False, extra=''):
    """One cutout north-up / east-right at its stored corners (pilot figure style)."""
    ext = [float(row['lon_sw']), float(row['lon_ne']), float(row['lat_sw']),
           float(row['lat_ne'])]
    if anomaly:
        a = img - img.mean()
        lim = float(np.abs(a).max())
        im = ax.imshow(a, origin='upper', extent=ext, cmap=ANOM_CMAP, vmin=-lim, vmax=lim,
                       interpolation='nearest', aspect=1 / np.cos(np.deg2rad(float(row['lat']))))
        fig.colorbar(im, ax=ax, label='SST - mean (degC)', shrink=0.8)
    else:
        im = ax.imshow(img, origin='upper', extent=ext, cmap=SST_CMAP, interpolation='nearest',
                       aspect=1 / np.cos(np.deg2rad(float(row['lat']))))
        fig.colorbar(im, ax=ax, label='SST (degC)', shrink=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlabel('longitude (deg E)')
    ax.set_ylabel('latitude (deg N)')
    ax.set_title(f"{extra}{pd.Timestamp(row['time']):%Y-%m-%d %H} UTC\n"
                 f"SW ({ext[2]:.2f}N, {ext[0]:.2f}E)  NE ({ext[3]:.2f}N, {ext[1]:.2f}E)\n"
                 f"mean {row['mean']:.2f}, std {row['std']:.2f} degC, "
                 f"grad_p90 {row['grad_p90']:.3f} degC/km", fontsize=8.5)


def plot_cutout_panel(ds, tab, picks, outfile, title, anomaly=False, labels=None, ncol=4):
    n = len(picks)
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.2 * nrow), squeeze=False,
                             constrained_layout=True)
    imgs = read_cutouts(ds, picks)
    for m, (ax, k) in enumerate(zip(axes.flat, picks)):
        extra = (labels[m] + '\n') if labels else ''
        draw_cutout(fig, ax, imgs[int(k)], tab.iloc[int(k)], anomaly=anomaly,
                    extra=f"{extra}#{k} ")
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=12)
    fig.savefig(outfile, dpi=110)
    plt.close(fig)


def pick_examples(times: pd.DatetimeIndex, n: int, rng) -> list:
    """One random cutout in each of *n* equal time intervals over the span."""
    t = times.asi8
    edges = np.linspace(t.min(), t.max() + 1, n + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        idx = np.nonzero((t >= a) & (t < b))[0]
        out.append(int(rng.choice(idx)))
    return out


def pick_extremes(tab: pd.DataFrame, n: int = 3):
    """Indices and labels of the *n* highest grad_p90, lowest std, warmest, coldest mean."""
    picks, labels = [], []
    for col, asc, lab in (('grad_p90', False, 'highest grad_p90'), ('std', True, 'lowest std'),
                          ('mean', False, 'warmest mean'), ('mean', True, 'coldest mean')):
        order = tab[col].sort_values(ascending=asc, kind='stable').index
        k = [int(i) for i in order if int(i) not in picks][:n]
        picks += k
        labels += [lab] * len(k)
    return picks, labels


# ---------------------------------------------------------------------------

def main(args) -> int:
    t_start = _time.time()
    path = args.file
    os.makedirs(args.outdir, exist_ok=True)
    fig = {k: os.path.join(args.outdir, f"{args.prefix}{k}.png")
           for k in ('time_histograms', 'yield', 'stat_histograms', 'coverage',
                     'example_cutouts', 'example_cutouts_extremes')}
    checks = {}

    print(f"== file ==\n{path}\n  size {os.path.getsize(path):,} bytes "
          f"({os.path.getsize(path) / 1e6:.1f} MB, {os.path.getsize(path) / 2**30:.2f} GiB)")

    print("\n== layout checks (st_sst.scripts.check_cutout_file) ==")
    checks['check_cutout_file layout checks'] = check_cutout_file.main(path,
                                                                      list_snapshots=False) == 0

    ds = xr.open_dataset(path)
    attrs = dict(ds.attrs)
    n, npix = ds.sizes['cutout'], ds.sizes['y']
    # Per-cutout table only (Dataset.to_dataframe would broadcast over y, x as well)
    tab = pd.DataFrame({c: ds[c].values for c in ds.data_vars if ds[c].dims == ('cutout',)})
    print(f"\n== layout ==\n  dims {dict(ds.sizes)}; sst {ds['sst'].dtype} "
          f"{ds['sst'].dims}, encoding chunks {ds['sst'].encoding.get('chunksizes')}, "
          f"zlib {ds['sst'].encoding.get('zlib')} level {ds['sst'].encoding.get('complevel')}")
    print(f"  table variables ({len(tab.columns)}): {', '.join(tab.columns)}")
    print("== global attributes ==")
    for k, v in attrs.items():
        s = str(v)
        print(f"  {k}: {s if len(s) < 110 else s[:107] + '...'}")
    checks['n_cutouts attr == count'] = int(attrs.get('n_cutouts', -1)) == n
    checks['credit names Dan Whitt'] = 'Dan Whitt' in attrs.get('credit', '')

    print("\n== invariants ==")
    t0 = _time.time()
    n_bad, vmin, vmax = stream_sst(path)
    print(f"  sst streamed in 1024-cutout blocks in {_time.time() - t0:.1f} s: "
          f"{n_bad} non-finite of {n * npix * npix:,} values; range {vmin:.3f} to {vmax:.3f} degC")
    checks['no NaN in sst (streamed)'] = n_bad == 0
    checks['UIDs unique'] = bool(tab['UID'].is_unique)
    checks['lat_ne > lat_sw'] = bool((tab['lat_ne'] > tab['lat_sw']).all())
    checks['lon_ne > lon_sw'] = bool((tab['lon_ne'] > tab['lon_sw']).all())
    ratio = np.minimum(tab['nj'], tab['ni']).values / npix
    print(f"  resampling ratio min(nj, ni)/npix {ratio.min():.3f}-{ratio.max():.3f}")
    checks['resampling ratio min(nj, ni)/npix >= 1'] = bool((ratio >= 1).all())
    checks['km_w, km_h >= km'] = bool((tab['km_w'] >= attrs['km']).all()
                                      and (tab['km_h'] >= attrs['km']).all())
    checks['centre in box'] = bool(tab['lat'].between(attrs['box_lat_min'],
                                                      attrs['box_lat_max']).all()
                                   and tab['lon'].between(attrs['box_lon_min'],
                                                          attrs['box_lon_max']).all())

    print("\n== time span ==")
    times = pd.DatetimeIndex(tab['time'])
    per = pd.Series(1, index=times).groupby(level=0).sum().sort_index()
    steps = tuple(int(s) for s in str(attrs.get('stride_hours', '9 10')).split())
    sched = pd.DatetimeIndex(cut.snapshot_hours(cut.RECORD_START, cut.RECORD_END, steps))
    if sched.tz is not None:                            # schedule is tz-aware UTC; file is naive
        sched = sched.tz_convert(None)
    print(f"  first {times.min()}, last {times.max()} UTC; {per.size} distinct snapshot times; "
          f"schedule {'/'.join(map(str, steps))} h from {cut.RECORD_START} to {cut.RECORD_END}: "
          f"{sched.size} hours ({sched[0]} .. {sched[-1]})")
    checks['distinct times == snapshot_hours() schedule'] = per.index.equals(sched)
    checks['n_snapshots attr == distinct times'] = int(attrs.get('n_snapshots', -1)) == per.size
    print(f"  n_snapshots attr {attrs.get('n_snapshots')}; equals 1,001: {per.size == 1001}")

    print("\n== UTC hour and month ==")
    hour, month = time_tables(times, per)
    print(hour.to_string(float_format=lambda v: f"{v:.1f}"))
    fh_c = flatness(hour['cutouts'])
    fh_s = flatness(hour['snapshots'])
    fh_e = flatness(hour['cutouts'], hour['snapshots'])
    print(fmt_flat('cutouts per hour vs uniform', fh_c))
    print(fmt_flat('snapshots per hour vs uniform', fh_s))
    print(fmt_flat('cutouts per hour vs snapshots per hour', fh_e))
    print(f"  mean yield per hour {hour['yield'].min():.1f}-{hour['yield'].max():.1f} "
          f"(overall {per.mean():.1f})")
    print(month.to_string(float_format=lambda v: f"{v:.1f}"))
    fm_d = flatness(month['cutouts'], month['days'])
    fm_e = flatness(month['cutouts'], month['snapshots'])
    print(fmt_flat('cutouts per month vs days in month', fm_d))
    print(fmt_flat('cutouts per month vs snapshots per month', fm_e))
    print(f"  cutouts per day {month['cutouts/day'].min():.1f}-{month['cutouts/day'].max():.1f}; "
          f"mean yield per month {month['yield'].min():.1f}-{month['yield'].max():.1f}")
    checks['all 24 UTC hours visited, snapshot counts within 2'] = bool(
        (hour['snapshots'] > 0).all() and np.ptp(hour['snapshots']) <= 2)
    checks[f'per-hour mean yield within {YIELD_TOL:.0%} of overall'] = bool(
        (np.abs(hour['yield'] / per.mean() - 1) <= YIELD_TOL).all())
    checks['all 13 months present'] = bool((month['snapshots'] > 0).all())
    checks[f'per-month mean yield within {YIELD_TOL:.0%} of overall'] = bool(
        (np.abs(month['yield'] / per.mean() - 1) <= YIELD_TOL).all())
    plot_time_histograms(hour, month, fig['time_histograms'])

    print("\n== per-snapshot yield ==")
    print(f"  min {per.min()}, max {per.max()}, mean {per.mean():.2f}, median {per.median():.0f}, "
          f"std {per.std():.2f}; total {per.sum():,} (= count {n:,}: {per.sum() == n})")
    proj = 204.0 * per.size
    print(f"  pilot projection 204.0 x {per.size} = {proj:,.0f}; actual is "
          f"{100 * (n / proj - 1):+.1f}%")
    lo, hi = per.nsmallest(3), per.nlargest(3)
    print("  lowest: " + ", ".join(f"{t:%Y-%m-%d %H} UTC {c}" for t, c in lo.items()))
    print("  highest: " + ", ".join(f"{t:%Y-%m-%d %H} UTC {c}" for t, c in hi.items()))
    strips = pd.Series(tab['i0'].values, index=times).groupby(level=0).nunique()
    by = per.groupby(strips).agg(['size', 'min', 'max', 'mean'])
    by.index.name = 'strips'
    print('  yield by number of tile strips (distinct i0) per snapshot:')
    print(by.to_string(float_format=lambda v: f'{v:.1f}'))
    checks['sum of per-snapshot yields == count'] = int(per.sum()) == n
    plot_yield(per, fig['yield'])

    print("\n== distributions ==")
    dt = dist_table(tab)
    print(dt.to_string(float_format=lambda v: f"{v:.4g}"))
    plot_stat_histograms(tab, fig['stat_histograms'])
    print(f"  lat {tab['lat'].min():.2f}..{tab['lat'].max():.2f}, "
          f"lon {tab['lon'].min():.2f}..{tab['lon'].max():.2f}")

    print("\n== spatial uniformity of cutout centres ==")
    uni = centre_uniformity(tab)
    print(f"  interior lat {uni['region'][0]:g}-{uni['region'][1]:g}, lon {uni['region'][2]:g} to "
          f"{uni['region'][3]:g}: {uni['n']:,} cutouts")
    for b in ('1deg', '2deg'):
        u = uni[b]
        print(f"  {b} bins ({u['bins']}): CV {u['cv']:.3f} vs Poisson {u['poisson_cv']:.3f}, "
              f"max/min {u['max/min']:.3f}")
        checks[f'large-scale uniformity: {b} CV < 2 x Poisson'] = bool(
            u['cv'] < 2 * u['poisson_cv'])
    for key in ('ci', 'cj'):
        u = uni[key]
        print(f"  per native {key}: {u['cells']} cells, {u['zero']} with no centre, mean "
              f"{u['mean']:.1f}, CV {u['cv']:.3f} (Poisson {u['poisson_cv']:.3f}), "
              f"range {u['min']}-{u['max']}")
    print("  NOTE (not a pass/fail check): at sub-tile scale the centres are modulated, most "
          "along native i (latitude), because tile_snapshot draws i_off/j_off from "
          "[0, typical tile size) and strips advance by integer widths that change with "
          "latitude; uniform at >= 1 deg.")
    plot_coverage(tab, attrs, uni, fig['coverage'])

    print("\n== example cutouts ==")
    rng = np.random.default_rng(args.seed)
    ex = pick_examples(times, args.n_examples, rng)
    print("  random (one per equal time interval): " + ", ".join(
        f"#{k} {times[k]:%Y-%m-%d %H}" for k in ex))
    plot_cutout_panel(ds, tab, ex, fig['example_cutouts'],
                      f"{len(ex)} random cutouts across the year (seed {args.seed}): 64 x 64 px, "
                      f"~2 km/px, north-up / east-right at their stored corners")
    xp, xl = pick_extremes(tab)
    print("  extremes: " + ", ".join(f"{lab} #{k}" for k, lab in zip(xp, xl)))
    plot_cutout_panel(ds, tab, xp, fig['example_cutouts_extremes'],
                      'Extremes, demeaned (each panel its own symmetric scale): rows = highest '
                      'grad_p90, lowest std, warmest mean, coldest mean', anomaly=True,
                      labels=xl, ncol=3)
    ds.close()
    for f in fig.values():
        print(f"  wrote {f}")

    print(f"\n== summary ({_time.time() - t_start:.0f} s) ==")
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    print('ALL CHECKS PASS' if all(checks.values()) else 'SOME CHECKS FAILED')
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main(parser()))
