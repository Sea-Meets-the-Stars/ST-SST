""" Validate the LLC4320_v2 Gulf Stream pilot cutouts (Implement prompt #3).

Reads the concatenated pilot file and

1. runs the invariant checks of the plan (``lat_ne > lat_sw``, ``lon_ne > lon_sw``,
   no NaN in ``sst``, unique UIDs, resampling ratio ``min(nj, ni) / npix >= 1``),
   plus the layout checks of `st_sst.scripts.check_cutout_file`;
2. spot-checks cutouts against ``Theta`` read directly with xarray from the
   hourly store at the stored ``(face, j0:j0+nj, i0:i0+ni)``:

   * every cutout of each pilot snapshot is rebuilt with the library's own
     orientation + ``resize_local_mean`` and compared with the stored ``sst``;
   * for a few cutouts, individual output pixels are recomputed independently
     of skimage -- an explicit area-weighted mean over the native cells, located
     through the orientation index map -- which tests orientation and indexing;
   * the stored SW / NE corners are compared with ``XC`` / ``YC`` of the SW / NE
     native cells, and the oriented tile is checked to be north-up / east-right;

3. draws, under ``--outdir``: the box with the SST snapshot(s) and tile outlines,
   a panel of cutouts with their corner coordinates, and histograms of
   ``km_w``, ``km_h``, ``ni``, ``nj``;
4. prints the measured yield and the snapshot count / expected cutouts for a
   set of candidate strides over the full record.

Exits non-zero if any check fails.

Usage (repository root)::

    conda run -n ocean14 python -m st_sst.scripts.plot_pilot_cutouts
    conda run -n ocean14 python -m st_sst.scripts.plot_pilot_cutouts --no-network
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402
from matplotlib.ticker import MaxNLocator             # noqa: E402

from st_sst.extract import llc_v2_cutouts as cut      # noqa: E402

PILOT_NAME = 'LLC4320v2_GulfStream_SST_128km_64px_pilot.nc'
MISSING_HOURS = ('2023-10-19 03:00', '2023-12-28 23:00')   # absent from the bucket
TARGET = 200_000
CANDIDATE_STEPS = ((9, 10), (8, 9), (10, 11), (9,), (10,), (11,), (8, 9, 10))


def default_file() -> str:
    root = os.environ.get('OS_OGCM', '/home/xavier/Oceanography/data/OGCM/')
    return os.path.join(root, 'LLC_v2', 'cutouts', PILOT_NAME)


def parser(options=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--file', default=default_file(), help='pilot cutout file')
    p.add_argument('--outdir', default='reports/figures', help='figure directory')
    p.add_argument('--n-cutouts', type=int, default=6, help='cutouts in the panel figure')
    p.add_argument('--n-spot', type=int, default=3, help='cutouts in the pixel-level spot check')
    p.add_argument('--seed', type=int, default=7, help='which cutouts to show / spot-check')
    p.add_argument('--no-network', action='store_true',
                   help='skip the S3 spot check and the map figure')
    return p.parse_args() if options is None else p.parse_args(options)


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def invariant_checks(ds) -> dict:
    npix = ds.sizes['y']
    ratio = np.minimum(ds['nj'].values, ds['ni'].values) / npix
    checks = {
        'lat_ne > lat_sw': bool((ds['lat_ne'] > ds['lat_sw']).all()),
        'lon_ne > lon_sw': bool((ds['lon_ne'] > ds['lon_sw']).all()),
        'no NaN in sst': bool(np.isfinite(ds['sst'].values).all()),
        'UIDs unique': pd.Index(ds['UID'].values).is_unique,
        'resampling ratio min(nj, ni)/npix >= 1': bool((ratio >= 1).all()),
        'km_w, km_h >= km': bool((ds['km_w'] >= ds.attrs['km']).all()
                                 and (ds['km_h'] >= ds.attrs['km']).all()),
        'centre in box': bool(((ds['lat'] >= ds.attrs['box_lat_min'])
                               & (ds['lat'] <= ds.attrs['box_lat_max'])
                               & (ds['lon'] >= ds.attrs['box_lon_min'])
                               & (ds['lon'] <= ds.attrs['box_lon_max'])).all()),
    }
    rj, ri = ds['nj'].values / npix, ds['ni'].values / npix
    print(f"resampling ratio: nj/npix {rj.min():.3f}-{rj.max():.3f}, "
          f"ni/npix {ri.min():.3f}-{ri.max():.3f}, min(nj,ni)/npix {ratio.min():.3f}-{ratio.max():.3f}")
    for name in ('km_w', 'km_h'):
        px = ds[name].values / npix
        print(f"realized pixel {name}/npix: {px.min():.4f}-{px.max():.4f} km "
              f"(median {np.median(px):.4f})")
    return checks


# ---------------------------------------------------------------------------
# Spot check against Theta
# ---------------------------------------------------------------------------

def area_weights(n_in: int, n_out: int) -> np.ndarray:
    """``(n_out, n_in)`` overlap of output cell r = [r, r+1)*n_in/n_out with input cells."""
    w = np.zeros((n_out, n_in))
    edges = np.arange(n_out + 1) * n_in / n_out
    for r in range(n_out):
        a, b = edges[r], edges[r + 1]
        for k in range(int(math.floor(a)), min(n_in, int(math.ceil(b)))):
            w[r, k] = max(0., min(b, k + 1) - max(a, k))
    return w


def oriented_index_map(j0, i0, nj, ni, orient):
    """Native face-absolute ``(J, I)`` of every cell of the oriented tile."""
    jj, ii = np.meshgrid(np.arange(j0, j0 + nj), np.arange(i0, i0 + ni), indexing='ij')
    return cut.orient_north_up(jj, orient), cut.orient_north_up(ii, orient)


def spot_check(ds, grid, stores, slices, rng, n_spot: int) -> dict:
    """Compare stored cutouts with Theta read directly. Returns check results."""
    npix = ds.sizes['y']
    sst = ds['sst'].values
    times = pd.DatetimeIndex(ds['time'].values)
    checks, maxdiff_all = {}, 0.

    face_cache = {}
    for t, store in stores.items():
        sel = np.nonzero(times == t)[0]
        for f in np.unique(ds['face'].values[sel]):
            js, isl = slices[int(f)]
            g = grid.isel(face=int(f), j=js, i=isl)
            orient = cut.face_orientation(g['XC'].values, g['YC'].values)
            face_cache[int(f)] = orient
            # Theta of the face rectangle, read directly
            th = store['Theta'].isel(face=int(f), j=js, i=isl).values
            for k in sel[ds['face'].values[sel] == f]:
                j0 = int(ds['j0'][k]) - js.start
                i0 = int(ds['i0'][k]) - isl.start
                img = cut.make_cutout(th, j0, i0, int(ds['nj'][k]), int(ds['ni'][k]), npix,
                                      orient)
                maxdiff_all = max(maxdiff_all, float(np.abs(img - sst[k]).max()))
        print(f"  {t:%Y-%m-%d %H:%M}: {sel.size} cutouts rebuilt from Theta with the library")
    print(f"  max |rebuilt - stored| over all cutouts: {maxdiff_all:.3e} degC")
    checks['all cutouts rebuilt == stored (|d| < 1e-5)'] = maxdiff_all < 1e-5

    # Pixel-level, skimage-independent check on a few cutouts from both snapshots
    tlist = sorted(stores)
    picks = []
    for m in range(n_spot):
        idx = np.nonzero(times == tlist[m % len(tlist)])[0]
        picks.append(int(rng.choice(idx)))
    w_pix, w_corner, ok_orient = 0., 0., True
    for k in picks:
        f, j0, i0 = int(ds['face'][k]), int(ds['j0'][k]), int(ds['i0'][k])
        nj, ni = int(ds['nj'][k]), int(ds['ni'][k])
        orient = face_cache[f]
        store = stores[times[k]]
        # Direct native read at (face, j0:j0+nj, i0:i0+ni)
        tile = store['Theta'].isel(face=f, j=slice(j0, j0 + nj), i=slice(i0, i0 + ni)).values
        g = grid.isel(face=f, j=slice(j0, j0 + nj), i=slice(i0, i0 + ni))
        xc, yc = g['XC'].values.astype(np.float64), g['YC'].values.astype(np.float64)
        J, I = oriented_index_map(j0, i0, nj, ni, orient)
        img_native = tile[J - j0, I - i0]                   # oriented native tile
        H, W = img_native.shape
        wr, wc = area_weights(H, npix), area_weights(W, npix)
        print(f"  cutout {k} ({times[k]:%Y-%m-%d %H} UTC, face {f}, j {j0}:{j0 + nj}, "
              f"i {i0}:{i0 + ni}; oriented {H}x{W} -> {npix}x{npix}, ratio "
              f"{H / npix:.3f} x {W / npix:.3f})")
        for (r, c) in ((0, 0), (0, npix - 1), (npix - 1, 0), (npix // 2, npix // 3),
                       (npix - 1, npix - 1)):
            rows, cols = np.nonzero(wr[r])[0], np.nonzero(wc[c])[0]
            ww = np.outer(wr[r, rows], wc[c, cols])
            val = float((ww * img_native[np.ix_(rows, cols)]).sum() / ww.sum())
            d = abs(val - float(sst[k, r, c]))
            w_pix = max(w_pix, d)
            jr = J[np.ix_(rows, cols)]
            ir = I[np.ix_(rows, cols)]
            print(f"    pixel (y={r:2d}, x={c:2d}): stored {sst[k, r, c]:.5f}, area mean of native "
                  f"j {jr.min()}-{jr.max()}, i {ir.min()}-{ir.max()} = {val:.5f}, |d| {d:.1e}")
        # Corners against XC/YC of the SW / NE native cells
        sw = (J[-1, 0], I[-1, 0])
        ne = (J[0, -1], I[0, -1])
        ref = {'lat_sw': yc[sw[0] - j0, sw[1] - i0], 'lon_sw': xc[sw[0] - j0, sw[1] - i0],
               'lat_ne': yc[ne[0] - j0, ne[1] - i0], 'lon_ne': xc[ne[0] - j0, ne[1] - i0]}
        dc = max(abs(float(ds[key][k]) - v) for key, v in ref.items())
        w_corner = max(w_corner, dc)
        print(f"    SW cell (j {sw[0]}, i {sw[1]}) = ({ref['lat_sw']:.4f}N, {ref['lon_sw']:.4f}E), "
              f"stored ({float(ds['lat_sw'][k]):.4f}, {float(ds['lon_sw'][k]):.4f}); "
              f"NE cell (j {ne[0]}, i {ne[1]}) = ({ref['lat_ne']:.4f}N, {ref['lon_ne']:.4f}E), "
              f"stored ({float(ds['lat_ne'][k]):.4f}, {float(ds['lon_ne'][k]):.4f}); "
              f"max |d| {dc:.1e} deg")
        # Orientation: lat decreasing down rows, lon increasing along columns, extremes at corners
        yo, xo = cut.orient_north_up(yc, orient), cut.orient_north_up(xc, orient)
        o = (np.all(np.diff(yo, axis=0) < 0) and np.all(np.diff(xo, axis=1) > 0)
             and np.isclose(yo.min(), ref['lat_sw']) and np.isclose(yo.max(), ref['lat_ne'])
             and np.isclose(xo.min(), ref['lon_sw']) and np.isclose(xo.max(), ref['lon_ne']))
        ok_orient &= bool(o)
    print(f"  orientation {cut.face_orientation(xc, yc)} (face {f})")
    checks['pixel-level area means == stored (|d| < 1e-4)'] = w_pix < 1e-4
    checks['corners == XC/YC of SW/NE cells (|d| < 1e-6)'] = w_corner < 1e-6
    checks['oriented tile north-up / east-right'] = ok_orient
    print(f"  pixel-level max |d| {w_pix:.2e} degC; corner max |d| {w_corner:.2e} deg")
    return checks


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_map(ds, grid, stores, slices, outfile, shown):
    times = pd.DatetimeIndex(ds['time'].values)
    box = [ds.attrs[k] for k in ('box_lat_min', 'box_lat_max', 'box_lon_min', 'box_lon_max')]
    tlist = sorted(stores)
    fig, axes = plt.subplots(1, len(tlist), figsize=(7.5 * len(tlist), 5.2), squeeze=False,
                             constrained_layout=True)
    for ax, t in zip(axes[0], tlist):
        sel = np.nonzero(times == t)[0]
        for f, (js, isl) in sorted(slices.items()):
            g = grid.isel(face=f, j=js, i=isl)
            th = stores[t]['Theta'].isel(face=f, j=js, i=isl).values
            s = 2                                            # every other cell, for size
            pm = ax.pcolormesh(g['XC'].values[::s, ::s], g['YC'].values[::s, ::s],
                               th[::s, ::s], cmap='RdYlBu_r', shading='auto',
                               vmin=np.nanpercentile(th, 1), vmax=np.nanpercentile(th, 99),
                               rasterized=True)
        for k in sel:
            lw, ec = (1.6, 'k') if k in shown else (0.5, '0.2')
            ax.add_patch(Rectangle((float(ds['lon_sw'][k]), float(ds['lat_sw'][k])),
                                   float(ds['lon_ne'][k] - ds['lon_sw'][k]),
                                   float(ds['lat_ne'][k] - ds['lat_sw'][k]),
                                   fill=False, lw=lw, ec=ec))
        for k in sel:
            if k in shown:
                ax.text(float(ds['lon'][k]), float(ds['lat'][k]), str(k), ha='center',
                        va='center', fontsize=8, weight='bold',
                        bbox=dict(fc='w', ec='none', alpha=0.7, pad=0.5))
        ax.add_patch(Rectangle((box[2], box[0]), box[3] - box[2], box[1] - box[0],
                               fill=False, lw=1.5, ec='m', ls='--'))
        ax.set_xlim(box[2] - 1.5, box[3] + 1.5)
        ax.set_ylim(box[0] - 1.5, box[1] + 1.5)
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(box[:2]))))
        ax.set_xlabel('longitude (deg E)')
        ax.set_ylabel('latitude (deg N)')
        ax.set_title(f"LLC4320_v2 SST {t:%Y-%m-%d %H:%M} UTC: {sel.size} tiles "
                     f"(box dashed; bold = shown in panel figure)", fontsize=10)
        fig.colorbar(pm, ax=ax, label='SST (degC)', shrink=0.85)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def plot_cutouts(ds, shown, outfile):
    n = len(shown)
    ncol = 3
    nrow = int(math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.1 * nrow), squeeze=False,
                             constrained_layout=True)
    times = pd.DatetimeIndex(ds['time'].values)
    for ax, k in zip(axes.flat, shown):
        img = ds['sst'].values[k]
        ext = [float(ds['lon_sw'][k]), float(ds['lon_ne'][k]),
               float(ds['lat_sw'][k]), float(ds['lat_ne'][k])]
        im = ax.imshow(img, origin='upper', extent=ext, cmap='RdYlBu_r', interpolation='nearest',
                       aspect=1 / np.cos(np.deg2rad(float(ds['lat'][k]))))
        fig.colorbar(im, ax=ax, label='SST (degC)', shrink=0.8)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.set_title(f"#{k} {times[k]:%Y-%m-%d %H} UTC\n"
                     f"SW ({ext[2]:.2f}N, {ext[0]:.2f}E)  NE ({ext[3]:.2f}N, {ext[1]:.2f}E)\n"
                     f"{float(ds['km_w'][k]):.1f} x {float(ds['km_h'][k]):.1f} km, "
                     f"native {int(ds['nj'][k])}x{int(ds['ni'][k])} (nj x ni)", fontsize=9)
        ax.set_xlabel('longitude (deg E)')
        ax.set_ylabel('latitude (deg N)')
    for ax in list(axes.flat)[n:]:
        ax.axis('off')
    fig.suptitle('Pilot cutouts, 64 x 64 px, drawn north-up / east-right at their stored corners',
                 fontsize=11)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


def plot_histograms(ds, outfile):
    times = pd.DatetimeIndex(ds['time'].values)
    tl = sorted(set(times))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (col, unit, integer) in zip(axes.flat, (('km_w', 'km', False), ('km_h', 'km', False),
                                                    ('ni', 'cells', True), ('nj', 'cells', True))):
        x = ds[col].values
        bins = (np.arange(x.min() - 0.5, x.max() + 1.5) if integer
                else np.linspace(x.min(), x.max(), 25))
        ax.hist([x[times == t] for t in tl], bins=bins, stacked=True,
                label=[f"{t:%Y-%m-%d %H} UTC" for t in tl], color=['#1f77b4', '#ff7f0e'][:len(tl)])
        if col in ('ni', 'nj'):
            ax.axvline(ds.sizes['y'], color='k', ls='--', lw=1, label='npix = 64')
        ax.set_xlabel(f"{col} ({unit})")
        ax.set_ylabel('cutouts')
        ax.set_title(f"{col}: {x.min():.4g}-{x.max():.4g}, median {np.median(x):.4g}", fontsize=10)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Pilot: realized extents (km_w west-east, km_h south-north) and native cells '
                 '(ni along i = latitude, nj along j = longitude on face 10)', fontsize=10)
    fig.savefig(outfile, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Yield and stride
# ---------------------------------------------------------------------------

def yield_and_stride(ds):
    per = pd.Series(pd.DatetimeIndex(ds['time'].values)).value_counts().sort_index()
    mean_yield = float(per.mean())
    print("tiles per snapshot: " + ", ".join(f"{t:%Y-%m-%d %H} UTC {c}" for t, c in per.items())
          + f"; mean {mean_yield:.1f}")
    missing = {cut.parse_utc(h) for h in MISSING_HOURS}
    rows = []
    for steps in CANDIDATE_STEPS:
        hrs = cut.snapshot_hours(cut.RECORD_START, cut.RECORD_END, steps)
        n_miss = sum(h in missing for h in hrs)
        n = len(hrs) - n_miss
        utc = np.bincount([h.hour for h in hrs], minlength=24)
        rows.append({'steps': '/'.join(map(str, steps)), 'snapshots': n, 'missing_hit': n_miss,
                     'expected': int(round(n * mean_yield)),
                     'utc_hours_visited': int((utc > 0).sum()),
                     'utc_count_range': f"{utc[utc > 0].min()}-{utc.max()}"})
    tab = pd.DataFrame(rows)
    print(f"expected cutouts over {cut.RECORD_START} .. {cut.RECORD_END} at {mean_yield:.1f} "
          f"per snapshot (missing hours {', '.join(MISSING_HOURS)} excluded):")
    print(tab.to_string(index=False))
    best = tab.loc[(tab['utc_hours_visited'] == 24)]
    best = best.iloc[(best['expected'] - TARGET).abs().argsort()].iloc[0]
    print(f"recommended stride: {best['steps']} h -> {best['snapshots']} snapshots, "
          f"~{best['expected']:,} cutouts (all 24 UTC hours visited)")
    return per, tab, best


# ---------------------------------------------------------------------------

def main(args) -> int:
    ds = xr.open_dataset(args.file).load()
    os.makedirs(args.outdir, exist_ok=True)
    n = ds.sizes['cutout']
    print(f"{args.file}: {n} cutouts, sst {ds['sst'].shape}")
    checks = {}

    print("\n== layout checks (st_sst.scripts.check_cutout_file) ==")
    from st_sst.scripts import check_cutout_file
    checks['check_cutout_file layout checks'] = check_cutout_file.main(args.file) == 0

    print("\n== invariant checks ==")
    checks.update(invariant_checks(ds))

    rng = np.random.default_rng(args.seed)
    times = pd.DatetimeIndex(ds['time'].values)
    tl = sorted(set(times))
    shown = []
    for m in range(args.n_cutouts):
        idx = np.setdiff1d(np.nonzero(times == tl[m % len(tl)])[0], shown)
        shown.append(int(rng.choice(idx)))

    if not args.no_network:
        print("\n== spot check against Theta read directly from S3 ==")
        from wrangler.ogcm import llc_v2_access as llc
        grid = llc.open_grid()
        box = [ds.attrs[k] for k in ('box_lat_min', 'box_lat_max', 'box_lon_min', 'box_lon_max')]
        slices = llc.region_slices(*box)
        stores = {t: llc.open_store(t.to_pydatetime()) for t in tl}
        checks.update(spot_check(ds, grid, stores, slices, rng, args.n_spot))
        f_map = os.path.join(args.outdir, 'pilot_map.png')
        plot_map(ds, grid, stores, slices, f_map, set(shown))
        print(f"wrote {f_map}")
    else:
        print("\n(--no-network: spot check and map skipped)")

    f_cut = os.path.join(args.outdir, 'pilot_cutouts.png')
    plot_cutouts(ds, shown, f_cut)
    print(f"wrote {f_cut} (cutouts {shown})")
    f_hist = os.path.join(args.outdir, 'pilot_extent_histograms.png')
    plot_histograms(ds, f_hist)
    print(f"wrote {f_hist}")

    print("\n== yield and stride ==")
    yield_and_stride(ds)

    print("\n== summary ==")
    for k, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {k}")
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    sys.exit(main(parser()))
