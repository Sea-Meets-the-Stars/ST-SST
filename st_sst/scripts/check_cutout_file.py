""" Sanity checks on a concatenated LLC4320_v2 cutout file.

Checks the layout and the invariants from the plan
(``reports/extract_llc4320v2_gulf_stream.md``): dims ``(cutout, y, x)``,
count, no NaN in ``sst``, unique int64 UIDs, ``lat_ne > lat_sw`` and
``lon_ne > lon_sw``, the sst chunking/compression, and prints the
per-snapshot yield and ranges of the realized extents and statistics.

Usage::

    conda run -n ocean14 python -m st_sst.scripts.check_cutout_file \
        $OS_OGCM/LLC_v2/cutouts/LLC4320v2_GulfStream_SST_128km_64px_pilot.nc
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from st_sst.extract import llc_v2_cutouts as cut


def sst_all_finite(path: str, block: int = 1024) -> bool:
    """True if every ``sst`` value is finite, streaming *block* cutouts at a time."""
    import netCDF4
    with netCDF4.Dataset(path) as nc:
        v = nc.variables['sst']
        v.set_auto_mask(False)
        n = v.shape[0]
        for k in range(0, n, block):
            if not np.isfinite(v[k:k + block]).all():
                return False
    return True


def main(path: str, list_snapshots: bool = True) -> int:
    import netCDF4
    ds = xr.open_dataset(path)
    n = ds.sizes['cutout']
    checks = {
        'dims (cutout, y, x)': ds['sst'].dims == ('cutout', 'y', 'x'),
        'all table variables present': set(cut.TABLE_COLUMNS) <= set(ds.data_vars),
        'n_cutouts attr matches': int(ds.attrs.get('n_cutouts', -1)) == n,
        'no NaN in sst': sst_all_finite(path),
        'UID int64': ds['UID'].dtype == np.int64,
        'UIDs unique': pd.Index(ds['UID'].values).is_unique,
        'lat_ne > lat_sw': bool((ds['lat_ne'] > ds['lat_sw']).all()),
        'lon_ne > lon_sw': bool((ds['lon_ne'] > ds['lon_sw']).all()),
        'centre inside corners': bool(((ds['lat'] > ds['lat_sw']) & (ds['lat'] < ds['lat_ne'])
                                       & (ds['lon'] > ds['lon_sw'])
                                       & (ds['lon'] < ds['lon_ne'])).all()),
        'credit attr names Dan Whitt': 'Dan Whitt' in ds.attrs.get('credit', ''),
    }
    with netCDF4.Dataset(path) as nc:
        v = nc.variables['sst']
        chunk = v.chunking()
        checks['sst chunks (min(1024,n), y, x)'] = chunk == [min(1024, n), ds.sizes['y'],
                                                             ds.sizes['x']]
        checks['sst zlib level 1'] = v.filters()['zlib'] and v.filters()['complevel'] == 1

    print(f"{path}: {n} cutouts, {os.path.getsize(path) / 1e6:.1f} MB, sst {ds['sst'].shape}, "
          f"chunks {chunk}")
    for k, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {k}")
    per = pd.Series(ds['time'].values).value_counts().sort_index()
    print(f"cutouts per snapshot: {per.size} snapshots, {per.min()}-{per.max()}, "
          f"mean {per.mean():.1f}")
    for t, c in (per.items() if list_snapshots else ()):
        print(f"  {t:%Y-%m-%d %H:%M} UTC  {c}")
    for col in ('km_w', 'km_h', 'nj', 'ni', 'min_depth', 'mean', 'std', 'grad_mean', 'grad_max'):
        x = ds[col].values
        print(f"  {col:9s} min {x.min():9.3f}  median {np.median(x):9.3f}  max {x.max():9.3f}")
    print(f"  lat {float(ds['lat'].min()):.2f}..{float(ds['lat'].max()):.2f}, "
          f"lon {float(ds['lon'].min()):.2f}..{float(ds['lon'].max()):.2f}")
    print(f"  git_commit {ds.attrs.get('git_commit')}, code_version {ds.attrs.get('code_version')}")
    ds.close()
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('path')
    sys.exit(main(p.parse_args().path))
