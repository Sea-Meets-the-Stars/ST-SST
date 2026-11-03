""" Tests for st_sst.extract.llc_v2_cutouts on a synthetic single-face grid.

No network: the grid is built in memory with the same dims and staggering as
``grid.zarr`` (``face, j, i, j_g, i_g``).  The default mimics face 10, which
holds the Gulf Stream: ``i`` runs along latitude (decreasing) and ``j`` along
longitude (increasing).
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from st_sst.extract import llc_v2_cutouts as cut

KM_DEG = 111.19


def synthetic_face(nj=240, ni=220, lat_top=46., lon_left=-81., dx_km=1.69, dy_km=1.83,
                   rotated=True, land=True):
    """A face-10-like (rotated) or face-2-like (unrotated) lat-lon face.

    Spacings are uniform; ``dx_km`` is along ``i`` and ``dy_km`` along ``j``.
    With ``land`` a block in the corner j<30, i<40 is masked.
    """
    if rotated:
        lat = lat_top - (np.arange(ni) + 0.5) * dx_km / KM_DEG              # along i
        lon = lon_left + (np.arange(nj) + 0.5) * dy_km / (KM_DEG * np.cos(np.radians(38.)))
        yc = np.broadcast_to(lat[None, :], (nj, ni)).copy()
        xc = np.broadcast_to(lon[:, None], (nj, ni)).copy()
    else:
        lat = (lat_top - 15.) + (np.arange(nj) + 0.5) * dy_km / KM_DEG     # along j, increasing
        lon = lon_left + (np.arange(ni) + 0.5) * dx_km / (KM_DEG * np.cos(np.radians(38.)))
        yc = np.broadcast_to(lat[:, None], (nj, ni)).copy()
        xc = np.broadcast_to(lon[None, :], (nj, ni)).copy()
    dxC = np.full((nj, ni), dx_km * 1000.)
    dyC = np.full((nj, ni), dy_km * 1000.)
    maskC = np.ones((nj, ni), dtype=bool)
    depth = np.full((nj, ni), 4000.)
    if land:
        maskC[:30, :40] = False
        depth[:30, :40] = 0.
    return xr.Dataset({
        'XC': (('face', 'j', 'i'), xc[None]), 'YC': (('face', 'j', 'i'), yc[None]),
        'dxC': (('face', 'j', 'i_g'), dxC[None]), 'dyC': (('face', 'j_g', 'i'), dyC[None]),
        'maskC': (('face', 'j', 'i'), maskC[None]), 'Depth': (('face', 'j', 'i'), depth[None])})


def arrays(ds):
    """(xc, yc, dx_km, dy_km, ocean) of face 0, with the staggered axes sliced explicitly."""
    g = ds.isel(face=0)
    return (g['XC'].values, g['YC'].values,
            ds['dxC'].isel(face=0).values / 1000., ds['dyC'].isel(face=0).values / 1000.,
            g['maskC'].values.astype(bool))


# ---------------------------------------------------------------------------
# Extents
# ---------------------------------------------------------------------------

def test_cells_for_km_reaches_km():
    s = np.full(200, 1.69)
    n, real = cut.cells_for_km(s, 10, 128.)
    assert n == 76                               # 75 * 1.69 = 126.75 < 128 <= 76 * 1.69
    assert real == pytest.approx(76 * 1.69)
    assert 128. <= real < 128. + 1.69


def test_cells_for_km_min_cells_and_edge():
    s = np.full(100, 2.01)
    n, real = cut.cells_for_km(s, 0, 128., min_cells=64)
    assert n == 64 and real == pytest.approx(64 * 2.01)
    n, real = cut.cells_for_km(np.full(100, 2.5), 0, 128., min_cells=64)
    assert n == 64                               # forced up from 52 cells
    assert cut.cells_for_km(s, 50, 128.) is None  # only 50 cells left
    assert cut.cells_for_km(s, 100, 128.) is None


def test_tile_extent():
    ds = synthetic_face()
    _, _, dx, dy, _ = arrays(ds)
    nj, ni, km_j, km_i = cut.tile_extent(dx[50, :], dy[:, 60], 50, 60, 128.)
    assert (nj, ni) == (70, 76)
    assert km_j == pytest.approx(70 * 1.83) and km_i == pytest.approx(76 * 1.69)
    assert cut.tile_extent(dx[200, :], dy[:, 60], 200, 60, 128.) is None   # off the j edge


@pytest.mark.parametrize('spacing', [1.53, 1.64, 1.69, 1.83, 1.85, 2.01])
def test_64px_never_upsamples_for_plan_spacings(spacing):
    """128 km at the Gulf Stream spacings (1.53-2.01 km) is always >= 64 cells."""
    s = np.full(300, spacing)
    n, real = cut.cells_for_km(s, 0, 128., min_cells=64)
    assert n >= 64
    assert real == pytest.approx(n * spacing)
    assert real < 128. + spacing or n == 64


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def test_face_orientation_rotated_is_plain_transpose():
    xc, yc, *_ = arrays(synthetic_face(rotated=True))
    assert cut.face_orientation(xc, yc) == {'transpose': True, 'flip_rows': False,
                                             'flip_cols': False}


def test_face_orientation_unrotated_flips_rows():
    xc, yc, *_ = arrays(synthetic_face(rotated=False))
    assert cut.face_orientation(xc, yc) == {'transpose': False, 'flip_rows': True,
                                             'flip_cols': False}


@pytest.mark.parametrize('rotated', [True, False])
def test_oriented_coordinates_are_north_up_east_right(rotated):
    xc, yc, *_ = arrays(synthetic_face(rotated=rotated))
    orient = cut.face_orientation(xc, yc)
    yo, xo = cut.orient_north_up(yc, orient), cut.orient_north_up(xc, orient)
    assert np.all(np.diff(yo, axis=0) < 0)        # latitude decreases down the rows
    assert np.all(np.diff(xo, axis=1) > 0)        # longitude increases along the columns


# ---------------------------------------------------------------------------
# Cutouts
# ---------------------------------------------------------------------------

def test_make_cutout_orientation_and_shape():
    ds = synthetic_face()
    xc, yc, dx, dy, _ = arrays(ds)
    orient = cut.face_orientation(xc, yc)
    j0, i0 = 40, 50
    nj, ni, km_j, km_i = cut.tile_extent(dx[j0, :], dy[:, i0], j0, i0, 128.)
    lat_img = cut.make_cutout(yc, j0, i0, nj, ni, 64, orient)     # the field IS latitude
    lon_img = cut.make_cutout(xc, j0, i0, nj, ni, 64, orient)
    assert lat_img.shape == (64, 64) and lat_img.dtype == np.float32
    assert np.all(np.diff(lat_img, axis=0) < 0)   # north up
    assert np.all(np.diff(lon_img, axis=1) > 0)   # east right
    km_w, km_h = cut.oriented_km(km_j, km_i, orient)
    assert km_w == pytest.approx(km_j) and km_h == pytest.approx(km_i)   # cols = j, rows = i
    corners = cut.tile_corners(xc, yc, j0, i0, nj, ni, orient)
    assert corners['lat_ne'] > corners['lat_sw'] and corners['lon_ne'] > corners['lon_sw']
    assert corners['lat_sw'] < corners['lat'] < corners['lat_ne']


def test_make_cutout_preserves_mean_and_refuses_upsampling():
    rng = np.random.default_rng(0)
    field = rng.normal(20., 1., (240, 220)).astype(np.float32)
    orient = {'transpose': True, 'flip_rows': False, 'flip_cols': False}
    img = cut.make_cutout(field, 10, 20, 70, 76, 64, orient)
    assert img.mean() == pytest.approx(field[10:80, 20:96].mean(), abs=1e-3)
    with pytest.raises(ValueError):
        cut.make_cutout(field, 10, 20, 60, 76, 64, orient)       # 60 rows < 64 px
    with pytest.raises(ValueError):
        cut.make_cutout(field, 200, 20, 70, 76, 64, orient)      # runs off the field


def test_tile_snapshot_tiles_are_disjoint_ocean_and_in_box():
    ds = synthetic_face()
    xc, yc, dx, dy, ocean = arrays(ds)
    box = (yc.min() + 0.2, yc.max() - 0.2, xc.min() + 0.2, xc.max() - 0.2)
    tiles = cut.tile_snapshot(xc, yc, dx, dy, ocean, box, 128., np.random.default_rng(1),
                              min_cells=64)
    assert len(tiles) >= 2
    occupancy = np.zeros(xc.shape, dtype=int)
    for t in tiles:
        occupancy[t['j0']:t['j0'] + t['nj'], t['i0']:t['i0'] + t['ni']] += 1
        assert ocean[t['j0']:t['j0'] + t['nj'], t['i0']:t['i0'] + t['ni']].all()
        assert box[0] <= t['lat'] <= box[1] and box[2] <= t['lon'] <= box[3]
        assert t['nj'] >= 64 and t['ni'] >= 64
        assert 128. <= t['km_j'] < 130. and 128. <= t['km_i'] < 130.
    assert occupancy.max() == 1                                   # no overlap
    # the land block (j<30, i<40) is never used
    assert occupancy[:30, :40].sum() == 0


def test_tile_snapshot_is_seeded():
    ds = synthetic_face()
    xc, yc, dx, dy, ocean = arrays(ds)
    box = (yc.min(), yc.max(), xc.min(), xc.max())
    a = cut.tile_snapshot(xc, yc, dx, dy, ocean, box, 128., np.random.default_rng(7))
    b = cut.tile_snapshot(xc, yc, dx, dy, ocean, box, 128., np.random.default_rng(7))
    c = cut.tile_snapshot(xc, yc, dx, dy, ocean, box, 128., np.random.default_rng(8))
    assert a == b
    assert (a[0]['j0'], a[0]['i0']) != (c[0]['j0'], c[0]['i0'])


def test_tile_snapshot_strip_ending_on_the_edge():
    """Regression: a strip whose tiles end exactly on the last column must not
    index past the slice (IndexError in 12 of 1,001 snapshots of the full run)."""
    rng_seed, km = 3, 128.
    probe = np.random.default_rng(rng_seed)
    probe.integers(0, int(np.ceil(km / 1.0)))                    # j offset
    i_off = int(probe.integers(0, int(np.ceil(km / 1.0))))      # i offset
    ds = synthetic_face(nj=300, ni=i_off + 2 * 128, dx_km=1.0, dy_km=1.0, land=False)
    xc, yc, dx, dy, ocean = arrays(ds)
    box = (yc.min(), yc.max(), xc.min(), xc.max())
    tiles = cut.tile_snapshot(xc, yc, dx, dy, ocean, box, km, np.random.default_rng(rng_seed),
                              min_cells=64)
    assert max(t['i0'] + t['ni'] for t in tiles) == xc.shape[1]


# ---------------------------------------------------------------------------
# Statistics and identifiers
# ---------------------------------------------------------------------------

def test_cutout_stats_on_a_ramp():
    slope = 0.1                                   # degC / km along x
    px = 2.0
    img = 15. + slope * px * np.arange(64)[None, :] * np.ones((64, 1))
    s = cut.cutout_stats(img, km_w=128., km_h=128.)
    assert set(s) == set(cut.STAT_KEYS)
    assert s['min'] == pytest.approx(15.) and s['max'] == pytest.approx(15. + slope * px * 63)
    assert s['mean'] == pytest.approx(img.mean()) and s['std'] == pytest.approx(img.std())
    assert s['p10'] < s['mean'] < s['p90']
    assert s['grad_mean'] == pytest.approx(slope) and s['grad_max'] == pytest.approx(slope)
    assert s['grad_p90'] == pytest.approx(slope)


def test_make_uid_is_unique_per_hour_and_position():
    times = pd.to_datetime(['2023-02-01 12:00', '2023-02-01 13:00', '2023-02-01 12:00'])
    lats = [35.123, 35.123, 35.124]
    lons = [-70.5, -70.5, -70.5]
    uids = cut.make_uid(times, lats, lons)
    assert uids.dtype == np.int64 and len(set(uids)) == 3
    assert cut.make_uid('2023-02-01 12:00', 35.123, -70.5) == uids[0]
    # the documented formula, with the time forced to nanoseconds
    ns = np.datetime64('2023-02-01T12:00', 'ns').astype(np.int64) // 10_000_000
    assert uids[0] == int(f"{str(ns)[:-5]}{round(125.123 * 1000)}{round(109.5 * 1000)}")


def test_make_uid_matches_wrangler_on_ns_datetimes():
    llc = pytest.importorskip('wrangler.ogcm.llc')
    df = pd.DataFrame({'datetime': pd.to_datetime(['2023-02-01 12:00', '2023-08-01 12:00']).as_unit('ns'),
                       'lat': [35.123456, 40.0], 'lon': [-70.5, -60.25]})
    expected = llc.add_uid(df.copy())
    got = cut.make_uid(df['datetime'], df['lat'], df['lon'])
    assert np.array_equal(got, expected)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def test_snapshot_hours_alternate_9_and_10():
    hrs = cut.snapshot_hours('2023-01-01 01:00', '2023-01-03 00:00')
    hours = [(h - hrs[0]).total_seconds() / 3600 for h in hrs]
    assert hours == [0, 9, 19, 28, 38, 47]
    assert all(h.tzinfo is not None for h in hrs)
    with pytest.raises(ValueError):
        cut.snapshot_hours('2023-01-02', '2023-01-01')


def test_snapshot_hours_full_record_is_uniform_in_utc_hour():
    hrs = cut.snapshot_hours()                    # the whole 13-month record
    assert 990 <= len(hrs) <= 1010
    counts = np.bincount([h.hour for h in hrs], minlength=24)
    assert counts.max() - counts.min() <= 2
    months = {(h.year, h.month) for h in hrs}
    assert len(months) == 13


# ---------------------------------------------------------------------------
# Per-snapshot extraction (synthetic grid and store, no network)
# ---------------------------------------------------------------------------

def synthetic_store(grid):
    """An hourly store on the synthetic face: Theta = latitude-based field, NaN on land."""
    yc, xc = grid['YC'].values, grid['XC'].values
    theta = (25. - 0.5 * (yc - 30.) + 0.05 * np.sin(xc)).astype(np.float32)
    theta[~grid['maskC'].values] = np.nan
    return xr.Dataset({'Theta': (('face', 'j', 'i'), theta)})


def _extract(seed=3, offset=(0, 0)):
    grid = synthetic_face()
    store = synthetic_store(grid)
    nj, ni = grid.sizes['j'], grid.sizes['i']
    yc, xc = grid['YC'].values[0], grid['XC'].values[0]
    box = (yc.min(), yc.max(), xc.min(), xc.max())
    slices = {0: (slice(offset[0], nj), slice(offset[1], ni))}
    return grid, store, cut.extract_snapshot('2023-02-01 12:00', grid=grid, box=box, km=128.,
                                             npix=64, seed=seed, store=store, slices=slices)


def test_extract_snapshot_on_synthetic_store():
    grid, store, (cube, table) = _extract()
    assert cube.shape == (len(table), 64, 64) and cube.dtype == np.float32
    assert len(table) >= 2
    assert list(table.columns) == list(cut.TABLE_COLUMNS)
    assert table['UID'].dtype == np.int64 and table['UID'].is_unique
    assert np.isfinite(cube).all()
    assert (table['time'] == pd.Timestamp('2023-02-01 12:00')).all()
    assert (table['lat_ne'] > table['lat_sw']).all() and (table['lon_ne'] > table['lon_sw']).all()
    assert (table['min_depth'] > 0).all()
    theta = store['Theta'].values[0]
    for k, r in table.iterrows():
        native = theta[r.j0:r.j0 + r.nj, r.i0:r.i0 + r.ni]
        assert cube[k].mean() == pytest.approx(native.mean(), abs=1e-3)
        assert r['mean'] == pytest.approx(cube[k].mean(), abs=1e-4)
        # north up: the field falls with latitude, so it rises down the rows
        assert cube[k][-1].mean() > cube[k][0].mean()
    assert table['km_w'].between(128., 130.).all() and table['km_h'].between(128., 130.).all()


def test_extract_snapshot_uses_face_indices_and_is_seeded():
    _, _, (cube_a, ta) = _extract(seed=3)
    _, _, (cube_b, tb) = _extract(seed=3)
    assert ta.equals(tb) and np.array_equal(cube_a, cube_b)
    # a sliced rectangle reports face-absolute indices, and the staggered
    # spacings are sliced along with it (uniform here, so the extents agree)
    _, store, (_, tc) = _extract(seed=3, offset=(10, 20))
    assert (tc['j0'] >= 10).all() and (tc['i0'] >= 20).all()
    assert set(tc['nj']) == set(ta['nj']) and set(tc['ni']) == set(ta['ni'])


def test_snapshot_seed_is_deterministic():
    a = cut.snapshot_seed(42, '2023-02-01 12:00')
    assert a == cut.snapshot_seed(42, '2023-02-01T12:00')
    assert a != cut.snapshot_seed(42, '2023-02-01 13:00')
    assert a != cut.snapshot_seed(43, '2023-02-01 12:00')
    assert cut.snapshot_tag('2023-02-01 12:00') == '20230201T12'


# ---------------------------------------------------------------------------
# NetCDF layer
# ---------------------------------------------------------------------------

def synthetic_cutouts(n, time, seed=0, npix=64):
    rng = np.random.default_rng(seed)
    cube = rng.normal(20., 1., (n, npix, npix)).astype(np.float32)
    lat = 30. + 15. * rng.random(n)
    lon = -80. + 30. * rng.random(n)
    t = pd.DatetimeIndex([pd.Timestamp(time)] * n).as_unit('ns')
    table = pd.DataFrame({'time': t, 'lat': lat, 'lon': lon,
                          'face': np.full(n, 10, np.int32),
                          'j0': rng.integers(2300, 3700, n).astype(np.int32),
                          'i0': rng.integers(2400, 3400, n).astype(np.int32),
                          'nj': np.full(n, 70, np.int32), 'ni': np.full(n, 76, np.int32),
                          'km_w': np.full(n, 128.5), 'km_h': np.full(n, 128.4),
                          'lat_sw': lat - 0.5, 'lon_sw': lon - 0.7,
                          'lat_ne': lat + 0.5, 'lon_ne': lon + 0.7,
                          'min_depth': np.full(n, 3000.)})
    for k in cut.STAT_KEYS:
        table[k] = rng.random(n)
    table.insert(0, 'UID', cut.make_uid(table['time'], lat, lon))
    return cube, table


@pytest.fixture
def attrs():
    return cut.global_attrs(seed=7, snapshot_time='2023-02-01T12:00:00Z', snapshot_seed=1)


def test_write_snapshot_nc_layout(tmp_path, attrs):
    netCDF4 = pytest.importorskip('netCDF4')
    cube, table = synthetic_cutouts(5, '2023-02-01 12:00')
    path = str(tmp_path / 'snapshots' / '20230201T12.nc')
    cut.write_snapshot_nc(path, cube, table, attrs)
    assert not (tmp_path / 'snapshots' / '20230201T12.nc.part').exists()
    ds = xr.open_dataset(path)
    assert ds['sst'].dims == ('cutout', 'y', 'x') and ds.sizes['cutout'] == 5
    assert set(cut.TABLE_COLUMNS) | {'sst'} <= set(ds.data_vars)
    assert np.array_equal(ds['sst'].values, cube)
    assert ds['UID'].dtype == np.int64 and np.array_equal(ds['UID'].values, table['UID'].values)
    assert (ds['time'].values == table['time'].values).all()
    assert ds.attrs['seed'] == 7 and 'Dan Whitt' in ds.attrs['credit']
    ds.close()
    with netCDF4.Dataset(path) as nc:
        assert nc.variables['sst'].chunking() == [5, 64, 64]      # capped at n
        assert nc.variables['sst'].filters()['zlib'] and nc.variables['sst'].filters()['complevel'] == 1
        assert nc.variables['UID'].dtype == np.int64


def test_concatenate_snapshots_round_trip(tmp_path, attrs):
    netCDF4 = pytest.importorskip('netCDF4')
    parts = [synthetic_cutouts(700, '2023-02-01 12:00', seed=1),
             synthetic_cutouts(0, '2023-05-01 00:00', seed=2),        # an empty snapshot
             synthetic_cutouts(600, '2023-08-01 12:00', seed=3)]
    paths = []
    for (cube, table), tag in zip(parts, ['20230201T12', '20230501T00', '20230801T12']):
        p = str(tmp_path / f'{tag}.nc')
        cut.write_snapshot_nc(p, cube, table, attrs)
        paths.append(p)
    out = str(tmp_path / 'all.nc')
    n = cut.concatenate_snapshots(paths[::-1], out, attrs={'pilot': 1})
    assert n == 1300
    ds = xr.open_dataset(out)
    assert ds['sst'].dims == ('cutout', 'y', 'x') and ds.sizes['cutout'] == 1300
    assert set(cut.TABLE_COLUMNS) | {'sst'} <= set(ds.data_vars)
    cube = np.concatenate([parts[0][0], parts[2][0]])
    table = pd.concat([parts[0][1], parts[2][1]], ignore_index=True)
    assert np.array_equal(ds['sst'].values, cube)               # sorted by file name
    assert np.array_equal(ds['UID'].values, table['UID'].values)
    assert ds['UID'].dtype == np.int64
    assert (ds['time'].values == table['time'].values).all()
    assert np.allclose(ds['lat'].values, table['lat'].values)
    assert ds['sst'].attrs['units'] == 'degC'
    a = ds.attrs
    assert a['n_cutouts'] == 1300 and a['n_snapshots'] == 3 and a['pilot'] == 1
    assert a['time_coverage_start'] == '2023-02-01T12:00:00Z'
    assert a['time_coverage_end'] == '2023-08-01T12:00:00Z'
    assert 'Dan Whitt' in a['credit'] and a['git_commit'] and a['code_version']
    assert a['km'] == 128. and a['npix'] == 64 and a['stride_hours'] == '9 10'
    assert 'snapshot_time' not in a and 'snapshot_seed' not in a
    ds.close()
    with netCDF4.Dataset(out) as nc:
        assert nc.variables['sst'].chunking() == [1024, 64, 64]
        f = nc.variables['sst'].filters()
        assert f['zlib'] and f['complevel'] == 1
