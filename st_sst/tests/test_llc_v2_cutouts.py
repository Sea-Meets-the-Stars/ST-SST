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
