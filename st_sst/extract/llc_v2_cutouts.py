""" Fixed-km SST cutouts from the LLC4320_v2 surface product.

Building blocks for the Gulf Stream cutout dataset specified in
``reports/extract_llc4320v2_gulf_stream.md``.  Everything here is a pure
function of numpy arrays, so it is tested on a synthetic grid without the
network.  On top of them sit the per-snapshot extractor (`extract_snapshot`,
reading through ``wrangler.ogcm.llc_v2_access``) and the NetCDF layer
(`write_snapshot_nc`, `concatenate_snapshots`); the driver is
``st_sst/scripts/extract_gulf_stream.py``.

Geometry conventions
--------------------
A face of the LLC grid is indexed ``[j, i]``.  Off the Arctic cap the grid is
lat-lon, but on the rotated faces (7-12, which hold the Gulf Stream) ``i`` runs
along latitude and ``j`` along longitude.  `face_orientation` works out, from
``XC``/``YC``, the transpose/flip that makes an image north-up and east-right,
and `orient_north_up` applies it.

Cutouts have a fixed physical size: from a tile origin ``(j0, i0)`` the local
spacings are accumulated until *km* is reached (`cells_for_km`,
`tile_extent`), after dbof's ``spatial_cutouts.extent_in_i`` / ``extent_in_j``
(Sea-Meets-the-Stars/llc4320-native-grid-preprocessing).  The native tile is
then oriented and area-averaged to ``npix x npix`` with
``skimage.transform.resize_local_mean`` (`make_cutout`); upsampling is refused.

Spacing arrays: ``dxC`` is the spacing along ``i`` and lives on
``(face, j, i_g)``; ``dyC`` is the spacing along ``j`` and lives on
``(face, j_g, i)``.  Slice ``i_g``/``j_g`` explicitly when you pull them from
``grid.zarr`` -- ``isel(j=, i=)`` leaves the staggered axes alone.
"""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Defaults from the plan document
DEFAULT_BOX = (30., 45., -80., -50.)        # lat_min, lat_max, lon_min, lon_max
DEFAULT_KM = 128.
DEFAULT_NPIX = 64
DEFAULT_STEPS = (9, 10)                     # alternating hours between snapshots
PILOT_HOURS = ('2023-02-01 12:00', '2023-08-01 12:00')
RECORD_START = '2023-01-01 01:00'
RECORD_END = '2024-01-31 23:00'

STAT_KEYS = ('mean', 'std', 'min', 'max', 'p10', 'p90',
             'grad_mean', 'grad_p90', 'grad_max')


# ---------------------------------------------------------------------------
# Fixed-km extents
# ---------------------------------------------------------------------------

def cells_for_km(spacing_km, start: int, km: float, min_cells: int = 1):
    """Cells from *start* (inclusive) whose cumulative spacing first reaches *km*.

    Args:
        spacing_km (array): 1-D cell spacings along the axis, in km.
        start (int): first cell of the run.
        km (float): physical length to cover.
        min_cells (int, optional): never return fewer cells than this (set it to
            the output pixel count so the cutout is never upsampled).

    Returns:
        tuple or None: ``(n_cells, realized_km)``; None if the run would leave
            the array.
    """
    s = np.asarray(spacing_km, dtype=np.float64)[start:]
    if s.size == 0:
        return None
    cum = np.cumsum(s)
    n = int(np.searchsorted(cum, km, side='left')) + 1
    n = max(n, int(min_cells))
    if n > s.size:
        return None
    return n, float(cum[n - 1])


def tile_extent(dx_1d, dy_1d, j0: int, i0: int, km: float, min_cells: int = 1):
    """Native extent of a *km* x *km* tile whose origin is ``(j0, i0)``.

    After dbof's ``extent_in_i`` / ``extent_in_j``, but measured from the tile
    origin rather than its centre so that tiles can be laid edge to edge.

    Args:
        dx_1d (array): spacing along ``i`` (km) on the tile's row ``j0``.
        dy_1d (array): spacing along ``j`` (km) on the tile's column ``i0``.
        j0, i0 (int): tile origin.
        km (float): tile side.
        min_cells (int, optional): see `cells_for_km`.

    Returns:
        tuple or None: ``(nj, ni, km_j, km_i)`` -- cells and realized km along
            ``j`` and along ``i``; None if the tile runs off the array.
    """
    ei = cells_for_km(dx_1d, i0, km, min_cells)
    ej = cells_for_km(dy_1d, j0, km, min_cells)
    if ei is None or ej is None:
        return None
    ni, km_i = ei
    nj, km_j = ej
    return nj, ni, km_j, km_i


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------

def _wrap_lon(d):
    return (np.asarray(d) + 180.) % 360. - 180.


def face_orientation(xc, yc) -> dict:
    """Transpose/flip that makes a ``[j, i]`` array north-up and east-right.

    Decided from the median steps of ``YC``/``XC`` along each axis, so it holds
    for any lat-lon face (not the Arctic cap).

    Returns:
        dict: ``{'transpose': bool, 'flip_rows': bool, 'flip_cols': bool}``,
            to be applied in that order by `orient_north_up`.
    """
    xc, yc = np.asarray(xc, dtype=np.float64), np.asarray(yc, dtype=np.float64)
    dlat_dj = np.nanmedian(np.diff(yc, axis=0))
    dlat_di = np.nanmedian(np.diff(yc, axis=1))
    dlon_dj = np.nanmedian(_wrap_lon(np.diff(xc, axis=0)))
    dlon_di = np.nanmedian(_wrap_lon(np.diff(xc, axis=1)))
    transpose = abs(dlat_di) > abs(dlat_dj)
    lat_step = dlat_di if transpose else dlat_dj      # along the rows after orienting
    lon_step = dlon_dj if transpose else dlon_di      # along the columns after orienting
    return {'transpose': bool(transpose),
            'flip_rows': bool(lat_step > 0),          # want latitude decreasing down the rows
            'flip_cols': bool(lon_step < 0)}          # want longitude increasing along columns


def orient_north_up(arr, orient: dict):
    """Apply a `face_orientation` result to a 2-D ``[j, i]`` array."""
    out = np.asarray(arr)
    if orient['transpose']:
        out = out.T
    if orient['flip_rows']:
        out = out[::-1, :]
    if orient['flip_cols']:
        out = out[:, ::-1]
    return out


def oriented_km(km_j: float, km_i: float, orient: dict):
    """``(km_w, km_h)`` of the oriented image from the realized km along j and i."""
    return (km_j, km_i) if orient['transpose'] else (km_i, km_j)


# ---------------------------------------------------------------------------
# Cutouts
# ---------------------------------------------------------------------------

def make_cutout(theta, j0: int, i0: int, nj: int, ni: int, npix: int, orient: dict):
    """Slice a native tile, orient it north-up, area-average to ``npix x npix``.

    Args:
        theta (array): the face's field, ``[j, i]``.
        j0, i0, nj, ni (int): tile origin and native extent (see `tile_extent`).
        npix (int): output side in pixels.
        orient (dict): from `face_orientation`.

    Returns:
        np.ndarray: ``(npix, npix)`` float32.

    Raises:
        ValueError: if the tile is not fully inside *theta* or would need upsampling.
    """
    from skimage.transform import resize_local_mean

    tile = np.asarray(theta)[j0:j0 + nj, i0:i0 + ni]
    if tile.shape != (nj, ni):
        raise ValueError(f"tile ({j0},{i0}) of {nj}x{ni} is not inside the {theta.shape} field")
    img = orient_north_up(tile, orient)
    if min(img.shape) < npix:
        raise ValueError(f"refusing to upsample a {img.shape} tile to {npix}x{npix}")
    out = resize_local_mean(np.ascontiguousarray(img, dtype=np.float32), (npix, npix))
    return np.ascontiguousarray(out, dtype=np.float32)


def tile_corners(xc, yc, j0: int, i0: int, nj: int, ni: int, orient: dict) -> dict:
    """Centre and corner coordinates (cell centres) of a tile, in map orientation.

    Returns:
        dict: ``lat``, ``lon`` (centre pixel), ``lat_sw``, ``lon_sw`` (bottom-left
            pixel), ``lat_ne``, ``lon_ne`` (top-right pixel).
    """
    yo = orient_north_up(np.asarray(yc)[j0:j0 + nj, i0:i0 + ni], orient)
    xo = orient_north_up(np.asarray(xc)[j0:j0 + nj, i0:i0 + ni], orient)
    nr, nc = yo.shape
    return {'lat': float(yo[nr // 2, nc // 2]), 'lon': float(xo[nr // 2, nc // 2]),
            'lat_sw': float(yo[-1, 0]), 'lon_sw': float(xo[-1, 0]),
            'lat_ne': float(yo[0, -1]), 'lon_ne': float(xo[0, -1])}


def tile_snapshot(xc, yc, dx, dy, ocean, box, km: float, rng, min_cells: int = 1) -> list:
    """Non-overlapping fixed-km tiling of a face slice with a random offset.

    Tiles are laid in strips of constant ``i`` origin; within a strip they are
    laid edge to edge along ``j``, each with its own extent from the local
    spacing, and the strip advances by the widest tile it holds, so tiles never
    overlap.  A tile is kept when its centre cell lies in *box* and every cell
    is *ocean*.

    Args:
        xc, yc (array): ``[j, i]`` longitude / latitude of the slice.
        dx, dy (array): ``[j, i]`` spacing along ``i`` / along ``j``, in km.
        ocean (bool array): ``[j, i]``; True where a cell may be used.
        box (tuple): ``(lat_min, lat_max, lon_min, lon_max)``.
        km (float): tile side.
        rng (numpy.random.Generator): for the tiling offset.
        min_cells (int, optional): see `cells_for_km`.

    Returns:
        list of dict: ``j0, i0, nj, ni, km_j, km_i, lat, lon`` per kept tile.
    """
    lat_min, lat_max, lon_min, lon_max = box
    xc, yc, ocean = np.asarray(xc), np.asarray(yc), np.asarray(ocean, dtype=bool)
    dx, dy = np.asarray(dx, dtype=np.float64), np.asarray(dy, dtype=np.float64)
    nj_typ = max(1, int(np.ceil(km / np.nanmedian(dy))))
    ni_typ = max(1, int(np.ceil(km / np.nanmedian(dx))))
    j_off = int(rng.integers(0, nj_typ))
    i_off = int(rng.integers(0, ni_typ))

    tiles = []
    i0 = i_off
    while i0 < dx.shape[1]:          # a strip ending exactly on the edge leaves i0 == width
        j0, max_ni, any_i = j_off, 0, False
        while True:
            ej = cells_for_km(dy[:, i0], j0, km, min_cells)
            if ej is None:
                break
            nj, km_j = ej
            ei = cells_for_km(dx[j0, :], i0, km, min_cells)
            if ei is not None:
                any_i = True
                ni, km_i = ei
                max_ni = max(max_ni, ni)
                cj, ci = j0 + nj // 2, i0 + ni // 2
                lat, lon = float(yc[cj, ci]), float(xc[cj, ci])
                if (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
                        and ocean[j0:j0 + nj, i0:i0 + ni].all()):
                    tiles.append({'j0': j0, 'i0': i0, 'nj': nj, 'ni': ni,
                                  'km_j': km_j, 'km_i': km_i, 'lat': lat, 'lon': lon})
            j0 += nj
        if not any_i:
            break
        i0 += max_ni
    return tiles


# ---------------------------------------------------------------------------
# Statistics and identifiers
# ---------------------------------------------------------------------------

def cutout_stats(img, km_w: float, km_h: float) -> dict:
    """Per-cutout statistics on the final image.

    ``mean, std, min, max, p10, p90`` of the values (the percentiles via
    ``wrangler.preproc.meta.stats`` when available) and ``grad_mean, grad_p90,
    grad_max`` of :math:`|\\nabla T|` in degC/km, from ``np.gradient`` with the
    realized pixel size.
    """
    img = np.asarray(img, dtype=np.float64)
    try:
        from wrangler.preproc import meta as wr_meta
        d = wr_meta.stats(img)
        mn, mx, p10, p90, mu = d['mn'], d['mx'], d['p10'], d['p90'], d['mu']
    except ImportError:                                   # pragma: no cover
        ff = np.sort(img.ravel())
        mn, mx = ff[0], ff[-1]
        p10, p90 = ff[int(0.1 * ff.size)], ff[int(0.9 * ff.size)]
        mu = ff.mean()
    px_h, px_w = km_h / img.shape[0], km_w / img.shape[1]
    grad = np.hypot(np.gradient(img, px_w, axis=1), np.gradient(img, px_h, axis=0))
    return {'mean': float(mu), 'std': float(img.std()), 'min': float(mn), 'max': float(mx),
            'p10': float(p10), 'p90': float(p90),
            'grad_mean': float(grad.mean()), 'grad_p90': float(np.percentile(grad, 90)),
            'grad_max': float(grad.max())}


def _to_ns(time) -> np.ndarray:
    """Any time-like (scalar or array) -> int64 nanoseconds since the epoch, UTC."""
    if np.ndim(time) == 0:
        ts = pd.to_datetime(time, utc=True)
        vals = np.array([ts.tz_localize(None).to_datetime64()])
    else:                                   # list, ndarray, Series, Index
        idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(time), utc=True))
        vals = idx.tz_localize(None).values
    return vals.astype('datetime64[ns]').astype(np.int64)


def make_uid(time, lat, lon):
    """Unique identifier from time and centre position, the wrangler/ulmo formula.

    Re-implements ``wrangler.ogcm.llc.add_uid``: the time in units of 10 ms with
    its last five digits dropped (so 1000 s resolution), then ``(lat+90)*1000``
    and ``(lon+180)*1000`` rounded, concatenated as decimal digits into an
    int64.  The time is forced to nanoseconds first; the original relies on
    the datetime64 unit and would silently lose the hour on microsecond
    datetimes (the pandas 3 default).

    Args:
        time: scalar or array of anything ``pandas.to_datetime`` accepts.
        lat, lon: scalar or array, degrees.

    Returns:
        np.int64 or np.ndarray of int64.
    """
    scalar = np.ndim(time) == 0 and np.ndim(lat) == 0
    t = _to_ns(time) // 10_000_000
    la = np.round((np.atleast_1d(np.asarray(lat, dtype=np.float64)) + 90.) * 1000).astype(np.int64)
    lo = np.round((np.atleast_1d(np.asarray(lon, dtype=np.float64)) + 180.) * 1000).astype(np.int64)
    t, la, lo = np.broadcast_arrays(t, la, lo)
    uid = np.array([np.int64(f"{str(tt)[:-5]}{a}{o}") for tt, a, o in zip(t, la, lo)],
                   dtype=np.int64)
    return uid[0] if scalar else uid


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def parse_utc(value) -> datetime:
    """datetime (naive = UTC) or ISO string ('2023-01-01 01:00', '2023-01-01T01') -> aware UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    t = datetime.fromisoformat(str(value).strip())
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def snapshot_hours(start=RECORD_START, end=RECORD_END, steps=DEFAULT_STEPS) -> list:
    """Snapshot times from *start* to *end* (inclusive) with cyclic hour *steps*.

    The default ``(9, 10)`` gives two snapshots per 19 h; 19 is coprime with 24,
    so every UTC hour is visited equally often over the record.
    """
    t, t1 = parse_utc(start), parse_utc(end)
    if t1 < t:
        raise ValueError(f"end ({t1}) precedes start ({t})")
    out, k = [], 0
    while t <= t1:
        out.append(t)
        t += timedelta(hours=int(steps[k % len(steps)]))
        k += 1
    return out


# ---------------------------------------------------------------------------
# Per-snapshot extraction
# ---------------------------------------------------------------------------

#: Columns of the per-cutout table, in output order.
TABLE_COLUMNS = ('UID', 'time', 'lat', 'lon', 'face', 'j0', 'i0', 'nj', 'ni',
                 'km_w', 'km_h', 'lat_sw', 'lon_sw', 'lat_ne', 'lon_ne', 'min_depth') + STAT_KEYS

#: Fixed time encoding, so per-snapshot files concatenate without re-encoding.
TIME_UNITS = 'hours since 2023-01-01 00:00:00'

CREDIT = ("The LLC4320_v2 simulation was produced and is owned by Dan Whitt (NASA Ames), "
          "who agreed to its public copy; please credit him in any publication that uses "
          "these data, and contact him regarding the simulation itself. The public copy was "
          "assembled by J. Xavier Prochaska with the wrangler package.")


def snapshot_seed(seed: int, date) -> int:
    """Deterministic per-snapshot seed from a base *seed* and the snapshot hour.

    Mixes the two through ``numpy.random.SeedSequence`` so neighbouring hours
    get unrelated tilings, and a rerun of any snapshot reproduces its own.
    """
    hours = int((parse_utc(date) - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
                // 3600)
    return int(np.random.SeedSequence([int(seed), hours]).generate_state(1, dtype=np.uint32)[0])


def snapshot_tag(date) -> str:
    """``'2023-02-01 12:00'`` -> ``'20230201T12'`` (the store and intermediate-file stem)."""
    return parse_utc(date).strftime('%Y%m%dT%H')


def _face_arrays(grid, face: int, js: slice, isl: slice) -> dict:
    """Grid fields of one face rectangle as numpy, staggered axes sliced explicitly.

    ``dxC`` lives on ``(face, j, i_g)`` and ``dyC`` on ``(face, j_g, i)``, so
    ``isel(j=, i=)`` alone would leave their staggered axis whole.
    """
    g = grid.isel(face=face, j=js, i=isl)
    return {'xc': g['XC'].values.astype(np.float64),
            'yc': g['YC'].values.astype(np.float64),
            'mask': g['maskC'].values.astype(bool),
            'depth': g['Depth'].values,
            'dx': grid['dxC'].isel(face=face, j=js, i_g=isl).values.astype(np.float64) / 1000.,
            'dy': grid['dyC'].isel(face=face, j_g=js, i=isl).values.astype(np.float64) / 1000.}


def extract_snapshot(date, grid=None, box=DEFAULT_BOX, km: float = DEFAULT_KM,
                     npix: int = DEFAULT_NPIX, seed: int = 0, store=None, slices=None,
                     root: str = None, variable: str = 'Theta'):
    """All fully-ocean fixed-km cutouts of one hourly snapshot.

    Reads only the face/index rectangle(s) that `region_slices` gives for *box*,
    tiles each with `tile_snapshot` (offset drawn from `snapshot_seed`), builds
    each cutout with `make_cutout`, and fills the per-cutout table.

    Args:
        date: the snapshot hour (UTC).
        grid (xarray.Dataset, optional): an open ``grid.zarr``; opened if None.
        box (tuple): ``(lat_min, lat_max, lon_min, lon_max)``.
        km (float): cutout side.
        npix (int): output side in pixels (also the minimum native cell count).
        seed (int): base seed; the tiling uses ``snapshot_seed(seed, date)``.
        store (xarray.Dataset, optional): the already-open hourly store (tests).
        slices (dict, optional): ``{face: (slice_j, slice_i)}``; from
            `region_slices` if None.
        root (str, optional): product root for the wrangler readers.
        variable (str): field to cut, default ``'Theta'``.

    Returns:
        tuple: ``(cube, table)`` -- ``(n, npix, npix)`` float32 and a
            ``pandas.DataFrame`` with `TABLE_COLUMNS`.

    Raises:
        Whatever ``open_store`` raises for a missing hour.
    """
    from wrangler.ogcm import llc_v2_access as llc
    kw = {} if root is None else {'root': root}
    lat_min, lat_max, lon_min, lon_max = box
    if slices is None:
        slices = llc.region_slices(lat_min, lat_max, lon_min, lon_max, **kw)
    if grid is None:
        grid = llc.open_grid(**kw)
    if store is None:
        store = llc.open_store(date, **kw)
    rng = np.random.default_rng(snapshot_seed(seed, date))
    t64 = np.datetime64(parse_utc(date).replace(tzinfo=None), 'ns')

    images, rows = [], []
    for face, (js, isl) in sorted(slices.items()):
        a = _face_arrays(grid, face, js, isl)
        theta = store[variable].isel(face=face, j=js, i=isl).values
        ocean = a['mask'] & np.isfinite(theta)
        orient = face_orientation(a['xc'], a['yc'])
        tiles = tile_snapshot(a['xc'], a['yc'], a['dx'], a['dy'], ocean, box, km, rng,
                              min_cells=npix)
        for t in tiles:
            j0, i0, nj, ni = t['j0'], t['i0'], t['nj'], t['ni']
            img = make_cutout(theta, j0, i0, nj, ni, npix, orient)
            km_w, km_h = oriented_km(t['km_j'], t['km_i'], orient)
            c = tile_corners(a['xc'], a['yc'], j0, i0, nj, ni, orient)
            row = {'time': t64, 'lat': t['lat'], 'lon': t['lon'], 'face': int(face),
                   'j0': int(js.start or 0) + j0, 'i0': int(isl.start or 0) + i0,
                   'nj': nj, 'ni': ni, 'km_w': km_w, 'km_h': km_h,
                   'lat_sw': c['lat_sw'], 'lon_sw': c['lon_sw'],
                   'lat_ne': c['lat_ne'], 'lon_ne': c['lon_ne'],
                   'min_depth': float(np.min(a['depth'][j0:j0 + nj, i0:i0 + ni]))}
            row.update(cutout_stats(img, km_w, km_h))
            images.append(img)
            rows.append(row)

    cube = (np.stack(images).astype(np.float32) if images
            else np.zeros((0, npix, npix), dtype=np.float32))
    table = pd.DataFrame(rows, columns=[c for c in TABLE_COLUMNS if c != 'UID'])
    table['time'] = pd.to_datetime(table['time']).astype('datetime64[ns]')
    table.insert(0, 'UID', make_uid(table['time'], table['lat'].values, table['lon'].values)
                 if len(table) else np.array([], dtype=np.int64))
    for col in ('face', 'j0', 'i0', 'nj', 'ni'):
        table[col] = table[col].astype(np.int32)
    logger.info("%s: %d cutouts", snapshot_tag(date), len(table))
    return cube, table


# ---------------------------------------------------------------------------
# NetCDF layer
# ---------------------------------------------------------------------------

_VAR_ATTRS = {
    'sst': {'long_name': 'sea surface temperature (Theta, top cell)', 'units': 'degC'},
    'UID': {'long_name': 'unique cutout identifier (wrangler datetime-lat-lon formula)'},
    'time': {'standard_name': 'time', 'long_name': 'UTC time of the snapshot'},
    'lat': {'long_name': 'latitude of the tile centre cell', 'units': 'degrees_north'},
    'lon': {'long_name': 'longitude of the tile centre cell', 'units': 'degrees_east'},
    'face': {'long_name': 'LLC face (0-12)'},
    'j0': {'long_name': 'native j of the tile origin on the face'},
    'i0': {'long_name': 'native i of the tile origin on the face'},
    'nj': {'long_name': 'native cells of the tile along j'},
    'ni': {'long_name': 'native cells of the tile along i'},
    'km_w': {'long_name': 'realized width (west-east) of the cutout', 'units': 'km'},
    'km_h': {'long_name': 'realized height (south-north) of the cutout', 'units': 'km'},
    'lat_sw': {'long_name': 'latitude of the south-west (bottom-left) native cell',
               'units': 'degrees_north'},
    'lon_sw': {'long_name': 'longitude of the south-west (bottom-left) native cell',
               'units': 'degrees_east'},
    'lat_ne': {'long_name': 'latitude of the north-east (top-right) native cell',
               'units': 'degrees_north'},
    'lon_ne': {'long_name': 'longitude of the north-east (top-right) native cell',
               'units': 'degrees_east'},
    'min_depth': {'long_name': 'minimum ocean depth over the native tile', 'units': 'm'},
    'mean': {'long_name': 'mean SST of the cutout', 'units': 'degC'},
    'std': {'long_name': 'standard deviation of SST of the cutout', 'units': 'degC'},
    'min': {'long_name': 'minimum SST of the cutout', 'units': 'degC'},
    'max': {'long_name': 'maximum SST of the cutout', 'units': 'degC'},
    'p10': {'long_name': '10th percentile of SST of the cutout', 'units': 'degC'},
    'p90': {'long_name': '90th percentile of SST of the cutout', 'units': 'degC'},
    'grad_mean': {'long_name': 'mean |grad SST| of the cutout', 'units': 'degC/km'},
    'grad_p90': {'long_name': '90th percentile |grad SST| of the cutout', 'units': 'degC/km'},
    'grad_max': {'long_name': 'maximum |grad SST| of the cutout', 'units': 'degC/km'},
}


def git_commit(path: str = None) -> str:
    """HEAD commit of the repository holding *path* (read-only), or ``'unknown'``."""
    import subprocess
    import os
    path = path or os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=path, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'],
                               cwd=path, capture_output=True, text=True, check=True).stdout
        return sha + ('-dirty' if dirty.strip() else '')
    except Exception:                                     # pragma: no cover
        return 'unknown'


def global_attrs(box=DEFAULT_BOX, km: float = DEFAULT_KM, npix: int = DEFAULT_NPIX,
                 steps=DEFAULT_STEPS, seed: int = 0, root: str = None, **extra) -> dict:
    """Global attributes of a cutout file (plan, design item 6)."""
    import st_sst
    from wrangler.ogcm import llc_v2_access as llc
    attrs = {
        'title': 'LLC4320_v2 Gulf Stream SST cutouts',
        'source': 'LLC4320_v2 hourly surface product (MITgcm LLC4320), field Theta',
        'source_bucket': root or llc.DEFAULT_ROOT,
        'source_endpoint': llc.DEFAULT_ENDPOINT,
        'box_lat_min': float(box[0]), 'box_lat_max': float(box[1]),
        'box_lon_min': float(box[2]), 'box_lon_max': float(box[3]),
        'km': float(km), 'npix': int(npix),
        'pixel_km_nominal': float(km) / int(npix),
        'resampling': ('fixed-km native tile (dxC/dyC accumulated from the origin), '
                       'transposed/flipped to north-up east-right, area-averaged with '
                       'skimage.transform.resize_local_mean; never upsampled'),
        'sampling': ('per snapshot: non-overlapping tiling in native index space with a '
                     'seeded random offset; tiles kept when the centre is in the box and '
                     'every cell is ocean (maskC and finite Theta)'),
        'stride_hours': ' '.join(str(int(s)) for s in steps),
        'seed': int(seed),
        'seed_rule': 'per-snapshot seed = snapshot_seed(seed, hour) via numpy SeedSequence',
        'code': 'st_sst.extract.llc_v2_cutouts',
        'code_version': st_sst.__version__,
        'git_commit': git_commit(),
        'credit': CREDIT,
        'creator': 'J. Xavier Prochaska (ST-SST)',
        'date_created': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'Conventions': 'CF-1.8',
    }
    attrs.update(extra)
    return attrs


def _encoding(n: int, npix: int) -> dict:
    """NetCDF encoding: sst chunks ``(1024, npix, npix)`` (capped at *n*), zlib 1 everywhere."""
    enc = {v: {'zlib': True, 'complevel': 1} for v in TABLE_COLUMNS}
    enc['sst'] = {'zlib': True, 'complevel': 1, 'dtype': 'float32'}
    if n > 0:
        enc['sst']['chunksizes'] = (min(1024, n), npix, npix)
    enc['time'].update({'units': TIME_UNITS, 'calendar': 'proleptic_gregorian',
                        'dtype': 'float64'})
    enc['UID']['dtype'] = 'int64'
    return enc


def to_dataset(cube, table, attrs: dict = None):
    """``(cube, table)`` -> ``xarray.Dataset`` with dims ``(cutout, y, x)``."""
    import xarray as xr
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3 or cube.shape[0] != len(table):
        raise ValueError(f"cube {cube.shape} does not match a table of {len(table)} rows")
    data = {'sst': (('cutout', 'y', 'x'), cube)}
    for col in TABLE_COLUMNS:
        vals = table[col].values
        if col == 'UID':
            vals = vals.astype(np.int64)
        elif col == 'time':
            vals = vals.astype('datetime64[ns]')
        data[col] = ('cutout', vals)
    ds = xr.Dataset(data, attrs=dict(attrs or {}))
    for v, a in _VAR_ATTRS.items():
        ds[v].attrs.update(a)
    ds['y'] = ('y', np.arange(cube.shape[1], dtype=np.int32),
               {'long_name': 'pixel row (north at y=0)'})
    ds['x'] = ('x', np.arange(cube.shape[2], dtype=np.int32),
               {'long_name': 'pixel column (west at x=0)'})
    return ds


def write_snapshot_nc(path: str, cube, table, attrs: dict = None) -> str:
    """Write one snapshot's cutouts to NetCDF4 (the plan's layout), atomically.

    Written to ``path + '.part'`` and renamed, so a file that exists is
    complete and the driver can resume by skipping it.
    """
    import os
    ds = to_dataset(cube, table, attrs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.part'
    ds.to_netcdf(tmp, format='NETCDF4', engine='netcdf4',
                 encoding=_encoding(len(table), cube.shape[1]))
    os.replace(tmp, path)
    return path


_SNAPSHOT_ONLY_ATTRS = ('snapshot_time', 'snapshot_seed', 'n_cutouts', 'date_created')


def concatenate_snapshots(paths, outfile: str, attrs: dict = None) -> int:
    """Concatenate per-snapshot files along ``cutout`` into one NetCDF4 file.

    Streams one snapshot at a time through ``netCDF4`` (the full dataset is a
    few GB), keeping each variable's dtype, attributes and encoded values (all
    files share `TIME_UNITS`), with ``sst`` chunked ``(1024, y, x)`` and zlib 1.
    Global attributes come from the first file, minus the per-snapshot ones,
    plus ``n_cutouts``, ``n_snapshots``, the time coverage and *attrs*.

    Returns:
        int: number of cutouts written.
    """
    import os
    import netCDF4

    paths = sorted(paths)
    if not paths:
        raise ValueError("no snapshot files to concatenate")
    sizes = []
    for p in paths:
        with netCDF4.Dataset(p) as src:
            sizes.append(len(src.dimensions['cutout']))
    n = int(sum(sizes))

    tmp = outfile + '.part'
    with netCDF4.Dataset(paths[0]) as first, netCDF4.Dataset(tmp, 'w', format='NETCDF4') as dst:
        ny, nx = len(first.dimensions['y']), len(first.dimensions['x'])
        dst.createDimension('cutout', n)
        dst.createDimension('y', ny)
        dst.createDimension('x', nx)
        out_vars = {}
        for name, v in first.variables.items():
            kw = {'zlib': True, 'complevel': 1}
            if name == 'sst' and n > 0:
                kw['chunksizes'] = (min(1024, n), ny, nx)
            fill = v.getncattr('_FillValue') if '_FillValue' in v.ncattrs() else None
            ov = dst.createVariable(name, v.dtype, v.dimensions, fill_value=fill, **kw)
            ov.setncatts({a: v.getncattr(a) for a in v.ncattrs() if a != '_FillValue'})
            if v.dimensions in (('y',), ('x',)):
                ov[:] = v[:]
            out_vars[name] = ov
        gatts = {a: first.getncattr(a) for a in first.ncattrs()
                 if a not in _SNAPSHOT_ONLY_ATTRS}

        k = 0
        tmin, tmax = None, None
        for p, m in zip(paths, sizes):
            if m == 0:
                continue
            with netCDF4.Dataset(p) as src:
                src.set_auto_maskandscale(False)
                for name, ov in out_vars.items():
                    if 'cutout' in src.variables[name].dimensions:
                        ov.set_auto_maskandscale(False)
                        ov[k:k + m] = src.variables[name][:]
                tv = src.variables['time'][:]
                tmin = tv.min() if tmin is None else min(tmin, tv.min())
                tmax = tv.max() if tmax is None else max(tmax, tv.max())
            k += m

        tunits = first.variables['time'].getncattr('units')
        cal = first.variables['time'].getncattr('calendar') \
            if 'calendar' in first.variables['time'].ncattrs() else 'standard'
        gatts.update({'n_cutouts': n, 'n_snapshots': len(paths),
                      'snapshot_files': ' '.join(os.path.basename(p) for p in paths)
                      if len(paths) <= 50 else f'{len(paths)} files',
                      'date_created': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')})
        if tmin is not None:
            fmt = '%Y-%m-%dT%H:%M:%SZ'
            gatts['time_coverage_start'] = netCDF4.num2date(tmin, tunits, cal).strftime(fmt)
            gatts['time_coverage_end'] = netCDF4.num2date(tmax, tunits, cal).strftime(fmt)
        gatts.update(attrs or {})
        dst.setncatts(gatts)
    os.replace(tmp, outfile)
    logger.info("wrote %s: %d cutouts from %d snapshots", outfile, n, len(paths))
    return n
