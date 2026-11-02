""" Fixed-km SST cutouts from the LLC4320_v2 surface product.

Building blocks for the Gulf Stream cutout dataset specified in
``reports/extract_llc4320v2_gulf_stream.md``.  Everything here is a pure
function of numpy arrays, so it is tested on a synthetic grid without the
network; the per-snapshot reader, the NetCDF layer and the driver script sit
on top of these.

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
    while True:
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
