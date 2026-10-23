""" How many fully-ocean, non-overlapping 128 km tiles fit in a Gulf Stream box per snapshot?

Sets the time-sampling budget for the cutout dataset: with N tiles per snapshot and a
target of M cutouts, M / N snapshots are needed out of the 9,501 hourly ones.

Works in native index space on the (single) LLC face that holds the box.  Tile extents
are fixed-km, found by accumulating ``dxC``/``dyC`` as in the dbof pipeline
(``spatial_cutouts.extent_in_i/j``).  A tile is kept only if every cell is ocean
(``maskC`` true) and its centre lies inside the box.  Tiles are laid on a regular grid
with no jitter, so the count is representative of one random tiling, not a maximum.

Usage::

    conda run -n ocean14 python st_sst/scripts/gulf_stream_tile_count.py
    conda run -n ocean14 python st_sst/scripts/gulf_stream_tile_count.py --box 30 45 -80 -40
"""

import argparse

import numpy as np

from wrangler.ogcm import llc_v2_access as llc


def parser(options=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--box', nargs=4, type=float, default=[30., 45., -80., -50.],
                   metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'))
    p.add_argument('--km', type=float, default=128.)
    p.add_argument('--target', type=int, default=200_000, help='total cutouts wanted')
    p.add_argument('--n_jitter', type=int, default=5, help='random tiling offsets to average over')
    p.add_argument('--seed', type=int, default=12345)
    return p.parse_args() if options is None else p.parse_args(options)


def cells_for_km(spacing_km_1d: np.ndarray, start: int, km: float) -> int:
    """Number of cells from *start* (inclusive) whose cumulative spacing first reaches *km*."""
    cum = np.cumsum(spacing_km_1d[start:])
    n = int(np.searchsorted(cum, km)) + 1
    return n


def main(args):
    lat_min, lat_max, lon_min, lon_max = args.box
    boxes = llc.region_slices(lat_min, lat_max, lon_min, lon_max)
    if len(boxes) != 1:
        raise RuntimeError(f"box spans faces {sorted(boxes)}; this script assumes one face")
    (face, (js, isl)), = boxes.items()
    grid = llc.open_grid()
    g = grid.isel(face=face, j=js, i=isl)
    xc, yc = g['XC'].values, g['YC'].values
    ocean = g['maskC'].values.astype(bool)
    # spacings at cell centres along each axis (km); staggered axes sliced explicitly
    dx = grid['dxC'].isel(face=face, j=js, i_g=isl).values / 1000.   # along i
    dy = grid['dyC'].isel(face=face, j_g=js, i=isl).values / 1000.   # along j
    nj, ni = xc.shape
    inside = (yc >= lat_min) & (yc <= lat_max) & (xc >= lon_min) & (xc <= lon_max)
    print(f"face {face}, slice ({nj} x {ni}); box cells {inside.sum()}, ocean fraction in box "
          f"{ocean[inside].mean():.3f}")

    # Native cell counts for one tile across the box (spacing varies with latitude)
    dx_row = np.median(dx, axis=0)   # dx varies along i (latitude) on this rotated face
    dy_col = np.median(dy, axis=1)   # dy varies along j? (longitude) -- should be ~constant
    n_i = [cells_for_km(dx_row, i0, args.km) for i0 in range(0, ni - 100, 50)]
    n_j = [cells_for_km(dy_col, j0, args.km) for j0 in range(0, nj - 100, 50)]
    print(f"native cells per {args.km:.0f} km: i (latitude axis) {min(n_i)}..{max(n_i)}, "
          f"j (longitude axis) {min(n_j)}..{max(n_j)}  -> 64 px is "
          f"{'never' if min(n_i + n_j) >= 64 else 'SOMETIMES'} an upsample")

    rng = np.random.default_rng(args.seed)
    counts = []
    for k in range(args.n_jitter):
        # Regular tiling in index space with a random offset, tile sizes from local spacing
        j0 = int(rng.integers(0, 40))
        tiles = 0
        while True:
            dj = cells_for_km(dy_col, j0, args.km)
            if j0 + dj > nj:
                break
            i0 = int(rng.integers(0, 40))
            while True:
                di = cells_for_km(dx_row, i0, args.km)
                if i0 + di > ni:
                    break
                jc, ic = j0 + dj // 2, i0 + di // 2
                if inside[jc, ic] and ocean[j0:j0 + dj, i0:i0 + di].all():
                    tiles += 1
                i0 += di
            j0 += dj
        counts.append(tiles)
    counts = np.array(counts)
    print(f"fully-ocean non-overlapping tiles per snapshot: mean {counts.mean():.0f} "
          f"(min {counts.min()}, max {counts.max()}) over {args.n_jitter} random tilings")
    n_snap = args.target / counts.mean()
    print(f"{args.target:,} cutouts -> ~{n_snap:.0f} snapshots -> one every "
          f"{9501 / n_snap:.1f} h of the 9,501 available (every {24 * 395 / n_snap / 24:.2f} days)")


if __name__ == '__main__':
    main(parser())
