""" Native LLC4320 grid geometry in the Gulf Stream region.

Answers the planning questions for the 128 km x 128 km / 128 x 128 pixel cutouts:

* which of the 13 LLC faces cover the region, and whether a cutout can straddle a face;
* the native cell spacing (dxC, dyC) there, hence how many native cells span 128 km;
* the ocean fraction and how many non-overlapping 128 km boxes fit in the region.

Reads only the public ``s3://llc4320-v2/SURFACE/grid.zarr`` (anonymous) through
``wrangler.ogcm.llc_v2_access``; the coarse XC/YC cache makes the face lookup offline.

Usage::

    conda run -n ocean14 python st_sst/scripts/gulf_stream_grid_check.py
    conda run -n ocean14 python st_sst/scripts/gulf_stream_grid_check.py --box 30 45 -80 -50
"""

import argparse

import numpy as np

from wrangler.ogcm import llc_v2_access as llc

R_EARTH_KM = 6371.0


def parser(options=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--box', nargs=4, type=float, default=[30., 45., -80., -50.],
                   metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'),
                   help='Gulf Stream region (default 30..45N, 80..50W)')
    p.add_argument('--km', type=float, default=128., help='cutout side in km (default 128)')
    p.add_argument('--npix', type=int, default=128, help='cutout side in pixels (default 128)')
    return p.parse_args() if options is None else p.parse_args(options)


def main(args):
    lat_min, lat_max, lon_min, lon_max = args.box
    print(f"Region: {lat_min}..{lat_max} N, {lon_min}..{lon_max} E; "
          f"cutout {args.km} km / {args.npix} px -> {args.km / args.npix:.3f} km per pixel")

    # Which faces, and the index rectangles (inclusive, from the coarse grid)
    boxes = llc.region_slices(lat_min, lat_max, lon_min, lon_max)
    print(f"Faces intersecting the box: {sorted(boxes)}")

    grid = llc.open_grid()
    for face, (js, isl) in sorted(boxes.items()):
        g = grid.isel(face=face, j=js, i=isl)
        xc, yc = g['XC'].values, g['YC'].values
        inside = (yc >= lat_min) & (yc <= lat_max) & (xc >= lon_min) & (xc <= lon_max)
        if inside.sum() == 0:
            print(f"  face {face}: slice j={js.start}:{js.stop} i={isl.start}:{isl.stop} -- no cells inside")
            continue
        # Cell spacing: dxC sits on west faces (face, j, i_g), dyC on south faces (face, j_g, i).
        # isel(j=, i=) leaves the staggered axes untouched, so slice those explicitly with the
        # same index ranges (the half-cell offset is irrelevant at this precision).
        dx = grid['dxC'].isel(face=face, j=js, i_g=isl).values / 1000.
        dy = grid['dyC'].isel(face=face, j_g=js, i=isl).values / 1000.
        dx_in, dy_in = dx[inside], dy[inside]
        depth = g['Depth'].values[inside]
        ocean = np.isfinite(depth) & (depth > 0)
        jj, ii = np.nonzero(inside)
        print(f"  face {face}: slice j={js.start}:{js.stop} i={isl.start}:{isl.stop}; "
              f"cells inside box: {inside.sum()} (j {jj.min()}..{jj.max()}, i {ii.min()}..{ii.max()} "
              f"within the slice); ocean fraction {ocean.mean():.2f}")
        print(f"    dxC [km]: min {dx_in.min():.2f}  median {np.median(dx_in):.2f}  max {dx_in.max():.2f}")
        print(f"    dyC [km]: min {dy_in.min():.2f}  median {np.median(dy_in):.2f}  max {dy_in.max():.2f}")
        n_x = args.km / np.median(dx_in)
        n_y = args.km / np.median(dy_in)
        print(f"    native cells per {args.km:.0f} km: ~{n_x:.0f} (i) x ~{n_y:.0f} (j); "
              f"{args.npix} px needs a {args.npix / n_x:.2f}x (i) / {args.npix / n_y:.2f}x (j) "
              f"{'UPSAMPLE' if args.npix > min(n_x, n_y) else 'downsample'}")
        # Orientation: which index tracks latitude on this face?
        dlat_dj = np.nanmedian(np.diff(yc, axis=0)); dlat_di = np.nanmedian(np.diff(yc, axis=1))
        dlon_dj = np.nanmedian(np.diff(xc, axis=0)); dlon_di = np.nanmedian(np.diff(xc, axis=1))
        print(f"    orientation: dYC/dj={dlat_dj:+.4f} dYC/di={dlat_di:+.4f} "
              f"dXC/dj={dlon_dj:+.4f} dXC/di={dlon_di:+.4f} deg/cell -> "
              f"{'j ~ latitude, i ~ longitude (unrotated)' if abs(dlat_dj) > abs(dlat_di) else 'ROTATED: i ~ latitude, j ~ longitude'}")
        # Full extent of this face, for reference
        fx, fy = grid['XC'].isel(face=face), grid['YC'].isel(face=face)
        c = [(float(fy.isel(j=jj, i=ii).values), float(fx.isel(j=jj, i=ii).values))
             for jj, ii in [(0, 0), (0, -1), (-1, 0), (-1, -1)]]
        print(f"    face {face} corners (lat, lon): j0i0={c[0][0]:.1f},{c[0][1]:.1f}  j0iN={c[1][0]:.1f},{c[1][1]:.1f}  "
              f"jNi0={c[2][0]:.1f},{c[2][1]:.1f}  jNiN={c[3][0]:.1f},{c[3][1]:.1f}")
        # Box edges in index space: how close is the region to the face edge?
        FS = grid.sizes['j']
        jmin, jmax = js.start + jj.min(), js.start + jj.max()
        imin, imax = isl.start + ii.min(), isl.start + ii.max()
        print(f"    full-face index extent: j {jmin}..{jmax} of {FS}, i {imin}..{imax} of {FS}")

    # Rough count of non-overlapping cutouts per snapshot
    km_lat = (lat_max - lat_min) * 2 * np.pi * R_EARTH_KM / 360.
    km_lon = (lon_max - lon_min) * 2 * np.pi * R_EARTH_KM / 360. * np.cos(np.radians(0.5 * (lat_min + lat_max)))
    n_tiles = int(km_lat // args.km) * int(km_lon // args.km)
    print(f"Region is ~{km_lat:.0f} km (N-S) x ~{km_lon:.0f} km (E-W): "
          f"~{n_tiles} non-overlapping {args.km:.0f} km tiles per snapshot before the land mask")


if __name__ == '__main__':
    main(parser())
