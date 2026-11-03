""" Extract fixed-km SST cutouts of the Gulf Stream from LLC4320_v2.

Implements the plan in ``reports/extract_llc4320v2_gulf_stream.md``: for each
scheduled hourly snapshot, every fully-ocean, non-overlapping *km* x *km*
tile in the box is cut, oriented north-up and area-averaged to *npix* x
*npix*, and written to ``{outdir}/snapshots/{YYYYMMDD}T{HH}.nc``.  Snapshots
whose file already exists are skipped, so the run is resumable.  A final
``--concatenate`` pass joins the per-snapshot files into one NetCDF file.

Usage (from the repository root, or anywhere once st_sst is installed)::

    conda run -n ocean14 python -m st_sst.scripts.extract_gulf_stream --pilot --dry-run
    conda run -n ocean14 python -m st_sst.scripts.extract_gulf_stream --pilot --n-cores 2
    conda run -n ocean14 python -m st_sst.scripts.extract_gulf_stream --pilot --concatenate
    conda run -n ocean14 python -m st_sst.scripts.extract_gulf_stream --n-cores 8
    conda run -n ocean14 python -m st_sst.scripts.extract_gulf_stream --concatenate

The per-snapshot seed is ``snapshot_seed(--seed, hour)``, so any snapshot
re-extracted with the same arguments reproduces its tiling.
"""

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from st_sst.extract import llc_v2_cutouts as cut

logger = logging.getLogger('extract_gulf_stream')


def parser(options=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--pilot', action='store_true',
                   help=f'only the pilot hours {", ".join(cut.PILOT_HOURS)} UTC')
    p.add_argument('--box', nargs=4, type=float, default=list(cut.DEFAULT_BOX),
                   metavar=('LAT_MIN', 'LAT_MAX', 'LON_MIN', 'LON_MAX'))
    p.add_argument('--km', type=float, default=cut.DEFAULT_KM, help='cutout side (km)')
    p.add_argument('--npix', type=int, default=cut.DEFAULT_NPIX, help='cutout side (pixels)')
    p.add_argument('--seed', type=int, default=12345, help='base seed for the tilings')
    p.add_argument('--stride', nargs='+', type=int, default=list(cut.DEFAULT_STEPS),
                   help='cyclic hour steps between snapshots (default: 9 10)')
    p.add_argument('--start', default=cut.RECORD_START, help='first snapshot (UTC)')
    p.add_argument('--end', default=cut.RECORD_END, help='last snapshot (UTC), inclusive')
    p.add_argument('--n-cores', type=int, default=8, help='parallel workers over snapshots')
    p.add_argument('--outdir', default=None,
                   help='output directory (default: $OS_OGCM/LLC_v2/cutouts)')
    p.add_argument('--dry-run', action='store_true',
                   help='list the snapshots and files that would be written; read nothing')
    p.add_argument('--concatenate', action='store_true',
                   help='concatenate existing per-snapshot files into the final file')
    return p.parse_args() if options is None else p.parse_args(options)


def default_outdir() -> str:
    root = os.environ.get('OS_OGCM')
    if not root:
        raise RuntimeError('$OS_OGCM is not set; pass --outdir')
    return os.path.join(root, 'LLC_v2', 'cutouts')


def final_name(km: float, npix: int, pilot: bool) -> str:
    return f"LLC4320v2_GulfStream_SST_{km:g}km_{npix}px{'_pilot' if pilot else ''}.nc"


def schedule(args) -> list:
    if args.pilot:
        return [cut.parse_utc(h) for h in cut.PILOT_HOURS]
    return cut.snapshot_hours(args.start, args.end, args.stride)


def snapshot_path(outdir: str, date) -> str:
    return os.path.join(outdir, 'snapshots', cut.snapshot_tag(date) + '.nc')


def run_one(date, path, box, km, npix, seed, steps):
    """Worker: extract one snapshot and write its file. Returns a status dict."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    t0 = time.time()
    tag = cut.snapshot_tag(date)
    try:
        cube, table = cut.extract_snapshot(date, box=tuple(box), km=km, npix=npix, seed=seed)
    except Exception as e:                       # a missing hour: open_store raises
        return {'tag': tag, 'status': 'failed', 'error': f'{type(e).__name__}: {e}',
                'seconds': time.time() - t0}
    attrs = cut.global_attrs(box=box, km=km, npix=npix, steps=steps, seed=seed,
                             snapshot_time=cut.parse_utc(date).strftime('%Y-%m-%dT%H:%M:%SZ'),
                             snapshot_seed=cut.snapshot_seed(seed, date),
                             n_cutouts=len(table))
    cut.write_snapshot_nc(path, cube, table, attrs)
    return {'tag': tag, 'status': 'ok', 'n': len(table), 'seconds': time.time() - t0}


def extract(args, outdir):
    hours = schedule(args)
    todo = [(h, snapshot_path(outdir, h)) for h in hours]
    done = [(h, p) for h, p in todo if os.path.exists(p)]
    todo = [(h, p) for h, p in todo if not os.path.exists(p)]
    logger.info("%d snapshots scheduled, %d already done, %d to extract",
                len(hours), len(done), len(todo))
    if args.dry_run:
        for h, p in todo:
            print(f"{h:%Y-%m-%d %H:%M} UTC -> {p}")
        for h, p in done:
            print(f"{h:%Y-%m-%d %H:%M} UTC    {p} (exists, skipped)")
        print(f"final file: {os.path.join(outdir, final_name(args.km, args.npix, args.pilot))}")
        return
    if not todo:
        return
    os.makedirs(os.path.join(outdir, 'snapshots'), exist_ok=True)

    # Build the coarse-grid cache once in the parent so workers do not race on it
    from wrangler.ogcm import llc_v2_access as llc
    llc.region_slices(*args.box)

    t0 = time.time()
    n_ok, n_cut, failed = 0, 0, []
    n_workers = max(1, min(args.n_cores, len(todo)))
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(run_one, h, p, args.box, args.km, args.npix, args.seed,
                            args.stride): h for h, p in todo}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if r['status'] == 'ok':
                n_ok += 1
                n_cut += r['n']
                logger.info("[%d/%d] %s: %d tiles in %.1f s", k, len(todo), r['tag'],
                            r['n'], r['seconds'])
            else:
                failed.append(r['tag'])
                logger.warning("[%d/%d] %s skipped (%s)", k, len(todo), r['tag'], r['error'])
    wall = time.time() - t0
    logger.info("extracted %d snapshots (%d cutouts, %.1f per snapshot) in %.1f s wall "
                "with %d workers; %d failed%s", n_ok, n_cut, n_cut / max(n_ok, 1), wall,
                n_workers, len(failed), f": {' '.join(failed)}" if failed else '')


def concatenate(args, outdir):
    hours = schedule(args)
    paths = [snapshot_path(outdir, h) for h in hours]
    have = [p for p in paths if os.path.exists(p)]
    if len(have) < len(paths):
        logger.warning("%d of %d scheduled snapshot files are missing",
                       len(paths) - len(have), len(paths))
    outfile = os.path.join(outdir, final_name(args.km, args.npix, args.pilot))
    if args.dry_run:
        print(f"would concatenate {len(have)} files -> {outfile}")
        return
    t0 = time.time()
    n = cut.concatenate_snapshots(have, outfile, attrs={'pilot': int(args.pilot)})
    logger.info("concatenated %d cutouts from %d snapshots into %s (%.1f MB) in %.1f s",
                n, len(have), outfile, os.path.getsize(outfile) / 1e6, time.time() - t0)


def main(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    outdir = args.outdir or default_outdir()
    if args.concatenate:
        concatenate(args, outdir)
    else:
        extract(args, outdir)


if __name__ == '__main__':
    main(parser())
