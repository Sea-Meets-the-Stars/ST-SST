# Plan: Gulf Stream SST cutouts from LLC4320_v2

*ST-SST planning document, 2026-10-23. Settled in `prompts/extract_sst.md` (Q&A/Planning,
rounds 1-2) between J. Xavier Prochaska and Claude Fable 5.1. Numbers come from
`st_sst/scripts/gulf_stream_grid_check.py` and `st_sst/scripts/gulf_stream_tile_count.py`.*

## Goal

A single NetCDF file of **200,000 SST cutouts**, each **128 km x 128 km** at
**64 x 64 pixels** (2 km/px), drawn uniformly over a Gulf Stream box from the hourly
LLC4320_v2 surface product, spanning its full 13 months with the diurnal cycle well
sampled. Raw °C, 100% ocean, north-up / east-right, with per-cutout statistics. It is
the model side of the Scattering Transform comparison with satellite SST; the
satellite side is decided later.

## Source

| item | value |
|---|---|
| store | `s3://llc4320-v2/SURFACE/{YYYYMMDD}T{HH}.zarr` + `grid.zarr`, endpoint `https://s3-west.nrp-nautilus.io`, anonymous |
| field | `Theta` (SST, °C, NaN on land), dims `(face, j, i)`, 13 faces of 4320², chunks `(1, 720, 720)` float32 |
| coverage | 2023-01-01 01:00 to 2024-01-31 23:00 UTC, 9,501 hourly snapshots (2023-10-19 03:00 and 2023-12-28 23:00 missing) |
| reader | `wrangler.ogcm.llc_v2_access` (`open_grid`, `open_store`, `region_slices`, `hours_between`) |
| credit | Dan Whitt (NASA Ames) owns the simulation; cite per `wrangler/docs/howto/llc4320_v2.rst` |

## Geometry (measured)

- Box **30-45N, 80-50W** (a `--box` parameter; extending east to 40W is the known
  alternative and raises the yield by ~45%). The box lies entirely on **face 10**,
  native indices j 2304-3743, i 2414-3400: no face seam, 6 of the face's 36 chunks,
  ~12 MB per snapshot.
- Face 10 is rotated: **`i` runs along latitude (decreasing), `j` along longitude
  (increasing)**, exactly, since the grid off the Arctic cap is lat-lon. A north-up /
  east-right image is the plain transpose `native[j, i].T` (rows = i, cols = j); no
  flip.
- Native spacing dx 1.53-1.85 km (median 1.69), dy 1.64-2.01 km (median 1.83). **128 km
  is 71-84 cells along latitude and ~70 along longitude**, so 64 px is a downsample
  everywhere (ratio 1.1-1.3); never an upsample.
- Ocean fraction 0.83; a random non-overlapping tiling yields **~201 fully-ocean tiles
  per snapshot** (±5%).

## Design decisions

1. **Cutout extent**: fixed km. From a tile origin, accumulate the local `dxC` (along i)
   and `dyC` (along j) until 128 km is reached, as in dbof's
   `spatial_cutouts.extent_in_i/j`. `dxC` lives on `(face, j, i_g)` and `dyC` on
   `(face, j_g, i)`; slice `i_g`/`j_g` explicitly (`isel(j=, i=)` does not).
2. **Resampling**: transpose to north-up, then `skimage.transform.resize_local_mean`
   (area average) to 64 x 64. Pixels are ~2.0 km and square to within ~2%.
3. **Sampling within a snapshot**: a regular tiling in native index space with a random
   (seeded) offset; keep every tile whose centre is in the box and whose cells are all
   ocean (`maskC` true and `Theta` finite). No overlap within a snapshot; fresh tiling
   each snapshot.
4. **Sampling in time**: nearly uniform over the 13 months with alternating **9 h and
   10 h steps** from 2023-01-01 01:00 (two snapshots per 19 h; 19 is coprime with 24 so
   all UTC hours are visited equally). That is ~1,000 snapshots x ~201 tiles ≈ 200,000.
   The stride is a parameter; the pilot yield fixes it before the full run. Missing hours
   are skipped (`open_store` raises; log and continue).
5. **Values**: raw °C float32; the analysis demeans. Statistics per cutout on the final
   64 x 64 image: mean, std, min, max, p10, p90 (via `wrangler.preproc.meta.stats` plus
   std) and |∇T| mean, p90, max in °C/km using `np.gradient` with the realized pixel
   size.
6. **Output**: one NetCDF4 file, dims `(cutout, y, x)`:
   - `sst(cutout, y, x)` float32, chunks `(1024, 64, 64)`, zlib level 1;
   - per cutout: `UID` (int64, ulmo/wrangler datetime-lat-lon formula), `time` (UTC),
     `lat`, `lon` (centre), `face`, `j0`, `i0`, `nj`, `ni` (native index box), `km_w`,
     `km_h` (realized), corners `lat_sw`, `lon_sw`, `lat_ne`, `lon_ne`, `min_depth`, and
     the statistics above;
   - global attrs: source bucket, box, km, px, resampling, stride, seed, code version
     and git commit, Dan Whitt credit. No per-pixel lat/lon, no native-resolution copy.
7. **Location**: `$OS_OGCM/LLC_v2/cutouts/LLC4320v2_GulfStream_SST_128km_64px.nc`
   (pilot: `..._pilot.nc`). Per-snapshot intermediates in
   `$OS_OGCM/LLC_v2/cutouts/snapshots/{YYYYMMDD}T{HH}.nc` make the run resumable; a
   final pass concatenates them. Full dataset ≈ 3.3 GB uncompressed.
8. **Code**: all new code in this repo; reuse wrangler for I/O and stats. The dbof
   extent logic (~40 lines) is re-implemented with attribution rather than installing
   `dbof` (not in `ocean14`; pulls torch, xgcm, ecco_v4_py). No edits to wrangler or
   llc4320-native-grid-preprocessing are needed.

## Code layout

| path | role |
|---|---|
| `st_sst/extract/llc_v2_cutouts.py` | library: tile extents, tiling, cutout + transpose + resample, stats, UID, NetCDF writer/concatenator; pure functions testable on synthetic grids |
| `st_sst/extract/schedule.py` (or in the module above) | snapshot schedule (alternating 9/10 h), pilot hours |
| `st_sst/scripts/extract_gulf_stream.py` | driver: `--pilot`, `--box`, `--km`, `--npix`, `--seed`, `--stride`, `--n-cores`, `--outdir`, `--dry-run`, resume by skipping existing snapshot files, then `--concatenate` |
| `st_sst/scripts/plot_pilot_cutouts.py` | validation figures (map + cutouts, km histograms, orientation check) to `reports/figures/` |
| `st_sst/tests/test_llc_v2_cutouts.py` | network-free tests on a synthetic single-face grid |

## Pilot

Two snapshots, **2023-02-01 12:00** and **2023-08-01 12:00 UTC**, every fully-ocean tile
(~400 cutouts) through the full pipeline to `..._pilot.nc`. Validation:

- a map of the box with the SST snapshot and the tile outlines, and a panel of ~6
  cutouts with their corner coordinates, to confirm north-up / east-right and 2 km px;
- histograms of `km_w`, `km_h`, `ni`, `nj`;
- spot checks: a few cutout pixels against `Theta` read directly with xarray at the
  same `(face, j, i)`; `lat_ne > lat_sw`, `lon_ne > lon_sw`; no NaN; unique UIDs;
- measured tiles per snapshot, used to set the stride for the full run.

## Full run and verification

~1,000 snapshots, parallel over snapshots (`ProcessPoolExecutor`, ~8 workers), ~12 GB
of reads, resumable. After concatenation verify: count, no NaN, unique UIDs, time span
and the histogram of UTC hour (flat) and month (flat), per-snapshot yield, and the
distribution of the statistics. Write the summary to this document.

## Risks and open items

- Consecutive snapshots 9-10 h apart share most mesoscale structure, so the effective
  number of independent fields is far below 200,000. Accepted; recorded here.
- The satellite side (matchups vs independent populations) is deferred; the index box
  and corners stored per cutout allow either later.
- If the full-run yield differs from the pilot, adjust the stride rather than allowing
  overlap.
