# Extract SST LLC4320_v2 data 

## Goals

The goal is to extract SST data from the LLC4320_v2 surface product for the 
Scattering Transform analysis.

## Context

- The LLC4320_v2 surface product is a 4320x2160 grid of SST data.  It is stored
as a zarr file in s3 -- `s3://llc4320-v2/SURFACE`.


- The `wrangler` repo used to create that product is located next to this Repo.
- The `llc4320-native-grid-preprocessing` repo which may be useful to create cutouts for the analysis is located next to this Repo.

## Prompts

### Planning

1. We are going to construct 10,000 cutouts of SST near the Gulf Stream from the
LLC4320_v2 surface product.  Each cutout will be 128km x 128km and 128x128 pixels 
on a side.  Read the files described in Context and then start a discussion with me
on how to go about doing this.  Put your questions in the Q&A/Planning section.
Use Fable if you can.  Log your work

2. I have answered your questions in the Q&A/Planning section.  Please review and let me know if you have any further questions.  
Use Fable if you can.  Log your work

3. I think we have done a proper enough discussion.  Please write a short
plan document named `reports/extract_llc4320v2_gulf_stream.md`.
Also create a set of prompts under Prompts/Implement for me to then execute.
I will use Opus 5.5 for the execution.
Use Fable for this prompt if you can.  Log your work

### Implement

Read `reports/extract_llc4320v2_gulf_stream.md` before each of these; it holds the
settled design. Use the `ocean14` environment. Log each prompt under `## Logs`.

1. Write the library module `st_sst/extract/llc_v2_cutouts.py` (and `st_sst/extract/__init__.py`)
   with pure, testable functions: `tile_extent(dx_1d, dy_1d, j0, i0, km)` (fixed-km extents
   from local spacing, after dbof's `extent_in_i/j`, with attribution), `tile_snapshot(...)`
   (random-offset regular tiling of a face slice, keeping fully-ocean tiles with centres in
   the box), `make_cutout(theta, j0, i0, nj, ni, npix)` (slice, transpose to north-up /
   east-right, `resize_local_mean` to `npix`), `cutout_stats(img, km_w, km_h)` (mean, std,
   min, max, p10, p90, grad_mean, grad_p90, grad_max in °C/km), `make_uid(time, lat, lon)`
   (the wrangler/ulmo formula from `wrangler.ogcm.llc.add_uid`, re-implemented), and the
   snapshot schedule `snapshot_hours(start, end, steps=(9, 10))`. Add
   `st_sst/tests/test_llc_v2_cutouts.py` with network-free tests on a synthetic single-face
   grid built in memory (dims `face, j, i, j_g, i_g`; `XC`, `YC`, `dxC`, `dyC`, `maskC`,
   `Depth`), including an orientation test (latitude must decrease down the rows,
   longitude increase along the columns) and a test that 64 px never upsamples for the
   spacings in the plan. Run `pytest`. Use Opus 5.5.

2. Add the per-snapshot extractor and the NetCDF layer to `st_sst/extract/llc_v2_cutouts.py`:
   `extract_snapshot(date, grid, box, km, npix, seed) -> (cube, table)` reading one hour
   with `wrangler.ogcm.llc_v2_access.open_store` sliced to the face and index rectangle
   from `region_slices`; `write_snapshot_nc(path, cube, table, attrs)` and
   `concatenate_snapshots(paths, outfile)` producing the layout in the plan (dims
   `(cutout, y, x)`, chunks `(1024, 64, 64)`, zlib 1, per-cutout variables, global
   attributes including the Dan Whitt credit). Write the driver
   `st_sst/scripts/extract_gulf_stream.py` with `--pilot`, `--box`, `--km`, `--npix`,
   `--seed`, `--stride`, `--n-cores`, `--outdir`, `--dry-run`, `--concatenate`, resuming by
   skipping existing per-snapshot files. Run the pilot (2023-02-01 12:00 and 2023-08-01
   12:00 UTC) to `$OS_OGCM/LLC_v2/cutouts/`, producing
   `LLC4320v2_GulfStream_SST_128km_64px_pilot.nc`. Report the tiles per snapshot and the
   wall time. Use Opus 5.5.

3. Validate the pilot. Write `st_sst/scripts/plot_pilot_cutouts.py` producing, under
   `reports/figures/`: the box with the SST snapshot and tile outlines plus ~6 cutouts with
   their corner coordinates; histograms of `km_w`, `km_h`, `ni`, `nj`; and a printed spot
   check of a few pixels against `Theta` read directly with xarray at the same
   `(face, j, i)`. Check `lat_ne > lat_sw`, `lon_ne > lon_sw`, no NaN, unique UIDs, and
   that the resampling ratio is >= 1 everywhere. Append a "Pilot results" section to
   `reports/extract_llc4320v2_gulf_stream.md` with the figures, the measured yield, and
   the stride that gives ~200,000 cutouts. Fix anything the checks reveal. Use Opus 5.5.

4. Full run. From the pilot yield set the default stride (alternating 9/10 h unless the
   yield says otherwise) and run `extract_gulf_stream.py` over 2023-01-01 01:00 to
   2024-01-31 23:00 with `--n-cores 8`, resumable, skipping the two missing hours with a
   logged warning. Then `--concatenate` to
   `$OS_OGCM/LLC_v2/cutouts/LLC4320v2_GulfStream_SST_128km_64px.nc`. Do not delete the
   per-snapshot files. Use Opus 5.5.

5. Verify the full dataset with a script `st_sst/scripts/verify_cutouts_nc.py`: total
   count, no NaN, unique UIDs, time span, histograms of UTC hour and month (should be
   flat), per-snapshot yield, distributions of the statistics, file size. Append a "Full
   dataset" section to the plan document with the numbers and figures, and add a short
   "Reading the cutouts" example (xarray) to `README.md`. Use Opus 5.5.

## Q&A

### Planning

#### What I found (facts the questions below rest on)

- **Product.** `s3://llc4320-v2/SURFACE` holds one Zarr store per hour, `Theta` only
  (SST, °C, NaN on land), on the native 13-face LLC grid, each face 4320 x 4320, chunked
  `(1, 720, 720)` float32. Coverage 2023-01-01 01:00 to 2024-01-31 23:00 UTC: 9,501 hourly
  snapshots (two single hours missing; 2024-02-01 to 02-08 not yet extracted). Anonymous
  listing from this machine takes ~4 s; `~/.cache/wrangler/llc4320_v2` already holds the
  coarse XC/YC cache, so lat/lon -> (face, j, i) lookups are offline. (The Context line
  above says "4320x2160"; the product is actually 13 x 4320 x 4320.)
- **Gulf Stream geometry** (`st_sst/scripts/gulf_stream_grid_check.py`, box 30-45N,
  80-50W): the whole box sits inside **face 10** (j 2304-3743, i 2414-3400 of 4320), so
  no cutout straddles a face seam. Face 10 spans 13.7-67.5N, 128W-38W, which is also the
  limit before a seam matters (east of 38W is a different face). Face 10 is one of the
  **rotated** faces: `i` runs along latitude (decreasing), `j` along longitude
  (increasing). Native spacing there is **dx ~1.69 km, dy ~1.83 km** (ranges 1.53-1.85 and
  1.64-2.01 km), ocean fraction 0.83.
- **The arithmetic that matters.** 128 km is only **~76 x 70 native cells**. A 128 x 128
  pixel cutout of 128 km therefore has to be **upsampled ~1.7-1.8x**; it would not add
  information and the two finest scattering scales would mostly measure the interpolant.
  The region holds ~260 non-overlapping 128 km tiles per snapshot (~215 after the land
  mask), so 10,000 non-overlapping cutouts need ~47 snapshots.
- **Reusable code.** `wrangler.ogcm.llc_v2_access` gives `open_grid`, `open_store`,
  `region_slices`, `nearest_index` (anonymous, read-only, tested). The dbof pipeline in
  `llc4320-native-grid-preprocessing` has the fixed-km / fixed-pixel cutout logic
  (`spatial_cutouts.extent_in_i/j`, area-average downsample, **upsampling refused by
  design**), gradient-weighted center sampling (`weighted_coordinate_sampling`, bias 1.3
  on log |grad b|^2), and a Zarr + Parquet dataset format with a reader. But it is built
  for the credentialed `s3://dbof` 2011-2012 LLC4320 (v1) copy on a stitched rectangular
  grid, with heavy dependencies (torch, dask, ecco_v4_py), so it is a source of ideas and
  snippets rather than a drop-in. wrangler's older `ogcm/llc.py` + `extract/ex_ogcm.py`
  is the ulmo-era v1 path (HEALPix uniform sampling, 64 x 64, local NetCDF) and does not
  read the v2 product.

#### Questions

**Q1. Pixel scale (the decision everything else hangs on).** 128 km / 128 px = 1 km per
pixel, finer than the ~1.7-1.8 km native grid. Which do you want?

- (a) 128 km, 128 px, bilinear/area upsampled ~1.75x (as specified; smallest ST scales
  are then interpolation, not ocean).
- (b) 128 x 128 native cells, no resampling: cutouts are ~216 km (i) x ~234 km (j).
- (c) 128 km at native resolution, ~76 x 70 px, then a mild area-average to a square
  64 x 64 (2 km/px). This is the dbof convention (fixed km, fixed px, downsample only).
- (d) 128 km resampled to a uniform 1.5 km grid -> ~85 x 85 px, or to 2 km -> 64 x 64.

My recommendation is (c) or (b) depending on whether "128 km" or "128 px" is the science
requirement. If the satellite side is VIIRS at ~0.75-1 km, (a) is defensible only as a
matched-grid convenience; say so and I will note the caveat in the metadata.

>A. Let us do (c)

**Q2. What does "near the Gulf Stream" mean operationally?** A fixed box (I used 30-45N,
80-50W; shelf included), or a swath around the Gulf Stream path (e.g. a band around the
time-mean maximum SST gradient or around a chosen SST isotherm)? Should centers be drawn
uniformly over the box, or weighted toward strong gradients as dbof does (bias 1.3 on
log |grad|^2)? Uniform over a fixed box is simplest and gives a clean comparison
population; weighting makes the sample front-rich but harder to compare to satellites.
I lean uniform in a box, with |grad SST| recorded per cutout so you can subselect later.

>A. Let us do uniform in a box.

**Q3. Land and the shelf.** Require 100% ocean cells (no NaN; no inpainting), or allow a
small land fraction with inpainting as in the VIIRS pipeline? And exclude the shelf
(e.g. Depth < 200 m or 500 m)? I recommend 100% ocean and no depth cut, with min Depth
stored per cutout.

>A. Let us do 100% ocean and no depth cut.

**Q4. Orientation and cell anisotropy.** Face 10 is rotated: a native (j, i) cutout has
west "up" and north to the "left" (i.e. transposed and flipped relative to a map).
Options: store native orientation and record it, or transpose/flip to north-up,
east-right. Also the cells are ~8% wider in j than in i (1.83 vs 1.69 km). For the ST
(wavelets with orientation, usually averaged over angle) neither matters much, but for
comparison with satellite cutouts a north-up, square-pixel grid is cleaner. I recommend
north-up/east-right and, if you pick Q1(c) or (d), resampling to square pixels.

>A. Let us do north-up/east-right and resampling to square pixels.

**Q5. Time sampling.** 9,501 hourly snapshots are available. For 10,000 cutouts:
- (a) ~50 snapshots (e.g. every 7-8 days at a fixed UTC hour, Jan 2023 - Jan 2024) with
  ~200 non-overlapping cutouts each;
- (b) daily (395 snapshots) with ~25 cutouts each, more independent in time;
- (c) include the diurnal/hourly axis deliberately (same centers at several hours).
Is 10,000 a target, a minimum, or a cap? Should seasons be balanced? I recommend (b),
one fixed hour (e.g. 12:00 UTC, or random hour per day if the diurnal cycle is a
nuisance you want averaged out), seasons balanced by construction.

>A. Let us make 200,000 cutouts the target, but start with a pilot run

**Q6. Overlap.** Within a snapshot: non-overlapping (random tiling with jitter) or random
centers with overlap allowed? Across snapshots: same centers every time, or fresh draws?
I recommend no overlap within a snapshot and fresh random centers per snapshot.

>A. Use your recommendation

**Q7. Preprocessing and what gets stored.** Raw °C float32 with per-cutout mean, std,
min, max, |grad| statistics in the table, or the ulmo-style preprocessed field
(mean-subtracted, optional Gaussian smoothing)? For the ST I would store raw °C and let
the analysis code demean. Do you also want a plane/linear trend removed, which some
spectral analyses do?

>A. Let us store raw °C and let the analysis code demean.  Do record the per-cutout mean, std,
min, max, |grad| statistics in the table.

**Q8. Output format and location.** The house convention (ulmo/wrangler) is an HDF5 cube
`valid` of shape (N, H, W) plus a pandas table (parquet/feather) with UID, datetime,
face/j/i, lat/lon of center, realized km width/height, native cell counts, and the
stats; dbof uses Zarr + Parquet with its own reader. Which do you want, and where should
the result live: local under `$OS_OGCM/LLC_v2/...`, the public `s3://llc4320-v2` bucket
under a new prefix (needs your credentials), or `s3://dbof`? I recommend HDF5 + parquet,
written locally, with an optional push to `s3://llc4320-v2/CUTOUTS/<name>/` later.

>A. If it is possible to generate one NetCDF file for the full dataset, let us do so.

**Q9. Where the code lives.** The reader stays in wrangler per CLAUDE.md. The cutout
extraction (region -> centers -> cutouts -> table) could go in (a) this repo,
`st_sst/extract/`, or (b) wrangler, `wrangler/extract/llc_v2_cutouts.py`, where it is
reusable by other projects and next to the VIIRS extractor. I lean (b) for the generic
extraction and (a) for the Gulf Stream run configuration and driver script. Your call,
since wrangler is a shared package.

>A. Re-use as much code as you can from `wrangler` and `llc4320-native-grid-preprocessing` 
and put all additional code in this Repo.  But if you can accomplish more by making small
edits in those Repos, let me know and we can discuss it.

**Q10. The satellite side, decided now or later?** If the model cutouts should be
matchups (same time and place as clear-sky VIIRS cutouts in 2023), the sampling design
must start from the satellite clear-sky catalog rather than from random model centers.
If the two populations are to be independent samples from the same region and season,
the plan above stands. Which is it?

>A. Decide later

**Q11. Scope of this first run.** Shall I first build a ~200-cutout pilot from a few
snapshots (minutes of I/O: the region is 6 chunks, ~12 MB per hour) to check
orientation, pixel scale and the table before the full 10,000? I will assume yes unless
you say otherwise.

>A. Yes, we want a pilot run.

#### Round 2 (2026-10-23): review of your answers and remaining questions

Your answers fix the design as: 128 km cutouts at native resolution, area-averaged to
64 x 64 (2 km/px), north-up / east-right, uniform random non-overlapping tiling of a
fixed box, 100% ocean, raw °C with stats in a table, one NetCDF file, 200,000 cutouts
after a pilot, code in this repo reusing wrangler and the dbof ideas. All of that is
consistent and buildable; nothing needs a change in wrangler or in
llc4320-native-grid-preprocessing. What follows is what the 200,000 target implies, the
concrete plan, and five remaining questions.

**New numbers** (`st_sst/scripts/gulf_stream_tile_count.py`, which tiles face 10 with
fixed-km extents from `dxC`/`dyC` and keeps only tiles whose every cell is ocean;
counts are ±5%, five random tilings each):

| box | ocean fraction | fully-ocean 128 km tiles per snapshot | snapshots for 200,000 | mean spacing |
|---|---|---|---|---|
| 30-45N, 80-50W (current) | 0.83 | ~201 | ~1,000 of 9,501 | one every ~9.5 h |
| 30-45N, 80-40W (to the Gulf Stream extension) | 0.87 | ~291 | ~690 | one every ~14 h |

Across the box 128 km is 71-84 native cells along latitude and ~70 along longitude, so
64 px is a downsample everywhere (ratio 1.1-1.3). The dataset is 200,000 x 64 x 64
float32 = 3.3 GB uncompressed.

**R1. Time sampling for 200,000.** Non-overlapping tiling caps a snapshot at ~200
cutouts, so 200,000 needs ~1,000 distinct hours, i.e. roughly 2-3 snapshots per day
over the 13 months. Two consequences to accept or not: (i) hours 9-10 apart share most of
their mesoscale structure, so the effective number of independent fields is well below
200,000 (the 10,000-cutout version had the same issue at a milder level); (ii) the
diurnal cycle is sampled, which is a feature for the satellite comparison. I propose
drawing ~1,000 hours at random (seeded, stratified so every day gets 2-3 hours and all
24 UTC hours are used) and taking every fully-ocean tile of a fresh random tiling per
hour, stopping at 200,000. Alternatives: a fixed stride (every 9 h hits only 8 of the
24 hours; every 7 h gives 1,357 snapshots, then subsample), or allowing overlap across
tilings within an hour. **Do you accept the random-hours plan, and shall I extend the
box east to 40W** (more tiles per hour, fewer hours, and it covers the extension)?

>A. Let's make the time sampling nearly uniform.  We wish to span the full 13 months
and get a good sample of the diurnal cycle.  Please make an effort to do so.

**R2. Single NetCDF: layout.** I propose dims `(cutout, y, x)` with `sst(cutout, y, x)`
float32 (°C, `_FillValue` NaN never used since cutouts are 100% ocean), chunked
`(1024, 64, 64)`, zlib level 1, plus per-cutout 1-D variables: `UID` (int64, the
wrangler/ulmo datetime-lat-lon formula), `time` (UTC), `lat`/`lon` of the centre,
`face`, `j0`, `i0`, `nj`, `ni` (native index box, so any pixel's position can be
rebuilt from `grid.zarr`), `km_w`/`km_h` (realized extent), `lat_sw`/`lon_sw`/`lat_ne`
/`lon_ne` corners, `min_depth`, and the stats: `mean`, `std`, `min`, `max`, `p10`,
`p90`, `grad_mean`, `grad_p90`, `grad_max` (|grad T| in °C/km on the 2 km image).
Global attributes record the source bucket, box, km, px, resampling method, git
commit, and the Dan Whitt credit line from the wrangler HOWTO. **Two choices for you:**
(a) per-pixel `lat(cutout,y,x)`/`lon(cutout,y,x)` arrays would add 6.6 GB; I recommend
the corners + index box instead. (b) Should the native-resolution cutout also be kept
(variable 71-84 x 70 cells, so padded to 84 x 84 with NaN; ~5.6 GB more)? I recommend
no for this dataset; it can be regenerated in minutes from the index box.

>A. (a) use the corners; (b) do not keep the native-resolution cutout.

**R3. Where to write.** `$OS_OGCM` is `/home/xavier/Oceanography/data/OGCM/` and has no
`LLC_v2/` yet. I propose
`$OS_OGCM/LLC_v2/cutouts/LLC4320v2_GulfStream_SST_128km_64px.nc` (pilot:
`..._pilot.nc`), built from per-snapshot intermediate NetCDFs in a scratch
subdirectory so the run is resumable and then concatenated once. **OK, or another
path?**

>A. Use your suggestion

**R4. Pilot definition.** Two hours, one winter and one summer (2023-02-01 12:00 and
2023-08-01 12:00 UTC), every fully-ocean tile, ~400 cutouts, through the full pipeline
including the NetCDF, plus a figure of a few cutouts over a map with the SST field
behind them to verify orientation and pixel scale, and a histogram of the realized
km_w/km_h. **Fine, or do you want different dates / more hours?**

>A. Fine

**R5. Gradient statistic.** |grad T| from `np.gradient` on the final 64 x 64 image with
the realized pixel size in km, reported as mean, p90, max in °C/km. If you would rather
have it on the native grid (sharper, but not what the ST will see), say so.

>A. Do it on the 64x64 grid.

**Plan and code reuse (for the record, no question).**
- I/O: `wrangler.ogcm.llc_v2_access` (`open_grid`, `open_store`, `region_slices`,
  `hours_between`); one `open_store` per hour, sliced to face 10 and the box's 6 chunks.
- Fixed-km extents: re-implemented here (~40 lines) after dbof's `extent_in_i/j`, with
  the staggered `i_g`/`j_g` slicing done right; dbof itself is not installed in
  `ocean14` and would drag in torch/xgcm/ecco_v4_py for the sake of those lines.
- Resampling: `skimage.transform.resize_local_mean` (area average, non-integer ratios;
  already used by `wrangler.preproc.pp_ogcm`), applied after the transpose that makes
  face 10 north-up/east-right (`i` -> rows, `j` -> columns; no flip needed, since the
  grid off face 6 is exactly lat-lon so the transpose is exact).
- Stats: `wrangler.preproc.meta.stats` (max, min, p10, p90, mean) plus std and the
  gradient terms; `wrangler.utils.calc_grad2` for the gradient.
- Land: a tile is rejected if `maskC` is false or `Theta` is NaN anywhere in it.
- Layout in this repo: `st_sst/extract/llc_v2_cutouts.py` (library: tiling, extents,
  cutout, stats, NetCDF writer), `st_sst/scripts/extract_gulf_stream.py` (driver with
  `--pilot`, `--seed`, `--box`, `--outfile`), tests on a synthetic face.

## Logs

### 2026-10-23 (Planning the 10,000 Gulf Stream SST cutouts)

Executed Planning prompt #1 (Claude Fable 5.1): read the Context repositories and opened
the discussion in Q&A/Planning (11 questions, each with a recommendation).

What was done:

- Read `wrangler/ogcm/llc_v2_access.py`, `docs/howto/llc4320_v2.rst`, the v1-era
  `ogcm/llc.py` / `extract/ex_ogcm.py` / `preproc/pp_ogcm.py`, and the llc4320_v2 design
  log; in `llc4320-native-grid-preprocessing` read the cutout geometry, weighted
  sampling, data organization and grid docs, `spatial_cutouts.py`, and the cutout /
  global / Gulf Stream transfer configs.
- Verified anonymous S3 access from this machine (9,503 entries listed in 4 s) and that
  the wrangler coarse-grid cache is already present.
- Wrote `st_sst/scripts/gulf_stream_grid_check.py` (reads only the public `grid.zarr`):
  for a lat/lon box it reports the intersecting faces, native dxC/dyC, how many native
  cells span the cutout, the face orientation, the face's corner coordinates, and the
  number of non-overlapping tiles. Its output for 30-45N, 80-50W is summarized at the
  top of Q&A/Planning.

What I learned about this repository and its data:

- **The headline issue:** 128 km at 128 px is 1 km/px, finer than the native ~1.7-1.8 km
  cells in the Gulf Stream, so the cutouts as specified require ~1.75x upsampling. This
  is Q1 and gates the rest of the design.
- The Gulf Stream box lies entirely on **face 10**, a rotated face (`i` ~ latitude,
  `j` ~ longitude), well inside the face (no seam) as long as we stay west of 38W. The
  region touches only 6 of the 36 chunks of a face, so I/O is ~12 MB per hourly snapshot;
  the full 10,000-cutout extraction is minutes of network, not hours.
- `dxC`/`dyC` live on the staggered axes `(face, j, i_g)` / `(face, j_g, i)`; an
  `isel(j=, i=)` on the grid dataset does not slice them. The first version of the check
  script got dx wrong (1.03 km) for exactly this reason; slice `i_g`/`j_g` explicitly.
- The dbof pipeline refuses to upsample by design, samples centers weighted by
  log |grad b|^2 (bias 1.3), and stores Zarr + Parquet; it targets the credentialed v1
  copy on a stitched grid, so it is a design reference, not a dependency, for this work.
- The prompt's Context line describes the product as "4320x2160"; it is 13 faces of
  4320 x 4320. Worth correcting in the prompt doc.

No git commands that change repository state were run.

### 2026-10-23 (Planning round 2: 200,000 cutouts, NetCDF, pilot)

Executed Planning prompt #2 (Claude Fable 5.1): reviewed the answers to Q1-Q11,
recorded the resulting design, and asked five follow-up questions (R1-R5) in
Q&A/Planning, each with a proposed default.

What was done:

- Wrote `st_sst/scripts/gulf_stream_tile_count.py`: tiles the face holding a lat/lon box
  with fixed-km extents from `dxC`/`dyC`, keeps only fully-ocean tiles (`maskC`), and
  reports tiles per snapshot and the number of snapshots needed for a target count.
  Results: ~201 tiles/snapshot for 30-45N, 80-50W (~1,000 snapshots for 200,000);
  ~291 for 30-45N, 80-40W (~690 snapshots). Across the box 128 km is 71-84 native cells
  along latitude and ~70 along longitude, so 64 px never upsamples.
- Checked `ocean14`: netCDF4 1.7.4, h5netcdf, h5py, scikit-image 0.26, pandas 3, pyarrow
  present; `dbof` and `xgcm` absent. Read `wrangler.preproc.meta.stats`,
  `wrangler.utils.calc_grad2` and `wrangler.extract.io` for reuse.

What I learned:

- The 200,000 target forces ~2-3 snapshots per day from the current box, so neighbouring
  cutouts in time are strongly correlated; this is flagged in R1 rather than decided.
- `wrangler.ogcm.llc` imports cartopy and astropy_healpix at module level, so its
  `add_uid` is better re-implemented than imported for a lightweight extractor.
- The dbof cutout code depends on a stitched rectangular grid and torch; re-implementing
  its ~40-line fixed-km extent logic is cheaper than installing the package in `ocean14`.
- In the tile-count script the j-extent uses a per-column median of `dyC`, which hides the
  latitude dependence of `dyC` on this rotated face; the count is therefore ±5%. The real
  extractor will use the local spacing at each tile.

No git commands that change repository state were run.

### 2026-10-23 (Plan document and Implement prompts)

Executed Planning prompt #3 (Claude Fable 5.1): wrote the plan and the implementation
prompts for Opus 5.5.

What was done:

- `reports/extract_llc4320v2_gulf_stream.md`: the settled design in one place (source,
  measured geometry, eight design decisions, code layout, pilot, full run and
  verification, risks). The R1 answer (nearly uniform in time, full 13 months, good
  diurnal sampling) is implemented as alternating 9 h / 10 h steps: two snapshots per
  19 h, and because 19 is coprime with 24 every UTC hour is visited equally; ~1,000
  snapshots x ~201 tiles ≈ 200,000. The box stays 30-45N, 80-50W since the eastward
  extension was not chosen; it is kept as a `--box` parameter.
- Five Implement prompts under Prompts/Implement: library + tests, extractor + NetCDF +
  pilot run, pilot validation, full run, verification + README.

What I learned:

- wrangler's `tests/test_llc_v2_access.py` builds a tiny local product with
  `llc_v2.write_grid_store` / `write_timestep_store`, and `llc_v2_access` works against a
  local directory; that fixture writes only XC, YC, Depth, so our tests build a synthetic
  grid in memory (with `dxC`, `dyC`, `maskC` on the staggered axes) instead.
- Sibling repos keep investigation reports as Markdown in `reports/` with figures in
  `reports/figures/` and report scripts in `reports/scripts/`; this plan follows that
  layout, with the analysis scripts kept under `st_sst/scripts/` because they are
  pipeline tools, not one-off report generators.

No git commands that change repository state were run.

### 2026-11-02 (Implement #1: cutout library and tests)

Executed Implement prompt #1 (Claude Fable 5.1, at the user's direction, rather than
Opus 5.5): wrote `st_sst/extract/llc_v2_cutouts.py` and its tests; 23 tests pass under
`conda run -n ocean14 python -m pytest -q` (1.4 s, no network).

What was done:

- `st_sst/extract/llc_v2_cutouts.py`: `cells_for_km` / `tile_extent` (fixed-km extents
  from local spacing, after dbof's `extent_in_i/j`, measured from the tile origin so
  tiles lay edge to edge; `min_cells` guards against upsampling), `face_orientation` /
  `orient_north_up` / `oriented_km` (the transpose/flip that makes any lat-lon face
  north-up and east-right, decided from XC/YC), `make_cutout` (slice, orient,
  `resize_local_mean`; refuses upsampling), `tile_corners`, `tile_snapshot`
  (random-offset strip tiling, non-overlapping, fully-ocean, centre in box),
  `cutout_stats` (wrangler's `meta.stats` for the percentiles plus std and |grad T|
  in °C/km with the realized pixel size), `make_uid`, `parse_utc`, `snapshot_hours`,
  and the plan's defaults (`DEFAULT_BOX`, `DEFAULT_KM`, `DEFAULT_NPIX`, `DEFAULT_STEPS`,
  `PILOT_HOURS`, record start/end).
- `st_sst/tests/test_llc_v2_cutouts.py`: a synthetic face built in memory with the
  `grid.zarr` dims and staggering (`face, j, i, j_g, i_g`), in a rotated (face-10-like)
  and an unrotated variant; tests for extents, the no-upsample guarantee at every
  spacing in the plan (1.53-2.01 km), orientation (latitude decreasing down the rows,
  longitude increasing along the columns, on both variants, for coordinates and for
  cutouts), mean preservation, disjoint tiling with land avoided, seeding, the ramp
  gradient statistic, UID uniqueness and equality with `wrangler.ogcm.llc.add_uid`, and
  the schedule (alternation, ~1,000 snapshots, flat UTC-hour histogram, 13 months).
- `scikit-image` added to `requirements.txt` and `setup.py`.

What I learned:

- **pandas 3 defaults datetimes to microseconds**, and `wrangler.ogcm.llc.add_uid`
  computes its time digits from `.values.astype(int64)` without fixing the unit, so on
  microsecond datetimes the UID's time part has ~11-day resolution and hourly snapshots
  would collide. `make_uid` forces nanoseconds first; the test against wrangler
  therefore uses `.as_unit('ns')`. Worth a one-line fix in wrangler at some point.
- `tz_localize(None)` on a Series acts on the index, not the values; convert through
  a `DatetimeIndex`.
- `resize_local_mean` keeps float32 in/out and preserves the mean to <1e-3 at the
  non-integer ratios in play (70x76 -> 64x64).
- On face 10 the spacing along the latitude axis (`dxC`, along `i`) is 1.53-1.85 km and
  along the longitude axis (`dyC`, along `j`) 1.64-2.01 km; at 2.01 km, 128 km is 63.7
  cells, so without `min_cells=64` the overshoot rule alone gives exactly 64 and the
  guard is what makes "never upsample" a guarantee rather than luck.
- `conda run` does not forward a heredoc on stdin; probe scripts go in a file.

No git commands that change repository state were run.

### 2026-11-02 (Implement #2: extractor, NetCDF layer, driver, pilot run)

Executed Implement prompt #2 (Claude Opus 5.5): added the per-snapshot extractor and the
NetCDF layer to `st_sst/extract/llc_v2_cutouts.py`, wrote the driver, and ran the pilot.
28 tests pass under `conda run -n ocean14 python -m pytest -q` (23 existing + 5 new,
~4 s, no network).

What was done:

- `st_sst/extract/llc_v2_cutouts.py`: `extract_snapshot(date, grid, box, km, npix, seed)`
  reads only the face rectangle(s) from `region_slices` (face 10, j 2288-3759,
  i 2400-3423 for the default box), with `dxC` sliced on `(j, i_g)` and `dyC` on
  `(j_g, i)`; ocean = `maskC` and finite `Theta`; tiles with `tile_snapshot`
  (`min_cells=npix`), cuts with `make_cutout`, and fills the plan's table (UID, time,
  centre lat/lon, face, face-absolute j0/i0, nj/ni, km_w/km_h, corners, min_depth and
  the nine statistics). Optional `store=`/`slices=`/`root=` let it run on in-memory
  datasets. Also `snapshot_seed` (per-snapshot seed from the base seed and the hour via
  `SeedSequence`), `snapshot_tag`, `global_attrs` (bucket, endpoint, box, km, npix,
  resampling, sampling, stride, seed, code version, git commit, Dan Whitt credit),
  `to_dataset`, `write_snapshot_nc` (written to `.part` then renamed, so an existing
  file is always complete) and `concatenate_snapshots` (streams files through
  `netCDF4`, keeping encoded values; `sst` chunks `(min(1024, n), 64, 64)`, zlib 1).
  Time is encoded with a fixed `hours since 2023-01-01 00:00:00` so files concatenate
  without re-encoding.
- `st_sst/scripts/extract_gulf_stream.py`: `--pilot`, `--box`, `--km`, `--npix`, `--seed`
  (default 12345), `--stride`, `--start/--end`, `--n-cores`, `--outdir` (default
  `$OS_OGCM/LLC_v2/cutouts`), `--dry-run`, `--concatenate`; resumes by skipping existing
  `snapshots/{YYYYMMDD}T{HH}.nc`; `ProcessPoolExecutor` over snapshots; a failed hour
  (e.g. a missing store) is logged as a warning and skipped. Run as
  `conda run -n ocean14 python -m st_sst.scripts.extract_gulf_stream ...` from the repo
  root (st_sst is not installed in ocean14). The full schedule is 1,001 snapshots.
- `st_sst/scripts/check_cutout_file.py`: layout and invariant checks on a concatenated
  file (dims, count, no NaN, unique int64 UIDs, corner ordering, chunking/compression),
  plus per-snapshot yield and ranges of extents and statistics.
- Tests: `extract_snapshot` on a synthetic store (cube/table shapes, columns, unique
  UIDs, cutout means equal native-tile means, north-up, extents 128-130 km), seeding and
  face-absolute indices with an offset slice, `snapshot_seed`, and a
  `write_snapshot_nc` + `concatenate_snapshots` round trip (including an empty
  snapshot) checking dims, variables, int64 UID, times, chunking `(1024, 64, 64)`, zlib 1
  and the global attributes.

Pilot (2023-02-01 12:00 and 2023-08-01 12:00 UTC, seed 12345, 2 workers):

- **206 and 202 tiles** (408 cutouts, 204 per snapshot, vs ~201 predicted).
- Wall time 7.4 s for the extraction (6.0 and 6.9 s per snapshot, mostly S3 reads),
  9.9 s end to end; concatenation 0.2 s.
- `$OS_OGCM/LLC_v2/cutouts/LLC4320v2_GulfStream_SST_128km_64px_pilot.nc`, 4.0 MB
  (snapshot files 2.1 and 2.0 MB), with `$OS_OGCM=/home/xavier/Oceanography/data/OGCM/`.
- All checks pass: dims `(cutout, y, x)`, 408 cutouts, no NaN, unique UIDs,
  `lat_ne > lat_sw`, `lon_ne > lon_sw`. km_w 128.0-129.9, km_h 128.0-129.4; nj 65-79,
  ni 70-83 (never below 64); mean SST 3.9-30.2 °C; min_depth down to 11 m (shelf tiles
  are kept, as the plan only requires all-ocean cells).
- At ~6.5 s per snapshot the full 1,001-snapshot run is ~15 min with 8 workers and
  ~204,000 cutouts at the 9/10 h stride, so the stride can stay as planned.

What I learned:

- The real `region_slices` rectangle (j 2288-3760, i 2400-3424) is the coarse-grid
  rectangle padded by one 16-cell step, a little larger than the measured 2304-3743 /
  2414-3400; the box test in `tile_snapshot` handles the padding.
- `Theta` in the hourly store is float32 on `(face, j, i)` with no time dimension;
  `grid.zarr` has `dxC` on `(face, j, i_g)` and `dyC` on `(face, j_g, i)`, in metres.
- Importing `wrangler.preproc.meta` pulls in argopy, which logs "Using ONLINE Argo
  Float implementation" in every worker; harmless.
- `st_sst` is not installed in `ocean14`, so scripts must be run with `python -m` from
  the repository root (or the package installed with `pip install -e .`).
- The `git_commit` attribute carries `-dirty` while the working tree has uncommitted
  changes, as it did for the pilot (HEAD 9571130); rerun after committing for a clean
  provenance string if wanted.
- `sst` chunks are capped at the cutout count, so the pilot file is chunked
  `(408, 64, 64)`; the full file will have the planned `(1024, 64, 64)`.

No git commands that change repository state were run.

### 2026-11-02 (Implement #3: pilot validation)

Executed Implement prompt #3 (Claude Opus 5.5): wrote `st_sst/scripts/plot_pilot_cutouts.py`,
validated the pilot file, and appended a "Pilot results" section to
`reports/extract_llc4320v2_gulf_stream.md`. All checks pass, so nothing was changed in
the library or driver and the pilot was not re-extracted. 28 tests still pass under
`conda run -n ocean14 python -m pytest -q`.

What was done:

- `st_sst/scripts/plot_pilot_cutouts.py` (`--file`, `--outdir`, `--n-cutouts`, `--n-spot`,
  `--seed`, `--no-network`; ~18 s with the network): reuses `check_cutout_file.main` for
  the layout checks, then checks `lat_ne > lat_sw`, `lon_ne > lon_sw`, no NaN, unique
  UIDs, `min(nj, ni)/npix >= 1`, km >= 128 and centre in box; prints PASS/FAIL and exits
  non-zero on any failure. Spot check: opens `grid.zarr` and the two hourly stores
  through `wrangler.ogcm.llc_v2_access`, rebuilds every cutout from `Theta` with
  `make_cutout`, recomputes 5 pixels in each of 3 cutouts with an explicit
  area-weighted mean over the native cells (located through the orientation index map,
  independent of skimage), and compares the stored corners with `XC`/`YC` of the SW/NE
  native cells. Writes `reports/figures/pilot_map.png`, `pilot_cutouts.png`,
  `pilot_extent_histograms.png`, and tabulates snapshots and expected cutouts for
  candidate strides.

Numbers:

- Yield 206 + 202 = 408, **204.0 per snapshot**.
- All 408 rebuilt cutouts are bit-identical to the stored ones (max |d| = 0); the
  independent pixel-level area means agree to 1.9e-6 °C; corners agree exactly.
- km_w 128.0-129.9, km_h 128.0-129.4; ni 70-83, nj 65-79; pixel 2.000-2.030 km
  (median 2.022) by 2.001-2.022 km (median 2.016); resampling ratio 1.016-1.234.
- Stride: 9/10 h gives 1,001 snapshots, ~204,200 cutouts, all 24 UTC hours (40-42
  each); 8/9 gives ~228,100, 10 gives ~194,000 but only 12 UTC hours, 11 gives ~176,300.
  **Keep 9/10.** Neither missing hour (2023-10-19 03, 2023-12-28 23) is on the schedule.

What I learned:

- On face 10 output pixel (y=0, x=0) is native `(j0, i0)` (NW); the SW cell is
  `(j0, i0+ni-1)` and the NE cell `(j0+nj-1, i0)`. `face_orientation` gives
  transpose only, no flips, as the plan said.
- At ratios of 1.0-1.3 each output pixel averages 1-2 x 2-3 native cells with
  fractional weights, so a pixel can only be checked against `Theta` through an area
  mean; the explicit overlap-weight formula reproduces `resize_local_mean` to float32
  precision.
- The stored corners are cell centres, so outlines drawn from them are about one cell
  (~2 km) smaller than the tile; tiles are kept by centre, so edge tiles reach ~0.2 deg
  outside the box (lat_sw down to 29.8N).
- An alternation `a/b` visits every UTC hour only if `a+b` is coprime with 24: 9/10 and
  8/9 do, 10/11 (21) does not.

No git commands that change repository state were run.

### 2026-11-03 (Implement #4: full run)

Executed Implement prompt #4 (Claude Opus 5.5): ran the full extraction over
2023-01-01 01:00 to 2024-01-31 23:00 at the 9/10 h stride with 8 workers, fixed an
edge-of-slice bug in `tile_snapshot` that crashed 12 snapshots, resumed them, and
concatenated to `$OS_OGCM/LLC_v2/cutouts/LLC4320v2_GulfStream_SST_128km_64px.nc`
(`$OS_OGCM=/home/xavier/Oceanography/data/OGCM/`). Per-snapshot files and the pilot
file were left in place. 29 tests pass under `conda run -n ocean14 python -m pytest -q`
(28 + 1 new).

What was done:

- Stride: `DEFAULT_STEPS = (9, 10)` in the library and the driver's `--stride` default
  (`list(cut.DEFAULT_STEPS)`) were already 9/10, as the pilot yield supports; no change.
  `--seed` defaults to 12345, `--start/--end` to the record bounds, `--outdir` to
  `$OS_OGCM/LLC_v2/cutouts`.
- `--dry-run`: 1,001 snapshots, first 2023-01-01 01:00, last 2024-01-31 21:00 UTC.
  Neither missing hour (2023-10-19 03:00, 2023-12-28 23:00; neighbours 10-18 23:00 /
  10-19 09:00 and 12-28 20:00) nor either pilot hour (2023-02-01 12:00, 2023-08-01 12:00)
  is on the schedule. `--concatenate` already builds its file list from the schedule
  (`snapshot_path` of each scheduled hour), not from a glob of `snapshots/`, so the two
  pilot files are excluded; no driver change was needed.
- Full run (`python -m st_sst.scripts.extract_gulf_stream --n-cores 8`, detached),
  08:18:04 to 08:28:10 local: 989 snapshots, 193,998 cutouts in 603 s wall (2.7-13.7 s
  per snapshot). **12 failed**, all with `IndexError: index 1024 is out of bounds for
  axis 1 with size 1024` (20230122T00, 20230210T19, 20230228T15, 20230529T21,
  20230619T20, 20230702T12, 20230707T16, 20230714T19, 20230720T08, 20230803T14,
  20230825T08, 20240129T02). There were no S3 or missing-hour failures.
- Bug and fix (`st_sst/extract/llc_v2_cutouts.py`, `tile_snapshot`): when the widest tile
  of a strip ends exactly on the last column of the face slice (`i0 + max_ni == 1024`),
  the next strip starts at `i0 = 1024` and `dy[:, i0]` raises before the "no tile fits"
  test can end the loop. Whether this happens depends only on the seeded offset, so it
  is deterministic. The outer loop is now `while i0 < dx.shape[1]`. The guard only matters
  where the old code raised, so the 989 completed snapshots are unchanged and were not
  re-extracted. Regression test `test_tile_snapshot_strip_ending_on_the_edge`
  (`st_sst/tests/test_llc_v2_cutouts.py`) builds a slice of width
  `i_off + 2*128` at 1 km spacing, which ends a strip exactly on the edge.
- Resume (same command): 12 snapshots, 2,480 cutouts, 15.1 s wall, 0 failed. All 1,001
  scheduled files exist (1,003 files in `snapshots/` with the two pilot files).
- `--concatenate`: 196,478 cutouts from 1,001 snapshots in 57.8 s; file **1,911.5 MB**.
- `python -m st_sst.scripts.check_cutout_file <file>`: all checks ok. Dims
  `(cutout, y, x)` = (196478, 64, 64); sst chunks `[1024, 64, 64]`, zlib 1; no NaN;
  int64 UIDs unique; `lat_ne > lat_sw`, `lon_ne > lon_sw`, centre inside corners;
  `n_cutouts` matches; credit names Dan Whitt. Global attrs: `stride_hours` "9 10",
  `seed` 12345, `n_snapshots` 1001, `pilot` 0, coverage 2023-01-01T01:00:00Z to
  2024-01-31T21:00:00Z, `git_commit` 95711303...-dirty (HEAD 9571130, uncommitted work
  from prompts #1-#4), `code_version` 0.1.0.

Numbers:

- **1,001 snapshots, 196,478 cutouts**, 184-209 tiles per snapshot (mean 196.3, median
  193, std 6.9). Monthly means are flat, from 195.0 to 197.6, so there is no seasonal
  signal in the yield.
- km_w 128.00-129.96, km_h 128.01-129.83; nj 65-79, ni 70-84; min_depth 11-5395 m;
  mean SST 1.0-31.3 °C; lat 30.20-44.61, lon -79.66 to -50.34.
- Wall time: 603 s extraction plus 15 s resume (about 10.3 min with 8 workers), then
  58 s concatenation.

What I learned:

- The mean yield (196.3) is below the pilot's 204.0. The pilot's two snapshots (206,
  202) happened to fall high in a 184-209 spread that is set mostly by the random offset,
  not by the date. The 12 offsets that end a strip on the slice edge fit an extra column
  of tiles and yield 205-209. The full set is about 3.8% short of the ~204,200 projected
  from the pilot, but still near the planned 200,000.
- About 1.2% of seeded offsets (12/1,001) hit the edge case. The synthetic tests missed
  it because their face sizes never lined up with a strip boundary.
- With 8 workers a snapshot takes 3-14 s (median about 5 s), mostly S3 reads; 1,001
  snapshots took about 10 min, a little faster than the 15 min projected from the pilot.
- The concatenated file takes its global attrs, including `git_commit`, from the first
  snapshot file. All snapshots were written at HEAD 9571130 with a dirty tree.
- The bug-fix edit and the new test were applied with a short
  system-`python3` text-substitution one-off, not the `ocean14` environment. It was a
  file edit, not a calculation, but it does not follow the convention.

No git commands that change repository state were run.
