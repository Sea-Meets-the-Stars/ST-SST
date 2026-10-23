# Extract SST LLC4320_v2 data 

## Goals

The goal is to extract SST data from the LLC4320_v2 surface product for the 
Scattering Transform analysis.

## Context

- The LLC4320_v2 surface product is a 4320x2160 grid of SST data.  It is stored
as a zarr file in s3.


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
