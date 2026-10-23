# Getting started

## Goals

This repository will be used to run the Scattering Transform on sea surface
temperature (SST) data, characterizing the multi-scale, non-Gaussian structure of
the SST field.  We will generate the analyses, metrics, and diagnostics to share
with the community.  We will also compare ocean model output (LLC4320_v2) to 
satellite data.

## Prompts

1. Read this file.  Execute the 1st task under "Claude/CLAUDE.md file"
2. Read this file.  Execute the 1st task under "Claude/Skills"
3. Read this file.  Execute the 1st task under "Claude/Settings"
4. Read this file.  Execute the 1st task under "Basic start up"


## Claude

### CLAUDE.md file

1. Please generate a basic CLAUDE.md file for this project.  Have it indicate:

- I will perform git commands
- Add to the CLAUDE.md file:  If you do any calculation, generate it as a python script and write it to disk so that I can add it to the Repository.
- Add to the CLAUDE.md file:  If you need to run Python, use the "ocean14" conda environment.
- Add to the CLAUDE.md file:  Prompt docs for this repository live in prompts/ (not claude_prompts/ as in the sibling repositories).

### Skills

1. Copy over the skills/ files from the `IOPtics` repository.

### Settings

1. Copy over the settings.json file from the `IOPtics` repository.  Copy the policy
   (the allow/deny/ask structure), not the accumulated path-specific entries that
   reference IOPtics' own scratchpad and test files.

## Basic start up

1. Generate the basic files that one needs for a Python GitHub repository, e.g. a file for dependencies.  Examine the other Repositories in Oceanography/python to see how I tend to organize things.  Also, make a suggestion of which of their settings.json files to copy into this one.


## Logging

The "Logs" section will record Claude's work.  Please use the following format:

### <Date> (Short summary of the work)

<Detailed description of the work and what you learned>

...

## Logs

### 2026-10-23 (Generate the root CLAUDE.md)

Executed prompt #1: generated `CLAUDE.md` at the repository root (Claude Opus 5.5).

What was done:

- Wrote `CLAUDE.md` in the house form used by the sibling repos (closest to
  `EPFT-UP/CLAUDE.md` and `wrangler/CLAUDE.md`): Project Overview (with layout),
  Working Conventions (git / calculations / python environment / logging), Related
  Repositories.
- All four directives from the task are present: the user performs git commands
  (read-only git permitted), calculations are written to disk as Python scripts,
  Python runs in the `ocean14` conda environment, and prompt docs live in `prompts/`
  rather than `claude_prompts/`.
- The Goals line about comparing LLC4320_v2 to satellite data was carried into the
  overview.

What I learned about this repository:

- **wrangler** (`/home/xavier/Oceanography/python/wrangler`) is the natural sibling.
  It already handles VIIRS SST from PODAAC *and* provides read-only, anonymous access
  to the public LLC4320_v2 surface product on Nautilus S3
  (`wrangler/ogcm/llc_v2_access.py`; one Zarr store per hour under
  `s3://llc4320-v2/SURFACE`, native 13-facet LLC grid, `(face, j, i)`, no time
  dimension). Both sides of the model-vs-satellite comparison should come through it
  rather than through new readers here. It is installed in `ocean14` from source.
- **llc4320-native-grid-preprocessing** is the upstream LLC4320 native-grid pipeline
  that wrangler's `ogcm/` is coupled to. (`llc4320-tiles-surface-only` is a worktree
  of the same repository, branch `tiles-surface-only`.)
- No sibling repository uses the Scattering Transform yet; ST-SST is listed in the
  Sea-Meets-the-Stars org profile as "Running the scattering transform on SST data".
- Unlike its siblings, this repo is in the `Sea-Meets-the-Stars` GitHub org (like
  wrangler, fronts, ulmo), not `ocean-colour` (like IOPtics, EPFT-UP).
- `/home/xavier/Oceanography` is a symlink to `/mnt/tank/Oceanography`; the
  `/home/xavier/...` form was used in `CLAUDE.md` for consistency with wrangler.

No git commands that change repository state were run.

### 2026-10-23 (Copy the skills from IOPtics)

Executed prompt #2: copied `.claude/skills/` from the IOPtics repository.

What was done:

- Created `.claude/` and copied `/home/xavier/Oceanography/python/IOPtics/.claude/skills/`
  into it verbatim: `critical-partner/SKILL.md` and `grill-me/SKILL.md`.
- Verified with `diff -r` that the copies are byte-identical to the IOPtics originals.

What I learned about this repository:

- The skill files contain nothing IOPtics-specific, so nothing needed pruning.
  `grill-me` is likely to be useful early here: the choice of scattering-transform
  backend and of the satellite/model matchup design is still wide open.

No git commands that change repository state were run.

### 2026-10-23 (Copy the settings.json policy from IOPtics)

Executed prompt #3: created `.claude/settings.json` from the IOPtics policy.

What was done:

- Used the already-pruned copy in `EPFT-UP/.claude/settings.json`, after confirming
  programmatically that every IOPtics `allow` entry it lacks is path-specific cruft
  (one-off Crossref `curl`s, `python -c` one-liners, absolute-path `pytest` /
  `sphinx` invocations naming IOPtics files and scratchpads). Its `deny` and `ask`
  blocks are identical to IOPtics'.
- Result: 37 generic `allow` entries (read-only shell, benign file ops, read-only git,
  conda/python/pip/pytest/jupyter incl. `conda run -n ocean14`, four publisher
  `WebFetch` domains); `deny` = `sudo`, `rm -rf /`, `rm -rf ~`, `git push` / `commit`
  / `reset` / `rebase`; `ask` = `rm:*`. Validated as JSON.

What I learned about this repository:

- The four allow-listed publisher domains are optics-oriented (Optica, ASLO/Wiley,
  Taylor & Francis, doi.org). Scattering-transform literature lives mostly on
  arXiv, so `WebFetch(domain:arxiv.org)` would be a sensible addition here. Not added
  — your call.

No git commands that change repository state were run.

### 2026-10-23 (Basic Python repository start up)

Executed prompt #4: generated the basic files for a Python GitHub repository,
following the house layout of the sibling repos (EPFT-UP as the template).

What was done — files created:

- `st_sst/` — the Python package (snake_case of the repo name), with `__init__.py`
  (`__version__ = '0.1.0'`) and `st_sst/tests/` holding a `test_version.py` import
  smoke test (keeps `pytest`, and therefore CI, from exiting with code 5 on an empty
  suite).
- `requirements.txt` — core scientific stack, gridded-data I/O (xarray, h5netcdf,
  h5py, netcdf4, cftime), and cloud Zarr access (zarr, s3fs, fsspec) for LLC4320_v2.
  Commented-out lines for a scattering-transform backend (`kymatio` + `torch`) and for
  `git+https://github.com/Sea-Meets-the-Stars/wrangler`.
- `setup.py` — the house `setup_keywords` template, `python_requires='>=3.12'`,
  `url` pointing at `Sea-Meets-the-Stars/ST-SST`, `install_requires` mirroring the
  uncommented requirements.
- `pytest.ini` — `testpaths = st_sst/tests`.
- `.github/workflows/ci.yml` — light advisory CI (pytest on py3.12 and py3.14 plus an
  import smoke check), EPFT-UP's workflow minus the docs job (there is no `docs/`).
- `README.md` — expanded to the house form: description, Installation, Layout,
  Related work, License.

Verified: `setup.py` parses, `ci.yml` is valid YAML, and
`conda run -n ocean14 python -m pytest -q` passes (1 test).

Suggestion requested by the task — which sibling `settings.json` to copy:

- **Keep the IOPtics-derived policy installed by prompt #3** (identical to EPFT-UP's).
  The closest siblings by science, wrangler and llc4320-native-grid-preprocessing,
  have no committed `settings.json` at all; fronts adds only `Skill(update-config)`
  and a `/tmp` read, neither worth copying.
- Generic entries worth considering: `WebFetch(domain:github.com)`,
  `WebFetch(domain:raw.githubusercontent.com)`, `WebFetch(domain:api.github.com)`
  (from PAB / retrieve-or-bust) if we pull wrangler or a scattering-transform package
  from GitHub, and `WebFetch(domain:arxiv.org)` for the literature. Not added.

What I learned about this repository:

- **No scattering-transform library is installed in `ocean14`** (`kymatio` and
  `scattering` are both missing; `torch` 2.12 is present). The backend is a real design
  choice — e.g. `kymatio` (classic wavelet scattering) vs. the
  scattering-covariance / wavelet-phase-harmonic codes used for non-Gaussian field
  statistics — so I left it commented out rather than pick one or install anything.
- `ocean14` already has everything else the LLC4320_v2 path needs (xarray 2025.9,
  zarr 3.4, s3fs/fsspec 2026.7), and wrangler is importable from its source checkout.
- `.gitignore` already covers `__pycache__/` and `.pytest_cache/`; the test-run caches
  were removed anyway.
- `docs/`, `notebooks/`, `reports/`, `data/` were not created — they should appear
  when there is something to put in them.

No git commands that change repository state were run.
