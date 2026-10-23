# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ST-SST runs the Scattering Transform on sea surface temperature (SST) data,
characterizing the multi-scale, non-Gaussian structure of the SST field. We will
generate the analyses, metrics, and diagnostics to share with the community, and
compare ocean model output (LLC4320_v2) to satellite data.

Repository layout:

- `prompts/` — prompts and task definitions that drive this work. Read the relevant
  prompt doc (starting with `start_up.md`) before acting, and execute only the
  numbered task you were pointed at. Note: this repository uses `prompts/`, not
  `claude_prompts/` as in most of the sibling repositories.

## Working Conventions

- **Git:** The user (J. Xavier Prochaska) will perform all git commands (add, commit,
  push, etc.). Do not run git commands that change repository state unless explicitly
  asked. Read-only git commands (e.g. `git status`, `git diff`, `git log`) are fine.
- **Calculations:** If you do any calculation, generate it as a Python script and write
  it to disk so that it can be added to the repository. Do not perform one-off
  calculations only in memory or in the chat.
- **Python environment:** If you need to run Python, use the `ocean14` conda environment
  (e.g. `conda run -n ocean14 python script.py`). Never use the system Python.
- **Logging:** Record completed work under the `## Logs` section of the prompt doc that
  drove it, dated, including what was learned about the repository.

## Related Repositories

All on this computer under `/home/xavier/Oceanography/python/`:

- **wrangler** — `/home/xavier/Oceanography/python/wrangler`. Downloads and processes
  SST data (VIIRS from PODAAC) and provides read-only access to the public LLC4320_v2
  surface product on Nautilus S3 (`wrangler/ogcm/llc_v2_access.py`,
  `s3://llc4320-v2/SURFACE`). Use it for both the satellite and the model side of
  the comparison rather than writing new readers here.
- **llc4320-native-grid-preprocessing** —
  `/home/xavier/Oceanography/python/llc4320-native-grid-preprocessing`. The pipeline
  that generates datasets from the LLC4320 model on its native grid.
- **IOPtics** — `/home/xavier/Oceanography/python/IOPtics`. Upstream source for this
  project's `.claude/skills/` and `.claude/settings.json`.
