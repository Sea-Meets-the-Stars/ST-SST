# ST-SST

Running the Scattering Transform on sea surface temperature (SST) data.

The goal is to characterize the multi-scale, non-Gaussian structure of the SST
field, generate analyses, metrics, and diagnostics to share with the community,
and compare ocean model output (LLC4320_v2) to satellite SST.

## Installation

```bash
git clone https://github.com/Sea-Meets-the-Stars/ST-SST.git
cd ST-SST
pip install -r requirements.txt
pip install -e .
```

Run the tests with `pytest`.

## Layout

- `st_sst/` — the Python package (`st_sst/tests/` holds the test suite)
- `prompts/` — task definitions for Claude-assisted development

## Related work

- [wrangler](https://github.com/Sea-Meets-the-Stars/wrangler) — SST download and
  processing, and access to the LLC4320_v2 surface product.
- [llc4320-native-grid-preprocessing](https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing)
  — LLC4320 native-grid dataset generation.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
