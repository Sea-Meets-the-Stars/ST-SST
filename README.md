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

## Reading the cutouts

`$OS_OGCM/LLC_v2/cutouts/LLC4320v2_GulfStream_SST_128km_64px.nc` (1.9 GB) holds
196,478 SST cutouts from the hourly LLC4320_v2 surface product. Each one is 128 km x
128 km at 64 x 64 pixels (~2 km/px), entirely ocean, north-up and east-right. They
come from 1,001 snapshots (2023-01-01 01:00 to 2024-01-31 21:00 UTC, alternating 9 h
and 10 h steps) over the Gulf Stream box 30-45N, 80-50W. The LLC4320_v2 simulation
was produced by and belongs to Dan Whitt (NASA Ames); please credit him in any
publication that uses these data (see the file's `credit` attribute).

`sst(cutout, y, x)` is raw °C float32: it is not demeaned, and the analysis removes
the mean itself. Row `y=0` is the north edge. The other variables are one value per
cutout: `UID`, `time`, centre `lat`/`lon`, the native index box (`face`, `j0`, `i0`,
`nj`, `ni`), realized `km_w`/`km_h`, the corner cell centres
`lat_sw, lon_sw, lat_ne, lon_ne`, `min_depth`, and the statistics `mean, std, min, max,
p10, p90, grad_mean, grad_p90, grad_max` (gradients in °C/km).

```python
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

path = os.path.join(os.environ['OS_OGCM'], 'LLC_v2', 'cutouts',
                    'LLC4320v2_GulfStream_SST_128km_64px.nc')
ds = xr.open_dataset(path)          # lazy: sst is read only when indexed
print(ds.sizes)                     # {'cutout': 196478, 'y': 64, 'x': 64}

# One cutout, drawn north-up at its corners
c = ds.isel(cutout=1000)
extent = [float(c.lon_sw), float(c.lon_ne), float(c.lat_sw), float(c.lat_ne)]
plt.imshow(c.sst, origin='upper', extent=extent, cmap='RdYlBu_r',
           aspect=1 / np.cos(np.deg2rad(float(c.lat))))
plt.colorbar(label='SST (°C)')
plt.title(f"{c.time.dt.strftime('%Y-%m-%d %H UTC').item()}  UID {int(c.UID)}")

# Select by the per-cutout variables (boolean indexing on the cutout dim)
summer = ds.isel(cutout=(ds.time.dt.month.isin([7, 8, 9])).values)
north = ds.isel(cutout=(ds.lat > 40).values)
fronts = ds.isel(cutout=(ds.grad_p90 > 0.1).values)
one = ds.isel(cutout=int(np.flatnonzero(ds.UID.values == int(c.UID))[0]))
anomaly = north.sst.isel(cutout=slice(0, 100)).load()
anomaly = anomaly - anomaly.mean(('y', 'x'))   # demean per cutout
```

The per-snapshot intermediates are in `$OS_OGCM/LLC_v2/cutouts/snapshots/{YYYYMMDD}T{HH}.nc`,
one file per snapshot with the same layout. The two-snapshot pilot is
`..._128km_64px_pilot.nc` (408 cutouts). The design, validation and figures are in
[reports/extract_llc4320v2_gulf_stream.md](reports/extract_llc4320v2_gulf_stream.md),
and `python -m st_sst.scripts.verify_cutouts_nc` re-runs the checks on the file.

## Related work

- [wrangler](https://github.com/Sea-Meets-the-Stars/wrangler) — SST download and
  processing, and access to the LLC4320_v2 surface product.
- [llc4320-native-grid-preprocessing](https://github.com/Sea-Meets-the-Stars/llc4320-native-grid-preprocessing)
  — LLC4320 native-grid dataset generation.

## License

BSD 3-Clause; see [LICENSE](LICENSE).
