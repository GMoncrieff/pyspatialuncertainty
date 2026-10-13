# R reference outputs

This folder holds the authors' original R script and the harness that turns it into
reference data for the Python parity tests.

The script is `Uncertainty_spatial_averages_and_totals.R`, the demo for
Wadoux & Heuvelink (2023), *Methods in Ecology and Evolution* 14:1320–1332. It is vendored
unchanged from [AlexandreWadoux/UncertaintySpatialAggregation](https://github.com/AlexandreWadoux/UncertaintySpatialAggregation)
at commit `934481f28e76f4b650130a8bf1b0c4b3e83ee954`, with SHA-256
`0217857456d6afce3841ef642c022abed05c97ce791228d9feac6b3e4889e32f`. It is licensed MIT; see
`LICENSE-UncertaintySpatialAggregation`.

`run_reference.R` runs the script and writes CSV files to `tests/data/r_reference/`.

## What the harness changes

Only the two lines that need RStudio are disabled (`rstudioapi::getActiveDocumentContext()`
and `setwd()`). Everything else runs exactly as published.

## What it exports

- The map and the 200 sample points.
- The gstat sample variograms and the hand-set nested variogram model.
- The exact 1000 point pairs drawn by the script's Monte Carlo integration, with their
  integrand values. They are recovered by re-drawing from the saved RNG state, and the
  harness asserts that this reproduces the script's estimate bit for bit.
- 200 replicate MC estimates with m = 1000, and one estimate with m = 10⁶.
- gstat fits of a few models.
- The script's scalar results.
- A version manifest.

See the header of `run_reference.R` for the full list of files.

## Regenerating the outputs

From the repository root:

```bash
Rscript r-reference/run_reference.R
```

The run takes about a minute with OpenBLAS and needs about 6 GB of RAM, because the
script builds dense 10⁴ × 10⁴ matrices.

It needs R ≥ 4.3 and these packages: gstat, ranger, raster, sp, spatstat, matrixStats,
ggplot2, gridExtra and rstudioapi. The committed outputs were produced with R 4.3.3 and the
Ubuntu 24.04 `r-cran-*` builds; the exact versions are in
`tests/data/r_reference/manifest.csv`.

To install them on Ubuntu 24.04:

```bash
apt-get install r-base-core r-cran-gstat r-cran-ranger r-cran-raster r-cran-sp \
  r-cran-spatstat r-cran-matrixstats r-cran-ggplot2 r-cran-gridextra r-cran-rstudioapi \
  libopenblas0-pthread
```

## Why the Python tests use these exports

R's random number streams and ranger's quantile regression forest cannot be reproduced
from Python. The Python tests therefore start from R's exported intermediates, such as the
QRF predictions and variances:

- **Steps 3–4** (standardized errors, sample variogram, correlogram) must match R exactly.
- **Steps 5–8** (the Monte Carlo integral and what follows from it) must match within
  Monte Carlo error.
