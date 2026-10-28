# pyspatialuncertainty

**Uncertainty of spatial averages and totals of maps.** This package is a Python
implementation of the Monte Carlo integration method of Wadoux & Heuvelink (2023),
*Uncertainty of spatial averages and totals of natural resource maps*, Methods in Ecology
and Evolution 14:1320–1332
([doi:10.1111/2041-210X.14106](https://doi.org/10.1111/2041-210X.14106)). It covers
continuous maps and a per-class analogue for maps of calibrated class probabilities.

When point-level map uncertainty is aggregated to a region, the result depends on how the
map errors are correlated in space:

- averaging the point standard deviations **overstates** the uncertainty of the regional
  average;
- assuming independent errors **understates** it, more and more as the region grows.

The method integrates σ(s)σ(u)ρ(|s − u|) over the block by Monte Carlo, where:

- σ is the point-support prediction-error SD;
- ρ is the correlation of the standardized errors, fitted to validation errors.

It works for maps made with any model that yields predictive distributions, including
machine learning such as quantile regression forests.

**One default differs from the paper.** The paper assumes the map is unbiased (or its
class probabilities calibrated), so the standardized errors have mean zero. By default
this package treats that mean as unknown: it estimates it from the validation points and
corrects the block predictions and their uncertainty for it, as ordinary kriging does.
The paper's procedure stays available as an option (see
[Reproducing Wadoux & Heuvelink](#reproducing-wadoux--heuvelink-2023)).

## Installation

The package needs Python ≥ 3.12. It is not on PyPI; install the 1.0.0 release straight
from GitHub:

```bash
uv add "pyspatialuncertainty @ git+https://github.com/GMoncrieff/pyspatialuncertainty@v1.0.0"
# or: pip install "git+https://github.com/GMoncrieff/pyspatialuncertainty@v1.0.0"
```

Add the `ml` extra (`pyspatialuncertainty[ml] @ git+...`) for the synthetic QRF pipeline.
To work on the package itself, clone it:

```bash
git clone https://github.com/GMoncrieff/pyspatialuncertainty
cd pyspatialuncertainty
uv sync --all-extras
```

## Continuous maps (steps 1–8 of Section 4.1)

```python
import pyspatialuncertainty as psu

# steps 2-3: σ maps and standardized validation errors (Polars in, Polars out)
val = psu.standardized_errors(cv_table, observed="obs", predicted="pred", sd="sd")

# step 4: correlation of the standardized errors (Gaussian likelihood fit, REML)
fit = psu.fit_correlation(val)
print(fit.model.sill)  # ≈ 1 if σ is calibrated; the method trusts σ for scale

# steps 5-8: Monte Carlo integration over blocks, corrected for the estimated mean error
res = psu.aggregate_continuous(
    sd_map,
    fit,  # the fit carries the validation points used to estimate the mean
    psu.blocks.tiles(sd_map, 10),  # or blocks.polygons(...), zones(...), domain(...)
    prediction=pred_map,
    n_mc=10_000,
    level=0.90,
    seed=1,
)
res.mean_estimate  # estimated mean of the standardized errors and its standard error
res.table  # Polars: map_mean, shift, mean_pred, sd_mean, lower, upper, total_pred, ...
res.to_xarray()  # block results as rasters
```

`sd_map` and `pred_map` are xarray DataArrays on a regular grid, in a projected CRS or
with no CRS.

### The default: an estimated mean of the errors

Maps are rarely unbiased, and class probabilities rarely calibrated, everywhere. A
systematic offset is shared by every pixel, so it never averages out: under the paper's
mean-zero assumption the interval of a large block keeps shrinking while its error does
not. By default the package therefore treats the mean μ of the standardized errors as
unknown (`pyspatialuncertainty.mean_correction`):

- step 4 fits the correlogram by REML, which does not mistake an offset for correlation;
- μ is estimated by generalised least squares from the validation points;
- each block prediction is shifted by x_B·μ̂, where x_B is the block mean of σ;
- the block variance gains the uncertainty of μ̂ and loses the part of the block error
  that the validation points already reveal: V_B + x_B²·var(μ̂) − 2·x_B·c_B.

In simulations with known truth this kept continuous intervals calibrated whether or not
the map was biased, and removed most of the undercoverage that miscalibrated class
probabilities cause. Practical notes:

- **Validation data.** Use a holdout collected independently of the training data,
  ideally by a probability design. With fewer than about 200 points (or a handful of
  clusters) on a map that is already calibrated, the correction adds more noise than it
  removes; `mean="zero"` is then slightly more accurate.
- **Recalibrate σ first.** The method takes the scale of σ (or of the class
  probabilities) on trust; no mean correction fixes a σ that is too small or too large.
- **Many validation points.** With thousands of points the corrected variance of a large
  block can be a small part of the Monte Carlo integral V_B; raise `n_mc` (the functions
  warn when Monte Carlo noise is large compared with the corrected variance).
- **Rare classes at landscape scale.** A stationary correlogram understates the
  uncertainty of a rare class's share of a large region; with a random sample, a
  design-based estimator (e.g. Olofsson et al. 2014) is safer there.
- **The likelihood fit's cost** grows with the cube of the number of validation points:
  about 1 s for 300 points and 30 s for 1,500. Above 2,000 points it warns; subsample
  the points or use `method="wls"`.

### Reproducing Wadoux & Heuvelink (2023)

The paper's procedure is a WLS fit of a gstat-compatible sample variogram and mean-zero
errors:

```python
fit = psu.fit_correlation(val, method="wls")  # or method="wls", model=psu.vgm(...)
res = psu.aggregate_continuous(sd_map, fit, blocks, prediction=pred_map, mean="zero")
```

This reproduces the authors' R script (see Verification). `mean="zero"` also works with
any correlation callable, and with the zero-mean likelihood fit,
`fit_correlation(val, mean="zero")`.

## Class probability maps

```python
psu.validate_probabilities(prob_map)  # sums to one, strictly inside (0, 1)
errors = psu.indicator_errors(val_table, {"forest": "p_forest", "grass": "p_grass"}, label="label")
fits = psu.fit_class_variograms(errors)  # one correlogram per class (REML fit)
res = psu.aggregate_categorical(
    prob_map,
    fits,  # the fits carry each class's validation points
    psu.blocks.polygons(regions, prob_map),
    area_scale=1e-4,
)
res.table  # map_fraction, shift, fraction, sd_fraction, expected_area, sd_area, ...
```

As for continuous maps, the mean of each class's standardized errors is estimated and the
fractions are corrected for it; `mean="zero"` trusts the probabilities' calibration.

## Verification

- `tests/` holds unit tests, hypothesis property tests and **parity tests against the
  authors' R script**. The R script was run under R 4.3.3, and its exported intermediates
  are in `tests/data/r_reference`.
  - Sample variograms and gstat model lines match to machine precision.
  - The integrand on the script's own Monte Carlo pairs matches to 1e-15.
  - The Monte Carlo estimates agree within Monte Carlo error (with `mean="zero"`).
- `tests/unit/test_mean_correction.py` checks the estimated-mean correction against the
  GLS formulas, a brute-force cross term and an exact simulation of the corrected error.
- `experiments/` holds the simulation studies with known truth: coverage of prediction
  intervals, estimated vs. true SD, naive baselines, and ablations of σ misspecification,
  probability miscalibration, signal-to-noise, correlation range and nugget.
- `report/report.html` is a self-contained report with the methods, the parity results,
  the verification and the ablation figures.

## Documentation

The `docs/` folder is a Quarto website: a quartodoc API reference, a method page and two
tutorials. To build it:

```bash
uv sync --all-groups
cd docs && uv run quartodoc build && uv run quarto render
```

## Development

```bash
uv run pytest                      # fast suite + R parity on committed reference data
uv run pytest -m slow              # statistical tests and the R harness re-run (needs Rscript)
uv run ruff check . && uv run ruff format --check . && uv run mypy
Rscript r-reference/run_reference.R   # regenerate the R reference outputs
```

## Citation and acknowledgements

Please cite the paper; `CITATION.cff` has the details. The R script in `r-reference/` is ©
Alexandre Wadoux, MIT licensed, and is vendored unchanged. This package is MIT licensed.
