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

## Installation

The package needs Python ≥ 3.12 and uses [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/GMoncrieff/pyspatialuncertainty
cd pyspatialuncertainty
uv sync                 # add --all-extras for the synthetic QRF pipeline
```

## Continuous maps (steps 1–8 of Section 4.1)

```python
import pyspatialuncertainty as psu

# steps 2-3: σ maps and standardized validation errors (Polars in, Polars out)
val = psu.standardized_errors(cv_table, observed="obs", predicted="pred", sd="sd")

# step 4: gstat-compatible sample variogram and WLS fit
ev = psu.empirical_variogram(val, "std_error", "x", "y")
fit = psu.fit_variogram(ev, psu.vgm(0.5, "Exp", 10, nugget=0.1))
print(fit.model.sill)  # ≈ 1 if σ is calibrated; the method trusts σ for scale

# steps 5-8: Monte Carlo integration over blocks
res = psu.aggregate_continuous(
    sd_map,
    fit.model,
    psu.blocks.tiles(sd_map, 10),  # or blocks.polygons(...), zones(...), domain(...)
    prediction=pred_map,
    n_mc=10_000,
    level=0.90,
    seed=1,
)
res.table  # Polars: mean_pred, sd_mean, lower, upper, total_pred, sd_total, mcse...
res.to_xarray()  # block results as rasters
```

`sd_map` and `pred_map` are xarray DataArrays on a regular grid, in a projected CRS or
with no CRS.

## Class probability maps

```python
psu.validate_probabilities(prob_map)  # sums to one, strictly inside (0, 1)
errors = psu.indicator_errors(val_table, {"forest": "p_forest", "grass": "p_grass"}, label="label")
fits = psu.fit_class_variograms(errors)  # one correlogram per class
res = psu.aggregate_categorical(
    prob_map,
    {c: f.model for c, f in fits.items()},
    psu.blocks.polygons(regions, prob_map),
    area_scale=1e-4,
)
res.table  # fraction, sd_fraction, expected_area, sd_area, area_lower, area_upper, ...
```

## Verification

- `tests/` holds unit tests, hypothesis property tests and **parity tests against the
  authors' R script**. The R script was run under R 4.3.3, and its exported intermediates
  are in `tests/data/r_reference`.
  - Sample variograms and gstat model lines match to machine precision.
  - The integrand on the script's own Monte Carlo pairs matches to 1e-15.
  - The Monte Carlo estimates agree within Monte Carlo error.
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
