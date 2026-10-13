"""Uncertainty of spatial averages and totals of maps.

A Python implementation of the Monte Carlo integration method of Wadoux & Heuvelink
(2023), *Uncertainty of spatial averages and totals of natural resource maps*, Methods in
Ecology and Evolution 14:1320-1332, https://doi.org/10.1111/2041-210X.14106, for continuous
maps, together with a per-class analogue for maps of class probabilities.

The method in eight steps (Section 4.1 of the paper):

1. quantify the point-support prediction uncertainty (e.g. a quantile regression forest);
2. compute the prediction-error standard deviation σ(s) (:func:`sd_from_samples`,
   :func:`sd_from_quantiles`);
3. standardize validation errors by σ (:func:`standardized_errors`);
4. model the correlation of the standardized errors with a variogram
   (:func:`empirical_variogram`, :func:`fit_variogram`, :func:`vgm`);
5. - 6. integrate σ(s)σ(u)ρ(|s − u|) over each block by Monte Carlo
   (:func:`aggregate_continuous`);
7. scale the variance by |B|² for block totals;
8. report standard deviations and prediction intervals.
"""

from importlib.metadata import version as _version

from pyspatialuncertainty import blocks
from pyspatialuncertainty.baselines import sd_average, sd_independent
from pyspatialuncertainty.blocks import BlockSet, ResolvedBlock, resolve_blocks
from pyspatialuncertainty.continuous import BlockUncertainty, aggregate_continuous
from pyspatialuncertainty.correlation import (
    CallableCorrelation,
    CorrelationFunction,
    VariogramCorrelation,
    as_correlation,
    from_variogram,
)
from pyspatialuncertainty.grid import GridError, GridSpec
from pyspatialuncertainty.integrate import Integrand, MCResult, mc_block_variance
from pyspatialuncertainty.intervals import normal_interval, z_value
from pyspatialuncertainty.standardize import sd_from_quantiles, sd_from_samples, standardized_errors
from pyspatialuncertainty.variogram import (
    EmpiricalVariogram,
    Exponential,
    Gaussian,
    Matern,
    MaternStein,
    Nugget,
    Spherical,
    Structure,
    VariogramFit,
    VariogramModel,
    empirical_variogram,
    fit_variogram,
    fit_variogram_auto,
    vgm,
)

__version__ = _version("pyspatialuncertainty")

__all__ = [
    "BlockSet",
    "BlockUncertainty",
    "CallableCorrelation",
    "CorrelationFunction",
    "EmpiricalVariogram",
    "Exponential",
    "Gaussian",
    "GridError",
    "GridSpec",
    "Integrand",
    "MCResult",
    "Matern",
    "MaternStein",
    "Nugget",
    "ResolvedBlock",
    "Spherical",
    "Structure",
    "VariogramCorrelation",
    "VariogramFit",
    "VariogramModel",
    "__version__",
    "aggregate_continuous",
    "as_correlation",
    "blocks",
    "empirical_variogram",
    "fit_variogram",
    "fit_variogram_auto",
    "from_variogram",
    "mc_block_variance",
    "normal_interval",
    "resolve_blocks",
    "sd_average",
    "sd_from_quantiles",
    "sd_from_samples",
    "sd_independent",
    "standardized_errors",
    "vgm",
    "z_value",
]
