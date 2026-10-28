"""Uncertainty of spatial averages and totals of maps.

A Python implementation of the Monte Carlo integration method of Wadoux & Heuvelink
(2023), *Uncertainty of spatial averages and totals of natural resource maps*, Methods in
Ecology and Evolution 14:1320-1332, https://doi.org/10.1111/2041-210X.14106, for continuous
maps, together with a per-class analogue for maps of class probabilities.

The method in eight steps (Section 4.1 of the paper):

1. quantify the point-support prediction uncertainty (e.g. a quantile regression forest);
2. compute the prediction-error standard deviation σ(s) (`sd_from_samples`,
   `sd_from_quantiles`);
3. standardize validation errors by σ (`standardized_errors`);
4. model the correlation of the standardized errors (`fit_correlation`: a Gaussian
   likelihood fit by default, or the paper's WLS variogram fit with ``method="wls"``;
   also `empirical_variogram`, `fit_variogram`, `vgm`);
5. - 6. integrate σ(s)σ(u)ρ(|s − u|) over each block by Monte Carlo
   (`aggregate_continuous`);
7. scale the variance by |B|² for block totals;
8. report standard deviations and prediction intervals.

For maps of class probabilities, `indicator_errors`, `fit_class_variograms` and
`aggregate_categorical` apply the same procedure per class to the indicator errors
I_k − p_k with σ_k = sqrt(p_k(1 − p_k)).

Defaults differ from the paper in one respect: the mean of the standardized errors is
treated as unknown. The correlogram is fitted by REML, the mean is estimated by GLS from
the validation points, and block predictions and variances are corrected for it
(`pyspatialuncertainty.mean_correction`). This keeps intervals honest when a map is
biased or its probabilities are miscalibrated. The paper's procedure is
``fit_correlation(..., method="wls")`` followed by
``aggregate_continuous(..., mean="zero")``.
"""

from importlib.metadata import version as _version

from pyspatialuncertainty import blocks
from pyspatialuncertainty.baselines import sd_average, sd_independent
from pyspatialuncertainty.blocks import BlockSet, ResolvedBlock, resolve_blocks
from pyspatialuncertainty.categorical import (
    ClassAreaUncertainty,
    ClassVariogram,
    ProbabilityError,
    aggregate_categorical,
    clip_probabilities,
    fit_class_variograms,
    indicator_errors,
    temperature_scale,
    validate_probabilities,
)
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
from pyspatialuncertainty.mean_correction import MeanEstimate, estimate_mean
from pyspatialuncertainty.standardize import sd_from_quantiles, sd_from_samples, standardized_errors
from pyspatialuncertainty.variogram import (
    CorrelationFit,
    EmpiricalVariogram,
    Exponential,
    Gaussian,
    LikelihoodFit,
    Matern,
    MaternStein,
    Nugget,
    Spherical,
    Structure,
    VariogramFit,
    VariogramModel,
    empirical_variogram,
    fit_correlation,
    fit_likelihood,
    fit_variogram,
    fit_variogram_auto,
    vgm,
)

__version__ = _version("pyspatialuncertainty")

__all__ = [
    "BlockSet",
    "BlockUncertainty",
    "CallableCorrelation",
    "ClassAreaUncertainty",
    "ClassVariogram",
    "CorrelationFit",
    "CorrelationFunction",
    "EmpiricalVariogram",
    "Exponential",
    "Gaussian",
    "GridError",
    "GridSpec",
    "Integrand",
    "LikelihoodFit",
    "MCResult",
    "Matern",
    "MaternStein",
    "MeanEstimate",
    "Nugget",
    "ProbabilityError",
    "ResolvedBlock",
    "Spherical",
    "Structure",
    "VariogramCorrelation",
    "VariogramFit",
    "VariogramModel",
    "__version__",
    "aggregate_categorical",
    "aggregate_continuous",
    "as_correlation",
    "blocks",
    "clip_probabilities",
    "empirical_variogram",
    "estimate_mean",
    "fit_class_variograms",
    "fit_correlation",
    "fit_likelihood",
    "fit_variogram",
    "fit_variogram_auto",
    "from_variogram",
    "indicator_errors",
    "mc_block_variance",
    "normal_interval",
    "resolve_blocks",
    "sd_average",
    "sd_from_quantiles",
    "sd_from_samples",
    "sd_independent",
    "standardized_errors",
    "temperature_scale",
    "validate_probabilities",
    "vgm",
    "z_value",
]
