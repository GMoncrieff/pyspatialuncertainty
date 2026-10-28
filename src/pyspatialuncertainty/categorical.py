"""Uncertainty of class fractions and areas from maps of class probabilities.

This is the categorical analogue of the continuous method, kept as close to it as
possible and restricted to the marginal uncertainty of each class. For class k with
calibrated probability p_k(s) and indicator I_k(s) (1 if s belongs to class k):

* the prediction error is e_k(s) = I_k(s) − p_k(s), which has mean zero and standard
  deviation σ_k(s) = sqrt(p_k(s)(1 − p_k(s))) when the probabilities are calibrated;
* standardized errors e_k / σ_k at validation locations (using out-of-sample probabilities)
  are used to fit one variogram, hence one correlation function ρ_k, per class;
* the variance of the error of the block fraction of class k is the Monte Carlo integral of
  σ_k(s)σ_k(u)ρ_k(|s − u|) over B × B, exactly as in the continuous case;
* the expected fraction of class k in B is the block mean of p_k and the expected class
  area is |B| times that fraction.

Probabilities are rarely perfectly calibrated, so by default the mean of each class's
standardized errors is estimated from the validation points and the fractions and their
variances are corrected for it, exactly as for continuous maps (see
`pyspatialuncertainty.mean_correction`). ``mean="zero"`` trusts the calibration.

Cross-class covariances (the class fractions in a block sum to one) and the joint
distribution of class areas are not modelled.
"""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import xarray as xr
from numpy.typing import ArrayLike

from pyspatialuncertainty._results import ids_to_series, to_xarray_cells, to_xarray_tiles
from pyspatialuncertainty._types import FloatArray, IntArray, SeedLike
from pyspatialuncertainty.blocks import BlockSet, resolve_blocks
from pyspatialuncertainty.continuous import block_means
from pyspatialuncertainty.correlation import CorrelationFunction, as_correlation
from pyspatialuncertainty.integrate import Integrand, mc_block_variance
from pyspatialuncertainty.intervals import normal_interval, z_value
from pyspatialuncertainty.mean_correction import (
    MeanEstimate,
    MeanTreatment,
    block_terms,
    check_fit_mean,
    estimate_mean,
    validation_points,
    variogram_of,
)
from pyspatialuncertainty.variogram.correlation_fit import (
    CorrelationFit,
    CorrelationMethod,
    fit_correlation,
)
from pyspatialuncertainty.variogram.empirical import EmpiricalVariogram, Estimator
from pyspatialuncertainty.variogram.fit import FitMethod, VariogramFit
from pyspatialuncertainty.variogram.likelihood import LikelihoodFit, MeanModel
from pyspatialuncertainty.variogram.models import VariogramModel


class ProbabilityError(ValueError):
    """Raised when class probabilities are not valid, calibrated-looking probabilities."""


def _class_axis(probabilities: xr.DataArray, class_dim: str) -> list[Hashable]:
    if class_dim not in probabilities.dims:
        raise ProbabilityError(f"the probabilities have no {class_dim!r} dimension")
    if class_dim in probabilities.coords:
        return [c.item() if hasattr(c, "item") else c for c in probabilities[class_dim].values]
    return list(range(probabilities.sizes[class_dim]))


def _check_matrix(p: FloatArray, atol: float, what: str) -> None:
    """Check an array of shape (n, K) of probabilities; rows that are all NaN are skipped."""
    if p.ndim != 2 or p.shape[1] < 2:
        raise ProbabilityError(f"{what}: at least two classes are required")
    finite = np.isfinite(p)
    rows_all = finite.all(axis=1)
    rows_none = ~finite.any(axis=1)
    if np.any(~rows_all & ~rows_none):
        raise ProbabilityError(f"{what}: some locations have probabilities for only some classes")
    q = p[rows_all]
    if q.size == 0:
        raise ProbabilityError(f"{what}: no location has valid probabilities")
    if np.any(q <= 0.0) or np.any(q >= 1.0):
        n_bad = int(np.sum((q <= 0.0) | (q >= 1.0)))
        raise ProbabilityError(
            f"{what}: {n_bad} probabilities are hard 0 or 1 (or outside (0, 1)). The method "
            "needs calibrated probabilities strictly between 0 and 1, since σ = sqrt(p(1 − p)) "
            "must be positive; use clip_probabilities() explicitly if appropriate"
        )
    dev = np.abs(q.sum(axis=1) - 1.0)
    if np.any(dev > atol):
        worst = float(dev.max())
        raise ProbabilityError(
            f"{what}: probabilities do not sum to one (max deviation {worst:.3g} > atol={atol})"
        )


def validate_probabilities(
    probabilities: xr.DataArray | ArrayLike,
    *,
    class_dim: str = "class",
    atol: float = 1e-6,
) -> None:
    """Check that class probabilities sum to one and are strictly between 0 and 1.

    Parameters
    ----------
    probabilities
        A DataArray with a class dimension (e.g. dims ``(class, y, x)``), or an array of shape
        ``(n, K)`` with one column per class. Locations where all classes are NaN are treated
        as unmapped and skipped.
    class_dim
        Name of the class dimension of a DataArray.
    atol
        Tolerance on the deviation of the class sum from one.

    Raises
    ------
    ProbabilityError
        If probabilities are missing for some classes only, do not sum to one, or are hard
        zeros or ones.
    """
    if isinstance(probabilities, xr.DataArray):
        _class_axis(probabilities, class_dim)
        arr = np.moveaxis(
            np.asarray(probabilities.values, dtype=np.float64),
            probabilities.get_axis_num(class_dim),
            -1,
        )
        _check_matrix(arr.reshape(-1, arr.shape[-1]), atol, "probability map")
    else:
        _check_matrix(np.asarray(probabilities, dtype=np.float64), atol, "probabilities")


def clip_probabilities(
    probabilities: xr.DataArray, eps: float = 1e-6, *, class_dim: str = "class"
) -> xr.DataArray:
    """Clip probabilities to [eps, 1 − eps] and renormalise them to sum to one.

    This is never applied automatically; call it explicitly when probabilities contain hard
    zeros or ones that you are willing to soften.
    """
    if not 0 < eps < 0.5:
        raise ValueError("eps must lie in (0, 0.5)")
    clipped = probabilities.clip(eps, 1.0 - eps)
    return clipped / clipped.sum(class_dim)


def temperature_scale(
    probabilities: xr.DataArray | ArrayLike,
    temperature: float,
    *,
    class_dim: str | int = "class",
) -> Any:
    """Temperature scaling p_k^(1/T) / Σ_j p_j^(1/T).

    T < 1 makes the probabilities more confident (sharper), T > 1 less confident; T = 1
    leaves them unchanged.

    Parameters
    ----------
    probabilities
        DataArray with a class dimension, or an array with classes along axis ``class_dim``.
    temperature
        The temperature T > 0.
    class_dim
        Class dimension name (DataArray) or axis (array).

    Returns
    -------
    DataArray or ndarray
        The rescaled probabilities.
    """
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    if isinstance(probabilities, xr.DataArray):
        if not isinstance(class_dim, str):
            raise TypeError("class_dim must be a dimension name for a DataArray")
        logp = probabilities.copy(data=np.log(np.asarray(probabilities.values, dtype=np.float64)))
        logp = logp / temperature
        logp = logp - logp.max(class_dim)
        weights = logp.copy(data=np.exp(logp.values))
        return weights / weights.sum(class_dim)
    axis = class_dim if isinstance(class_dim, int) else -1
    lp = np.log(np.asarray(probabilities, dtype=np.float64)) / temperature
    lp -= lp.max(axis=axis, keepdims=True)
    w = np.exp(lp)
    return w / w.sum(axis=axis, keepdims=True)


def indicator_errors(
    data: pl.DataFrame,
    probability_columns: Mapping[Hashable, str],
    label: str = "label",
    *,
    x: str = "x",
    y: str = "y",
    atol: float = 1e-6,
) -> pl.DataFrame:
    """Indicator prediction errors and their standardized values at validation locations.

    Parameters
    ----------
    data
        Validation table with coordinates, the observed class and one column of
        out-of-sample (e.g. cross-validated) probabilities per class.
    probability_columns
        Mapping from class label to the column holding that class's probability.
    label
        Column with the observed class labels.
    x, y
        Coordinate columns.
    atol
        Tolerance for the sum-to-one check.

    Returns
    -------
    polars.DataFrame
        Long table with one row per location and class and columns ``x``, ``y``,
        ``class``, ``indicator`` (I_k), ``probability`` (p_k), ``error`` (I_k − p_k),
        ``sd`` (sqrt(p_k(1 − p_k))) and ``std_error`` (error / sd).
    """
    classes = list(probability_columns)
    cols = [probability_columns[c] for c in classes]
    p = data.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()
    if np.any(~np.isfinite(p)):
        raise ProbabilityError("validation probabilities must all be finite")
    _check_matrix(p, atol, "validation probabilities")
    labels = data[label].to_list()
    unknown = {v for v in labels if v not in set(classes)}
    if unknown:
        raise ProbabilityError(
            f"observed labels {sorted(map(str, unknown))} are not among the classes"
        )
    xs = data[x].cast(pl.Float64).to_numpy()
    ys = data[y].cast(pl.Float64).to_numpy()
    frames = []
    lab = np.asarray(labels, dtype=object)
    for k, cls in enumerate(classes):
        ind = (lab == cls).astype(np.float64)
        pk = p[:, k]
        sd = np.sqrt(pk * (1.0 - pk))
        err = ind - pk
        frames.append(
            pl.DataFrame(
                {
                    "x": xs,
                    "y": ys,
                    "class": pl.Series([cls] * len(xs)),
                    "indicator": ind,
                    "probability": pk,
                    "error": err,
                    "sd": sd,
                    "std_error": err / sd,
                }
            )
        )
    return pl.concat(frames, how="vertical")


@dataclass(frozen=True)
class ClassVariogram:
    """Sample variogram and fitted model of the standardized errors of one class.

    Attributes
    ----------
    cls
        The class label.
    empirical
        Sample variogram of the standardized indicator errors.
    fit
        The `LikelihoodFit` (default) or the WLS `VariogramFit`.
    x, y, z
        Coordinates and standardized errors of the class's validation points, used by
        `aggregate_categorical` to estimate the mean of the errors.
    """

    cls: Hashable
    empirical: EmpiricalVariogram
    fit: LikelihoodFit | VariogramFit
    x: FloatArray | None = field(default=None, repr=False)
    y: FloatArray | None = field(default=None, repr=False)
    z: FloatArray | None = field(default=None, repr=False)

    @property
    def model(self) -> VariogramModel:
        """The fitted variogram model."""
        return self.fit.model


def fit_class_variograms(
    errors: pl.DataFrame,
    *,
    method: CorrelationMethod | None = None,
    mean: MeanModel = "constant",
    model: VariogramModel | Mapping[Hashable, VariogramModel] | None = None,
    cutoff: float | None = None,
    width: float | None = None,
    estimator: Estimator = "matheron",
    fit_method: FitMethod = 7,
) -> dict[Hashable, ClassVariogram]:
    """Fit one correlation model per class to standardized indicator errors (step 4).

    Each class is fitted with `fit_correlation`: by default a Gaussian likelihood fit with
    an unknown mean of the errors (REML), which goes with the estimated-mean correction of
    `aggregate_categorical`. ``mean="zero"`` fixes the mean at zero, as perfectly
    calibrated probabilities imply.

    Parameters
    ----------
    errors
        Output of `indicator_errors`.
    method
        ``"likelihood"`` or ``"wls"`` (the paper's weighted least-squares variogram fit).
        The default is ``"likelihood"``, or ``"wls"`` when a starting ``model`` is given.
    mean
        Likelihood only: ``"constant"`` (REML, default) or ``"zero"`` (ML).
    model
        WLS only: starting model(s), one for all classes or a mapping per class. Without
        one, the WLS fit chooses among nugget + exponential/spherical/Gaussian models
        (`fit_variogram_auto`).
    cutoff, width, estimator
        Passed to `empirical_variogram`.
    fit_method
        WLS only: gstat weighting scheme.

    Returns
    -------
    dict
        Class label → `ClassVariogram`.
    """
    if method is None:
        method = "likelihood" if model is None else "wls"
    out: dict[Hashable, ClassVariogram] = {}
    for cls in errors["class"].unique(maintain_order=True).to_list():
        sub = errors.filter(pl.col("class") == cls)
        start = model[cls] if isinstance(model, Mapping) else model
        fit = fit_correlation(
            sub,
            "std_error",
            "x",
            "y",
            method=method,
            mean=mean,
            model=start,
            fit_method=fit_method,
            cutoff=cutoff,
            width=width,
            estimator=estimator,
        )
        out[cls] = ClassVariogram(
            cls=cls, empirical=fit.empirical, fit=fit.details, x=fit.x, y=fit.y, z=fit.z
        )
    return out


@dataclass(frozen=True)
class ClassAreaUncertainty:
    """Expected class fractions and areas per block, with their uncertainty.

    Attributes
    ----------
    table
        One row per class and block with columns ``class``, ``block_id``, ``area`` (|B|),
        ``n_cells``, ``map_fraction`` (block mean of p_k), ``shift`` (correction for the
        estimated mean error, zero with ``mean="zero"``), ``fraction`` (``map_fraction +
        shift``), ``var_fraction`` and ``mcse_var``, ``var_known_mean`` (the Monte Carlo
        integral alone), ``sd_fraction``, ``lower`` and ``upper`` (prediction interval of
        the fraction, clipped to [0, 1]), ``expected_area`` (|B| · fraction), ``sd_area``,
        ``area_lower`` and ``area_upper``.
    blockset
        The blocks.
    classes
        Class labels, in the order of the probability map.
    n_mc
        Number of Monte Carlo point pairs per block.
    level
        Nominal coverage of the prediction intervals.
    z
        Normal quantile used for the intervals.
    mean
        ``"constant"`` (the mean of each class's standardized errors was estimated) or
        ``"zero"``.
    mean_estimates
        Class label → estimated mean and its uncertainty (empty with ``mean="zero"``).
    """

    table: pl.DataFrame
    blockset: BlockSet
    classes: tuple[Hashable, ...]
    n_mc: int
    level: float
    z: float
    cells: tuple[IntArray, ...] = field(repr=False, default=())
    mean: MeanTreatment = "constant"
    mean_estimates: dict[Hashable, MeanEstimate] = field(default_factory=dict, repr=False)

    def to_xarray(
        self, columns: Sequence[str] | None = None, *, coarse: bool = False
    ) -> xr.Dataset:
        """Results as an xarray Dataset with a ``class`` dimension (see
        `to_xarray`)."""
        cols = (
            list(columns)
            if columns is not None
            else [
                "fraction",
                "sd_fraction",
                "lower",
                "upper",
                "expected_area",
                "sd_area",
            ]
        )
        if coarse:
            return to_xarray_tiles(self.blockset, self.table, cols, group_col="class")
        return to_xarray_cells(self.blockset, self.cells, self.table, cols, group_col="class")


def aggregate_categorical(
    probabilities: xr.DataArray,
    correlations: CorrelationFunction
    | VariogramModel
    | Mapping[Hashable, ClassVariogram | CorrelationFit | CorrelationFunction | VariogramModel],
    blocks: BlockSet,
    *,
    class_dim: str = "class",
    mean: MeanTreatment = "constant",
    validation: pl.DataFrame | None = None,
    cross_term: bool = True,
    n_mc: int = 10_000,
    level: float = 0.90,
    z: float | None = None,
    seed: SeedLike = None,
    area_scale: float = 1.0,
    atol: float = 1e-6,
) -> ClassAreaUncertainty:
    """Expected class fractions and areas per block and their uncertainty.

    Parameters
    ----------
    probabilities
        Calibrated class probabilities on the grid of ``blocks``, with a class dimension.
        They must sum to one and lie strictly between 0 and 1 (checked).
    correlations
        Correlation model of the standardized indicator errors for each class: the
        mapping returned by `fit_class_variograms` (which also holds the validation
        points), a mapping from class label to a variogram model or fit, or one model used
        for all classes. With ``mean="zero"`` any correlation callables will do.
    blocks
        Blocks to aggregate over.
    class_dim
        Name of the class dimension.
    mean
        ``"constant"`` (default): the mean of each class's standardized errors is
        estimated by GLS from the validation points; fractions are shifted by it and its
        uncertainty is added to their variances (see
        `pyspatialuncertainty.mean_correction`). ``"zero"``: the probabilities are trusted
        to be calibrated and the errors to have mean zero.
    validation
        Standardized indicator errors at the validation points, as returned by
        `indicator_errors` (columns ``x``, ``y``, ``class`` and ``std_error``). Defaults to
        the points stored on the `ClassVariogram` fits.
    cross_term
        With ``mean="constant"``, subtract the covariance between the block error and the
        estimated mean (default). Without it the intervals are wider.
    n_mc
        Number of pairs of uniform points per block; all classes share the same pairs.
    level, z
        Coverage or explicit normal multiplier of the prediction intervals.
    seed
        Random seed.
    area_scale
        Factor converting squared CRS units to the reported area unit.
    atol
        Tolerance of the sum-to-one check.

    Returns
    -------
    ClassAreaUncertainty
        Per class and block: expected fraction and area, standard deviations and prediction
        intervals.
    """
    if mean not in ("constant", "zero"):
        raise ValueError("mean must be 'constant' or 'zero'")
    classes = _class_axis(probabilities, class_dim)
    validate_probabilities(probabilities, class_dim=class_dim, atol=atol)
    if isinstance(correlations, Mapping):
        missing = [c for c in classes if c not in correlations]
        if missing:
            raise ValueError(f"no correlation function for classes {missing}")
        given = [correlations[c] for c in classes]
    else:
        given = [correlations] * len(classes)
    corrs = [as_correlation(c) for c in given]
    estimates: dict[Hashable, MeanEstimate] = {}
    if mean == "constant":
        for cls, c in zip(classes, given, strict=True):
            model = variogram_of(c)
            if model is None:
                raise ValueError(
                    f"estimating the mean (mean='constant') needs a variogram model or fit for "
                    f"class {cls!r}; use mean='zero' with plain correlation functions"
                )
            check_fit_mean(c)
            if validation is not None:
                if "class" not in validation.columns:
                    raise ValueError("validation needs a 'class' column (see indicator_errors)")
                pts = validation_points(None, validation.filter(pl.col("class") == cls))
            else:
                pts = validation_points(c, None, what="the fits of fit_class_variograms")
            estimates[cls] = estimate_mean(model, *pts)

    g = blocks.grid
    p_flat = [g.values(probabilities.isel({class_dim: k})) for k in range(len(classes))]
    valid = np.all([np.isfinite(p) for p in p_flat], axis=0)
    resolved = resolve_blocks(blocks, valid)
    sds = [np.sqrt(p * (1.0 - p)) for p in p_flat]
    mc = mc_block_variance(
        resolved, [Integrand(s, c) for s, c in zip(sds, corrs, strict=True)], n_mc=n_mc, seed=seed
    )
    mult = z_value(level) if z is None else float(z)
    area = np.array([b.area for b in resolved]) * area_scale
    ids = ids_to_series([b.id for b in resolved])
    frames = []
    for k, cls in enumerate(classes):
        map_frac = block_means(resolved, p_flat[k])
        var_known = mc.variance[k]
        shift = np.zeros(len(resolved))
        var = var_known
        if cls in estimates:
            terms = block_terms(estimates[cls], sds[k], g, resolved, cross_term=cross_term)
            shift = terms.shift
            var = var_known + terms.var_beta - terms.cross
            with np.errstate(invalid="ignore"):
                noisy = np.isfinite(var) & (var < 3.0 * mc.mcse[k])
            if np.any(noisy):
                warnings.warn(
                    f"class {cls!r}: for {int(noisy.sum())} block(s) the corrected variance is "
                    "less than three Monte Carlo standard errors (it can even be negative); "
                    "increase n_mc",
                    stacklevel=2,
                )
        frac = map_frac + shift
        sd = np.sqrt(np.maximum(var, 0.0))
        lower, upper = normal_interval(frac, sd, z=mult, lower_bound=0.0, upper_bound=1.0)
        frames.append(
            pl.DataFrame(
                [
                    pl.Series("class", [cls] * len(resolved)),
                    ids,
                    pl.Series("area", area),
                    pl.Series("n_cells", [b.n_cells for b in resolved], dtype=pl.Int64),
                    pl.Series("map_fraction", map_frac),
                    pl.Series("shift", shift),
                    pl.Series("fraction", frac),
                    pl.Series("var_fraction", var),
                    pl.Series("mcse_var", mc.mcse[k]),
                    pl.Series("var_known_mean", var_known),
                    pl.Series("sd_fraction", sd),
                    pl.Series("lower", lower),
                    pl.Series("upper", upper),
                    pl.Series("expected_area", area * frac),
                    pl.Series("sd_area", area * sd),
                    pl.Series("area_lower", area * lower),
                    pl.Series("area_upper", area * upper),
                ]
            )
        )
    return ClassAreaUncertainty(
        table=pl.concat(frames, how="vertical"),
        blockset=blocks,
        classes=tuple(classes),
        n_mc=n_mc,
        level=level,
        z=mult,
        cells=tuple(b.cells for b in resolved),
        mean=mean,
        mean_estimates=estimates,
    )
