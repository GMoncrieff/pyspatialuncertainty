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

Cross-class covariances (the class fractions in a block sum to one) and the joint
distribution of class areas are not modelled.
"""

from __future__ import annotations

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
from pyspatialuncertainty.variogram.empirical import (
    EmpiricalVariogram,
    Estimator,
    empirical_variogram,
)
from pyspatialuncertainty.variogram.fit import (
    FitMethod,
    VariogramFit,
    fit_variogram,
    fit_variogram_auto,
)
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
    """Sample variogram and fitted model of the standardized errors of one class."""

    cls: Hashable
    empirical: EmpiricalVariogram
    fit: VariogramFit

    @property
    def model(self) -> VariogramModel:
        """The fitted variogram model."""
        return self.fit.model


def fit_class_variograms(
    errors: pl.DataFrame,
    *,
    model: VariogramModel | Mapping[Hashable, VariogramModel] | None = None,
    cutoff: float | None = None,
    width: float | None = None,
    estimator: Estimator = "matheron",
    fit_method: FitMethod = 7,
) -> dict[Hashable, ClassVariogram]:
    """Fit one variogram per class to standardized indicator errors (step 4, per class).

    Parameters
    ----------
    errors
        Output of `indicator_errors`.
    model
        Initial model(s): one model for all classes, a mapping per class, or ``None`` to
        select automatically among nugget + exponential/spherical/Gaussian models
        (`fit_variogram_auto`).
    cutoff, width, estimator
        Passed to `empirical_variogram`.
    fit_method
        gstat weighting scheme for the fit.

    Returns
    -------
    dict
        Class label → `ClassVariogram`.
    """
    out: dict[Hashable, ClassVariogram] = {}
    for cls in errors["class"].unique(maintain_order=True).to_list():
        sub = errors.filter(pl.col("class") == cls)
        ev = empirical_variogram(
            sub, "std_error", "x", "y", cutoff=cutoff, width=width, estimator=estimator
        )
        if model is None:
            fit = fit_variogram_auto(ev, fit_method=fit_method)
        else:
            start = model[cls] if isinstance(model, Mapping) else model
            fit = fit_variogram(ev, start, fit_method=fit_method)
        out[cls] = ClassVariogram(cls=cls, empirical=ev, fit=fit)
    return out


@dataclass(frozen=True)
class ClassAreaUncertainty:
    """Expected class fractions and areas per block, with their uncertainty.

    Attributes
    ----------
    table
        One row per class and block with columns ``class``, ``block_id``, ``area`` (|B|),
        ``n_cells``, ``fraction`` (block mean of p_k), ``var_fraction`` and ``mcse_var``,
        ``sd_fraction``, ``lower`` and ``upper`` (prediction interval of the fraction, clipped
        to [0, 1]), ``expected_area`` (|B| · fraction), ``sd_area``, ``area_lower`` and
        ``area_upper``.
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
    """

    table: pl.DataFrame
    blockset: BlockSet
    classes: tuple[Hashable, ...]
    n_mc: int
    level: float
    z: float
    cells: tuple[IntArray, ...] = field(repr=False, default=())

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
    | Mapping[Hashable, CorrelationFunction | VariogramModel],
    blocks: BlockSet,
    *,
    class_dim: str = "class",
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
        Correlation function of the standardized indicator errors for each class (a mapping
        from class label), or one function or variogram model used for all classes.
    blocks
        Blocks to aggregate over.
    class_dim
        Name of the class dimension.
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
    classes = _class_axis(probabilities, class_dim)
    validate_probabilities(probabilities, class_dim=class_dim, atol=atol)
    if isinstance(correlations, Mapping):
        missing = [c for c in classes if c not in correlations]
        if missing:
            raise ValueError(f"no correlation function for classes {missing}")
        corrs = [as_correlation(correlations[c]) for c in classes]
    else:
        shared = as_correlation(correlations)
        corrs = [shared] * len(classes)

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
        frac = block_means(resolved, p_flat[k])
        var = mc.variance[k]
        sd = np.sqrt(np.maximum(var, 0.0))
        lower, upper = normal_interval(frac, sd, z=mult, lower_bound=0.0, upper_bound=1.0)
        frames.append(
            pl.DataFrame(
                [
                    pl.Series("class", [cls] * len(resolved)),
                    ids,
                    pl.Series("area", area),
                    pl.Series("n_cells", [b.n_cells for b in resolved], dtype=pl.Int64),
                    pl.Series("fraction", frac),
                    pl.Series("var_fraction", var),
                    pl.Series("mcse_var", mc.mcse[k]),
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
    )
