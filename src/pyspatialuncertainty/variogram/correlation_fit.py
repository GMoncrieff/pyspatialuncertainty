"""Step 4 in one call: fit a model of the correlation of the standardized errors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from pyspatialuncertainty._types import FloatArray
from pyspatialuncertainty.correlation import VariogramCorrelation
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
from pyspatialuncertainty.variogram.likelihood import (
    LikelihoodFit,
    MeanModel,
    _column,
    fit_likelihood,
)
from pyspatialuncertainty.variogram.models import ModelType, VariogramModel

CorrelationMethod = Literal["likelihood", "wls"]


@dataclass(frozen=True)
class CorrelationFit:
    """A fitted model of the spatial correlation of standardized errors.

    Attributes
    ----------
    model
        The fitted variogram model of the standardized errors. Pass it to
        `aggregate_continuous` or `aggregate_categorical`, which use its correlogram
        ρ(h) = (sill − γ(h)) / sill.
    method
        ``"likelihood"`` or ``"wls"``.
    empirical
        The sample variogram of the standardized errors. It is computed for either method
        so that the fit can be checked visually.
    details
        The underlying `LikelihoodFit` or `VariogramFit`.
    x, y, z
        Coordinates and standardized errors of the validation points used for the fit
        (rows with missing values removed). `aggregate_continuous` uses them to estimate
        the mean of the errors.
    """

    model: VariogramModel
    method: CorrelationMethod
    empirical: EmpiricalVariogram
    details: LikelihoodFit | VariogramFit
    x: FloatArray | None = field(default=None, repr=False)
    y: FloatArray | None = field(default=None, repr=False)
    z: FloatArray | None = field(default=None, repr=False)

    @property
    def correlation(self) -> VariogramCorrelation:
        """The correlogram ρ(h) = (sill − γ(h)) / sill of the fitted model."""
        return VariogramCorrelation(self.model)


def fit_correlation(
    data: pl.DataFrame | None = None,
    value: str | ArrayLike = "std_error",
    x: str | ArrayLike = "x",
    y: str | ArrayLike = "y",
    *,
    method: CorrelationMethod = "likelihood",
    mean: MeanModel = "constant",
    models: Sequence[ModelType] = ("Exp", "Sph", "Gau"),
    model: VariogramModel | None = None,
    fit_method: FitMethod = 7,
    cutoff: float | None = None,
    width: float | None = None,
    estimator: Estimator = "matheron",
) -> CorrelationFit:
    """Fit the spatial correlation of standardized prediction errors (step 4).

    Two methods are available.

    * ``"likelihood"`` (default): a Gaussian likelihood fit of a nugget plus an
      exponential, spherical or Gaussian structure (`fit_likelihood`). By default the
      mean of the errors is treated as unknown (REML, ``mean="constant"``), which goes
      with the estimated-mean correction that `aggregate_continuous` and
      `aggregate_categorical` apply by default. ``mean="zero"`` (ML with the mean fixed
      at zero) goes with ``mean="zero"`` there.
    * ``"wls"``: the procedure of the paper and the authors' R script. A gstat-style
      sample variogram is fitted by weighted least squares, either automatically among
      nugget + Exp/Sph/Gau (`fit_variogram_auto`) or from the starting ``model``
      (`fit_variogram`). Together with ``mean="zero"`` in `aggregate_continuous` this
      reproduces Wadoux & Heuvelink (2023).

    Parameters
    ----------
    data
        Polars data frame holding the columns named by ``value``, ``x`` and ``y``. If
        ``None``, these must be arrays.
    value, x, y
        Column names (or arrays) of the standardized errors and the coordinates.
    method
        ``"likelihood"`` or ``"wls"``.
    mean
        Likelihood only: ``"constant"`` (REML with an unknown mean, the default) or
        ``"zero"`` (ML with the mean fixed at zero).
    models
        Structure types to choose from (likelihood, and WLS without a starting model).
    model
        WLS only: a starting model, fitted with `fit_variogram` instead of the automatic
        choice.
    fit_method
        WLS only: gstat weighting scheme (default 7).
    cutoff, width, estimator
        Passed to `empirical_variogram`.

    Returns
    -------
    CorrelationFit
        The fitted model, the sample variogram and the fit details.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> rng = np.random.default_rng(0)
    >>> val = pl.DataFrame({"x": rng.uniform(0, 100, 200), "y": rng.uniform(0, 100, 200),
    ...                     "std_error": rng.normal(size=200)})
    >>> fit = fit_correlation(val)
    >>> fit.method, round(float(fit.correlation(np.array([0.0]))[0]), 6)
    ('likelihood', 1.0)
    """
    if method not in ("likelihood", "wls"):
        raise ValueError("method must be 'likelihood' or 'wls'")
    if model is not None and method != "wls":
        raise ValueError("a starting model is only used by the WLS fit; pass method='wls'")
    ev = empirical_variogram(data, value, x, y, cutoff=cutoff, width=width, estimator=estimator)
    details: LikelihoodFit | VariogramFit
    if method == "likelihood":
        details = fit_likelihood(data, value, x, y, mean=mean, models=models)
    elif model is None:
        details = fit_variogram_auto(ev, models=models, fit_method=fit_method)
    else:
        details = fit_variogram(ev, model, fit_method=fit_method)
    zs = _column(data, value, "value")
    xs = _column(data, x, "x")
    ys = _column(data, y, "y")
    ok = np.isfinite(zs) & np.isfinite(xs) & np.isfinite(ys)
    return CorrelationFit(
        model=details.model,
        method=method,
        empirical=ev,
        details=details,
        x=xs[ok],
        y=ys[ok],
        z=zs[ok],
    )
