"""Gaussian likelihood fits of the correlation of standardized prediction errors.

The model for the standardized errors z at n validation locations is

    z ~ N(μ·1, s²·[(1 − η)·R_a + η·I]),

where R_a is the correlation matrix of an exponential, spherical or Gaussian structure
with range a, and η is the nugget share. The sill s² is profiled out analytically. η and
a are found by a coarse grid search refined with L-BFGS-B. The model type with the
largest likelihood is returned as a ``Nug + <model>`` variogram model.

The mean can be handled in two ways.

* ``mean="constant"`` (default): restricted maximum likelihood (REML) with an unknown
  constant mean, which, like the sample variogram, ignores a common offset. This goes
  with the estimated-mean correction that the aggregation functions apply by default
  (see `pyspatialuncertainty.mean_correction`): the offset is estimated and propagated
  there instead of being read as correlation.
* ``mean="zero"``: maximum likelihood with μ = 0, as the original method assumes
  (unbiased predictions, or calibrated probabilities). A common offset of the validation
  errors then counts as evidence of long-range correlation, the error component that
  dominates the uncertainty of large blocks. Use it with ``mean="zero"`` in the
  aggregation functions.

In simulation studies on known truth (``experiments/improvements`` in the repository),
both likelihood fits gave better-calibrated block intervals than the WLS variogram fit,
most of all for large blocks, and REML with the estimated-mean correction was the most
robust to miscalibrated maps.

Each likelihood evaluation factorises a dense n × n matrix, so the cost grows as n³:
about 1 s for 300 points and 30 s for 1,500 points on one core. Above
`LIKELIHOOD_WARN_N` points a warning suggests subsampling or the WLS variogram fit.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import ArrayLike
from scipy import linalg, optimize, special

from pyspatialuncertainty._types import FloatArray
from pyspatialuncertainty.variogram.models import ModelType, VariogramModel, vgm

MeanModel = Literal["zero", "constant"]

LIKELIHOOD_WARN_N = 2000
"""Number of points above which `fit_likelihood` warns about its run time."""

_PRACTICAL = {"Exp": 3.0, "Sph": 1.0, "Gau": math.sqrt(3.0)}
_GRID_NUGGET = (0.03, 0.15, 0.35, 0.6, 0.85, 0.97)
_GRID_RANGES = 9
_LOGIT_BOUND = 6.0  # nugget share between about 0.0025 and 0.9975
_FAILED = 1e300  # objective value where the covariance matrix cannot be factorised


@dataclass(frozen=True)
class LikelihoodFit:
    """Result of a Gaussian likelihood fit.

    Attributes
    ----------
    model
        The fitted ``Nug + <model>`` variogram model. Its sill is the estimated variance of
        the standardized errors.
    loglik
        Maximised log-likelihood (restricted log-likelihood for ``mean="constant"``).
    mean
        ``"zero"`` (ML with mean zero) or ``"constant"`` (REML with a constant mean).
    n_obs
        Number of observations used.
    converged
        Whether the optimiser reported convergence for the selected model type.
    """

    model: VariogramModel
    loglik: float
    mean: MeanModel
    n_obs: int
    converged: bool


def _column(data: pl.DataFrame | None, value: str | ArrayLike, name: str) -> FloatArray:
    if isinstance(value, str):
        if data is None:
            raise ValueError(f"{name}={value!r} is a column name but no data frame was given")
        return data[value].cast(pl.Float64).to_numpy()
    return np.asarray(value, dtype=np.float64).ravel()


def fit_likelihood(
    data: pl.DataFrame | None = None,
    value: str | ArrayLike = "std_error",
    x: str | ArrayLike = "x",
    y: str | ArrayLike = "y",
    *,
    mean: MeanModel = "constant",
    models: Sequence[ModelType] = ("Exp", "Sph", "Gau"),
    max_practical_range: float | None = None,
) -> LikelihoodFit:
    """Fit ``Nug + <model>`` to standardized errors by Gaussian (restricted) maximum likelihood.

    Parameters
    ----------
    data
        Polars data frame holding the columns named by ``value``, ``x`` and ``y``. If
        ``None``, these must be arrays.
    value, x, y
        Column names (or arrays) of the standardized errors and the coordinates.
    mean
        ``"constant"`` (default) estimates a constant mean (REML). ``"zero"`` fixes the
        mean of the errors at zero (ML).
    models
        Structure types to try: any of ``"Exp"``, ``"Sph"`` and ``"Gau"``.
    max_practical_range
        Upper bound on the practical range. Defaults to twice the diagonal of the bounding
        box of the data.

    Returns
    -------
    LikelihoodFit
        The fitted model of the structure type with the largest likelihood.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> xy = rng.uniform(0, 100, size=(200, 2))
    >>> fit = fit_likelihood(None, rng.normal(size=200), xy[:, 0], xy[:, 1], mean="zero")
    >>> # independent errors: (almost) no correlation beyond the point spacing
    >>> bool(fit.model.correlation(np.array([10.0]))[0] < 0.1)
    True
    """
    if mean not in ("zero", "constant"):
        raise ValueError("mean must be 'zero' or 'constant'")
    unknown = [m for m in models if m not in _PRACTICAL]
    if unknown or not models:
        raise ValueError(f"models must be a non-empty subset of {sorted(_PRACTICAL)}")
    z = _column(data, value, "value")
    xs = _column(data, x, "x")
    ys = _column(data, y, "y")
    if not (z.size == xs.size == ys.size):
        raise ValueError("value, x and y must have the same length")
    ok = np.isfinite(z) & np.isfinite(xs) & np.isfinite(ys)
    z, xs, ys = z[ok], xs[ok], ys[ok]
    n = z.size
    if n < 3:
        raise ValueError("at least three observations are needed")
    if not np.any(z != (0.0 if mean == "zero" else z[0])):
        raise ValueError("the standardized errors carry no variation to fit")
    if n > LIKELIHOOD_WARN_N:
        warnings.warn(
            f"fit_likelihood with {n} points factorises an {n} x {n} matrix many times and "
            "may be slow; consider a random subsample of the validation points or the WLS "
            "variogram fit (method='wls')",
            stacklevel=2,
        )

    D = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    positive = D[D > 0]
    if positive.size == 0:
        raise ValueError("all observations are at the same location")
    diag_len = math.hypot(float(np.ptp(xs)), float(np.ptp(ys)))
    hi_practical = 2.0 * diag_len if max_practical_range is None else float(max_practical_range)
    lo_practical = 0.1 * float(positive.min())
    if not hi_practical > lo_practical:
        raise ValueError("max_practical_range is too small for these data")
    one = np.ones(n)
    diag = np.diag_indices(n)

    def profile(code: ModelType, theta: FloatArray) -> tuple[float, float]:
        """Negative profile (restricted) log-likelihood without its constant, and the sill.

        The constant is left out of the optimised objective because L-BFGS-B's stopping
        rule is relative to the size of the objective.
        """
        eta = float(special.expit(theta[0]))
        K = (1.0 - eta) * (1.0 - vgm(1.0, code, math.exp(theta[1])).gamma(D))
        K[diag] = 1.0 + 1e-10
        try:
            c = linalg.cho_factor(K, lower=True, check_finite=False)
        except linalg.LinAlgError:
            return _FAILED, math.nan
        kz = linalg.cho_solve(c, z, check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(c[0]))))
        if mean == "zero":
            q = max(float(z @ kz), 1e-300)
            return 0.5 * (n * math.log(q) + logdet), q / n
        k1 = linalg.cho_solve(c, one, check_finite=False)
        s11 = float(one @ k1)
        q = max(float(z @ kz) - float(one @ kz) ** 2 / s11, 1e-300)
        return 0.5 * ((n - 1) * math.log(q) + logdet + math.log(s11)), q / (n - 1)

    best: tuple[float, ModelType, FloatArray, float, bool] | None = None
    for code in models:
        bounds = [
            (-_LOGIT_BOUND, _LOGIT_BOUND),
            (math.log(lo_practical / _PRACTICAL[code]), math.log(hi_practical / _PRACTICAL[code])),
        ]
        grid = [
            np.array([float(special.logit(e)), la])
            for e in _GRID_NUGGET
            for la in np.linspace(bounds[1][0], bounds[1][1], _GRID_RANGES)
        ]
        values = [profile(code, t)[0] for t in grid]
        start = grid[int(np.argmin(values))]

        def objective(t: FloatArray, code: ModelType = code) -> float:
            return profile(code, t)[0]

        res = optimize.minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 60},
        )
        theta = res.x if res.fun <= min(values) else start
        nll, s2 = profile(code, theta)
        if best is None or nll < best[0]:
            best = (nll, code, theta, s2, bool(res.success))
    assert best is not None
    nll, code, theta, s2, converged = best
    if not nll < _FAILED:
        raise ValueError("the likelihood could not be evaluated for any candidate model")
    m = n if mean == "zero" else n - 1
    loglik = -nll - 0.5 * m * (math.log(2.0 * math.pi / m) + 1.0)
    eta = float(special.expit(theta[0]))
    model = vgm((1.0 - eta) * s2, code, math.exp(theta[1]), nugget=eta * s2)
    return LikelihoodFit(model=model, loglik=loglik, mean=mean, n_obs=n, converged=converged)
