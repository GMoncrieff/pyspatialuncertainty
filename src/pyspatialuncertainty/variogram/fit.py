"""Weighted least-squares fitting of variogram models, following ``gstat::fit.variogram()``.

Weights per lag class j (gstat's ``fit.method``):

====  ==============================  =====================================
code  weight w_j                      notes
====  ==============================  =====================================
1     N_j                             number of pairs
2     N_j / γ(h_j)²                   Cressie (1985); model-based, iterated
6     1                               ordinary least squares
7     N_j / h_j²                      gstat's default
====  ==============================  =====================================

The objective Σ_j w_j (γ̂_j − γ(h_j))² is minimised over the partial sills (constrained to
be non-negative) and the ranges (constrained to be positive) that are not held fixed.
For methods 1, 6 and 7 the solutions agree with gstat to within about 1e-4 (tested against
gstat 2.1-1). For method 2 the weights depend on the model; here they are updated until a
fixed point is reached, which can differ from gstat's internal iteration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import optimize

from pyspatialuncertainty._types import FloatArray
from pyspatialuncertainty.variogram.empirical import EmpiricalVariogram
from pyspatialuncertainty.variogram.models import ModelType, Structure, VariogramModel

FitMethod = Literal[1, 2, 6, 7]


@dataclass(frozen=True)
class VariogramFit:
    """Result of a variogram fit.

    Attributes
    ----------
    model
        The fitted variogram model.
    sserr
        Weighted sum of squared errors at the solution (gstat's ``SSErr`` attribute).
    fit_method
        The gstat ``fit.method`` code used for the weights.
    converged
        Whether the optimiser reported convergence.
    n_iter
        Number of (re-weighting) iterations.
    message
        Optimiser message.
    """

    model: VariogramModel
    sserr: float
    fit_method: int
    converged: bool
    n_iter: int
    message: str


def _flags(value: bool | Sequence[bool], n: int, name: str) -> list[bool]:
    if isinstance(value, bool):
        return [value] * n
    flags = [bool(v) for v in value]
    if len(flags) != n:
        raise ValueError(f"{name} must be a bool or have one entry per structure ({n})")
    return flags


def fit_weights(ev: EmpiricalVariogram, model: VariogramModel, fit_method: int) -> FloatArray:
    """Weights of the lag classes for a gstat ``fit.method`` code."""
    npairs = ev.n_pairs.astype(np.float64)
    if fit_method == 1:
        return npairs
    if fit_method == 2:
        g = model.gamma(ev.dist)
        g_pos: FloatArray = np.where(g > 0, g, np.finfo(np.float64).tiny ** 0.25)
        return npairs / g_pos**2
    if fit_method == 6:
        return np.ones_like(npairs)
    if fit_method == 7:
        d: FloatArray = np.where(ev.dist > 0, ev.dist, np.finfo(np.float64).tiny ** 0.25)
        return npairs / d**2
    raise ValueError(f"unsupported fit_method {fit_method}; use 1, 2, 6 or 7")


def weighted_sse(ev: EmpiricalVariogram, model: VariogramModel, fit_method: int) -> float:
    """Weighted sum of squared errors of ``model`` against ``ev`` (gstat's ``SSErr``)."""
    w = fit_weights(ev, model, fit_method)
    return float(np.sum(w * (ev.gamma - model.gamma(ev.dist)) ** 2))


def fit_variogram(
    ev: EmpiricalVariogram,
    model: VariogramModel,
    *,
    fit_method: FitMethod = 7,
    fit_sills: bool | Sequence[bool] = True,
    fit_ranges: bool | Sequence[bool] = True,
    max_iter: int = 100,
    tol: float = 1e-10,
    max_range: float | None = None,
    max_psill: float | None = None,
) -> VariogramFit:
    """Fit a (nested) variogram model to a sample variogram by weighted least squares.

    Parameters
    ----------
    ev
        Sample variogram, e.g. from `empirical_variogram`.
    model
        Initial model; its structure types and smoothness parameters are kept fixed.
    fit_method
        gstat weighting scheme: 1, 2, 6 or 7 (default, N_j / h_j²).
    fit_sills, fit_ranges
        Whether to fit the partial sills and ranges, either for all structures or one flag
        per structure (as in gstat). Ranges of nugget structures are never fitted.
    max_iter
        Maximum number of re-weighting iterations (only used by ``fit_method=2``).
    tol
        Convergence tolerance.
    max_range, max_psill
        Optional upper bounds on the range parameters and partial sills. Without them a
        sample variogram that keeps rising up to the cutoff can be fitted with an
        arbitrarily large sill and range, which turns the correlogram into ρ ≈ 1.

    Returns
    -------
    VariogramFit
        The fitted model and fit diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> from pyspatialuncertainty.variogram import empirical_variogram, vgm
    >>> rng = np.random.default_rng(0)
    >>> xy = rng.uniform(0, 100, size=(300, 2))
    >>> ev = empirical_variogram(None, rng.normal(size=300), xy[:, 0], xy[:, 1])
    >>> fit = fit_variogram(ev, vgm(1.0, "Exp", 10.0, nugget=0.1))
    >>> fit.model.sill > 0
    True
    """
    structures = model.structures
    k = len(structures)
    fs = _flags(fit_sills, k, "fit_sills")
    fr = [
        f and s.model != "Nug"
        for f, s in zip(_flags(fit_ranges, k, "fit_ranges"), structures, strict=True)
    ]
    psill0 = np.array([s.psill for s in structures])
    range0 = np.array([s.range if s.model != "Nug" else 0.0 for s in structures])
    n_sill = int(sum(fs))
    n_range = int(sum(fr))

    if n_sill + n_range == 0 or len(ev) == 0:
        return VariogramFit(
            model, weighted_sse(ev, model, fit_method), fit_method, True, 0, "nothing to fit"
        )

    max_dist = float(np.max(ev.dist)) if len(ev) else 1.0
    sill_idx = np.flatnonzero(fs)
    range_idx = np.flatnonzero(fr)

    def unpack(theta: FloatArray) -> VariogramModel:
        ps = psill0.copy()
        rg = range0.copy()
        ps[sill_idx] = theta[:n_sill]
        rg[range_idx] = np.exp(theta[n_sill:])
        return model.replace_parameters(ps, rg)

    theta = np.concatenate(
        [psill0[sill_idx], np.log(np.maximum(range0[range_idx], 1e-12 * max_dist))]
    )
    lower = np.concatenate([np.zeros(n_sill), np.full(n_range, math.log(1e-9 * max_dist))])
    sill_hi = np.inf if max_psill is None else float(max_psill)
    range_hi = math.log(1e6 * max_dist) if max_range is None else math.log(float(max_range))
    upper = np.concatenate([np.full(n_sill, sill_hi), np.full(n_range, range_hi)])
    theta = np.clip(theta, lower, upper)

    weights = fit_weights(ev, model, fit_method)
    converged = False
    message = ""
    n_iter = 0
    for iteration in range(1, (max_iter if fit_method == 2 else 1) + 1):
        n_iter = iteration
        sqrt_w = np.sqrt(weights)

        def residuals(t: FloatArray, sqrt_w: FloatArray = sqrt_w) -> FloatArray:
            return sqrt_w * (unpack(t).gamma(ev.dist) - ev.gamma)

        res = optimize.least_squares(
            residuals,
            theta,
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=tol,
            xtol=tol,
            gtol=tol,
            max_nfev=2000,
        )
        converged = bool(res.status > 0)
        message = str(res.message)
        change = float(np.max(np.abs(res.x - theta) / np.maximum(np.abs(theta), 1e-12)))
        theta = res.x
        if fit_method != 2:
            break
        weights = fit_weights(ev, unpack(theta), fit_method)
        if change < 1e-9:
            break

    fitted = unpack(theta)
    return VariogramFit(
        model=fitted,
        sserr=weighted_sse(ev, fitted, fit_method),
        fit_method=fit_method,
        converged=converged,
        n_iter=n_iter,
        message=message,
    )


def fit_variogram_auto(
    ev: EmpiricalVariogram,
    *,
    models: Sequence[ModelType] = ("Exp", "Sph", "Gau"),
    nugget: bool = True,
    fit_method: FitMethod = 7,
    range_fractions: Sequence[float] = (0.05, 0.15, 0.35, 0.7),
    bounded: bool = True,
) -> VariogramFit:
    """Fit ``Nug + <model>`` for several model types and starting ranges; keep the best.

    The candidate with the smallest weighted sum of squared errors (for the same
    ``fit_method``) is returned. Starting ranges are fractions of the largest lag distance,
    converted to each model's range parameter (e.g. divided by 3 for ``"Exp"``).

    Parameters
    ----------
    ev
        Sample variogram.
    models
        Structure types to try in addition to the nugget.
    nugget
        Whether to include a nugget structure.
    fit_method
        gstat weighting scheme (see `fit_variogram`).
    range_fractions
        Starting (practical) ranges as fractions of the largest lag distance.
    bounded
        Constrain each fit to a practical range of at most twice the largest lag distance
        and partial sills of at most twice the largest sample semivariance, which keeps
        the automatic fit from extrapolating an unbounded trend.

    Returns
    -------
    VariogramFit
        The best fit.
    """
    if len(ev) == 0:
        raise ValueError("the sample variogram is empty")
    max_dist = float(np.max(ev.dist))
    gmax = float(np.max(ev.gamma))
    g_first = float(ev.gamma[0])
    nug0 = max(min(0.5 * g_first, 0.5 * gmax), 1e-3 * max(gmax, 1e-12))
    psill0 = max(gmax - (nug0 if nugget else 0.0), 1e-3 * max(gmax, 1e-12))
    practical = {"Exp": 3.0, "Gau": math.sqrt(3.0), "Sph": 1.0, "Mat": 3.0, "Ste": 1.0, "Nug": 1.0}
    best: VariogramFit | None = None
    for code in models:
        if code == "Nug":
            continue
        for frac in range_fractions:
            a0 = frac * max_dist / practical[code]
            structs: list[Structure] = []
            if nugget:
                structs.append(Structure("Nug", nug0))
            structs.append(Structure(code, psill0, a0))
            fit = fit_variogram(
                ev,
                VariogramModel(tuple(structs)),
                fit_method=fit_method,
                max_range=2.0 * max_dist / practical[code] if bounded else None,
                max_psill=2.0 * gmax if bounded else None,
            )
            if best is None or fit.sserr < best.sserr:
                best = fit
    assert best is not None
    return best
