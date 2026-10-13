"""Correlation functions of standardized prediction errors (step 4 of the method).

The method needs the correlation ρ(h) of the *standardized* point-support prediction
errors as a function of the Euclidean distance h. It is usually derived from a variogram
model fitted to the standardized errors through ρ(h) = (sill − γ(h)) / sill
(Wadoux & Heuvelink, 2023, Section 4.1, step 4), but any callable that maps an array of
distances to correlations can be used, for instance an expert-elicited correlogram.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

from pyspatialuncertainty._types import FloatArray

if TYPE_CHECKING:
    from pyspatialuncertainty.variogram.models import VariogramModel


@runtime_checkable
class CorrelationFunction(Protocol):
    """A stationary, isotropic correlation function ρ(h) with ρ(0) = 1."""

    def __call__(self, h: FloatArray, /) -> FloatArray:
        """Evaluate the correlation at the distances ``h``."""
        ...


@dataclass(frozen=True)
class VariogramCorrelation:
    """Correlation function derived from a variogram model, ρ(h) = (sill − γ(h)) / sill.

    For h > 0 a nugget lowers the correlation below one; at h = 0 the correlation is one.
    """

    model: VariogramModel

    def __call__(self, h: FloatArray, /) -> FloatArray:
        return self.model.correlation(h)


@dataclass(frozen=True)
class CallableCorrelation:
    """Wraps a user-supplied callable as a `CorrelationFunction`."""

    func: Callable[[FloatArray], ArrayLike]

    def __call__(self, h: FloatArray, /) -> FloatArray:
        out = np.asarray(self.func(h), dtype=np.float64)
        return np.broadcast_to(out, np.shape(h)).astype(np.float64, copy=False)


def from_variogram(model: VariogramModel) -> VariogramCorrelation:
    """Correlation function of a variogram model, ρ(h) = (sill − γ(h)) / sill."""
    return VariogramCorrelation(model)


def as_correlation(obj: object, *, check: bool = True) -> CorrelationFunction:
    """Coerce a variogram model or a callable to a `CorrelationFunction`.

    Parameters
    ----------
    obj
        A `VariogramModel` (converted with
        `from_variogram`) or any callable mapping distances to correlations.
    check
        Sanity-check the function: ρ(0) must be 1 and ρ(h) must lie in [−1, 1] on a few
        test distances. Violations raise ``ValueError``.

    Returns
    -------
    CorrelationFunction
        A callable object evaluating ρ(h) on arrays.
    """
    from pyspatialuncertainty.variogram.models import VariogramModel

    func: CorrelationFunction
    if isinstance(obj, VariogramModel):
        func = VariogramCorrelation(obj)
    elif isinstance(obj, VariogramCorrelation | CallableCorrelation):
        func = obj
    elif callable(obj):
        func = CallableCorrelation(obj)
    else:
        raise TypeError("expected a VariogramModel or a callable correlation function")
    if check:
        _check_correlation(func)
    return func


def _check_correlation(func: CorrelationFunction) -> None:
    h = np.array([0.0, 1e-9, 1e-3, 0.1, 1.0, 10.0, 100.0, 1e4], dtype=np.float64)
    rho = np.asarray(func(h), dtype=np.float64)
    if rho.shape != h.shape:
        raise ValueError("the correlation function must return one value per distance")
    if not np.all(np.isfinite(rho)):
        raise ValueError("the correlation function returned non-finite values")
    if not np.isclose(rho[0], 1.0, rtol=0, atol=1e-8):
        raise ValueError(f"ρ(0) must equal 1, got {rho[0]}")
    if np.any(np.abs(rho) > 1 + 1e-12):
        raise ValueError("correlations must lie in [-1, 1]")
    if np.any(np.diff(rho[1:]) > 1e-12):
        warnings.warn(
            "the correlation function is not monotonically non-increasing on the test "
            "distances; make sure it is a valid (positive definite) correlation function",
            stacklevel=3,
        )
