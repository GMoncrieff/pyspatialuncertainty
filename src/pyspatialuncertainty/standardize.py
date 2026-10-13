"""Point-support standard deviations and standardized prediction errors (steps 2-3).

Step 2 of the method derives the point-support prediction-error standard deviation σ(s)
from a predictive distribution, for instance by drawing samples from a quantile regression
forest (`sd_from_samples`) or from a dense set of predicted quantiles
(`sd_from_quantiles`). Step 3 divides validation errors by σ at the validation
locations to obtain standardized errors (`standardized_errors`), whose spatial
correlation is then modelled with a variogram.
"""

from __future__ import annotations

from typing import TypeVar, overload

import numpy as np
import polars as pl
import xarray as xr
from numpy.typing import ArrayLike

from pyspatialuncertainty._types import FloatArray

T = TypeVar("T", xr.DataArray, FloatArray)


@overload
def sd_from_samples(
    samples: xr.DataArray, dim: str = "sample", *, ddof: int = 1
) -> xr.DataArray: ...
@overload
def sd_from_samples(samples: ArrayLike, dim: int = 0, *, ddof: int = 1) -> FloatArray: ...
def sd_from_samples(
    samples: xr.DataArray | ArrayLike, dim: str | int = "sample", *, ddof: int = 1
) -> xr.DataArray | FloatArray:
    """Standard deviation of samples from the predictive distribution (step 2).

    Parameters
    ----------
    samples
        Samples from the conditional distribution at each location, e.g. 200–500 draws from a
        quantile regression forest, as a DataArray with a sample dimension or an array.
    dim
        Name (DataArray) or axis (array) of the sample dimension.
    ddof
        Delta degrees of freedom (1 gives the usual sample standard deviation).

    Returns
    -------
    DataArray or ndarray
        The standard deviation with the sample dimension removed.
    """
    if isinstance(samples, xr.DataArray):
        if not isinstance(dim, str):
            raise TypeError("dim must be a dimension name for a DataArray")
        return samples.std(dim=dim, ddof=ddof)
    if not isinstance(dim, int):
        raise TypeError("dim must be an axis number for an array")
    return np.asarray(np.std(np.asarray(samples, dtype=np.float64), axis=dim, ddof=ddof))


def _quantile_weights(levels: FloatArray) -> FloatArray:
    if levels.ndim != 1 or levels.size < 2:
        raise ValueError("at least two quantile levels are needed")
    if np.any(np.diff(levels) <= 0) or levels[0] <= 0 or levels[-1] >= 1:
        raise ValueError("quantile levels must be strictly increasing and lie in (0, 1)")
    edges = np.concatenate([[0.0], 0.5 * (levels[1:] + levels[:-1]), [1.0]])
    return np.diff(edges)


def sd_from_quantiles(
    quantiles: xr.DataArray | ArrayLike,
    levels: ArrayLike,
    dim: str | int = "quantile",
) -> xr.DataArray | FloatArray:
    """Standard deviation implied by a dense set of predicted quantiles (step 2).

    Each quantile q_j at level τ_j represents the probability mass of the interval between
    the midpoints to its neighbouring levels (0 and 1 at the ends). For equally spaced levels
    (j − ½)/J this equals the standard deviation of J equally weighted samples.

    Parameters
    ----------
    quantiles
        Predicted quantiles, with the quantile levels along ``dim``.
    levels
        The quantile levels, strictly increasing in (0, 1).
    dim
        Name (DataArray) or axis (array) of the quantile dimension.

    Returns
    -------
    DataArray or ndarray
        The standard deviation with the quantile dimension removed.
    """
    w = _quantile_weights(np.asarray(levels, dtype=np.float64))
    if isinstance(quantiles, xr.DataArray):
        if not isinstance(dim, str):
            raise TypeError("dim must be a dimension name for a DataArray")
        wda = xr.DataArray(w, dims=(dim,))
        mean = (quantiles * wda).sum(dim)
        var = (((quantiles - mean) ** 2) * wda).sum(dim)
        return np.sqrt(var)
    if not isinstance(dim, int):
        raise TypeError("dim must be an axis number for an array")
    q = np.moveaxis(np.asarray(quantiles, dtype=np.float64), dim, -1)
    if q.shape[-1] != w.size:
        raise ValueError("the number of quantiles does not match the number of levels")
    q_mean = q @ w
    q_var = ((q - q_mean[..., None]) ** 2) @ w
    return np.sqrt(q_var)


def standardized_errors(
    data: pl.DataFrame,
    observed: str = "observed",
    predicted: str = "predicted",
    sd: str = "sd",
    *,
    name: str = "std_error",
) -> pl.DataFrame:
    """Add standardized prediction errors (observed − predicted) / σ to a table (step 3).

    The predictions and σ at the validation locations must come from a model that did not
    use the observation at that location, e.g. from cross-validation or an independent
    validation set.

    Parameters
    ----------
    data
        Table with the validation observations, predictions and standard deviations.
    observed, predicted, sd
        Column names.
    name
        Name of the new column.

    Returns
    -------
    polars.DataFrame
        ``data`` with the standardized errors added.
    """
    s = data[sd].cast(pl.Float64)
    if (s.is_null() | s.is_nan() | (s <= 0)).any():
        raise ValueError("standard deviations at validation locations must be positive and finite")
    return data.with_columns(
        (
            (pl.col(observed).cast(pl.Float64) - pl.col(predicted).cast(pl.Float64)) / pl.col(sd)
        ).alias(name)
    )
