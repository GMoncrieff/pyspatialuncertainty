"""Prediction intervals for block averages and totals (step 8 of the method).

Block-averaged prediction errors are approximately normally distributed by the central
limit theorem, so a prediction interval is the block prediction plus or minus a normal
quantile times the block standard deviation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from pyspatialuncertainty._types import FloatArray


def z_value(level: float) -> float:
    """Two-sided standard-normal quantile for a central interval with coverage ``level``."""
    if not 0 < level < 1:
        raise ValueError("level must lie strictly between 0 and 1")
    return float(stats.norm.ppf(0.5 + level / 2.0))


def normal_interval(
    prediction: ArrayLike,
    sd: ArrayLike,
    *,
    level: float = 0.90,
    z: float | None = None,
    lower_bound: float | None = None,
    upper_bound: float | ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Normal prediction interval ``prediction ± z · sd``.

    Parameters
    ----------
    prediction, sd
        Block predictions and block standard deviations.
    level
        Nominal coverage of the central interval (default 0.90, as in the authors' script).
    z
        Explicit multiplier, overriding ``level`` (e.g. ``1.64`` as in the R script).
    lower_bound, upper_bound
        Optional bounds to clip the interval to (e.g. 0 and 1 for class fractions).

    Returns
    -------
    tuple of arrays
        Lower and upper interval limits.
    """
    mult = z_value(level) if z is None else float(z)
    pred = np.asarray(prediction, dtype=np.float64)
    s = np.asarray(sd, dtype=np.float64)
    lower = pred - mult * s
    upper = pred + mult * s
    if lower_bound is not None:
        lower = np.maximum(lower, lower_bound)
        upper = np.maximum(upper, lower_bound)
    if upper_bound is not None:
        ub = np.asarray(upper_bound, dtype=np.float64)
        lower = np.minimum(lower, ub)
        upper = np.minimum(upper, ub)
    return lower, upper
