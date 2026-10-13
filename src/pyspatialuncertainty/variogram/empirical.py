"""Sample (empirical) variograms with gstat's conventions.

Defaults follow ``gstat::variogram()`` for an omnidirectional variogram:

* ``cutoff`` defaults to 0.33333 times the diagonal of the bounding box of the data (gstat's
  default ``fraction`` is the five-digit value 0.33333, not exactly 1/3);
* ``width`` defaults to ``cutoff / 15``;
* a pair at distance d (0 < d ≤ cutoff) falls in lag class k when
  k·width < d ≤ (k+1)·width (exact multiples of ``width`` go to the lower class), and
  zero-distance pairs go to the first class;
* each lag class reports the number of pairs (``np``), the mean pair distance (``dist``)
  and the semivariance (``gamma``); empty classes are dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from pyspatialuncertainty._types import FloatArray, IntArray

Estimator = Literal["matheron", "cressie"]

GSTAT_CUTOFF_FRACTION = 0.33333
"""Fraction of the bounding-box diagonal used as default cutoff (gstat's ``fraction``)."""


@dataclass(frozen=True)
class EmpiricalVariogram:
    """A sample variogram.

    Attributes
    ----------
    n_pairs
        Number of point pairs in each lag class (gstat's ``np``).
    dist
        Mean distance of the pairs in each lag class.
    gamma
        Estimated semivariance in each lag class.
    boundaries
        Lag-class boundaries used for binning.
    cutoff
        Maximum distance considered.
    width
        Lag-class width (``None`` when explicit boundaries were supplied).
    estimator
        ``"matheron"`` (method of moments) or ``"cressie"`` (Cressie–Hawkins robust estimator).
    n_obs
        Number of observations used.
    """

    n_pairs: IntArray
    dist: FloatArray
    gamma: FloatArray
    boundaries: FloatArray
    cutoff: float
    width: float | None
    estimator: Estimator
    n_obs: int

    def to_polars(self) -> pl.DataFrame:
        """Return the variogram as a table with columns ``np``, ``dist`` and ``gamma``."""
        return pl.DataFrame(
            {"np": self.n_pairs, "dist": self.dist, "gamma": self.gamma},
            schema={"np": pl.Int64, "dist": pl.Float64, "gamma": pl.Float64},
        )

    def __len__(self) -> int:
        return int(self.n_pairs.size)


def _column(data: pl.DataFrame | None, value: str | ArrayLike, name: str) -> FloatArray:
    if isinstance(value, str):
        if data is None:
            raise ValueError(f"{name}={value!r} is a column name but no data frame was given")
        return data[value].cast(pl.Float64).to_numpy()
    return np.asarray(value, dtype=np.float64).ravel()


def empirical_variogram(
    data: pl.DataFrame | None = None,
    value: str | ArrayLike = "value",
    x: str | ArrayLike = "x",
    y: str | ArrayLike = "y",
    *,
    cutoff: float | None = None,
    width: float | None = None,
    boundaries: ArrayLike | None = None,
    estimator: Estimator = "matheron",
    max_chunk_elements: int = 4_000_000,
) -> EmpiricalVariogram:
    """Compute an omnidirectional sample variogram, matching ``gstat::variogram()``.

    Parameters
    ----------
    data
        Polars data frame holding the columns named by ``value``, ``x`` and ``y``. If
        ``None``, these must be arrays.
    value, x, y
        Column names (or arrays) of the variable and the coordinates.
    cutoff
        Maximum pair distance. Defaults to 0.33333 times the bounding-box diagonal, as in gstat.
    width
        Lag-class width. Defaults to ``cutoff / 15``.
    boundaries
        Explicit lag-class boundaries (overrides ``cutoff`` and ``width``).
    estimator
        ``"matheron"`` (default) or ``"cressie"`` for the Cressie–Hawkins estimator.
    max_chunk_elements
        Upper bound on the size of the pairwise-distance blocks held in memory.

    Returns
    -------
    EmpiricalVariogram
        The sample variogram.
    """
    z = _column(data, value, "value")
    xs = _column(data, x, "x")
    ys = _column(data, y, "y")
    if not (z.size == xs.size == ys.size):
        raise ValueError("value, x and y must have the same length")
    ok = np.isfinite(z) & np.isfinite(xs) & np.isfinite(ys)
    z, xs, ys = z[ok], xs[ok], ys[ok]
    n = z.size
    if n < 2:
        raise ValueError("at least two observations are needed")
    if estimator not in ("matheron", "cressie"):
        raise ValueError(f"unknown estimator {estimator!r}")

    explicit = boundaries is not None
    if explicit:
        bounds = np.asarray(boundaries, dtype=np.float64)
        if bounds.ndim != 1 or bounds.size < 2 or np.any(np.diff(bounds) <= 0):
            raise ValueError("boundaries must be an increasing sequence of at least two values")
        lag_cutoff = float(bounds[-1])
        lag_width: float | None = None
        n_bins = bounds.size - 1
    else:
        if cutoff is None:
            cutoff = GSTAT_CUTOFF_FRACTION * math.hypot(float(np.ptp(xs)), float(np.ptp(ys)))
        if width is None:
            width = cutoff / 15.0
        if not (cutoff > 0 and width > 0):
            raise ValueError("cutoff and width must be positive")
        lag_cutoff = float(cutoff)
        lag_width = float(width)
        n_bins = max(1, math.ceil(lag_cutoff / lag_width - 1e-12))
        bounds = np.arange(n_bins + 1, dtype=np.float64) * lag_width
        bounds[-1] = min(bounds[-1], lag_cutoff)

    counts = np.zeros(n_bins, dtype=np.int64)
    sum_dist = np.zeros(n_bins)
    sum_stat = np.zeros(n_bins)

    chunk = max(1, min(n, max_chunk_elements // n))
    for i0 in range(0, n - 1, chunk):
        i1 = min(i0 + chunk, n - 1)
        rows = np.arange(i0, i1)
        cols = np.arange(i0 + 1, n)
        dx = xs[rows, None] - xs[None, cols]
        dy = ys[rows, None] - ys[None, cols]
        d = np.hypot(dx, dy)
        upper = cols[None, :] > rows[:, None]
        sel = upper & (d <= lag_cutoff)
        if not np.any(sel):
            continue
        dsel = d[sel]
        dz = (z[rows, None] - z[None, cols])[sel]
        if explicit:
            idx = np.searchsorted(bounds, dsel, side="left") - 1
            idx[dsel == 0] = 0
            keep = (idx >= 0) & (idx < n_bins)
        else:
            assert lag_width is not None
            frac = dsel / lag_width
            idx = np.floor(frac).astype(np.int64)
            idx[(dsel > 0) & (frac == idx)] -= 1
            keep = (idx >= 0) & (idx < n_bins)
        idx, dsel, dz = idx[keep], dsel[keep], dz[keep]
        stat = dz**2 if estimator == "matheron" else np.sqrt(np.abs(dz))
        counts += np.bincount(idx, minlength=n_bins)[:n_bins]
        sum_dist += np.bincount(idx, weights=dsel, minlength=n_bins)[:n_bins]
        sum_stat += np.bincount(idx, weights=stat, minlength=n_bins)[:n_bins]

    nonempty = counts > 0
    npairs = counts[nonempty]
    mean_dist = sum_dist[nonempty] / npairs
    if estimator == "matheron":
        gamma = sum_stat[nonempty] / (2.0 * npairs)
    else:
        gamma = 0.5 * (sum_stat[nonempty] / npairs) ** 4 / (0.457 + 0.494 / npairs)
    return EmpiricalVariogram(
        n_pairs=npairs.astype(np.int64),
        dist=mean_dist,
        gamma=gamma,
        boundaries=bounds,
        cutoff=lag_cutoff,
        width=lag_width,
        estimator=estimator,
        n_obs=int(n),
    )
