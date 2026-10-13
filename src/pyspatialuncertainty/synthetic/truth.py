"""Exact block-error variances for the synthetic scenarios.

These are used to check the brute-force (simulation) truth and the Monte Carlo
integration of the method:

* :func:`exact_block_variance` computes (1/N²) Σ_j Σ_l w_j w_l C(d_jl) over the nodes of a
  block on a regular grid by FFT convolution, which is the exact variance of the block mean
  of a field w(s)·S(s) with stationary covariance C for S;
* :func:`bvn_cdf` and :func:`indicator_block_variance` give the exact variance of the block
  fraction of a class in a thresholded latent Gaussian field, using bivariate normal
  probabilities computed with Owen's T function.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal, special

from pyspatialuncertainty._types import BoolArray, FloatArray

Covariance = Callable[[FloatArray], FloatArray]


def exact_block_variance(
    weights: FloatArray, mask: BoolArray, spacing: float, covariance: Covariance
) -> float:
    """Variance of the block mean of w(s)·S(s) over the masked nodes of a regular grid.

    Parameters
    ----------
    weights
        2-D array of node weights w (e.g. the error standard deviation σ at each node).
    mask
        2-D boolean array selecting the nodes of the block.
    spacing
        Node spacing.
    covariance
        Covariance function C(h) of the stationary field S.

    Returns
    -------
    float
        (1/N²) Σ_{j,l ∈ B} w_j w_l C(|s_j − s_l|), with N the number of nodes in the block.
    """
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        raise ValueError("empty block")
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    w = np.where(mask, weights, 0.0)[r0 : r1 + 1, c0 : c1 + 1]
    h, wd = w.shape
    di = np.arange(-(h - 1), h)
    dj = np.arange(-(wd - 1), wd)
    kernel = covariance(spacing * np.hypot(di[:, None], dj[None, :]))
    conv = signal.fftconvolve(w, kernel, mode="full")[h - 1 : 2 * h - 1, wd - 1 : 2 * wd - 1]
    return float(np.sum(w * conv)) / float(rows.size) ** 2


def bvn_cdf(h: ArrayLike, k: ArrayLike, r: ArrayLike) -> FloatArray:
    """Standard bivariate normal CDF P(X ≤ h, Y ≤ k) with correlation r (vectorised).

    Uses Owen's T function; infinite limits and |r| = 1 are handled exactly.
    """
    h_arr, k_arr, r_arr = np.broadcast_arrays(
        np.asarray(h, dtype=np.float64),
        np.asarray(k, dtype=np.float64),
        np.asarray(r, dtype=np.float64),
    )
    out = np.empty(h_arr.shape)
    hh = h_arr.ravel()
    kk = k_arr.ravel()
    rr = np.clip(r_arr.ravel(), -1.0, 1.0)
    res = out.ravel()

    neg_inf = (hh == -np.inf) | (kk == -np.inf)
    h_inf = (hh == np.inf) & ~neg_inf
    k_inf = (kk == np.inf) & ~neg_inf & ~h_inf
    res[neg_inf] = 0.0
    res[h_inf] = special.ndtr(kk[h_inf])
    res[k_inf] = special.ndtr(hh[k_inf])
    rest = ~(neg_inf | h_inf | k_inf)

    one = rest & (rr >= 1.0 - 1e-14)
    res[one] = special.ndtr(np.minimum(hh[one], kk[one]))
    mone = rest & (rr <= -1.0 + 1e-14)
    res[mone] = np.maximum(special.ndtr(hh[mone]) + special.ndtr(kk[mone]) - 1.0, 0.0)
    gen = rest & ~one & ~mone
    if np.any(gen):
        x = hh[gen]
        y = kk[gen]
        rho = rr[gen]
        x = np.where(x == 0.0, 1e-300, x)
        y = np.where(y == 0.0, 1e-300, y)
        s = np.sqrt(1.0 - rho * rho)
        a_x = (y - rho * x) / (x * s)
        a_y = (x - rho * y) / (y * s)
        delta = np.where(np.sign(x) == np.sign(y), 0.0, 0.5)
        res[gen] = (
            0.5 * special.ndtr(x)
            + 0.5 * special.ndtr(y)
            - special.owens_t(x, a_x)
            - special.owens_t(y, a_y)
            - delta
        )
    return np.clip(out, 0.0, 1.0)


def bvn_rectangle(
    a1: ArrayLike, b1: ArrayLike, a2: ArrayLike, b2: ArrayLike, r: ArrayLike
) -> FloatArray:
    """P(a1 < X ≤ b1, a2 < Y ≤ b2) for a standard bivariate normal with correlation r."""
    val = bvn_cdf(b1, b2, r) - bvn_cdf(a1, b2, r) - bvn_cdf(b1, a2, r) + bvn_cdf(a1, a2, r)
    return np.clip(val, 0.0, 1.0)


def indicator_block_variance(
    lower: FloatArray,
    upper: FloatArray,
    coords: FloatArray,
    correlation: Callable[[FloatArray], FloatArray],
    *,
    chunk: int = 512,
) -> float:
    """Exact variance of the block fraction of a class in a thresholded latent Gaussian field.

    The latent variable at node j is standard normal after standardisation; node j belongs to
    the class when ``lower[j] < Y_j ≤ upper[j]``. The block fraction is the mean of the class
    indicators (or of the class probabilities given a latent structured component; see
    :class:`~pyspatialuncertainty.synthetic.categorical.LatentClassScenario`).

    Parameters
    ----------
    lower, upper
        Standardised class limits per node (may be ±inf).
    coords
        Node coordinates, shape (N, 2).
    correlation
        Correlation r(d) between the latent variables of two nodes at distance d, used for
        all pairs including j = l (pass r(0) < 1 when the indicator is averaged over an
        independent point-scale nugget).
    chunk
        Number of rows processed at once.

    Returns
    -------
    float
        (1/N²) Σ_j Σ_l [P(Y_j ∈ A_j, Y_l ∈ A_l) − p_j p_l].
    """
    n = coords.shape[0]
    p = special.ndtr(upper) - special.ndtr(lower)
    total = 0.0
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d = np.hypot(
            coords[i0:i1, None, 0] - coords[None, :, 0], coords[i0:i1, None, 1] - coords[None, :, 1]
        )
        r = correlation(d)
        joint = bvn_rectangle(
            lower[i0:i1, None], upper[i0:i1, None], lower[None, :], upper[None, :], r
        )
        total += float(np.sum(joint - p[i0:i1, None] * p[None, :]))
    return total / float(n) ** 2
