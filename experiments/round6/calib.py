"""Draws of the calibrator, to carry its uncertainty into the block estimates.

Four ways to draw a calibrator fitted on a calibration sample, 50 draws each; the block
estimate is then recomputed per draw with the error model held fixed:

* ``ci``: iid bootstrap of the calibration points (design-agnostic; ignores dependence).
* ``cg``: pair-correlation sandwich draws θ_b ~ N(θ̂, H⁻¹MH⁻¹) (design-agnostic). With
  per-point scores sᵢ,

      M = Σᵢ sᵢsᵢᵀ + Σ_{i<j} wᵢⱼ (sᵢsⱼᵀ + sⱼsᵢᵀ),

  where wᵢⱼ is the probability that the pair was sampled as one unit, from a cluster-process
  (Thomas) fit to the excess of the sample's pair distances over a simple random sample of
  the landscape's cells (``pair_weights``; none unless the excess is clear). A clustered
  sample then gets the cluster bootstrap's meat Σ_c (Σ_{i∈c} sᵢ)(Σ_{i∈c} sᵢ)ᵀ in
  expectation and a random sample the iid one. Only the coordinates are used.
* ``cs``: model sandwich, with the fitted correlation of the standardized errors between
  every pair, M = Σᵢ sᵢsᵢᵀ + Σ_{i≠j} ρᵢⱼ uᵢuⱼᵀ, uᵢ = JᵢᵀWᵢ^½. This is the variance over
  realizations of the field. The field also drives the truth, so it overstates the
  calibration noise; it is kept as a diagnostic.
* ``cc``: round 5's bootstrap of clusters (points for random samples); a reference only,
  since it needs the sampling design.

Class calibrators are round 5's temperature and vector scaling (ridge 1 on a − 1 and b);
continuous ones are the linear correction b0 + b1·m with a σ factor, where only (b0, b1)
move the estimate.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import numpy as np
from scipy import special
from scipy.spatial import cKDTree

from experiments.round5 import shared as sh

N_DRAWS = 50
N_BINS = 24


def cluster_groups(clusters: np.ndarray) -> list[np.ndarray]:
    _, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.cumsum(np.bincount(inv))[:-1])


def resample(kind: str, clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Indices of one bootstrap sample: ``ci`` resamples points, ``cc`` clusters."""
    if kind == "ci":
        return rng.integers(0, clusters.size, clusters.size)
    groups = cluster_groups(clusters)
    return np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])


def correlation_matrix(
    rho: Callable[[np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """ρ(|xᵢ − xⱼ|) between distinct points, zero on the diagonal (built in row chunks)."""
    n = x.size
    R = np.empty((n, n))
    step = 250
    for a in range(0, n, step):
        b = min(a + step, n)
        R[a:b] = rho(np.hypot(x[a:b, None] - x[None, :], y[a:b, None] - y[None, :]))
    R[np.diag_indices(n)] = 0.0
    return R


@functools.cache
def lattice_pair_share(side: int, d_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Bin edges and the share of pairs of distinct cells of a side × side grid per bin: the
    pair distances of a simple random sample of cells (cell centres sit on a lattice, so the
    continuous formula would show spurious excesses at short lags)."""
    edges = np.concatenate([[0.0], np.geomspace(1.0, d_max, N_BINS)])
    r = math.ceil(d_max)
    off = np.arange(-r, r + 1)
    dx, dy = np.meshgrid(off, off, indexing="ij")
    count = np.clip(side - np.abs(dx), 0, None) * np.clip(side - np.abs(dy), 0, None)
    count[r, r] = 0  # a cell with itself
    per_bin = np.histogram(np.hypot(dx, dy).ravel(), edges, weights=count.ravel())[0] / 2.0
    n_cells = float(side) ** 2
    return edges, per_bin / (n_cells * (n_cells - 1) / 2.0)


def _pair_hist(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, ...]:
    pairs = cKDTree(np.column_stack([x, y])).query_pairs(r=edges[-1], output_type="ndarray")
    if pairs.size == 0:
        return pairs, np.zeros(0), np.zeros(edges.size - 1)
    d = np.hypot(x[pairs[:, 0]] - x[pairs[:, 1]], y[pairs[:, 0]] - y[pairs[:, 1]])
    return pairs, d, np.histogram(d, edges)[0].astype(float)


@functools.cache
def null_pair_cov(side: int, m: int, n_sim: int = 60) -> np.ndarray:
    """Covariance of the pair counts per distance bin over simple random samples of m cells.
    Pairs share points, so the counts vary far more than Poisson counts, and together."""
    edges, _ = lattice_pair_share(side, side / 4.0)
    rng = np.random.default_rng([side, m])
    counts = np.array(
        [
            _pair_hist((c % side) + 0.5, side - (c // side) - 0.5, edges)[2]
            for c in (rng.choice(side * side, m, replace=False) for _ in range(n_sim))
        ]
    )
    return np.cov(counts, rowvar=False)


def pair_weights(
    x: np.ndarray, y: np.ndarray, side: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs (i, j) and the probability that each was sampled as one unit.

    The excess of pair counts per distance bin over a simple random sample of the side × side
    landscape's cells (exact lattice mean) is fitted by A·ΔF with F(r) = 1 − exp(−r²/4s²),
    the within-cluster distance distribution of a Thomas process, by generalized least
    squares with the simulated covariance of the counts plus the Poisson variance, over
    scales s up to 1/16 of the landscape side. With z = A/SE(A) > 4 a pair at distance d gets
    w = A f(d) / (A f(d) + random-pair density); otherwise there are no pairs.
    Coordinates are cell centres."""
    m = x.size
    edges, share = lattice_pair_share(int(side), side / 4.0)
    pairs, d, obs = _pair_hist(x, y, edges)
    none = np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    if pairs.size == 0:
        return none
    expected = m * (m - 1) / 2.0 * share
    C = null_pair_cov(int(side), m) + np.diag(np.maximum(expected, 1.0))
    Ci = np.linalg.inv(C)
    ex = obs - expected
    best = None
    for s in np.geomspace(0.5, side / 16.0, 48):
        dF = np.diff(1.0 - np.exp(-(edges**2) / (4.0 * s * s)))
        info = float(dF @ Ci @ dF)
        A = float(dF @ Ci @ ex) / info
        r = ex - A * dF
        crit = float(r @ Ci @ r)
        if best is None or crit < best[0]:
            best = (crit, A, A * math.sqrt(info), dF)
    _, A, z, dF = best
    if not z > 4.0:
        return none
    wb = A * dF / np.maximum(A * dF + expected, 1e-300)
    w = wb[np.clip(np.searchsorted(edges, d, side="right") - 1, 0, wb.size - 1)]
    keep = w > 1e-3
    return pairs[keep, 0], pairs[keep, 1], w[keep]


def pair_meat(s: np.ndarray, pairs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Σᵢ sᵢsᵢᵀ + Σ_pairs w (sᵢsⱼᵀ + sⱼsᵢᵀ) for scores s (m × q)."""
    i, j, w = pairs
    M = s.T @ s
    if w.size:
        C = (s[i] * w[:, None]).T @ s[j]
        M += C + C.T
    return M


def _draw(theta: np.ndarray, cov: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    lam, Q = np.linalg.eigh((cov + cov.T) / 2)
    L = Q * np.sqrt(np.clip(lam, 0.0, None))
    return theta[None, :] + rng.standard_normal((n, theta.size)) @ L.T


# ---------------------------------------------------------------------------
# Class calibrators
# ---------------------------------------------------------------------------


def class_theta(kind: str, par: dict) -> np.ndarray:
    if kind == "vector":
        return np.concatenate([np.asarray(par["a"]), np.asarray(par["b"])])
    return np.array([np.log(par["T"])])


def class_par(kind: str, theta: np.ndarray) -> dict:
    if kind == "vector":
        k = theta.size // 2
        return {"a": theta[:k].tolist(), "b": theta[k:].tolist()}
    return {"T": float(np.exp(theta[0]))}


def class_parts(
    kind: str, theta: np.ndarray, lp: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores s (m × q), uᵢ = JᵢᵀWᵢ^½ stacked as U (m × q × K) and the Hessian H (q × q)."""
    k, m = lp.shape
    Y = np.eye(k)[labels]  # (m, K)
    if kind == "vector":
        a, b = theta[:k], theta[k:]
        z = (a[:, None] * lp + b[:, None]).T  # (m, K)
        J = np.zeros((m, k, 2 * k))  # dz/dθ per point
        J[:, np.arange(k), np.arange(k)] = lp.T
        J[:, np.arange(k), k + np.arange(k)] = 1.0
        ridge = sh.VEC_RIDGE * np.eye(2 * k)
    else:
        z = (lp / np.exp(theta[0])).T
        J = -z[:, :, None]  # d(lp/T)/d log T
        ridge = np.zeros((1, 1))
    P = np.exp(z - special.logsumexp(z, axis=1, keepdims=True))
    s = np.einsum("mkq,mk->mq", J, P - Y)
    W = np.einsum("mk,kl->mkl", P, np.eye(k)) - P[:, :, None] * P[:, None, :]
    lam, Q = np.linalg.eigh(W)
    Wh = np.einsum("mkl,ml,mjl->mkj", Q, np.sqrt(np.clip(lam, 0.0, None)), Q)
    U = np.einsum("mkq,mkj->mqj", J, Wh)
    H = np.einsum("mqj,mrj->qr", U, U) + ridge
    return s, U, H


def class_sandwich(
    kind: str,
    theta: np.ndarray,
    lp: np.ndarray,
    labels: np.ndarray,
    *,
    R: np.ndarray | None = None,
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """H⁻¹MH⁻¹: the model meat with R (``cs``), the pair-correlation meat with pairs (``cg``),
    or the robust iid meat with neither."""
    s, U, H = class_parts(kind, theta, lp, labels)
    if pairs is not None:
        M = pair_meat(s, pairs)
    else:
        M = s.T @ s
        if R is not None:
            for j in range(U.shape[2]):
                Uj = U[:, :, j]
                M += Uj.T @ (R @ Uj)
    Hi = np.linalg.inv(H)
    return Hi @ M @ Hi


def class_draws(
    kind: str,
    par: dict,
    lp_cal: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    *,
    R: np.ndarray | None = None,
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    n: int = N_DRAWS,
) -> list[dict]:
    theta = class_theta(kind, par)
    cov = class_sandwich(kind, theta, lp_cal, labels, R=R, pairs=pairs)
    return [class_par(kind, t) for t in _draw(theta, cov, n, rng)]


# ---------------------------------------------------------------------------
# Continuous recalibration (linear part)
# ---------------------------------------------------------------------------


def linear_sandwich(
    e: np.ndarray,
    m: np.ndarray,
    sd: np.ndarray,
    *,
    R: np.ndarray | None = None,
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(b0, b1) by least squares and their sandwich covariance: pair-correlation meat
    (``cg``) or model meat with Cov(eᵢ, eⱼ) = ρᵢⱼσᵢσⱼ (``cs``)."""
    X = np.column_stack([np.ones(e.size), m])
    XtX_i = np.linalg.inv(X.T @ X)
    b = XtX_i @ X.T @ e
    r = e - X @ b
    if pairs is not None:
        meat = pair_meat(X * r[:, None], pairs)
    else:
        meat = (X * r[:, None] ** 2).T @ X
        if R is not None:
            Xs = X * sd[:, None]
            meat += Xs.T @ (R @ Xs)
    return b, XtX_i @ meat @ XtX_i


def linear_draws(
    e: np.ndarray,
    m: np.ndarray,
    sd: np.ndarray,
    rng: np.random.Generator,
    *,
    R: np.ndarray | None = None,
    pairs: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    n: int = N_DRAWS,
) -> np.ndarray:
    b, cov = linear_sandwich(e, m, sd, R=R, pairs=pairs)
    return _draw(b, cov, n, rng)
