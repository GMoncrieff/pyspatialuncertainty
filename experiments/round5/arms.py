"""The arithmetic of the round-5 arms.

Every estimator here is "map share + shift" with a variance

    V_B + x̄_Bᵀ A x̄_B − 2 x̄_Bᵀ G c_B

(the estimated-mean correction of the package, generalised to a design X): V_B is the
exact block integral of σσρ, A and G come from the GLS of the standardized errors z on X
(β̂ = G z, A = (XᵀK⁻¹X)⁻¹), x̄_B is the block mean of σ(s)x(s), and c_B,i the block mean of
σ(s)ρ(|s − x_i|). X = 1 is the package's ``ok``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize, special

import pyspatialuncertainty as psu
from experiments.round5 import shared as sh

LOGIT_BOUND = 6.0  # the package's bound on the logit of the nugget share
PRACTICAL = {"Exp": 3.0, "Sph": 1.0, "Gau": math.sqrt(3.0)}


# ---------------------------------------------------------------------------
# GLS of the standardized errors on a design
# ---------------------------------------------------------------------------


def gls_factor(corr: sh.Corr, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, bool]:
    """Cholesky factor of the correlation matrix of the validation errors (nugget on the
    diagonal), as `mean_correction.estimate_mean` builds it."""
    n = x.size
    K = np.empty((n, n))
    step = 250  # rows at a time: one n × n matrix in memory, not several temporaries
    for a in range(0, n, step):
        b = min(a + step, n)
        K[a:b] = corr.struct(np.hypot(x[a:b, None] - x[None, :], y[a:b, None] - y[None, :]))
    K[np.diag_indices_from(K)] = 1.0 + 1e-10
    return linalg.cho_factor(K, lower=True, overwrite_a=True, check_finite=False)


@dataclass(frozen=True)
class GLS:
    beta: np.ndarray  # (q,)
    A: np.ndarray  # (q, q), covariance of β̂ in z units
    G: np.ndarray  # (q, n), β̂ = G z


def gls(
    chol: tuple[np.ndarray, bool], X: np.ndarray, z: np.ndarray, prior_prec: float = 0.0
) -> GLS:
    """GLS of z on X; ``prior_prec`` adds a N(0, 1/prior_prec) prior on each coefficient
    (A is then the posterior variance and G the posterior-mean weights)."""
    X = X.reshape(X.shape[0], -1)
    kx = linalg.cho_solve(chol, X, check_finite=False)
    A = np.linalg.inv(X.T @ kx + prior_prec * np.eye(X.shape[1]))
    G = A @ kx.T
    return GLS(G @ z, A, G)


@dataclass(frozen=True)
class Corrected:
    est: np.ndarray  # block estimates
    var: np.ndarray  # block variances
    xb: np.ndarray  # (n_blocks, q)
    C: np.ndarray  # (q, n_blocks)
    vb: np.ndarray
    gls: GLS


def linear_correction(
    layout: sh.Layout,
    kernels: sh.Kernels,
    sd: np.ndarray,
    share: np.ndarray,
    cells: np.ndarray,
    cols_cells: list[np.ndarray],
    fit: GLS,
    vb: np.ndarray,
    *,
    cross: bool = True,
) -> Corrected:
    """Shift x̄_Bᵀβ̂ and variance V_B + x̄ᵀAx̄ − 2x̄ᵀGc_B for a design given on all cells."""
    xb = np.column_stack([layout.bmat @ (sd * col) for col in cols_cells])
    C = (
        sh.cross_c(layout, kernels.centre, sd, cells, fit.G)
        if cross
        else np.zeros((fit.G.shape[0], layout.n_blocks))
    )
    added = np.einsum("bi,ij,bj->b", xb, fit.A, xb) - 2.0 * np.einsum("bq,qb->b", xb, C)
    return Corrected(share + xb @ fit.beta, vb + added, xb, C, vb, fit)


# ---------------------------------------------------------------------------
# Empirical-Bayes shrinkage of a constant mean
# ---------------------------------------------------------------------------


def eb_lambda(beta: float, A: float, tau2: float | None = None) -> tuple[float, float]:
    """Shrinkage factor λ = τ²/(τ² + A) with the type-II ML τ̂² = max(0, β̂² − A)."""
    if tau2 is None:
        tau2 = max(0.0, beta**2 - A)
    return (tau2 / (tau2 + A) if tau2 > 0 else 0.0), tau2


def eb_correction(ok: Corrected, share: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """μ ~ N(0, τ²): shift λx_Bμ̂, variance V_B + λx_B²A − 2λx_Bc_B (``ok`` at λ = 1)."""
    xb = ok.xb[:, 0]
    A = float(ok.gls.A[0, 0])
    est = share + lam * xb * float(ok.gls.beta[0])
    var = ok.vb + lam * xb**2 * A - 2.0 * lam * xb * ok.C[0]
    return est, var


# ---------------------------------------------------------------------------
# Exact logit intercept (calibration-in-the-large)
# ---------------------------------------------------------------------------


CIL_PRIOR_VAR = 4.0  # N(0, 2²) prior on the logit intercept: the estimate exists without labels


def cil_delta(
    chol: tuple[np.ndarray, bool], logit_p: np.ndarray, y: np.ndarray, fallback: float
) -> tuple[float, bool]:
    """δ̂ solving σ(δ)ᵀK⁻¹z(δ) − δ/v₀ = 0 with p(δ) = expit(logit p + δ) at the validation
    points: the posterior mode of the intercept under a N(0, v₀) prior (quasi-likelihood)."""

    def f(delta: float) -> float:
        p = np.clip(special.expit(logit_p + delta), 1e-12, 1 - 1e-12)
        s = np.sqrt(p * (1.0 - p))
        z = (y - p) / s
        return float(s @ linalg.cho_solve(chol, z, check_finite=False)) - delta / CIL_PRIOR_VAR

    lo, hi = -30.0, 30.0
    f_lo, f_hi = f(lo), f(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return fallback, False
    return float(optimize.brentq(f, lo, hi, xtol=1e-8)), True


# ---------------------------------------------------------------------------
# Correlogram-parameter uncertainty: REML profile, curvature, unscented sigma points
# ---------------------------------------------------------------------------


def reml_profile(z: np.ndarray, D: np.ndarray, code: str, theta: np.ndarray) -> float:
    """The package's REML profile objective (`variogram.likelihood.fit_likelihood`)."""
    n = z.size
    eta = float(special.expit(theta[0]))
    K = (1.0 - eta) * (1.0 - psu.vgm(1.0, code, math.exp(theta[1])).gamma(D))
    K[np.diag_indices(n)] = 1.0 + 1e-10
    try:
        c = linalg.cho_factor(K, lower=True, check_finite=False)
    except linalg.LinAlgError:
        return 1e300
    one = np.ones(n)
    kz = linalg.cho_solve(c, z, check_finite=False)
    k1 = linalg.cho_solve(c, one, check_finite=False)
    logdet = 2.0 * float(np.sum(np.log(np.diag(c[0]))))
    s11 = float(one @ k1)
    q = max(float(z @ kz) - float(one @ kz) ** 2 / s11, 1e-300)
    return 0.5 * ((n - 1) * math.log(q) + logdet + math.log(s11))


def theta_of(model: psu.VariogramModel) -> tuple[str, np.ndarray]:
    eta = min(max(sh.nugget_share(model), special.expit(-LOGIT_BOUND)), special.expit(LOGIT_BOUND))
    return sh.family(model), np.array([float(special.logit(eta)), math.log(sh.main_range(model))])


def model_of(code: str, theta: np.ndarray) -> psu.VariogramModel:
    eta = float(special.expit(theta[0]))
    return psu.vgm(1.0 - eta, code, math.exp(theta[1]), nugget=eta)


def sigma_points(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    model: psu.VariogramModel,
    *,
    step: float = 0.05,
    kappa: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray, str, str]:
    """Unscented sigma points of θ = (logit η, log a) from the curvature of the REML profile.

    Coordinates at a bound of the package's fit are held fixed. Returns the points (the
    first is θ̂), their weights, the family and a status string.
    """
    code, th = theta_of(model)
    if code not in PRACTICAL:
        return [th], np.array([1.0]), code, "nugget"
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    pos = D[D > 0]
    diag_len = math.hypot(float(np.ptp(x)), float(np.ptp(y)))
    lo_a = math.log(0.1 * float(pos.min()) / PRACTICAL[code])
    hi_a = math.log(2.0 * diag_len / PRACTICAL[code])
    bounds = np.array([[-LOGIT_BOUND, LOGIT_BOUND], [lo_a, hi_a]])
    free = [i for i in range(2) if bounds[i, 0] + 2 * step < th[i] < bounds[i, 1] - 2 * step]
    if not free:
        return [th], np.array([1.0]), code, "bound"
    f0 = reml_profile(z, D, code, th)

    def f(d: np.ndarray) -> float:
        return reml_profile(z, D, code, th + d)

    d = len(free)
    H = np.empty((d, d))
    for a, i in enumerate(free):
        ei = np.zeros(2)
        ei[i] = step
        H[a, a] = (f(ei) - 2 * f0 + f(-ei)) / step**2
        for b in range(a):
            ej = np.zeros(2)
            ej[free[b]] = step
            H[a, b] = H[b, a] = (f(ei + ej) - f(ei - ej) - f(-ei + ej) + f(-ei - ej)) / (
                4 * step**2
            )
    try:
        L = np.linalg.cholesky(np.linalg.inv(H))
        if not np.all(np.isfinite(L)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        return [th], np.array([1.0]), code, "not_pd"
    scale = math.sqrt(d + kappa)
    pts = [th]
    for a in range(d):
        for sgn in (1.0, -1.0):
            t = th.copy()
            t[free] += sgn * scale * L[:, a]
            pts.append(np.clip(t, bounds[:, 0], bounds[:, 1]))
    w = np.full(len(pts), 1.0 / (2.0 * (d + kappa)))
    w[0] = kappa / (d + kappa)
    return pts, w, code, f"ok{d}"


# ---------------------------------------------------------------------------
# Design-based benchmarks
# ---------------------------------------------------------------------------


def ppi_block(resid: np.ndarray, psu_ids: np.ndarray, n_min: int = 20) -> tuple[float, float]:
    """Difference estimator: mean residual and its SE with primary sampling units ``psu_ids``
    (each point for a random sample, each cluster for a clustered one)."""
    n = resid.size
    if n < n_min:
        return math.nan, math.nan
    ids, inv = np.unique(psu_ids, return_inverse=True)
    m = ids.size
    if m < 2:
        return math.nan, math.nan
    rbar = float(resid.mean())
    R = np.bincount(inv, weights=resid, minlength=m)
    k = np.bincount(inv, minlength=m)
    var = m / (m - 1) * float(np.sum((R - k * rbar) ** 2)) / n**2
    return rbar, math.sqrt(var)
