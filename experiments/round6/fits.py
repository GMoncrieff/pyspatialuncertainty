"""Correlogram fits of the standardized errors, and their parameter uncertainty.

Every fit returns a `Fit`: the correlation function used for the block integrals and the
GLS, and, for two-parameter fits with a constant mean (REML), the profile objective in
θ = (logit η, log a) with its bounds, so that the parameter uncertainty can be propagated.

* Package fits (``fit_likelihood``): free family (Exp/Sph/Gau), Exp/Sph, Exp only, or the
  zero-mean ML fit of the paper's method.
* Threshold-shaped REML: the correlation of the indicator 1{Y > t} of a latent standard
  normal field Y with correlation (1 − η)·exp(−h/a), cut at the class's mean threshold
  t = Φ⁻¹(1 − p̄). For a pair with latent correlation r,

      Cov(1{Y₁ > t}, 1{Y₂ > t}) = Φ(t)(1 − Φ(t)) − 2·T(t, √((1 − r)/(1 + r)))

  (Owen's T), so the indicator correlation is g(r) = 1 − 2T/(p̄(1 − p̄)), which drops fast at
  short lags and keeps the latent range in its tail. At t = 0 it is (2/π)·asin(r).

Parameter uncertainty, applied the same way at every sample size:

* Laplace: unscented sigma points from the curvature of the profile at the fit (round 5).
* Grid posterior: the profile on a 13 × 13 grid (±4 Laplace SDs around the fit, or the
  whole bound range where the curvature is not positive), a flat prior, and sigma points
  from the posterior mean and covariance.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import linalg, optimize, special, stats

from experiments.round5 import arms as ar
from experiments.round5 import shared as sh
from experiments.round5.run import fit as package_fit

LOGIT_BOUND = ar.LOGIT_BOUND
GRID_NUGGET = (0.03, 0.15, 0.35, 0.6, 0.85, 0.97)  # the package's start grid
GRID_RANGES = 9
N_GRID = 13
GRID_WIDTH = 4.0
KAPPA = 1.0
STEP = 0.05


@dataclass
class Fit:
    label: str
    corr: sh.Corr
    family: str
    sill: float  # REML (or ML) estimate of the variance of the standardized errors
    status: str = ""  # "" or the fallback message
    theta: np.ndarray | None = None  # (logit η, log a) for two-parameter REML fits
    bounds: np.ndarray | None = None  # (2, 2)
    profile: Callable[[np.ndarray], float] | None = None
    corr_of: Callable[[np.ndarray], sh.Corr] | None = None
    practical: float = math.nan
    nugget: float = math.nan

    def describe(self) -> dict[str, object]:
        return {
            f"{self.label}_family": self.family,
            f"{self.label}_range": self.practical,
            f"{self.label}_nugget": self.nugget,
            f"{self.label}_status": self.status,
        }


def distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])


def gls_factor(corr: sh.Corr, D: np.ndarray) -> tuple[np.ndarray, bool]:
    """``arms.gls_factor`` from precomputed distances (built in row chunks)."""
    n = D.shape[0]
    K = np.empty((n, n))
    step = 250
    for a in range(0, n, step):
        b = min(a + step, n)
        K[a:b] = corr.struct(D[a:b])
    K[np.diag_indices(n)] = 1.0 + 1e-10
    return linalg.cho_factor(K, lower=True, overwrite_a=True, check_finite=False)


def package(
    label: str,
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    D: np.ndarray,
    *,
    mean: str = "constant",
    models: tuple[str, ...] = ("Exp", "Sph", "Gau"),
) -> Fit:
    """A package likelihood fit (nugget only when the errors carry no variation)."""
    model, status = package_fit(z, x, y, mean=mean, models=models)
    corr = sh.corr_of_model(model)
    code = sh.family(model)
    out = Fit(
        label,
        corr,
        code,
        float(model.sill),
        status,
        practical=sh.practical_range(model),
        nugget=sh.nugget_share(model),
    )
    if mean == "constant" and not status and code in ar.PRACTICAL:
        _, th = ar.theta_of(model)
        diag_len = math.hypot(float(np.ptp(x)), float(np.ptp(y)))
        pos = D[D > 0]
        out.theta = th
        out.bounds = np.array(
            [
                [-LOGIT_BOUND, LOGIT_BOUND],
                [
                    math.log(0.1 * float(pos.min()) / ar.PRACTICAL[code]),
                    math.log(2.0 * diag_len / ar.PRACTICAL[code]),
                ],
            ]
        )
        out.profile = lambda t, c=code: ar.reml_profile(z, D, c, t)
        out.corr_of = lambda t, c=code: sh.corr_of_model(ar.model_of(c, t))
    return out


# ---------------------------------------------------------------------------
# Threshold-shaped correlogram
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Threshold:
    """Indicator correlation g(r) for latent correlation r, tabulated in u = √(1 − r)."""

    t: float
    p: float
    u: np.ndarray
    g: np.ndarray

    @classmethod
    def at(cls, p_bar: float, m: int = 4001) -> Threshold:
        p_bar = float(np.clip(p_bar, 1e-6, 1 - 1e-6))
        t = float(stats.norm.ppf(1.0 - p_bar))
        u = np.linspace(0.0, 1.0, m)
        a = u / np.sqrt(2.0 - u**2)  # √((1 − r)/(1 + r)) at r = 1 − u²
        g = 1.0 - 2.0 * special.owens_t(t, a) / (p_bar * (1.0 - p_bar))
        return cls(t, p_bar, u, g)

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=np.float64)
        return np.interp(np.sqrt(np.clip(1.0 - r, 0.0, 1.0)), self.u, self.g)

    def corr(self, theta: np.ndarray) -> sh.Corr:
        s = 1.0 - float(special.expit(theta[0]))
        a = math.exp(float(theta[1]))
        return sh.Corr(
            lambda d: self(s * np.exp(-np.asarray(d, dtype=np.float64) / a)),
            float(self(np.array([s]))[0]),
            "thr",
        )


def threshold_profile(
    z: np.ndarray, Diu: np.ndarray, iu: tuple[np.ndarray, np.ndarray], thr: Threshold
) -> Callable[[np.ndarray], tuple[float, float]]:
    """Negative REML profile (constant mean, sill profiled out) and the sill, as the package."""
    n = z.size
    one = np.ones(n)
    lower = (iu[1], iu[0])

    def f(theta: np.ndarray) -> tuple[float, float]:
        s = 1.0 - float(special.expit(theta[0]))
        vals = thr(s * np.exp(-Diu / math.exp(float(theta[1]))))
        K = np.empty((n, n))
        K[iu] = vals
        K[lower] = vals
        K[np.diag_indices(n)] = 1.0 + 1e-10
        try:
            c = linalg.cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
        except linalg.LinAlgError:
            return 1e300, math.nan
        kz = linalg.cho_solve(c, z, check_finite=False)
        k1 = linalg.cho_solve(c, one, check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(c[0]))))
        s11 = float(one @ k1)
        q = max(float(z @ kz) - float(one @ kz) ** 2 / s11, 1e-300)
        return 0.5 * ((n - 1) * math.log(q) + logdet + math.log(s11)), q / (n - 1)

    return f


def threshold(
    label: str, z: np.ndarray, x: np.ndarray, y: np.ndarray, D: np.ndarray, p_bar: float
) -> Fit:
    """Threshold-shaped REML fit of the standardized indicator errors of one class."""
    thr = Threshold.at(p_bar)
    if np.ptp(z) == 0:
        return Fit(label, sh.corr_of_model(sh.NUGGET_ONLY), "Nug", 1.0, "fallback")
    n = z.size
    iu = np.triu_indices(n, 1)
    Diu = D[iu]
    pos = Diu[Diu > 0]
    if pos.size == 0:
        return Fit(label, sh.corr_of_model(sh.NUGGET_ONLY), "Nug", 1.0, "fallback")
    diag_len = math.hypot(float(np.ptp(x)), float(np.ptp(y)))
    bounds = np.array(
        [
            [-LOGIT_BOUND, LOGIT_BOUND],
            [math.log(0.1 * float(pos.min()) / 3.0), math.log(2.0 * diag_len / 3.0)],
        ]
    )
    prof = threshold_profile(z, Diu, iu, thr)
    grid = [
        np.array([float(special.logit(e)), la])
        for e in GRID_NUGGET
        for la in np.linspace(bounds[1, 0], bounds[1, 1], GRID_RANGES)
    ]
    values = [prof(t)[0] for t in grid]
    start = grid[int(np.argmin(values))]
    res = optimize.minimize(
        lambda t: prof(t)[0],
        start,
        method="L-BFGS-B",
        bounds=[tuple(b) for b in bounds],
        options={"maxiter": 60},
    )
    theta = np.asarray(res.x if res.fun <= min(values) else start, dtype=np.float64)
    nll, s2 = prof(theta)
    if not nll < 1e300:
        return Fit(label, sh.corr_of_model(sh.NUGGET_ONLY), "Nug", 1.0, "fallback: no fit")
    corr = thr.corr(theta)
    return Fit(
        label,
        corr,
        "thr",
        float(s2),
        "",
        theta,
        bounds,
        lambda t: prof(t)[0],
        thr.corr,
        practical=3.0 * math.exp(float(theta[1])),
        nugget=1.0 - corr.rho0,
    )


# ---------------------------------------------------------------------------
# Parameter uncertainty
# ---------------------------------------------------------------------------


def curvature(fit: Fit, step: float = STEP) -> tuple[np.ndarray | None, list[int]]:
    """Finite-difference Hessian of the profile in the coordinates away from the bounds."""
    assert fit.theta is not None
    assert fit.bounds is not None
    assert fit.profile is not None
    th, b = fit.theta, fit.bounds
    free = [i for i in range(2) if b[i, 0] + 2 * step < th[i] < b[i, 1] - 2 * step]
    if not free:
        return None, free
    f0 = fit.profile(th)

    def f(d: np.ndarray) -> float:
        return fit.profile(th + d)  # type: ignore[misc]

    H = np.empty((len(free), len(free)))
    for a, i in enumerate(free):
        ei = np.zeros(2)
        ei[i] = step
        H[a, a] = (f(ei) - 2 * f0 + f(-ei)) / step**2
        for c in range(a):
            ej = np.zeros(2)
            ej[free[c]] = step
            H[a, c] = H[c, a] = (f(ei + ej) - f(ei - ej) - f(-ei + ej) + f(-ei - ej)) / (
                4 * step**2
            )
    return H, free


def _points(
    center: np.ndarray, L: np.ndarray, free: list[int], bounds: np.ndarray, kappa: float
) -> tuple[list[np.ndarray], np.ndarray]:
    d = len(free)
    scale = math.sqrt(d + kappa)
    pts = [center.copy()]
    for a in range(d):
        for sgn in (1.0, -1.0):
            t = center.copy()
            t[free] += sgn * scale * L[:, a]
            pts.append(np.clip(t, bounds[:, 0], bounds[:, 1]))
    w = np.full(len(pts), 1.0 / (2.0 * (d + kappa)))
    w[0] = kappa / (d + kappa)
    return pts, w


def laplace(fit: Fit, kappa: float = KAPPA) -> tuple[list[np.ndarray], np.ndarray, str]:
    """Round 5's unscented sigma points; the first point is the fit itself."""
    if fit.theta is None:
        return [], np.array([1.0]), "none"
    H, free = curvature(fit)
    if H is None:
        return [fit.theta], np.array([1.0]), "bound"
    try:
        L = np.linalg.cholesky(np.linalg.inv(H))
        if not np.all(np.isfinite(L)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        return [fit.theta], np.array([1.0]), "not_pd"
    assert fit.bounds is not None
    pts, w = _points(fit.theta, L, free, fit.bounds, kappa)
    return pts, w, f"ok{len(free)}"


def grid_posterior(
    fit: Fit, n_grid: int = N_GRID, width: float = GRID_WIDTH, kappa: float = KAPPA
) -> tuple[list[np.ndarray], np.ndarray, str]:
    """Sigma points of the flat-prior posterior of θ on a grid around the fit."""
    if fit.theta is None:
        return [], np.array([1.0]), "none"
    assert fit.bounds is not None
    assert fit.profile is not None
    th, b = fit.theta, fit.bounds
    lo, hi = b[:, 0].copy(), b[:, 1].copy()
    H, free = curvature(fit)
    sd = np.full(2, np.nan)
    if H is not None:
        try:
            cov = np.linalg.inv(H)
            dg = np.diag(cov)
            if np.all(dg > 0) and np.all(np.isfinite(dg)):
                sd[free] = np.sqrt(dg)
        except np.linalg.LinAlgError:
            pass
    for i in range(2):
        if np.isfinite(sd[i]):
            lo[i] = max(lo[i], th[i] - width * sd[i])
            hi[i] = min(hi[i], th[i] + width * sd[i])
    g0, g1 = np.linspace(lo[0], hi[0], n_grid), np.linspace(lo[1], hi[1], n_grid)
    nll = np.array([[fit.profile(np.array([a, c])) for c in g1] for a in g0])
    w = np.exp(-(nll - nll.min()))
    w /= w.sum()
    A, C = np.meshgrid(g0, g1, indexing="ij")
    mean = np.array([float((w * A).sum()), float((w * C).sum())])
    dA, dC = A - mean[0], C - mean[1]
    S = np.array(
        [
            [float((w * dA * dA).sum()), float((w * dA * dC).sum())],
            [float((w * dA * dC).sum()), float((w * dC * dC).sum())],
        ]
    )
    keep = [i for i in range(2) if S[i, i] > 1e-12]
    if not keep:
        return [th], np.array([1.0]), "narrow"
    try:
        L = np.linalg.cholesky(S[np.ix_(keep, keep)])
    except np.linalg.LinAlgError:
        return [th], np.array([1.0]), "not_pd"
    pts, wts = _points(mean, L, keep, b, kappa)
    return pts, wts, f"grid{len(keep)}"


# ---------------------------------------------------------------------------
# Dispersion of class maps
# ---------------------------------------------------------------------------


def brier_ratio(labels: np.ndarray, p_pts: np.ndarray) -> float:
    """Pooled Σ(y − p)² / Σ p(1 − p) over the classes and the validation points."""
    Y = np.eye(p_pts.shape[0])[labels].T
    return float(((Y - p_pts) ** 2).sum() / (p_pts * (1.0 - p_pts)).sum())
