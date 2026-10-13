"""Candidate improvements to the correlogram step of the method (experiment code only).

The current procedure fits a correlogram in three steps:

1. a Matheron sample variogram of the standardized errors z = e/σ, with gstat's defaults;
2. an automatic, bounded WLS fit of Nug + Exp/Sph/Gau with ``fit.method = 7``;
3. ρ = (sill − γ)/sill.

Each candidate here replaces one of these parts and keeps the rest; intervals stay normal,
prediction ± z·SD. Nothing in this module is used by the package.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import linalg, optimize, special

import pyspatialuncertainty as psu
from pyspatialuncertainty.variogram.empirical import GSTAT_CUTOFF_FRACTION, EmpiricalVariogram
from pyspatialuncertainty.variogram.models import VariogramModel

MODELS: tuple[str, ...] = ("Exp", "Sph", "Gau")
PRACTICAL = {"Exp": 3.0, "Sph": 1.0, "Gau": math.sqrt(3.0)}


# ---------------------------------------------------------------------------
# Sample variograms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pairs:
    """Point pairs within gstat's default cutoff, with their lag classes."""

    i: np.ndarray
    j: np.ndarray
    d: np.ndarray
    lag: np.ndarray
    n_bins: int
    boundaries: np.ndarray
    cutoff: float
    width: float
    n_obs: int


def pairs(x: np.ndarray, y: np.ndarray) -> Pairs:
    """Pairs and lag classes with gstat's defaults (cutoff 0.33333 × diagonal, 15 classes)."""
    cutoff = GSTAT_CUTOFF_FRACTION * math.hypot(float(np.ptp(x)), float(np.ptp(y)))
    width = cutoff / 15.0
    n_bins = max(1, math.ceil(cutoff / width - 1e-12))
    i, j = np.triu_indices(x.size, 1)
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    keep = d <= cutoff
    i, j, d = i[keep], j[keep], d[keep]
    frac = d / width
    lag = np.floor(frac).astype(np.int64)
    lag[(d > 0) & (frac == lag)] -= 1
    ok = (lag >= 0) & (lag < n_bins)
    bounds = np.arange(n_bins + 1, dtype=np.float64) * width
    bounds[-1] = min(bounds[-1], cutoff)
    return Pairs(i[ok], j[ok], d[ok], lag[ok], n_bins, bounds, cutoff, width, x.size)


def _ratio_variogram(p: Pairs, num: np.ndarray, den: np.ndarray) -> EmpiricalVariogram:
    """Sample variogram with γ = Σ num / Σ den in each lag class."""
    cnt = np.bincount(p.lag, minlength=p.n_bins)
    s_d = np.bincount(p.lag, weights=p.d, minlength=p.n_bins)
    s_num = np.bincount(p.lag, weights=num, minlength=p.n_bins)
    s_den = np.bincount(p.lag, weights=den, minlength=p.n_bins)
    ok = (cnt > 0) & (s_den > 0)
    return EmpiricalVariogram(
        n_pairs=cnt[ok].astype(np.int64),
        dist=s_d[ok] / cnt[ok],
        gamma=np.maximum(s_num[ok] / s_den[ok], 0.0),
        boundaries=p.boundaries,
        cutoff=p.cutoff,
        width=p.width,
        estimator="matheron",
        n_obs=p.n_obs,
    )


def matheron(p: Pairs, z: np.ndarray) -> EmpiricalVariogram:
    """Matheron's estimator, Σ (z_i − z_j)² / (2N), as ``psu.empirical_variogram``."""
    dz = z[p.i] - z[p.j]
    return _ratio_variogram(p, dz**2, np.full(dz.size, 2.0))


def sigma_weighted(p: Pairs, e: np.ndarray, s: np.ndarray) -> EmpiricalVariogram:
    """σσ-weighted variogram of the raw errors.

    γ_w(h) = Σ [(e_i − e_j)² − (σ_i − σ_j)²] / (2 Σ σ_i σ_j).

    Since E(e_i − e_j)² = (σ_i − σ_j)² + 2 σ_i σ_j (1 − ρ_ij), γ_w estimates the
    σσ-weighted mean of 1 − ρ at lag h. The block integral weights correlations the same
    way. For constant σ, γ_w is Matheron's variogram of e/σ.
    """
    num = (e[p.i] - e[p.j]) ** 2 - (s[p.i] - s[p.j]) ** 2
    return _ratio_variogram(p, num, 2.0 * s[p.i] * s[p.j])


def pooled(p: Pairs, zs: Sequence[np.ndarray]) -> EmpiricalVariogram:
    """Matheron's variogram pooled over several variables (e.g. the classes of a map)."""
    num = np.zeros(p.i.size)
    for z in zs:
        num += (z[p.i] - z[p.j]) ** 2
    return _ratio_variogram(p, num, np.full(p.i.size, 2.0 * len(zs)))


# ---------------------------------------------------------------------------
# Model fits
# ---------------------------------------------------------------------------


def wls(
    ev: EmpiricalVariogram, *, fit_method: int = 7, models: Sequence[str] = MODELS
) -> VariogramModel:
    """The package's automatic bounded WLS fit (the current default with the defaults)."""
    return psu.fit_variogram_auto(ev, models=models, fit_method=fit_method).model  # type: ignore[arg-type]


def correlation(model: VariogramModel) -> psu.CorrelationFunction:
    """ρ = (sill − γ)/sill; a zero-sill fit is treated as pure nugget (ρ = 0 for h > 0)."""
    if not model.sill > 0:
        return psu.as_correlation(psu.Nugget(1.0), check=False)
    return psu.as_correlation(model, check=False)


def distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])


def likelihood_fit(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    models: Sequence[str] = MODELS,
    max_practical: float | None = None,
    mean: Literal["constant", "zero"] = "constant",
) -> VariogramModel:
    """Gaussian likelihood fit of Nug + model; the best model by likelihood is returned.

    The covariance is s²·[(1 − η)ρ_a(h) + η·1(h = 0)]. The sill s² is profiled out
    analytically, and η and a are optimised: first on a coarse grid, then with L-BFGS-B.
    Practical ranges are bounded by twice the bounding-box diagonal.

    With ``mean="constant"`` this is REML with an unknown constant mean, which, like the
    sample variogram, ignores any common offset of the errors. With ``mean="zero"`` it is
    ML with the mean fixed at zero, as the method assumes for standardized errors (unbiased
    predictions, calibrated probabilities). A common offset in the validation errors then
    counts as evidence of long-range correlation, which is the error component that
    dominates large blocks.
    """
    n = z.size
    D = distance_matrix(x, y)
    diag_len = math.hypot(float(np.ptp(x)), float(np.ptp(y)))
    if max_practical is None:
        max_practical = 2.0 * diag_len
    min_d = float(D[D > 0].min())
    one = np.ones(n)
    idx = np.diag_indices(n)

    def profile(code: str, theta: np.ndarray) -> tuple[float, float]:
        eta = float(special.expit(theta[0]))
        a = math.exp(theta[1])
        K = (1.0 - eta) * (1.0 - psu.vgm(1.0, code, a).gamma(D))  # type: ignore[arg-type]
        K[idx] = 1.0 + 1e-10
        try:
            c = linalg.cho_factor(K, lower=True, check_finite=False)
        except linalg.LinAlgError:
            return 1e300, math.nan
        kz = linalg.cho_solve(c, z, check_finite=False)
        logdet = 2.0 * float(np.sum(np.log(np.diag(c[0]))))
        if mean == "zero":
            q = max(float(z @ kz), 1e-300)
            return 0.5 * (n * math.log(q) + logdet), q / n
        k1 = linalg.cho_solve(c, one, check_finite=False)
        s11 = float(one @ k1)
        s1z = float(one @ kz)
        q = max(float(z @ kz) - s1z**2 / s11, 1e-300)
        return 0.5 * ((n - 1) * math.log(q) + logdet + math.log(s11)), q / (n - 1)

    best: tuple[float, str, np.ndarray, float] | None = None
    for code in models:
        lo_a = math.log(0.1 * min_d / PRACTICAL[code])
        hi_a = math.log(max_practical / PRACTICAL[code])
        bounds = [(-6.0, 6.0), (lo_a, hi_a)]
        grid = [
            np.array([special.logit(e), la])
            for e in (0.03, 0.15, 0.35, 0.6, 0.85, 0.97)
            for la in np.linspace(lo_a, hi_a, 9)
        ]
        vals = [profile(code, t)[0] for t in grid]
        t0 = grid[int(np.argmin(vals))]
        res = optimize.minimize(
            lambda t, code=code: profile(code, t)[0],
            t0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 60},
        )
        theta = res.x if res.fun <= min(vals) else t0
        nll, s2 = profile(code, theta)
        if best is None or nll < best[0]:
            best = (nll, code, theta, s2)
    assert best is not None
    _, code, theta, s2 = best
    eta = float(special.expit(theta[0]))
    a = math.exp(theta[1])
    return psu.vgm((1.0 - eta) * s2, code, a, nugget=eta * s2)  # type: ignore[arg-type]
