"""Tests of the Gaussian likelihood fit and of `fit_correlation`."""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl
import pytest
from scipy import stats

import pyspatialuncertainty as psu
from pyspatialuncertainty.variogram import likelihood as lik

TRUE = psu.vgm(0.75, "Exp", 8.0, nugget=0.25)


def correlated_errors(n: int, n_draws: int, seed: int) -> tuple[np.ndarray, ...]:
    """Coordinates and draws (n, n_draws) of unit-variance errors with correlogram TRUE."""
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    R = TRUE.correlation(D)
    R[np.diag_indices(n)] = 1.0
    z = np.linalg.cholesky(R) @ rng.standard_normal((n, n_draws))
    return x, y, z, D


def test_recovers_nugget_share_and_range() -> None:
    x, y, z, _ = correlated_errors(300, 5, seed=1)
    shares, ranges, sills = [], [], []
    for b in range(z.shape[1]):
        m = psu.fit_likelihood(None, z[:, b], x, y, mean="zero", models=("Exp",)).model
        shares.append(m.nugget / m.sill)
        ranges.append(m.structures[-1].range)
        sills.append(m.sill)
    assert 0.15 < np.median(shares) < 0.35
    assert 5.0 < np.median(ranges) < 12.0
    assert 0.8 < np.median(sills) < 1.2


@pytest.mark.parametrize("mean", ["zero", "constant"])
def test_loglik_is_the_gaussian_log_density(mean: str) -> None:
    x, y, z, D = correlated_errors(120, 1, seed=2)
    fit = psu.fit_likelihood(None, z[:, 0], x, y, mean=mean)  # type: ignore[arg-type]
    C = fit.model.covariance(D)
    if mean == "zero":
        expected = stats.multivariate_normal(np.zeros(z.shape[0]), C).logpdf(z[:, 0])
    else:
        # restricted likelihood: log-density of the n - 1 error contrasts
        n = z.shape[0]
        Ci = np.linalg.inv(C)
        one = np.ones(n)
        s11 = one @ Ci @ one
        mu = (one @ Ci @ z[:, 0]) / s11
        r = z[:, 0] - mu
        _, logdet = np.linalg.slogdet(C)
        expected = -0.5 * ((n - 1) * np.log(2 * np.pi) + logdet + np.log(s11) + r @ Ci @ r)
    assert fit.loglik == pytest.approx(float(expected), abs=1e-6)
    assert fit.mean == mean
    assert fit.n_obs == z.shape[0]


def test_reml_ignores_an_offset_but_zero_mean_ml_does_not() -> None:
    x, y, z, _ = correlated_errors(200, 1, seed=3)
    base = psu.fit_likelihood(None, z[:, 0], x, y, mean="constant").model
    shifted = psu.fit_likelihood(None, z[:, 0] + 2.0, x, y, mean="constant").model
    assert shifted.sill == pytest.approx(base.sill, rel=1e-6)
    zero = psu.fit_likelihood(None, z[:, 0], x, y, mean="zero").model
    zero_shifted = psu.fit_likelihood(None, z[:, 0] + 2.0, x, y, mean="zero").model
    # a common offset is read as strong long-range correlation
    far = np.array([90.0])
    assert zero_shifted.correlation(far)[0] > zero.correlation(far)[0] + 0.3


@pytest.mark.parametrize("seed", [4, 5, 6, 7])
def test_independent_errors_give_negligible_correlation(seed: int) -> None:
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 100, 250), rng.uniform(0, 100, 250)
    m = psu.fit_likelihood(None, rng.standard_normal(250), x, y).model
    # either a nugget or a structure shorter than the point spacing: no correlation at the
    # distances the data resolve
    assert np.all(m.correlation(np.array([3.0, 10.0, 30.0])) < 0.1)
    psu.as_correlation(m)  # a valid correlation function


def test_input_checks() -> None:
    rng = np.random.default_rng(5)
    x, y, z = rng.uniform(size=10), rng.uniform(size=10), rng.standard_normal(10)
    with pytest.raises(ValueError, match="mean"):
        psu.fit_likelihood(None, z, x, y, mean="median")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="models"):
        psu.fit_likelihood(None, z, x, y, models=("Mat",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="three"):
        psu.fit_likelihood(None, z[:2], x[:2], y[:2])
    with pytest.raises(ValueError, match="no variation"):
        psu.fit_likelihood(None, np.zeros(10), x, y)
    with pytest.raises(ValueError, match="same length"):
        psu.fit_likelihood(None, z, x[:5], y)
    with pytest.raises(ValueError, match="no data frame"):
        psu.fit_likelihood(None, "std_error", x, y)


def test_missing_values_are_dropped() -> None:
    x, y, z, _ = correlated_errors(80, 1, seed=6)
    zz = z[:, 0].copy()
    zz[:5] = np.nan
    fit = psu.fit_likelihood(None, zz, x, y)
    assert fit.n_obs == 75


def test_warns_for_many_points(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lik, "LIKELIHOOD_WARN_N", 20)
    x, y, z, _ = correlated_errors(30, 1, seed=7)
    with pytest.warns(UserWarning, match="may be slow"):
        lik.fit_likelihood(None, z[:, 0], x, y)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lik.fit_likelihood(None, z[:20, 0], x[:20], y[:20])


def test_fit_correlation_methods() -> None:
    x, y, z, _ = correlated_errors(200, 1, seed=8)
    val = pl.DataFrame({"x": x, "y": y, "std_error": z[:, 0]})
    ev = psu.empirical_variogram(val, "std_error", "x", "y")

    default = psu.fit_correlation(val)
    assert default.method == "likelihood"
    assert isinstance(default.details, psu.LikelihoodFit)
    assert default.details.mean == "constant"
    assert default.z is not None
    np.testing.assert_array_equal(default.z, z[:, 0])
    assert default.model == psu.fit_likelihood(val).model
    np.testing.assert_array_equal(default.empirical.gamma, ev.gamma)
    assert float(default.correlation(np.array([0.0]))[0]) == 1.0

    wls = psu.fit_correlation(val, method="wls")
    assert isinstance(wls.details, psu.VariogramFit)
    assert wls.model == psu.fit_variogram_auto(ev).model

    start = psu.vgm(1.0, "Sph", 20.0, nugget=0.2)
    given = psu.fit_correlation(val, method="wls", model=start)
    assert given.model == psu.fit_variogram(ev, start).model

    with pytest.raises(ValueError, match="method='wls'"):
        psu.fit_correlation(val, model=start)
    with pytest.raises(ValueError, match="method must be"):
        psu.fit_correlation(val, method="ols")  # type: ignore[arg-type]
