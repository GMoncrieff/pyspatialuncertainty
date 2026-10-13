from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from pyspatialuncertainty.variogram import (
    Exponential,
    Gaussian,
    Matern,
    MaternStein,
    Nugget,
    Spherical,
    Structure,
    VariogramModel,
    vgm,
)


def test_closed_forms() -> None:
    a = 7.0
    assert Exponential(2.0, a).gamma(a)[()] == pytest.approx(2.0 * (1 - math.exp(-1)))
    assert Spherical(1.0, a).gamma(a / 2)[()] == pytest.approx(0.6875)
    assert Spherical(1.0, a).gamma(np.array([a, 2 * a, 100 * a])) == pytest.approx([1.0, 1.0, 1.0])
    assert Gaussian(1.0, a).gamma(a)[()] == pytest.approx(1 - math.exp(-1))


def test_gamma_zero_at_origin_and_nugget_jump() -> None:
    for m in (Nugget(1.0), Exponential(1, 3), Spherical(1, 3), Gaussian(1, 3), Matern(1, 3, 1.5)):
        assert m.gamma(0.0)[()] == 0.0
    nug = Nugget(0.3)
    assert nug.gamma(np.array([0.0, 1e-12, 5.0])) == pytest.approx([0.0, 0.3, 0.3])


def test_matern_half_equals_exponential() -> None:
    h = np.linspace(0, 50, 101)
    np.testing.assert_allclose(
        Matern(1.0, 4.0, 0.5).gamma(h), Exponential(1.0, 4.0).gamma(h), atol=1e-12
    )
    # Stein's parameterisation with kappa = 1/2 is exponential with range a / sqrt(2)
    np.testing.assert_allclose(
        MaternStein(1.0, 4.0, 0.5).gamma(h),
        Exponential(1.0, 4.0 / math.sqrt(2)).gamma(h),
        atol=1e-12,
    )


def test_matern_large_kappa_tends_to_gaussian_shape() -> None:
    h = np.linspace(0, 20, 50)
    g = MaternStein(1.0, 5.0, 50.0).gamma(h)
    assert np.all(np.diff(g) >= -1e-12)
    assert g[-1] == pytest.approx(1.0, abs=1e-6)


def test_nested_model_sill_and_correlation() -> None:
    m = vgm(0.46, "Exp", 2, nugget=0.05) + vgm(0.08, "Sph", 15) + vgm(0.06, "Exp", 60)
    assert m.sill == pytest.approx(0.65)
    assert m.nugget == pytest.approx(0.05)
    assert len(m) == 4
    rho = m.correlation(np.array([0.0, 1e-9, 1000.0]))
    assert rho[0] == 1.0
    assert rho[1] == pytest.approx(1 - 0.05 / 0.65, rel=1e-6)
    assert rho[2] == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(m.covariance(np.array([3.0])), m.sill - m.gamma(np.array([3.0])))
    corr = m.to_correlation()
    np.testing.assert_allclose(corr(np.array([1.0, 2.0])), m.correlation(np.array([1.0, 2.0])))


def test_table_round_trip() -> None:
    m = vgm(1.0, "Mat", 3.0, nugget=0.2, kappa=1.5) + vgm(0.5, "Gau", 10.0)
    back = VariogramModel.from_table(m.to_polars())
    assert back == m


def test_from_gstat_table_with_zero_nuggets() -> None:
    tab = pl.DataFrame(
        {
            "model": ["Nug", "Exp", "Nug", "Sph"],
            "psill": [0.0, 0.06, 0.05, 0.08],
            "range": [0.0, 60.0, 0.0, 15.0],
            "kappa": [0.0, 0.5, 0.0, 0.5],
        }
    )
    m = VariogramModel.from_table(tab)
    assert m.sill == pytest.approx(0.19)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"model": "Exp", "psill": -1.0, "range": 1.0}, "psill"),
        ({"model": "Exp", "psill": 1.0, "range": 0.0}, "range"),
        ({"model": "Mat", "psill": 1.0, "range": 1.0, "kappa": 0.0}, "kappa"),
        ({"model": "Foo", "psill": 1.0, "range": 1.0}, "unknown model"),
    ],
)
def test_invalid_structures(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Structure(**kwargs)  # type: ignore[arg-type]


def test_negative_lags_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Exponential(1, 1).gamma(np.array([-1.0]))


def test_correlation_requires_positive_sill() -> None:
    with pytest.raises(ValueError, match="zero sill"):
        Nugget(0.0).correlation(np.array([1.0]))
