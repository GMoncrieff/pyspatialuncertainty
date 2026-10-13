from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr
from scipy import integrate

from pyspatialuncertainty.baselines import sd_average, sd_independent
from pyspatialuncertainty.blocks import box, resolve_blocks, tiles
from pyspatialuncertainty.grid import GridSpec
from pyspatialuncertainty.integrate import Integrand, mc_block_variance
from pyspatialuncertainty.intervals import normal_interval, z_value


def square_distance_pdf(t: float) -> float:
    """Density of the distance between two uniform points in the unit square."""
    if 0 <= t <= 1:
        return 2 * t * (t * t - 4 * t + math.pi)
    if 1 < t <= math.sqrt(2):
        return 2 * t * (4 * math.sqrt(t * t - 1) - (t * t + 2 - math.pi) - 4 * math.acos(1 / t))
    return 0.0


def expected_corr_square(side: float, corr: object) -> float:
    def f(t: float) -> float:
        return float(corr(np.array([side * t]))[0]) * square_distance_pdf(t)  # type: ignore[operator]

    a = integrate.quad(f, 0, 1, limit=200)[0]
    b = integrate.quad(f, 1, math.sqrt(2), limit=200)[0]
    return a + b


def test_square_distance_pdf_is_a_density() -> None:
    total = (
        integrate.quad(square_distance_pdf, 0, 1)[0]
        + integrate.quad(square_distance_pdf, 1, 2**0.5)[0]
    )
    assert total == pytest.approx(1.0, abs=1e-10)
    mean = (
        integrate.quad(lambda t: t * square_distance_pdf(t), 0, 1)[0]
        + integrate.quad(lambda t: t * square_distance_pdf(t), 1, 2**0.5)[0]
    )
    assert mean == pytest.approx(0.521405433, abs=1e-8)


@pytest.fixture
def grid() -> GridSpec:
    x = np.arange(40) + 0.5
    y = (np.arange(40) + 0.5)[::-1]
    return GridSpec.from_dataarray(
        xr.DataArray(np.zeros((40, 40)), coords={"y": y, "x": x}, dims=("y", "x"))
    )


def exp_corr(a: float) -> object:
    return lambda h: np.exp(-np.asarray(h) / a)


def test_perfect_correlation_gives_sigma_squared(grid: GridSpec) -> None:
    sd = np.full(grid.size, 1.7)
    blocks = resolve_blocks(tiles(grid, 10), np.ones(grid.size, dtype=bool))
    res = mc_block_variance(blocks, [Integrand(sd, lambda h: np.ones_like(h))], n_mc=500, seed=1)
    np.testing.assert_allclose(res.variance, 1.7**2, rtol=1e-12)
    np.testing.assert_allclose(res.mcse, 0.0, atol=1e-12)


@pytest.mark.parametrize("sampling", ["cells", "geometry"])
def test_matches_exact_square_integral(grid: GridSpec, sampling: str) -> None:
    corr = exp_corr(4.0)
    sd = np.full(grid.size, 2.0)
    valid = np.ones(grid.size, dtype=bool)
    if sampling == "cells":
        blocks = resolve_blocks(tiles(grid, 20), valid)[:1]
    else:
        blocks = resolve_blocks(box(0, 20, 20, 40, grid), valid)
    exact = 4.0 * expected_corr_square(20.0, corr)
    res = mc_block_variance(blocks, [Integrand(sd, corr)], n_mc=200_000, seed=7)  # type: ignore[arg-type]
    assert abs(res.variance[0, 0] - exact) < 4 * res.mcse[0, 0]
    assert res.mcse[0, 0] / exact < 0.01


def test_mcse_is_calibrated(grid: GridSpec) -> None:
    corr = exp_corr(3.0)
    sd = np.full(grid.size, 1.0)
    blocks = resolve_blocks(tiles(grid, 10), np.ones(grid.size, dtype=bool))
    exact = expected_corr_square(10.0, corr)
    res = mc_block_variance(blocks, [Integrand(sd, corr)], n_mc=2000, seed=11)  # type: ignore[arg-type]
    z = (res.variance[0] - exact) / res.mcse[0]
    # 16 independent blocks; with a well-calibrated MCSE these are standard normal
    assert np.mean(np.abs(z) < 1.96) >= 0.75
    assert abs(float(np.mean(z))) < 1.0


def test_shared_pairs_and_reproducibility(grid: GridSpec) -> None:
    corr = exp_corr(5.0)
    rng = np.random.default_rng(0)
    sd = rng.uniform(0.5, 2.0, grid.size)
    blocks = resolve_blocks(tiles(grid, 8), np.ones(grid.size, dtype=bool))
    a = mc_block_variance(blocks, [Integrand(sd, corr), Integrand(2 * sd, corr)], n_mc=3000, seed=5)  # type: ignore[arg-type]
    np.testing.assert_allclose(a.variance[1], 4 * a.variance[0], rtol=1e-12)
    b = mc_block_variance(blocks, [Integrand(sd, corr)], n_mc=3000, seed=5)  # type: ignore[arg-type]
    np.testing.assert_array_equal(a.variance[0], b.variance[0])
    # streams are per block: a block's estimate does not depend on the other blocks
    c = mc_block_variance(blocks[:3], [Integrand(sd, corr)], n_mc=3000, seed=5)  # type: ignore[arg-type]
    np.testing.assert_array_equal(c.variance[0], a.variance[0, :3])


def test_nan_sd_raises(grid: GridSpec) -> None:
    sd = np.ones(grid.size)
    sd[0] = np.nan
    blocks = resolve_blocks(tiles(grid, 40), np.ones(grid.size, dtype=bool))
    with pytest.raises(ValueError, match="non-finite"):
        mc_block_variance(blocks, [Integrand(sd, exp_corr(2.0))], n_mc=5000, seed=1)  # type: ignore[arg-type]


def test_baselines_bracket_the_mc_estimate(grid: GridSpec) -> None:
    corr = exp_corr(3.0)
    rng = np.random.default_rng(2)
    sd = rng.uniform(0.5, 1.5, grid.size)
    blocks = resolve_blocks(tiles(grid, 10), np.ones(grid.size, dtype=bool))
    res = mc_block_variance(blocks, [Integrand(sd, corr)], n_mc=5000, seed=3)  # type: ignore[arg-type]
    sd_mc = np.sqrt(res.variance[0])
    avg = sd_average(blocks, sd)
    ind = sd_independent(blocks, sd)
    assert np.all(ind < sd_mc)
    assert np.all(sd_mc < avg)
    np.testing.assert_allclose(avg[0], sd[blocks[0].cells].mean())
    np.testing.assert_allclose(ind[0], np.sqrt(np.sum(sd[blocks[0].cells] ** 2)) / 100)


def test_intervals() -> None:
    assert z_value(0.9) == pytest.approx(1.6448536269514722)
    lo, hi = normal_interval([10.0], [1.0], level=0.95)
    assert (lo[0], hi[0]) == pytest.approx((10 - 1.959964, 10 + 1.959964), abs=1e-6)
    lo, hi = normal_interval([0.05], [0.1], z=1.64, lower_bound=0.0, upper_bound=1.0)
    assert lo[0] == 0.0
    assert hi[0] == pytest.approx(0.214)
    with pytest.raises(ValueError, match="level"):
        z_value(1.0)
