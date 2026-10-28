"""Parity of steps 3 and 5-8 with the authors' R script on its own exported inputs."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import pytest
import xarray as xr
from scipy import stats

import pyspatialuncertainty as psu

pytestmark = pytest.mark.r_parity


@pytest.fixture(scope="module")
def model(read_ref: Callable[[str], pl.DataFrame]) -> psu.VariogramModel:
    return psu.VariogramModel.from_table(read_ref("variogram_model.csv"))


@pytest.fixture(scope="module")
def sd_map(r_maps: dict[str, xr.DataArray]) -> xr.DataArray:
    return np.sqrt(r_maps["predVarRF"])


def test_standardized_errors(read_ref: Callable[[str], pl.DataFrame]) -> None:
    pt = read_ref("pt_all.csv").with_columns(pl.col("predVarRF").sqrt().alias("sd"))
    out = psu.standardized_errors(pt, observed="y", predicted="predRF", sd="sd")
    np.testing.assert_allclose(out["std_error"].to_numpy(), pt["stdErrRF"].to_numpy(), rtol=1e-12)


def test_integrand_on_the_scripts_own_pairs(
    read_ref: Callable[[str], pl.DataFrame], sd_map: xr.DataArray, model: psu.VariogramModel
) -> None:
    pairs = read_ref("mc_pairs.csv")
    grid = psu.GridSpec.from_dataarray(sd_map)
    sd = grid.values(sd_map)
    x1, y1, x2, y2 = (pairs[c].to_numpy() for c in ("x1", "y1", "x2", "y2"))
    c1, c2 = grid.flat_index_of(x1, y1), grid.flat_index_of(x2, y2)
    np.testing.assert_allclose(sd[c1], pairs["sd1"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(sd[c2], pairs["sd2"].to_numpy(), rtol=1e-12)
    rho = model.correlation(np.hypot(x1 - x2, y1 - y2))
    np.testing.assert_allclose(rho, pairs["rho"].to_numpy(), rtol=1e-12)
    integrand = sd[c1] * sd[c2] * rho
    np.testing.assert_allclose(integrand, pairs["integrand"].to_numpy(), rtol=1e-12)


def test_block_mean_prediction_and_area(
    r_maps: dict[str, xr.DataArray],
    sd_map: xr.DataArray,
    model: psu.VariogramModel,
    r_summary: dict[str, float],
) -> None:
    res = psu.aggregate_continuous(
        sd_map,
        model,
        psu.blocks.box(0, 0, 100, 100, sd_map),
        prediction=r_maps["predRF"],
        mean="zero",
        n_mc=1000,
        seed=0,
    )
    row = res.table.row(0, named=True)
    assert row["mean_pred"] == pytest.approx(r_summary["mean_predRF"], rel=1e-12)
    assert row["area"] == pytest.approx(r_summary["B"], rel=1e-12)
    assert row["total_pred"] == pytest.approx(r_summary["sum_predRF"], rel=1e-12)
    assert row["n_cells"] == 10_000


@pytest.fixture(scope="module")
def python_highm(
    r_maps: dict[str, xr.DataArray], sd_map: xr.DataArray, model: psu.VariogramModel
) -> dict[str, float]:
    res = psu.aggregate_continuous(
        sd_map,
        model,
        psu.blocks.box(0, 0, 100, 100, sd_map),
        prediction=r_maps["predRF"],
        mean="zero",
        n_mc=1_000_000,
        z=1.64,
        seed=20231320,
    )
    return res.table.row(0, named=True)


def test_mc_estimate_agrees_with_the_script(
    python_highm: dict[str, float], r_summary: dict[str, float]
) -> None:
    se = np.hypot(r_summary["mcse_script"], python_highm["mcse_var"])
    assert abs(python_highm["var_mean"] - r_summary["var_average"]) < 3 * se


def test_mc_estimate_agrees_with_high_precision_r(
    python_highm: dict[str, float], read_ref: Callable[[str], pl.DataFrame]
) -> None:
    hm = read_ref("mc_highm.csv").row(0, named=True)
    se = np.hypot(hm["mcse"], python_highm["mcse_var"])
    assert abs(python_highm["var_mean"] - hm["var_average"]) < 3 * se
    # and the Monte Carlo standard errors themselves agree (same integrand distribution)
    assert python_highm["mcse_var"] == pytest.approx(hm["mcse"], rel=0.02)


def test_replicate_distribution_matches_r(
    read_ref: Callable[[str], pl.DataFrame], sd_map: xr.DataArray, model: psu.VariogramModel
) -> None:
    r_reps = read_ref("mc_replicates.csv")["var_average"].to_numpy()
    grid = psu.GridSpec.from_dataarray(sd_map)
    sd = grid.values(sd_map)
    blocks = psu.resolve_blocks(psu.blocks.box(0, 0, 100, 100, grid), np.isfinite(sd))
    integrand = psu.Integrand(sd, psu.from_variogram(model))
    py_reps = np.array(
        [
            psu.mc_block_variance(blocks, [integrand], n_mc=1000, seed=s).variance[0, 0]
            for s in range(r_reps.size)
        ]
    )
    se_diff = np.hypot(r_reps.std(ddof=1), py_reps.std(ddof=1)) / np.sqrt(r_reps.size)
    assert abs(py_reps.mean() - r_reps.mean()) < 3 * se_diff
    f = py_reps.var(ddof=1) / r_reps.var(ddof=1)
    p = 2 * min(
        stats.f.cdf(f, py_reps.size - 1, r_reps.size - 1),
        stats.f.sf(f, py_reps.size - 1, r_reps.size - 1),
    )
    assert p > 0.001


def test_interval_and_total_arithmetic(r_summary: dict[str, float]) -> None:
    sd = np.sqrt(r_summary["var_average"])
    lo, hi = psu.normal_interval([r_summary["mean_predRF"]], [sd], z=1.64)
    assert lo[0] == pytest.approx(r_summary["pi90_lower_z164"], rel=1e-12)
    assert hi[0] == pytest.approx(r_summary["pi90_upper_z164"], rel=1e-12)
    assert r_summary["var_total"] == pytest.approx(
        r_summary["var_average"] * r_summary["B"] ** 2, rel=1e-12
    )
    # the true block mean of the example lies inside the script's 90% interval
    assert lo[0] <= r_summary["mean_y"] <= hi[0]
