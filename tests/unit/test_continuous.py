from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import shapely
import xarray as xr
from scipy import stats

import pyspatialuncertainty as psu


def make_maps(n: int = 40, sd_value: float = 2.0) -> tuple[xr.DataArray, xr.DataArray]:
    x = np.arange(n) + 0.5
    y = x[::-1]
    sd = xr.DataArray(np.full((n, n), sd_value), coords={"y": y, "x": x}, dims=("y", "x"))
    pred = xr.DataArray(np.add.outer(np.zeros(n), x), coords=sd.coords, dims=sd.dims)
    return sd, pred


def test_constant_sd_tiles() -> None:
    sd, pred = make_maps()
    res = psu.aggregate_continuous(
        sd,
        psu.vgm(1.0, "Exp", 4.0),
        psu.blocks.tiles(sd, 20),
        prediction=pred,
        mean="zero",
        n_mc=20_000,
        seed=1,
    )
    t = res.table
    assert t.height == 4
    assert t["block_id"].dtype == pl.Int64
    np.testing.assert_allclose(t["area"].to_numpy(), 400.0)
    np.testing.assert_array_equal(t["n_cells"].to_numpy(), 400)
    np.testing.assert_allclose(t["mean_pred"].to_numpy(), [10.0, 30.0, 10.0, 30.0])
    np.testing.assert_allclose(t["total_pred"].to_numpy(), 400 * t["mean_pred"].to_numpy())
    np.testing.assert_allclose(t["sd_total"].to_numpy(), 400 * t["sd_mean"].to_numpy())
    np.testing.assert_allclose(t["var_total"].to_numpy(), 400**2 * t["var_mean"].to_numpy())
    z = stats.norm.ppf(0.95)
    np.testing.assert_allclose(
        t["upper"].to_numpy() - t["mean_pred"].to_numpy(), z * t["sd_mean"].to_numpy()
    )
    # constant sigma: identical blocks give statistically identical variances
    v = t["var_mean"].to_numpy()
    assert np.ptp(v) < 6 * t["mcse_var"].max()
    assert np.all(t["sd_mean"].to_numpy() < 2.0)


def test_nan_cells_are_excluded() -> None:
    sd, pred = make_maps()
    sd = sd.copy()
    sd[0:5, 0:5] = np.nan
    res = psu.aggregate_continuous(
        sd,
        psu.vgm(1.0, "Sph", 10.0),
        psu.blocks.tiles(sd, 20),
        prediction=pred,
        mean="zero",
        n_mc=2000,
        seed=2,
    )
    row = res.table.row(0, named=True)
    assert row["n_cells"] == 400 - 25
    assert row["area"] == pytest.approx(375.0)


def test_without_prediction_and_area_scale() -> None:
    sd, _ = make_maps()
    res = psu.aggregate_continuous(
        sd,
        psu.vgm(1.0, "Exp", 4.0),
        psu.blocks.domain(sd),
        mean="zero",
        n_mc=2000,
        seed=3,
        area_scale=0.01,
    )
    row = res.table.row(0, named=True)
    assert np.isnan(row["mean_pred"])
    assert row["area"] == pytest.approx(16.0)
    assert row["sd_total"] == pytest.approx(16.0 * row["sd_mean"])


def test_polygons_with_string_ids_and_callable_correlation() -> None:
    sd, pred = make_maps()
    polys = [shapely.box(0, 0, 20, 10), shapely.Polygon([(0, 0), (40, 0), (40, 40)])]
    blocks = psu.blocks.polygons(polys, sd, ids=["strip", "triangle"])
    res = psu.aggregate_continuous(
        sd, lambda h: np.exp(-h / 3.0), blocks, prediction=pred, mean="zero", n_mc=5000, seed=4
    )
    assert res.table["block_id"].to_list() == ["strip", "triangle"]
    assert res.table["area"].to_numpy() == pytest.approx([200.0, 800.0])


def test_to_xarray_cells_and_coarse() -> None:
    sd, pred = make_maps()
    res = psu.aggregate_continuous(
        sd,
        psu.vgm(1.0, "Exp", 4.0),
        psu.blocks.tiles(sd, 10),
        prediction=pred,
        mean="zero",
        n_mc=1000,
        seed=5,
    )
    fine = res.to_xarray()
    assert fine["sd_mean"].shape == (40, 40)
    assert float(fine["mean_pred"].isel(y=0, x=0)) == pytest.approx(5.0)
    coarse = res.to_xarray(coarse=True)
    assert coarse["sd_mean"].shape == (4, 4)
    np.testing.assert_allclose(coarse.x.values, [5.0, 15.0, 25.0, 35.0])
    np.testing.assert_allclose(coarse.y.values, [35.0, 25.0, 15.0, 5.0])
    zres = psu.aggregate_continuous(
        sd, psu.vgm(1.0, "Exp", 4.0), psu.blocks.domain(sd), mean="zero", n_mc=500, seed=1
    )
    with pytest.raises(ValueError, match="only available for tiles"):
        zres.to_xarray(coarse=True)


def test_reproducible_with_seed() -> None:
    sd, pred = make_maps()
    args = (sd, psu.vgm(1.0, "Exp", 4.0), psu.blocks.tiles(sd, 20))
    a = psu.aggregate_continuous(*args, prediction=pred, mean="zero", n_mc=1000, seed=9).table
    b = psu.aggregate_continuous(*args, prediction=pred, mean="zero", n_mc=1000, seed=9).table
    assert a.equals(b)


def test_invalid_correlation_rejected() -> None:
    sd, _ = make_maps()
    with pytest.raises(ValueError, match=r"ρ\(0\) must equal 1"):
        psu.aggregate_continuous(
            sd, lambda h: 0.5 * np.ones_like(h), psu.blocks.domain(sd), mean="zero", n_mc=10
        )
