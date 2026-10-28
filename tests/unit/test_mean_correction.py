from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr

import pyspatialuncertainty as psu
from pyspatialuncertainty.mean_correction import block_terms, estimate_mean, structured_correlation


def grid_map(n: int = 12, value: float = 1.0) -> xr.DataArray:
    x = np.arange(n) + 0.5
    return xr.DataArray(np.full((n, n), value), coords={"y": x[::-1], "x": x}, dims=("y", "x"))


def validation(n_pts: int = 30, n: int = 12, seed: int = 0, offset: float = 0.0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cells = rng.choice(n * n, n_pts, replace=False)
    return pl.DataFrame(
        {
            "x": (cells % n) + 0.5,
            "y": (n - 1 - cells // n) + 0.5,
            "std_error": offset + rng.normal(size=n_pts),
        }
    )


def test_estimate_mean_is_the_gls_mean() -> None:
    val = validation()
    x, y, z = (val[c].to_numpy() for c in ("x", "y", "std_error"))
    # independent errors: plain mean, variance 1/n
    est = estimate_mean(psu.vgm(1.0, "Nug"), x, y, z)
    assert est.beta == pytest.approx(z.mean())
    assert est.var_beta == pytest.approx(1 / z.size)
    # correlated errors: the textbook GLS formulas
    model = psu.vgm(0.8, "Exp", 3.0, nugget=0.2)
    est = estimate_mean(model, x, y, z)
    D = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    K = structured_correlation(model, D)
    np.fill_diagonal(K, 1.0)
    Ki1 = np.linalg.solve(K, np.ones(z.size))
    assert est.var_beta == pytest.approx(1 / Ki1.sum(), rel=1e-8)
    assert est.beta == pytest.approx(Ki1 @ z / Ki1.sum(), rel=1e-8)
    assert est.weights.sum() == pytest.approx(1.0)


def test_block_terms_match_brute_force() -> None:
    sd = grid_map()
    sd = sd + 0.05 * sd.x  # sigma varies across the map
    val = validation()
    model = psu.vgm(0.8, "Exp", 3.0, nugget=0.2)
    est = estimate_mean(model, *(val[c].to_numpy() for c in ("x", "y", "std_error")))
    blocks = psu.blocks.tiles(sd, 6)
    grid = blocks.grid
    flat = grid.values(sd)
    resolved = psu.resolve_blocks(blocks, np.isfinite(flat))
    terms = block_terms(est, flat, grid, resolved)
    for b, blk in enumerate(resolved):
        cx, cy = grid.cell_centres(blk.cells)
        s = flat[blk.cells]
        xb = s.mean()
        h = np.hypot(cx[:, None] - est.x[None, :], cy[:, None] - est.y[None, :])
        cb = np.mean(s * (structured_correlation(model, h) @ est.weights))
        assert terms.sd_mean[b] == pytest.approx(xb)
        assert terms.shift[b] == pytest.approx(xb * est.beta)
        assert terms.var_beta[b] == pytest.approx(xb**2 * est.var_beta)
        assert terms.cross[b] == pytest.approx(2 * xb * cb)
    assert np.all(block_terms(est, flat, grid, resolved, cross_term=False).cross == 0.0)


def test_aggregate_continuous_applies_the_correction() -> None:
    sd = grid_map(value=2.0)
    pred = sd * 0 + 10.0
    val = validation(offset=0.4)
    fit = psu.fit_correlation(val)
    assert fit.details.mean == "constant"
    blocks = psu.blocks.tiles(sd, 6)
    res = psu.aggregate_continuous(sd, fit, blocks, prediction=pred, n_mc=4000, seed=1)
    zero = psu.aggregate_continuous(
        sd, fit, blocks, prediction=pred, mean="zero", n_mc=4000, seed=1
    )
    t, t0 = res.table, zero.table
    est = res.mean_estimate
    assert est is not None
    assert zero.mean_estimate is None
    assert res.mean == "constant"
    # constant sigma: every block is shifted by 2 * beta
    np.testing.assert_allclose(t["shift"].to_numpy(), 2.0 * est.beta)
    np.testing.assert_allclose(t["mean_pred"].to_numpy(), 10.0 + 2.0 * est.beta)
    np.testing.assert_allclose(t["map_mean"].to_numpy(), t0["mean_pred"].to_numpy())
    # same Monte Carlo pairs: the known-mean part equals the zero-mean variance
    np.testing.assert_allclose(t["var_known_mean"].to_numpy(), t0["var_mean"].to_numpy())
    assert np.all(t0["shift"].to_numpy() == 0.0)
    # explicit validation table gives the same result as the points stored on the fit
    again = psu.aggregate_continuous(
        sd, fit.model, blocks, prediction=pred, validation=val, n_mc=4000, seed=1
    )
    assert again.table.equals(t)
    # without the cross term the variance is larger
    nc = psu.aggregate_continuous(
        sd, fit, blocks, prediction=pred, cross_term=False, n_mc=4000, seed=1
    )
    assert np.all(nc.table["var_mean"].to_numpy() > t["var_mean"].to_numpy())


def test_an_offset_of_the_errors_only_moves_the_estimate() -> None:
    sd = grid_map()
    model = psu.vgm(0.7, "Exp", 3.0, nugget=0.3)
    blocks = psu.blocks.tiles(sd, 6)
    args = {"prediction": sd * 0, "n_mc": 2000, "seed": 2}
    a = psu.aggregate_continuous(sd, model, blocks, validation=validation(), **args).table
    b = psu.aggregate_continuous(sd, model, blocks, validation=validation(offset=1.5), **args).table
    np.testing.assert_allclose(b["shift"].to_numpy() - a["shift"].to_numpy(), 1.5)
    np.testing.assert_allclose(b["var_mean"].to_numpy(), a["var_mean"].to_numpy())


def test_corrected_variance_matches_simulation() -> None:
    """Exact check: with the true correlogram and an unknown mean, the corrected block
    estimate's error variance is V_B + x_B²A − 2·x_B·c_B."""
    n, n_pts = 10, 25
    sd = grid_map(n)
    model = psu.vgm(1.0, "Exp", 2.5)
    val = validation(n_pts, n, seed=3)
    px, py = val["x"].to_numpy(), val["y"].to_numpy()
    blocks = psu.blocks.tiles(sd, 5)
    res = psu.aggregate_continuous(
        sd, model, blocks, prediction=sd * 0, validation=val, n_mc=400_000, seed=4
    )
    est = res.mean_estimate
    assert est is not None
    grid = blocks.grid
    cx, cy = grid.cell_centres(np.arange(grid.size))
    allx, ally = np.concatenate([cx, px]), np.concatenate([cy, py])
    C = model.correlation(np.hypot(allx[:, None] - allx[None, :], ally[:, None] - ally[None, :]))
    L = np.linalg.cholesky(C + 1e-10 * np.eye(C.shape[0]))
    rng = np.random.default_rng(5)
    sims = 4000
    field = (L @ rng.normal(size=(C.shape[0], sims))) + 3.0  # unknown mean 3
    cell_err, pt_z = field[: grid.size], field[grid.size :]
    beta = est.weights @ pt_z
    resolved = psu.resolve_blocks(blocks, np.ones(grid.size, bool))
    for b, blk in enumerate(resolved):
        err = cell_err[blk.cells].mean(axis=0) - beta  # sigma = 1: x_B = 1
        assert err.mean() == pytest.approx(0.0, abs=0.05)
        assert err.var() == pytest.approx(res.table["var_mean"][b], rel=0.12)


def test_errors_and_warnings() -> None:
    sd = grid_map()
    blocks = psu.blocks.domain(sd)
    with pytest.raises(ValueError, match="mean='zero'"):
        psu.aggregate_continuous(sd, psu.vgm(1.0, "Exp", 3.0), blocks, n_mc=10)
    with pytest.raises(ValueError, match="variogram model"):
        psu.aggregate_continuous(sd, lambda h: np.exp(-h), blocks, validation=validation(), n_mc=10)
    with pytest.raises(ValueError, match="mean must be"):
        psu.aggregate_continuous(sd, psu.vgm(1.0, "Exp", 3.0), blocks, mean="ok", n_mc=10)  # type: ignore[arg-type]
    ml = psu.fit_correlation(validation(), mean="zero")
    with pytest.warns(UserWarning, match="fitted with the mean fixed at zero"):
        psu.aggregate_continuous(sd, ml, blocks, n_mc=100, seed=1)


def test_aggregate_categorical_applies_the_correction() -> None:
    rng = np.random.default_rng(6)
    n = 16
    x = np.arange(n) + 0.5
    raw = rng.dirichlet([3, 2, 1], size=(n, n))
    probs = xr.DataArray(
        np.moveaxis(raw, -1, 0),
        coords={"class": ["a", "b", "c"], "y": x[::-1], "x": x},
        dims=("class", "y", "x"),
    )
    m = 120
    pts = rng.dirichlet([3, 2, 1], size=m)
    labels = [["a", "b", "c"][rng.choice(3, p=[0.2, 0.3, 0.5])] for _ in range(m)]  # miscalibrated
    df = pl.DataFrame(
        {
            "x": rng.uniform(0, n, m),
            "y": rng.uniform(0, n, m),
            "label": labels,
            "pa": pts[:, 0],
            "pb": pts[:, 1],
            "pc": pts[:, 2],
        }
    )
    errors = psu.indicator_errors(df, {"a": "pa", "b": "pb", "c": "pc"})
    fits = psu.fit_class_variograms(errors)
    blocks = psu.blocks.tiles(probs, 8)
    res = psu.aggregate_categorical(probs, fits, blocks, n_mc=2000, seed=7)
    zero = psu.aggregate_categorical(probs, fits, blocks, mean="zero", n_mc=2000, seed=7)
    t = res.table
    assert set(res.mean_estimates) == {"a", "b", "c"}
    # the rare-in-the-map class c is under-predicted: its fractions move up
    c = t.filter(pl.col("class") == "c")
    assert res.mean_estimates["c"].beta > 0
    assert np.all(c["fraction"].to_numpy() > c["map_fraction"].to_numpy())
    np.testing.assert_allclose(
        t["fraction"].to_numpy(), t["map_fraction"].to_numpy() + t["shift"].to_numpy()
    )
    np.testing.assert_allclose(
        t["var_known_mean"].to_numpy(), zero.table["var_fraction"].to_numpy()
    )
    np.testing.assert_allclose(zero.table["fraction"].to_numpy(), t["map_fraction"].to_numpy())
    # the errors table can be passed explicitly instead
    models = {k: f.model for k, f in fits.items()}
    again = psu.aggregate_categorical(probs, models, blocks, validation=errors, n_mc=2000, seed=7)
    assert again.table.equals(t)
    with pytest.raises(ValueError, match="mean='zero'"):
        psu.aggregate_categorical(probs, models, blocks, n_mc=10)
