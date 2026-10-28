from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

import pyspatialuncertainty as psu
from pyspatialuncertainty.categorical import ProbabilityError


def dirichlet_map(k: int = 3, n: int = 30, seed: int = 0) -> xr.DataArray:
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(k), size=(n, n)).transpose(2, 0, 1)
    x = np.arange(n) + 0.5
    return xr.DataArray(
        p, coords={"class": ["a", "b", "c"][:k], "y": x[::-1], "x": x}, dims=("class", "y", "x")
    )


def test_validate_accepts_valid_and_skips_unmapped_cells() -> None:
    p = dirichlet_map()
    psu.validate_probabilities(p)
    p = p.copy()
    p[:, 0, 0] = np.nan
    psu.validate_probabilities(p)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.__setitem__((0, 1, 1), 0.0), "hard 0 or 1"),
        (lambda p: p.__setitem__((slice(None), 2, 2), [1.0, 0.0, 0.0]), "hard 0 or 1"),
        (lambda p: p.__setitem__((0, 3, 3), float(p[0, 3, 3]) + 0.01), "sum to one"),
        (lambda p: p.__setitem__((1, 4, 4), np.nan), "only some classes"),
    ],
)
def test_validate_rejects(mutate: object, match: str) -> None:
    p = dirichlet_map().values.copy()
    mutate(p)  # type: ignore[operator]
    da = dirichlet_map().copy(data=p)
    with pytest.raises(ProbabilityError, match=match):
        psu.validate_probabilities(da)


def test_validate_requires_class_dim_and_two_classes() -> None:
    with pytest.raises(ProbabilityError, match="no 'class' dimension"):
        psu.validate_probabilities(dirichlet_map().rename({"class": "k"}))
    with pytest.raises(ProbabilityError, match="two classes"):
        psu.validate_probabilities(np.full((5, 1), 1.0))


@settings(max_examples=50, deadline=None)
@given(
    arrays(np.float64, (20, 4), elements=st.floats(0.01, 10.0)),
    st.floats(0.2, 5.0),
)
def test_temperature_scaling_properties(raw: np.ndarray, temperature: float) -> None:
    p = raw / raw.sum(axis=1, keepdims=True)
    q = psu.temperature_scale(p, temperature, class_dim=1)
    np.testing.assert_allclose(q.sum(axis=1), 1.0, rtol=1e-12)
    assert np.all((q > 0) & (q < 1))
    # order of classes within a location is preserved (ties may swap by rounding)
    order = np.argsort(p, axis=1, kind="stable")
    assert np.all(np.diff(np.take_along_axis(q, order, axis=1), axis=1) >= -1e-12)
    # sharper for T < 1, flatter for T > 1
    if temperature < 1:
        assert np.all(q.max(axis=1) >= p.max(axis=1) - 1e-12)
    else:
        assert np.all(q.max(axis=1) <= p.max(axis=1) + 1e-12)
    np.testing.assert_allclose(psu.temperature_scale(p, 1.0, class_dim=1), p, rtol=1e-12)


def test_temperature_scale_dataarray() -> None:
    p = dirichlet_map()
    q = psu.temperature_scale(p, 0.5)
    np.testing.assert_allclose(q.sum("class").values, 1.0)
    psu.validate_probabilities(q)


@settings(max_examples=30, deadline=None)
@given(st.floats(1e-9, 0.1))
def test_clip_probabilities(eps: float) -> None:
    p = dirichlet_map().copy()
    p[:, 0, 0] = [1.0, 0.0, 0.0]
    c = psu.clip_probabilities(p, eps)
    psu.validate_probabilities(c)
    assert float(c.min()) > 0


def test_indicator_errors() -> None:
    df = pl.DataFrame(
        {"x": [0.0, 1.0], "y": [0.0, 0.0], "label": ["a", "b"], "pa": [0.8, 0.3], "pb": [0.2, 0.7]}
    )
    e = psu.indicator_errors(df, {"a": "pa", "b": "pb"})
    assert e.height == 4
    a = e.filter(pl.col("class") == "a")
    np.testing.assert_allclose(a["error"].to_numpy(), [0.2, -0.3])
    np.testing.assert_allclose(a["sd"].to_numpy(), np.sqrt([0.16, 0.21]))
    np.testing.assert_allclose(a["std_error"].to_numpy(), [0.2 / 0.4, -0.3 / np.sqrt(0.21)])
    with pytest.raises(ProbabilityError, match="not among the classes"):
        psu.indicator_errors(df.with_columns(pl.lit("z").alias("label")), {"a": "pa", "b": "pb"})
    with pytest.raises(ProbabilityError, match="hard 0 or 1"):
        psu.indicator_errors(
            df.with_columns(pl.lit(1.0).alias("pa"), pl.lit(0.0).alias("pb")),
            {"a": "pa", "b": "pb"},
        )


def test_fit_class_variograms_auto_and_given_model() -> None:
    rng = np.random.default_rng(4)
    n = 300
    p = rng.dirichlet([2, 2, 2], size=n)
    labels = [["a", "b", "c"][rng.choice(3, p=pi)] for pi in p]
    df = pl.DataFrame(
        {
            "x": rng.uniform(0, 100, n),
            "y": rng.uniform(0, 100, n),
            "label": labels,
            "pa": p[:, 0],
            "pb": p[:, 1],
            "pc": p[:, 2],
        }
    )
    e = psu.indicator_errors(df, {"a": "pa", "b": "pb", "c": "pc"})
    fits = psu.fit_class_variograms(e)
    assert set(fits) == {"a", "b", "c"}
    for f in fits.values():
        # default: REML likelihood fit, keeping the validation points
        assert isinstance(f.fit, psu.LikelihoodFit)
        assert f.fit.mean == "constant"
        assert f.x is not None
        assert f.z is not None
        assert f.z.size == n
        # independent errors with calibrated probabilities: sill near 1, no correlation
        assert 0.5 < f.model.sill < 1.6
        assert np.all(f.model.correlation(np.array([5.0, 20.0])) < 0.15)
    wls = psu.fit_class_variograms(e, method="wls")
    for cls, f in wls.items():
        assert isinstance(f.fit, psu.VariogramFit)
        assert f.model == psu.fit_variogram_auto(f.empirical).model
        np.testing.assert_array_equal(f.empirical.gamma, fits[cls].empirical.gamma)
    # a starting model implies the WLS fit
    fixed = psu.fit_class_variograms(e, model=psu.vgm(1.0, "Exp", 10.0, nugget=0.1))
    assert all(isinstance(f.fit, psu.VariogramFit) for f in fixed.values())
    assert all(len(f.model) == 2 for f in fixed.values())
    with pytest.raises(ValueError, match="method='wls'"):
        psu.fit_class_variograms(
            e, method="likelihood", model=psu.vgm(1.0, "Exp", 10.0, nugget=0.1)
        )


def test_aggregate_categorical_basic_properties() -> None:
    p = dirichlet_map(seed=1)
    corr = psu.vgm(1.0, "Exp", 3.0)
    res = psu.aggregate_categorical(
        p, corr, psu.blocks.tiles(p, 15), mean="zero", n_mc=4000, seed=2
    )
    t = res.table
    assert t.height == 3 * 4
    assert res.classes == ("a", "b", "c")
    # fractions are block means of the probabilities and sum to one over classes
    sums = (
        t.group_by("block_id").agg(pl.col("fraction").sum()).sort("block_id")["fraction"].to_numpy()
    )
    np.testing.assert_allclose(sums, 1.0)
    first = p.isel(y=slice(0, 15), x=slice(0, 15)).mean(("y", "x")).values
    np.testing.assert_allclose(t.filter(pl.col("block_id") == 0)["fraction"].to_numpy(), first)
    np.testing.assert_allclose(t["expected_area"].to_numpy(), 225 * t["fraction"].to_numpy())
    np.testing.assert_allclose(t["sd_area"].to_numpy(), 225 * t["sd_fraction"].to_numpy())
    assert (t["lower"] >= 0).all()
    assert (t["upper"] <= 1).all()
    ds = res.to_xarray(coarse=True)
    assert ds["fraction"].dims == ("class", "y", "x")
    assert ds["fraction"].shape == (3, 2, 2)


def test_binary_symmetry_with_shared_pairs() -> None:
    p = dirichlet_map(k=2, seed=3)
    res = psu.aggregate_categorical(
        p, psu.vgm(1.0, "Sph", 8.0), psu.blocks.tiles(p, 10), mean="zero", n_mc=2000, seed=5
    )
    a = res.table.filter(pl.col("class") == "a")["var_fraction"].to_numpy()
    b = res.table.filter(pl.col("class") == "b")["var_fraction"].to_numpy()
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_aggregate_categorical_requires_every_class() -> None:
    p = dirichlet_map()
    with pytest.raises(ValueError, match="no correlation function"):
        psu.aggregate_categorical(p, {"a": psu.vgm(1.0, "Exp", 3.0)}, psu.blocks.domain(p), n_mc=10)
    bad = p.copy()
    bad[0, 0, 0] = 0.0
    with pytest.raises(ProbabilityError):
        psu.aggregate_categorical(bad, psu.vgm(1.0, "Exp", 3.0), psu.blocks.domain(p), n_mc=10)
