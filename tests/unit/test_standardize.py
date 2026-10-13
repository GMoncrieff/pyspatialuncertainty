from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import xarray as xr
from scipy import stats

import pyspatialuncertainty as psu


def test_sd_from_samples() -> None:
    rng = np.random.default_rng(0)
    s = rng.normal(0, 3, size=(500, 4, 5))
    da = xr.DataArray(s, dims=("sample", "y", "x"))
    np.testing.assert_allclose(psu.sd_from_samples(da).values, s.std(axis=0, ddof=1))
    np.testing.assert_allclose(psu.sd_from_samples(s, 0), s.std(axis=0, ddof=1))


def test_sd_from_quantiles_equal_spacing_matches_samples() -> None:
    rng = np.random.default_rng(1)
    q = np.sort(rng.normal(size=(3, 200)), axis=1)
    levels = (np.arange(200) + 0.5) / 200
    np.testing.assert_allclose(psu.sd_from_quantiles(q, levels, dim=1), q.std(axis=1, ddof=0))


def test_sd_from_quantiles_of_a_normal() -> None:
    levels = (np.arange(999) + 0.5) / 999
    q = stats.norm.ppf(levels, loc=5.0, scale=2.0)
    da = xr.DataArray(np.tile(q, (2, 1)), dims=("cell", "quantile"))
    out = psu.sd_from_quantiles(da, levels)
    np.testing.assert_allclose(out.values, 2.0, rtol=0.01)


def test_sd_from_quantiles_validates_levels() -> None:
    with pytest.raises(ValueError, match="increasing"):
        psu.sd_from_quantiles(np.zeros((2, 3)), [0.1, 0.05, 0.9], dim=1)


def test_standardized_errors() -> None:
    df = pl.DataFrame({"observed": [1.0, 2.0], "predicted": [0.5, 3.0], "sd": [0.5, 2.0]})
    out = psu.standardized_errors(df)
    assert out["std_error"].to_list() == [1.0, -0.5]
    with pytest.raises(ValueError, match="positive"):
        psu.standardized_errors(df.with_columns(pl.lit(0.0).alias("sd")))
