"""Shared fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

R_REFERENCE = Path(__file__).parent / "data" / "r_reference"


def read_reference(name: str, directory: Path = R_REFERENCE) -> pl.DataFrame:
    """Read one of the CSV files exported by ``r-reference/run_reference.R``."""
    return pl.read_csv(directory / name, infer_schema_length=None)


@pytest.fixture(scope="session")
def r_ref() -> Path:
    """Directory with the outputs exported by ``r-reference/run_reference.R``."""
    if not (R_REFERENCE / "summary.csv").exists():
        pytest.skip("R reference outputs are not available")
    return R_REFERENCE


@pytest.fixture(scope="session")
def read_ref(r_ref: Path) -> Callable[[str], pl.DataFrame]:
    return lambda name: read_reference(name, r_ref)


@pytest.fixture(scope="session")
def r_summary(read_ref: Callable[[str], pl.DataFrame]) -> dict[str, float]:
    df = read_ref("summary.csv")
    return dict(zip(df["key"].to_list(), df["value"].to_list(), strict=True))


@pytest.fixture(scope="session")
def r_maps(read_ref: Callable[[str], pl.DataFrame]) -> dict[str, xr.DataArray]:
    """The R example's maps as DataArrays (x = x1, y = x2, ascending)."""
    grid = read_ref("grid.csv")
    x = np.sort(grid["x1"].unique().to_numpy())
    y = np.sort(grid["x2"].unique().to_numpy())
    out = {}
    for col in ("y", "predRF", "predVarRF"):
        # expand.grid(x1, x2): x1 varies fastest, so rows are x2 and columns x1
        values = grid[col].to_numpy().reshape(y.size, x.size)
        out[col] = xr.DataArray(values, coords={"y": y, "x": x}, dims=("y", "x"))
    return out
