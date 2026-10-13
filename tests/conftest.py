"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

R_REFERENCE = Path(__file__).parent / "data" / "r_reference"


@pytest.fixture(scope="session")
def r_ref() -> Path:
    """Directory with the outputs exported by ``r-reference/run_reference.R``."""
    if not (R_REFERENCE / "summary.csv").exists():
        pytest.skip("R reference outputs are not available")
    return R_REFERENCE


@pytest.fixture(scope="session")
def r_summary(r_ref: Path) -> dict[str, float]:
    df = pl.read_csv(r_ref / "summary.csv")
    return dict(zip(df["key"].to_list(), df["value"].to_list(), strict=True))
