"""Re-run the R harness and compare with the committed reference outputs (needs R)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests.conftest import R_REFERENCE, read_reference

pytestmark = [pytest.mark.slow, pytest.mark.requires_r, pytest.mark.r_parity]

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is not installed")
def test_harness_reproduces_committed_outputs(tmp_path: Path) -> None:
    proc = subprocess.run(
        ["Rscript", str(REPO / "r-reference" / "run_reference.R"), str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if proc.returncode != 0 and "there is no package called" in proc.stderr:
        pytest.skip("required R packages are not installed")
    assert proc.returncode == 0, proc.stderr[-2000:]
    for name in (
        "summary.csv",
        "mc_pairs.csv",
        "sample_variogram.csv",
        "pt_all.csv",
        "mc_highm.csv",
    ):
        new = read_reference(name, tmp_path)
        old = read_reference(name, R_REFERENCE)
        assert new.columns == old.columns
        for col in new.columns:
            if new[col].dtype in (pl.Float64, pl.Int64):
                np.testing.assert_allclose(new[col].to_numpy(), old[col].to_numpy(), rtol=1e-10)
            else:
                assert new[col].to_list() == old[col].to_list()
