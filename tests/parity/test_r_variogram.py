"""Parity of the variogram code with gstat, using outputs of r-reference/run_reference.R."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pyspatialuncertainty.variogram import VariogramModel, empirical_variogram, fit_variogram, vgm

pytestmark = pytest.mark.r_parity


@pytest.fixture(scope="module")
def pt_all(r_ref: Path) -> pl.DataFrame:
    return pl.read_csv(r_ref / "pt_all.csv")


@pytest.mark.parametrize(
    ("fname", "kwargs"),
    [
        ("sample_variogram.csv", {}),
        ("sample_variogram_cressie.csv", {"estimator": "cressie"}),
        ("sample_variogram_cutoff30_width2.csv", {"cutoff": 30.0, "width": 2.0}),
    ],
)
def test_sample_variogram_matches_gstat(
    r_ref: Path, pt_all: pl.DataFrame, fname: str, kwargs: dict[str, object]
) -> None:
    ev = empirical_variogram(pt_all, "stdErrRF", "x1", "x2", **kwargs)  # type: ignore[arg-type]
    ref = pl.read_csv(r_ref / fname)
    np.testing.assert_array_equal(ev.n_pairs, ref["np"].to_numpy())
    np.testing.assert_allclose(ev.dist, ref["dist"].to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(ev.gamma, ref["gamma"].to_numpy(), rtol=1e-10)


def test_default_boundaries_match_gstat(r_ref: Path, pt_all: pl.DataFrame) -> None:
    ev = empirical_variogram(pt_all, "stdErrRF", "x1", "x2")
    ref = pl.read_csv(r_ref / "sample_variogram_boundaries.csv")["boundary"].to_numpy()
    np.testing.assert_allclose(ev.boundaries, ref, rtol=1e-12, atol=1e-12)


def test_script_variogram_model_line(r_ref: Path) -> None:
    model = VariogramModel.from_table(pl.read_csv(r_ref / "variogram_model.csv"))
    assert model.sill == pytest.approx(0.65)
    line = pl.read_csv(r_ref / "variogram_line.csv")
    d = line["dist"].to_numpy()
    np.testing.assert_allclose(model.gamma(d), line["gamma"].to_numpy(), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(model.covariance(d), line["covariance"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(model.correlation(d), line["correlation"].to_numpy(), rtol=1e-12)


def test_script_correlogram(r_ref: Path) -> None:
    model = VariogramModel.from_table(pl.read_csv(r_ref / "variogram_model.csv"))
    cor = pl.read_csv(r_ref / "correlogram_script.csv")
    np.testing.assert_allclose(
        model.correlation(cor["dist"].to_numpy()), cor["correlation"].to_numpy(), rtol=1e-12
    )


@pytest.mark.parametrize(
    ("case", "model"),
    [
        ("Exp", vgm(1, "Exp", 10)),
        ("Sph", vgm(1, "Sph", 10)),
        ("Gau", vgm(1, "Gau", 10)),
        ("Mat", vgm(1, "Mat", 10, kappa=1.5)),
        ("Ste", vgm(1, "Ste", 10, kappa=1.5)),
        ("Ste05", vgm(1, "Ste", 10, kappa=0.5)),
        ("Nug", vgm(1, "Nug")),
    ],
)
def test_single_structure_parameterisation(r_ref: Path, case: str, model: VariogramModel) -> None:
    ref = pl.read_csv(r_ref / "variogram_line_single_models.csv").filter(pl.col("case") == case)
    np.testing.assert_allclose(
        model.gamma(ref["dist"].to_numpy()), ref["gamma"].to_numpy(), rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize(
    ("case", "start", "method"),
    [
        ("exp_m7", vgm(0.5, "Exp", 10, nugget=0.1), 7),
        ("exp_m6", vgm(0.5, "Exp", 10, nugget=0.1), 6),
        ("exp_m1", vgm(0.5, "Exp", 10, nugget=0.1), 1),
        ("sph_m7", vgm(0.5, "Sph", 20, nugget=0.1), 7),
        ("ste_m7", vgm(0.7, "Ste", 25, nugget=0.1, kappa=0.5), 7),
    ],
)
def test_fit_matches_gstat(
    r_ref: Path, pt_all: pl.DataFrame, case: str, start: VariogramModel, method: int
) -> None:
    ev = empirical_variogram(pt_all, "stdErrRF", "x1", "x2")
    fit = fit_variogram(ev, start, fit_method=method)  # type: ignore[arg-type]
    ref = pl.read_csv(r_ref / "gstat_fits.csv").filter(pl.col("case") == case)
    # Same optimum: our weighted SSE is not worse than gstat's, and parameters agree closely.
    assert fit.sserr <= ref["sserr"][0] * (1 + 1e-6)
    np.testing.assert_allclose(
        [s.psill for s in fit.model.structures], ref["psill"].to_numpy(), rtol=2e-3, atol=1e-6
    )
    ranges = [s.range for s in fit.model.structures if s.model != "Nug"]
    ref_ranges = ref.filter(pl.col("model") != "Nug")["range"].to_numpy()
    np.testing.assert_allclose(ranges, ref_ranges, rtol=2e-3)
