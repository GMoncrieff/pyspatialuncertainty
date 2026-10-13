from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from pyspatialuncertainty.variogram import (
    EmpiricalVariogram,
    empirical_variogram,
    fit_variogram,
    fit_variogram_auto,
    vgm,
)


def test_hand_computed_matheron() -> None:
    # three points on a line at 0, 1, 3 with values 0, 1, 3
    ev = empirical_variogram(
        None, [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 0.0, 0.0], cutoff=4, width=1
    )
    # pairs: d=1 (dz=1) -> class 0 (exact multiple goes to the lower class); d=2 (dz=2) -> class 1;
    # d=3 (dz=3) -> class 2
    assert ev.n_pairs.tolist() == [1, 1, 1]
    np.testing.assert_allclose(ev.dist, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(ev.gamma, [0.5, 2.0, 4.5])


def test_zero_distance_pairs_go_to_first_class() -> None:
    # x = 0, 0, 0.5: one pair at distance 0 and two at 0.5, all in the first class
    ev = empirical_variogram(
        None, [0.0, 2.0, 5.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0], cutoff=1, width=1
    )
    assert ev.n_pairs.tolist() == [3]
    np.testing.assert_allclose(ev.gamma, [(2**2 + 5**2 + 3**2) / 6])


def test_cressie_hand_computed() -> None:
    ev = empirical_variogram(
        None, [0.0, 4.0], [0.0, 1.0], [0.0, 0.0], cutoff=2, width=2, estimator="cressie"
    )
    expected = 0.5 * (math.sqrt(4.0)) ** 4 / (0.457 + 0.494)
    np.testing.assert_allclose(ev.gamma, [expected])


def test_explicit_boundaries() -> None:
    ev = empirical_variogram(
        None, [0.0, 1.0, 3.0], [0.0, 1.0, 3.0], [0.0, 0.0, 0.0], boundaries=[0.0, 1.5, 4.0]
    )
    assert ev.n_pairs.tolist() == [1, 2]
    assert ev.width is None


def test_chunking_invariance() -> None:
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 50, size=(400, 2))
    z = rng.normal(size=400)
    a = empirical_variogram(None, z, xy[:, 0], xy[:, 1])
    b = empirical_variogram(None, z, xy[:, 0], xy[:, 1], max_chunk_elements=1000)
    np.testing.assert_array_equal(a.n_pairs, b.n_pairs)
    np.testing.assert_allclose(a.gamma, b.gamma, rtol=1e-12)
    np.testing.assert_allclose(a.dist, b.dist, rtol=1e-12)


def test_polars_columns_and_nan_dropping() -> None:
    df = pl.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0] * 4, "e": [0.0, 1.0, float("nan"), 2.0]}
    )
    ev = empirical_variogram(df, "e", "x", "y", cutoff=10, width=10)
    assert ev.n_obs == 3
    out = ev.to_polars()
    assert out.columns == ["np", "dist", "gamma"]
    assert out["np"].to_list() == [3]


def _exact_variogram(model: object, dist: np.ndarray, npairs: int = 100) -> EmpiricalVariogram:
    g = model.gamma(dist)  # type: ignore[attr-defined]
    return EmpiricalVariogram(
        n_pairs=np.full(dist.size, npairs, dtype=np.int64),
        dist=dist,
        gamma=g,
        boundaries=np.concatenate([[0.0], dist + 0.5]),
        cutoff=float(dist[-1] + 0.5),
        width=1.0,
        estimator="matheron",
        n_obs=1000,
    )


@pytest.mark.parametrize("fit_method", [1, 2, 6, 7])
def test_fit_recovers_noise_free_parameters(fit_method: int) -> None:
    truth = vgm(0.8, "Exp", 6.0, nugget=0.2)
    ev = _exact_variogram(truth, np.linspace(0.5, 40, 30))
    fit = fit_variogram(ev, vgm(0.5, "Exp", 15.0, nugget=0.05), fit_method=fit_method)  # type: ignore[arg-type]
    assert fit.converged
    got = {s.model: s for s in fit.model.structures}
    assert got["Nug"].psill == pytest.approx(0.2, rel=1e-5)
    assert got["Exp"].psill == pytest.approx(0.8, rel=1e-5)
    assert got["Exp"].range == pytest.approx(6.0, rel=1e-5)
    assert fit.sserr == pytest.approx(0.0, abs=1e-12)


def test_fixed_parameters_are_kept() -> None:
    truth = vgm(0.8, "Sph", 12.0, nugget=0.2)
    ev = _exact_variogram(truth, np.linspace(0.5, 40, 30))
    start = vgm(0.5, "Sph", 20.0, nugget=0.1)
    fit = fit_variogram(ev, start, fit_ranges=False)
    assert fit.model.structures[1].range == 20.0
    nothing = fit_variogram(ev, start, fit_sills=False, fit_ranges=False)
    assert nothing.model == start
    assert nothing.message == "nothing to fit"
    per_structure = fit_variogram(ev, start, fit_sills=[False, True])
    assert per_structure.model.structures[0].psill == 0.1


def test_flags_length_checked() -> None:
    ev = _exact_variogram(vgm(1.0, "Exp", 3.0), np.linspace(1, 10, 10))
    with pytest.raises(ValueError, match="one entry per structure"):
        fit_variogram(ev, vgm(1.0, "Exp", 3.0, nugget=0.1), fit_sills=[True])


@pytest.mark.parametrize("family", ["Exp", "Sph", "Gau"])
def test_auto_fit_selects_generating_family(family: str) -> None:
    truth = vgm(1.0, family, 10.0, nugget=0.1)  # type: ignore[arg-type]
    ev = _exact_variogram(truth, np.linspace(0.5, 45, 40))
    fit = fit_variogram_auto(ev)
    assert fit.model.structures[-1].model == family
    np.testing.assert_allclose(fit.model.gamma(ev.dist), ev.gamma, atol=1e-5)


def test_auto_fit_is_bounded_for_unbounded_variograms() -> None:
    dist = np.linspace(1, 40, 20)
    ev = EmpiricalVariogram(
        n_pairs=np.full(20, 100, dtype=np.int64),
        dist=dist,
        gamma=0.5 + 0.02 * dist,
        boundaries=np.linspace(0, 41, 21),
        cutoff=41.0,
        width=2.0,
        estimator="matheron",
        n_obs=300,
    )
    fit = fit_variogram_auto(ev)
    assert fit.model.sill <= 2 * ev.gamma.max() * 2 + 1e-9
    main = fit.model.structures[-1]
    practical = {"Exp": 3.0, "Gau": math.sqrt(3.0), "Sph": 1.0}[main.model]
    assert main.range <= 2 * 40 / practical * (1 + 1e-9)
    unbounded = fit_variogram_auto(ev, bounded=False)
    assert unbounded.sserr <= fit.sserr + 1e-12
