from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from scipy import stats

import pyspatialuncertainty as psu
from pyspatialuncertainty.synthetic import (
    CirculantEmbedding,
    ErrorModelScenario,
    LatentClassScenario,
    PipelineScenario,
    SubGridLayout,
    bvn_cdf,
    bvn_rectangle,
    cholesky_sample,
    evaluation_blocks,
    exact_block_variance,
    indicator_block_variance,
)


def exp_cov(a: float, sill: float = 1.0):  # type: ignore[no-untyped-def]
    return lambda h: sill * np.exp(-np.asarray(h) / a)


def test_circulant_embedding_reproduces_covariance() -> None:
    emb = CirculantEmbedding((24, 30), 1.0, exp_cov(5.0, 2.0))
    fields = emb.sample(np.random.default_rng(0), 4000)
    assert emb.variance == pytest.approx(2.0)
    v = fields.reshape(4000, -1)
    np.testing.assert_allclose(v.var(axis=0).mean(), 2.0, rtol=0.02)
    for lag in (1, 3, 7):
        c = np.mean(fields[:, :, :-lag] * fields[:, :, lag:])
        assert c == pytest.approx(2.0 * math.exp(-lag / 5.0), rel=0.05)
        c = np.mean(fields[:, :-lag, :] * fields[:, lag:, :])
        assert c == pytest.approx(2.0 * math.exp(-lag / 5.0), rel=0.05)


def test_circulant_embedding_pads_for_long_ranges() -> None:
    emb = CirculantEmbedding((16, 16), 1.0, lambda h: np.exp(-((np.asarray(h) / 8.0) ** 2)))
    assert emb.embedding_shape[0] >= 32


def test_cholesky_sampler_matches_covariance() -> None:
    xy = np.column_stack([np.arange(6.0), np.zeros(6)])
    f = cholesky_sample(xy, exp_cov(2.0), np.random.default_rng(1), 20_000)
    np.testing.assert_allclose(
        np.cov(f.T), np.exp(-np.abs(np.subtract.outer(xy[:, 0], xy[:, 0])) / 2.0), atol=0.03
    )


def test_exact_block_variance_matches_direct_sum() -> None:
    rng = np.random.default_rng(2)
    w = rng.uniform(0.5, 2, (12, 15))
    mask = np.zeros((12, 15), dtype=bool)
    mask[2:9, 4:13] = True
    mask[5, 5] = False
    cov = exp_cov(3.0)
    rr, cc = np.nonzero(mask)
    d = np.hypot(np.subtract.outer(rr, rr), np.subtract.outer(cc, cc)) * 0.5
    direct = float(w[rr, cc] @ cov(d) @ w[rr, cc]) / rr.size**2
    assert exact_block_variance(w, mask, 0.5, cov) == pytest.approx(direct, rel=1e-10)


def test_bvn_against_scipy() -> None:
    rng = np.random.default_rng(3)
    h, k, r = rng.normal(size=50), rng.normal(size=50), rng.uniform(-0.9, 0.95, 50)
    ref = [
        stats.multivariate_normal([0, 0], [[1, ri], [ri, 1]]).cdf([hi, ki])
        for hi, ki, ri in zip(h, k, r, strict=True)
    ]
    np.testing.assert_allclose(bvn_cdf(h, k, r), ref, atol=1e-7)
    assert bvn_cdf(0.0, 0.0, 0.5)[()] == pytest.approx(1 / 4 + math.asin(0.5) / (2 * math.pi))
    assert bvn_rectangle(-np.inf, np.inf, -np.inf, np.inf, 0.3)[()] == pytest.approx(1.0)


def test_indicator_block_variance_against_simulation() -> None:
    rng = np.random.default_rng(4)
    coords = np.column_stack([np.repeat(np.arange(4.0), 4), np.tile(np.arange(4.0), 4)])
    lower = np.full(16, -0.3)
    upper = np.full(16, 1.0)
    corr = lambda d: 0.9 * np.exp(-np.asarray(d) / 2.0) + 0.1 * (np.asarray(d) == 0)  # noqa: E731
    exact = indicator_block_variance(lower, upper, coords, corr)
    y = cholesky_sample(coords, corr, rng, 50_000)
    frac = ((y > lower) & (y <= upper)).mean(axis=1)
    assert frac.var() == pytest.approx(exact, rel=0.03)


@pytest.fixture(scope="module")
def small_error_scenario() -> ErrorModelScenario:
    return ErrorModelScenario(
        layout=SubGridLayout(n=24, f=3), correlation=psu.vgm(0.7, "Exp", 3.0, nugget=0.3)
    )


def test_error_scenario_truth_and_oracle(small_error_scenario: ErrorModelScenario) -> None:
    sc = small_error_scenario
    blocks = evaluation_blocks(sc.layout, tile_sizes=(2, 6, 24), max_per_size=3, n_polygons=2)
    exact = sc.exact_block_sd(blocks)
    rng = np.random.default_rng(5)
    reals = [sc.simulate(rng) for _ in range(1500)]
    errs = np.array([blocks.block_means(r.error_sub) for r in reals])
    # brute-force SD agrees with the exact discretised value
    np.testing.assert_allclose(errs.std(axis=0), exact, rtol=0.08)
    # the method with the true sigma and correlation (oracle) agrees with the truth
    sd_map = sc.sigma_map()
    res = psu.aggregate_continuous(
        sd_map, sc.true_correlation(), blocks.blockset, n_mc=40_000, seed=6
    )
    np.testing.assert_allclose(res.table["sd_mean"].to_numpy(), exact, rtol=0.03)
    # validation errors: standardized variance near one, nugget share visible at short lags
    std = np.concatenate(
        [
            (r.validation["error"] / sc.sigma_true.ravel()[r.validation["cell"]]).to_numpy()
            for r in reals[:50]
        ]
    )
    assert std.var() == pytest.approx(1.0, rel=0.05)


def test_sigma_map_variants(small_error_scenario: ErrorModelScenario) -> None:
    sc = small_error_scenario
    assert float(sc.sigma_map(2.0).mean()) == pytest.approx(2 * sc.sigma_true.mean())
    const = sc.sigma_map(1.0, "constant")
    assert float(const.std()) == pytest.approx(0.0)


def test_latent_scenario_calibration_and_prevalence() -> None:
    sc = LatentClassScenario(layout=SubGridLayout(n=32, f=2), snr=1.0, n_validation=2000)
    p = sc.probabilities()
    psu.validate_probabilities(p)
    np.testing.assert_allclose(p.mean(("y", "x")).values, sc.prevalence, atol=1e-6)
    rng = np.random.default_rng(7)
    labels = []
    probs = []
    for _ in range(20):
        r = sc.simulate(rng)
        labels.append(r.validation["label"].to_numpy())
        probs.append(sc.p_true.reshape(3, -1)[:, r.validation["cell"].to_numpy()].T)
    lab = np.concatenate(labels)
    pr = np.concatenate(probs)
    # calibration: observed class frequencies match mean predicted probabilities in bins
    for k, cls in enumerate(sc.classes):
        ind = (lab == cls).astype(float)
        bins = np.quantile(pr[:, k], [0, 0.25, 0.5, 0.75, 1.0])
        for lo, hi in itertools.pairwise(bins):
            sel = (pr[:, k] >= lo) & (pr[:, k] <= hi)
            assert ind[sel].mean() == pytest.approx(pr[sel, k].mean(), abs=0.03)
    q = sc.probabilities(temperature=0.5)
    assert float(q.max()) > float(p.max())


def test_latent_scenario_exact_vs_bruteforce() -> None:
    sc = LatentClassScenario(
        layout=SubGridLayout(n=16, f=2), latent=psu.vgm(0.75, "Exp", 2.0, nugget=0.25)
    )
    blocks = evaluation_blocks(sc.layout, tile_sizes=(2, 4), max_per_size=2, n_polygons=0)
    rng = np.random.default_rng(8)
    fr = np.array(
        [
            [blocks.block_means(r.fraction_sub[k]) for k in range(3)]
            for r in (sc.simulate(rng) for _ in range(3000))
        ]
    )
    for b, idx in enumerate(blocks.sub_index):
        for k in range(3):
            assert fr[:, k, b].std() == pytest.approx(sc.exact_block_sd(k, idx), rel=0.08)


def test_oracle_correlogram_is_a_valid_correlation() -> None:
    sc = LatentClassScenario(layout=SubGridLayout(n=16, f=2))
    corr = sc.oracle_correlation(2, n=20)
    psu.as_correlation(corr)  # passes the checks (rho(0) = 1, |rho| <= 1)
    vals = corr(np.array([0.0, 0.5, 2.0, 50.0]))
    assert vals[0] == 1.0
    assert 0 < vals[1] < 1


@pytest.mark.slow
def test_pipeline_scenario_smoke() -> None:
    sc = PipelineScenario(n_trees=50, n_train=120)
    r = sc.simulate(np.random.default_rng(9))
    assert r.prediction.shape == (100, 100)
    assert r.validation.height == 120
    assert float(r.sd.min()) > 0
    assert PipelineScenario.with_trend_r2(0.5).trend_r2 == pytest.approx(0.5)
