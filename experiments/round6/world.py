"""The round-6 test bed: class and continuous landscapes of any side length.

Round 5 (``experiments.round5.shared``) fixed the landscape at 1024 × 1024 cells. Here the
side n is a parameter. At n = 1024 every draw (blocks, truth, samples, labels, calibration
samples, correlogram subsets) reproduces round 5 and the Landscape Check exactly; other
sides use the same seeds with the side appended.

Blocks are tiles of 4, 64 and 256 cells per side (those smaller than the landscape) and the
landscape itself, with at most 256 tiles per side chosen at random (round 4's rule). The
latent field, its ranges, the signal and the class prevalences do not depend on n.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from experiments.round5 import shared as sh
from pyspatialuncertainty.synthetic import ErrorModelScenario, LatentClassScenario, SubGridLayout
from pyspatialuncertainty.synthetic.categorical import interval_probability
from pyspatialuncertainty.synthetic.grf import CirculantEmbedding

TILE_SIDES = (4, 64, 256)
K = len(sh.CLASSES)


def seed(n: int, *key: int) -> list[int]:
    """Round 5's seed sequence at n = 1024, with the side appended for other sides."""
    return [sh.SEED, *key] if n == 1024 else [sh.SEED, *key, n]


def block_sides(n: int) -> tuple[int, ...]:
    return (*(s for s in TILE_SIDES if s < n), n)


def make_layout(n: int) -> sh.Layout:
    rng = np.random.default_rng(sh.SEED if n == 1024 else [sh.SEED, 5, n])
    tiles: dict[int, np.ndarray] = {}
    for side in block_sides(n):
        n_tiles = (n // side) ** 2
        tiles[side] = (
            np.arange(n_tiles)
            if n_tiles <= sh.N_SMALL
            else np.sort(rng.choice(n_tiles, sh.N_SMALL, replace=False))
        )
    return sh.make_layout(n, block_sides(n), tiles)


def sample_cells(
    n: int, design: str, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``shared.sample_cells`` on an n × n landscape (identical draws at n = 1024)."""
    if design == "random":
        return rng.choice(n * n, k, replace=False), np.arange(k)
    taken = np.zeros(n * n, dtype=bool)
    out, ids, got, r, c = [], [], 0, math.ceil(sh.CLUSTER_RADIUS), 0
    while got < k:
        cy, cx = rng.uniform(0, n, 2)
        rows = np.arange(max(0, int(cy) - r), min(n, int(cy) + r + 1))
        cols = np.arange(max(0, int(cx) - r), min(n, int(cx) + r + 1))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        cells = (rr * n + cc)[np.hypot(rr + 0.5 - cy, cc + 0.5 - cx) <= sh.CLUSTER_RADIUS]
        cells = cells[~taken[cells]]
        pick = rng.choice(cells, min(sh.CLUSTER_SIZE, cells.size, k - got), replace=False)
        taken[pick] = True
        out.append(pick)
        ids.append(np.full(pick.size, c))
        got += pick.size
        c += 1
    return np.concatenate(out), np.concatenate(ids)


# ---------------------------------------------------------------------------
# Class maps
# ---------------------------------------------------------------------------


@dataclass
class ClassWorld:
    n: int
    sc: LatentClassScenario
    layout: sh.Layout
    p_true: np.ndarray  # (K, n²) calibrated map, clipped and renormalised
    p_raw: np.ndarray  # (K, n²) unclipped true probabilities


def class_world(n: int) -> ClassWorld:
    sc = LatentClassScenario(
        layout=SubGridLayout(n=n, f=1),
        latent=sh.CORRELOGRAM,
        signal_range=sh.SIGNAL_RANGE,
        prevalence=sh.PREVALENCE,
        classes=sh.CLASSES,
    )
    raw = sc.p_true.reshape(K, -1)
    return ClassWorld(n, sc, make_layout(n), sh.clip_norm(raw), raw)


def class_draws(w: ClassWorld, r: int) -> sh.Draws:
    """``shared.class_draws`` on an n × n landscape (identical draws at n = 1024)."""
    sc = w.sc
    rng = np.random.default_rng(seed(w.n, 7, r))
    s = sc.sampler.sample(rng, 1)[0].ravel()
    sd_n = math.sqrt(sc.nugget)
    mu = sc.mu.ravel()
    lim = sc.limits
    y_struct = mu + s
    frac = np.stack(
        [
            interval_probability((lim[j] - y_struct) / sd_n, (lim[j + 1] - y_struct) / sd_n)
            for j in range(K)
        ]
    )
    truth = np.stack([w.layout.bmat @ frac[j] for j in range(K)])
    n_max = max(sh.SIZES)
    cells, clusters = {}, {}
    for d in sh.DESIGNS:
        cells[d], clusters[d] = sample_cells(w.n, d, n_max, rng)
    eps = {d: rng.standard_normal(n_max) for d in sh.DESIGNS}
    perms = {(d, k): rng.permutation(k)[: sh.N_FIT] for d in sh.DESIGNS for k in sh.SIZES}
    labels = {
        d: np.searchsorted(sc.thresholds, mu[cells[d]] + s[cells[d]] + sd_n * eps[d], side="left")
        for d in sh.DESIGNS
    }
    rng_c = np.random.default_rng(seed(w.n, 8, r))
    cal_cells, cal_clusters = {}, {}
    for d in sh.DESIGNS:
        cal_cells[d], cal_clusters[d] = sample_cells(w.n, d, n_max, rng_c)
    cal_eps = {d: rng_c.standard_normal(n_max) for d in sh.DESIGNS}
    cal_labels = {
        d: np.searchsorted(
            sc.thresholds, mu[cal_cells[d]] + s[cal_cells[d]] + sd_n * cal_eps[d], side="left"
        )
        for d in sh.DESIGNS
    }
    return sh.Draws(
        r, s, truth, cells, clusters, labels, perms, cal_cells, cal_clusters, cal_labels
    )


# map name → (miscalibration or None, calibrator or None); M7 is not recalibrated
CLASS_MAPS: dict[str, tuple[str | None, str | None]] = sh.CLASS_MAPS | {"M7": ("combined", None)}


# ---------------------------------------------------------------------------
# Continuous maps
# ---------------------------------------------------------------------------


@dataclass
class ContWorld:
    n: int
    sc: ErrorModelScenario
    layout: sh.Layout
    sd_true: np.ndarray
    m: np.ndarray  # standardized prediction field (round 4's m)


def cont_world(n: int) -> ContWorld:
    sc = ErrorModelScenario(layout=SubGridLayout(n=n, f=1), correlation=sh.CORRELOGRAM)
    sd_true = np.asarray(sc.sigma_true).ravel()
    emb = CirculantEmbedding((n, n), 1.0, lambda h: np.exp(-h / sh.SIGNAL_RANGE))
    rng = np.random.default_rng(sh.SEED + 1 if n == 1024 else [sh.SEED, 4, n])
    f = emb.sample(rng, 1)[0].ravel()
    return ContWorld(n, sc, make_layout(n), sd_true, (f - f.mean()) / f.std())


@dataclass
class ContDraws:
    """One realization of the continuous error landscape and its samples (round 5's order)."""

    s: np.ndarray
    cells: dict[str, np.ndarray]
    clusters: dict[str, np.ndarray]
    eps: dict[str, np.ndarray]
    perms: dict[tuple[str, int], np.ndarray]
    cal_cells: dict[str, np.ndarray]
    cal_clusters: dict[str, np.ndarray]
    cal_eps: dict[str, np.ndarray]


def cont_draws(w: ContWorld, r: int) -> ContDraws:
    rng = np.random.default_rng(seed(w.n, 10, r))
    s = w.sc.sampler.sample(rng, 1)[0].ravel()
    n_max = max(sh.SIZES)
    cells, clusters, eps = {}, {}, {}
    for d in sh.DESIGNS:
        cells[d], clusters[d] = sample_cells(w.n, d, n_max, rng)
    for d in sh.DESIGNS:
        eps[d] = rng.standard_normal(n_max)
    perms = {(d, k): rng.permutation(k)[: sh.N_FIT] for d in sh.DESIGNS for k in sh.SIZES}
    rng_c = np.random.default_rng(seed(w.n, 11, r))
    cal_cells, cal_clusters, cal_eps = {}, {}, {}
    for d in sh.DESIGNS:
        cal_cells[d], cal_clusters[d] = sample_cells(w.n, d, n_max, rng_c)
    for d in sh.DESIGNS:
        cal_eps[d] = rng_c.standard_normal(n_max)
    return ContDraws(s, cells, clusters, eps, perms, cal_cells, cal_clusters, cal_eps)


# map name → (bias kind or None, recalibration or None); C4 is not recalibrated
CONT_MAPS: dict[str, tuple[str | None, str | None]] = {
    "C1": (None, None),
    "C2s": ("prop", "sigma"),
    "C2l": ("prop", "linear"),
    "C3s": ("const", "sigma"),
    "C3l": ("const", "linear"),
    "C4": ("prop", None),
}
