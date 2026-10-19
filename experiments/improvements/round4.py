"""Round 4: the chosen corrections on a 1024 × 1024 landscape.

Round 3 recommended the unknown-mean correction with a constant mean: ``ok`` (with the
cross term) for continuous maps and ``ok_nc`` (without it) for class maps. This round
tests them at a realistic scale against the package as it is (``current``), two oracles
and the design-based estimators:

* landscape: 1024 × 1024 cells (validation points at cell centres); error correlogram
  Nug 0.2 + Exp 0.8 with range parameter 32 cells (practical range ≈ 96);
* blocks: 256 random 4 × 4 tiles, all 256 tiles of 64 × 64, all 16 of 256 × 256 and the
  whole landscape;
* holdouts of 100, 1000 and 5000 points, simple random or clustered (clusters of 10 cells
  within 16 cells); the samples are nested (100 ⊂ 1000 ⊂ 5000);
* maps: calibrated and three miscalibrations per mode. Class maps (classes a/b/c at
  67/30/3 %, probabilities clipped to [0.001, 0.999]): ``offset`` (rare-class logit
  − 0.7: the rare class is under-predicted), ``slope`` (rare-class logit × 1.3: too steep)
  and ``scale`` (temperature 0.7: over-confident). Continuous maps: ``offset`` (the map
  over-predicts by 0.3 σ), ``slope`` (mean error −0.3·σ·m(s), m a fixed standardized
  field with range 64, as for a map that over-stretches its predictions; zero landscape
  mean) and ``scale`` (the σ map is 0.75 × the true σ).

Arms: ``current`` (the package: zero-mean ML correlogram, Monte Carlo block variance,
known mean), ``ok`` / ``ok_nc`` (REML correlogram, GLS estimate of a constant mean of the
standardized errors from all holdout points, corrected block estimate), ``oracle`` (true σ
and correlogram, zero mean), ``oracle_ok`` (true correlogram, the map's σ, unknown mean,
with cross term), ``olofsson`` (class maps: post-stratified by mapped class per block;
strata with fewer than 2 points are merged into the block's largest stratum) and ``ppi``
(continuous maps: mean residual of the points in the block). The design-based arms need
20 points in a block. Correlograms are fitted on a random subsample of at most 500 holdout
points, as the package advises for large n.

Scaling: the cross term's c_B,i (block mean of σ(s)ρ(|s − x_i|)) enters only through
Σ_i G_i c_B,i, the block mean of σ(s)·v(s) with v(s) = Σ_i G_i ρ(|s − x_i|). v is one
FFT convolution of the GLS weights with the correlogram on the cell grid, so the cell
sums of round 3 stay exact without a cells × points distance matrix. The oracle block
variances (fixed σ and correlogram) are the exact cell double sums, also by FFT.

The truth, holdout points and Monte Carlo pairs are shared across maps, sizes and designs
within a realization. Per-block sums over realizations go to
``results/round4/{mode}_r{start}_{stop}.parquet`` (with fitted parameters in
``..._fits.parquet``); ``--summarize`` combines them into ``scorecard.md`` and
``summary.csv``.

Usage::

    uv run --group experiments python -m experiments.improvements.round4 --quick
    uv run --group experiments python -m experiments.improvements.round4 --skip-existing \\
        --max-chunks 3
    uv run python -m experiments.improvements.round4 --summarize
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import fft, linalg, sparse

import pyspatialuncertainty as psu
from experiments.common import chunks, run_parallel
from experiments.improvements import candidates as cand
from experiments.improvements.round3 import Z
from experiments.improvements.round3_ideal import P_CLIP, transform_rare
from pyspatialuncertainty.blocks import Block
from pyspatialuncertainty.synthetic import ErrorModelScenario, LatentClassScenario, SubGridLayout
from pyspatialuncertainty.synthetic.categorical import interval_probability
from pyspatialuncertainty.synthetic.grf import CirculantEmbedding

RESULTS4 = Path(__file__).resolve().parent / "results" / "round4"
SEED4 = 20261019
N_SIDE = 1024
CORRELOGRAM = psu.vgm(0.8, "Exp", 32.0, nugget=0.2)
SIGNAL_RANGE = 64.0  # class maps: range of the known signal μ; continuous: of m(s)
PREVALENCE = (0.67, 0.30, 0.03)
CLASSES = ("a", "b", "c")
SIDES = (4, 64, 256, 1024)
N_SMALL = 256  # tiles per side at most (random subset of the 4 × 4 tiles)
SIZES = (100, 1000, 5000)
DESIGNS = ("random", "clustered")
CLUSTER_SIZE = 10
CLUSTER_RADIUS = 16.0
N_FIT = 500  # correlogram fits use at most this many holdout points
N_MC = 10_000
N_MIN = 20  # design-based arms: points needed in a block
N_WORKERS = 4
MODES = ("continuous", "categorical")
CALS = ("calibrated", "offset", "slope", "scale")
BIAS = 0.3
SD_SCALE = 0.75
RARE_SHIFT = -0.7
RARE_SLOPE = 1.3
TEMPERATURE = 0.7
KEYS = ["mode", "calibration", "design", "n", "arm", "class"]
STATS = ("n_valid", "n_cov", "s_res", "s_res2", "s_sd2", "s_logsd", "s_logsd2")


# ---------------------------------------------------------------------------
# Blocks, exact convolutions and sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Blocks:
    """Square tiles of several sizes on an n × n grid, as one list of blocks.

    ``lookup[g]`` maps the tile ids of side ``sides[g]`` (row-major) to block indices
    (−1 for tiles that are not used); ``tiles[g]`` lists the tile ids in use.
    """

    n: int
    sides: tuple[int, ...]
    tiles: tuple[np.ndarray, ...]
    lookup: tuple[np.ndarray, ...]
    groups: tuple[str, ...]
    resolved: list[psu.ResolvedBlock]
    bmat: sparse.csr_matrix

    def membership(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(point, block) index pairs for the blocks that contain each cell."""
        row, col = np.divmod(cells, self.n)
        pts, blk = [], []
        for side, lk in zip(self.sides, self.lookup, strict=True):
            b = lk[(row // side) * (self.n // side) + col // side]
            ok = np.flatnonzero(b >= 0)
            pts.append(ok)
            blk.append(b[ok])
        return np.concatenate(pts), np.concatenate(blk)


def tile_cells(n: int, side: int, tile: int) -> np.ndarray:
    r0, c0 = (tile // (n // side)) * side, (tile % (n // side)) * side
    rr, cc = np.meshgrid(np.arange(r0, r0 + side), np.arange(c0, c0 + side), indexing="ij")
    return (rr * n + cc).ravel()


def make_blocks(n: int, sides: tuple[int, ...], n_small: int, seed: int = SEED4) -> Blocks:
    rng = np.random.default_rng(seed)
    grid = SubGridLayout(n=n, f=1).grid
    blocks: list[Block] = []
    groups: list[str] = []
    tiles, lookup = [], []
    for side in sides:
        n_tiles = (n // side) ** 2
        chosen = (
            np.arange(n_tiles)
            if n_tiles <= n_small
            else np.sort(rng.choice(n_tiles, n_small, replace=False))
        )
        lk = np.full(n_tiles, -1, dtype=np.int64)
        lk[chosen] = len(blocks) + np.arange(chosen.size)
        blocks += [Block(id=f"{side}_{t}", cells=tile_cells(n, side, int(t))) for t in chosen]
        groups += [str(side)] * chosen.size
        tiles.append(chosen)
        lookup.append(lk)
    resolved = psu.resolve_blocks(
        psu.BlockSet(grid=grid, blocks=tuple(blocks), kind="custom"), np.ones(grid.size, bool)
    )
    return Blocks(
        n=n,
        sides=sides,
        tiles=tuple(tiles),
        lookup=tuple(lookup),
        groups=tuple(groups),
        resolved=resolved,
        bmat=cand.block_average_matrix(resolved, grid.size),
    )


class Kernel:
    """A correlation kernel ρ(|Δ|) on the cell offsets of an L × L window, for exact
    (cell-sum) convolutions by FFT. ``fn`` gives ρ at the offsets' distances, including
    its value for coincident cells at distance 0."""

    def __init__(self, side: int, fn: Callable[[np.ndarray], np.ndarray]):
        self.side = side
        p = 2 * side
        i = np.arange(p)
        off = np.where(i < side, i, i - p).astype(np.float64)
        self.hat = fft.rfft2(np.asarray(fn(np.hypot(off[:, None], off[None, :]))))

    def conv(self, arr: np.ndarray) -> np.ndarray:
        """Σ_p arr(p) ρ(|q − p|) for every cell q of the window(s) (last two axes)."""
        p = 2 * self.side
        out = fft.irfft2(fft.rfft2(arr, s=(p, p)) * self.hat, s=(p, p))
        return out[..., : self.side, : self.side]


def exact_block_variance(
    blocks: Blocks, sd: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """(1/|B|²) Σ_s Σ_u σ(s)σ(u)ρ(|s − u|) over the cells of every block."""
    n = blocks.n
    out = np.empty(len(blocks.groups))
    sd2 = sd.reshape(n, n)
    for side, chosen, lk in zip(blocks.sides, blocks.tiles, blocks.lookup, strict=True):
        per = n // side
        patches = sd2.reshape(per, side, per, side).transpose(0, 2, 1, 3).reshape(-1, side, side)
        patches = patches[chosen]
        conv = Kernel(side, fn).conv(patches)
        out[lk[chosen]] = (patches * conv).sum(axis=(1, 2)) / side**4
    return out


def cross_field(kernel: Kernel, cells: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """v(s) = Σ_i w_i ρ(|s − x_i|) on all cells (flat), for points at cells ``cells``."""
    n = kernel.side
    raster = np.bincount(cells, weights=weights, minlength=n * n).reshape(n, n)
    return kernel.conv(raster).ravel()


def sample_cells(design: str, n_side: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Distinct cells; every prefix of the result is a sample of the same design."""
    if design == "random":
        return rng.choice(n_side * n_side, n, replace=False)
    if design != "clustered":
        raise ValueError(f"unknown design {design!r}")
    taken = np.zeros(n_side * n_side, dtype=bool)
    out: list[np.ndarray] = []
    got = 0
    r = math.ceil(CLUSTER_RADIUS)
    while got < n:
        cy, cx = rng.uniform(0, n_side, 2)
        rows = np.arange(max(0, int(cy) - r), min(n_side, int(cy) + r + 1))
        cols = np.arange(max(0, int(cx) - r), min(n_side, int(cx) + r + 1))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        near = np.hypot(rr + 0.5 - cy, cc + 0.5 - cx) <= CLUSTER_RADIUS
        cells = (rr * n_side + cc)[near]
        cells = cells[~taken[cells]]
        pick = rng.choice(cells, min(CLUSTER_SIZE, cells.size, n - got), replace=False)
        taken[pick] = True
        out.append(pick)
        got += pick.size
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Landscapes
# ---------------------------------------------------------------------------


def standardized_field(n: int, range_: float, seed: int) -> np.ndarray:
    emb = CirculantEmbedding((n, n), 1.0, lambda h: np.exp(-h / range_))
    f = emb.sample(np.random.default_rng(seed), 1)[0].ravel()
    return (f - f.mean()) / f.std()


@functools.cache
def landscape(mode: str, n: int = N_SIDE, sides: tuple[int, ...] = SIDES) -> dict[str, Any]:
    """Everything that is fixed across realizations: blocks, maps per calibration, the
    oracle correlograms and the exact oracle block variances."""
    lay = SubGridLayout(n=n, f=1)
    blocks = make_blocks(n, sides, N_SMALL)
    cx, cy = lay.grid.cell_centres(np.arange(n * n))
    out: dict[str, Any] = {"layout": lay, "blocks": blocks, "cx": cx, "cy": cy}
    bmat = blocks.bmat
    if mode == "continuous":
        sc = ErrorModelScenario(layout=lay, correlation=CORRELOGRAM)
        sd_true = sc.sigma_true.ravel()
        m = standardized_field(n, SIGNAL_RANGE, SEED4 + 1)
        m = m - float(np.mean(sd_true * m)) / float(sd_true.mean())  # zero mean bias
        bias = {
            "calibrated": np.zeros(n * n),
            "offset": -BIAS * sd_true,
            "slope": -BIAS * sd_true * m,
            "scale": np.zeros(n * n),
        }
        sd_user = {c: (SD_SCALE if c == "scale" else 1.0) * sd_true for c in CALS}

        def fn(d: np.ndarray) -> np.ndarray:
            return cand.structured_correlation(sc.correlogram, d)

        v_true = exact_block_variance(blocks, sd_true, fn)
        out |= {
            "sc": sc,
            "sampler": sc.sampler,
            "nugget": sc.nugget,
            "sd_true": sd_true,
            "bias": bias,
            "sd_user": sd_user,
            "x_b": {c: np.asarray(bmat @ sd_user[c]) for c in CALS},
            "v_true": v_true,
            "v_user": {c: v_true * (sd_user[c][0] / sd_true[0]) ** 2 for c in CALS},
            "oracle_fn": fn,
            "oracle_kernel": Kernel(n, fn),
        }
        return out
    sc = LatentClassScenario(
        layout=lay,
        latent=CORRELOGRAM,
        signal_range=SIGNAL_RANGE,
        prevalence=PREVALENCE,
        classes=CLASSES,
    )
    k = len(CLASSES)
    p_true = sc.p_true.reshape(k, -1)
    p_user = {}
    for c in CALS:
        if c == "offset":
            p = transform_rare(p_true, shift=RARE_SHIFT)
        elif c == "slope":
            p = transform_rare(p_true, slope=RARE_SLOPE)
        elif c == "scale":
            p = psu.temperature_scale(p_true, TEMPERATURE, class_dim=0)
        else:
            p = p_true
        p = np.clip(p, P_CLIP, 1 - P_CLIP)
        p_user[c] = p / p.sum(axis=0, keepdims=True)
    sd_true = np.sqrt(p_true * (1 - p_true))
    sd_user = {c: np.sqrt(p_user[c] * (1 - p_user[c])) for c in CALS}
    oracle_fn, v_true, kernels = [], [], []
    for j in range(k):
        corr = sc.oracle_correlation(j)
        rho0 = float(np.asarray(corr(np.array([1e-9])))[0])

        def fn(d: np.ndarray, corr: Any = corr, rho0: float = rho0) -> np.ndarray:
            return np.where(d > 0, np.asarray(corr(d)), rho0)

        oracle_fn.append(fn)
        v_true.append(exact_block_variance(blocks, sd_true[j], fn))
        kernels.append(Kernel(n, fn))
    map_class = {c: np.argmax(p_user[c], axis=0) for c in CALS}
    onehot = {c: np.eye(k)[map_class[c]] for c in CALS}
    out |= {
        "sc": sc,
        "sampler": sc.sampler,
        "nugget": sc.nugget,
        "mu": sc.mu.ravel(),
        "p_user": p_user,
        "sd_user": sd_user,
        "x_b": {c: np.stack([np.asarray(bmat @ sd_user[c][j]) for j in range(k)]) for c in CALS},
        "pred": {c: np.stack([np.asarray(bmat @ p_user[c][j]) for j in range(k)]) for c in CALS},
        "map_class": map_class,
        "weights": {c: np.asarray(bmat @ onehot[c]) for c in CALS},  # (blocks, strata)
        "v_true": np.stack(v_true),
        "v_user": {
            c: np.stack(
                [exact_block_variance(blocks, sd_user[c][j], oracle_fn[j]) for j in range(k)]
            )
            for c in CALS
        },
        "oracle_fn": oracle_fn,
        "oracle_kernel": kernels,
    }
    return out


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GLSOnes:
    """GLS mean of z with design X = 1: β̂ = G z, A = (1ᵀK⁻¹1)⁻¹, G = A·(K⁻¹1)ᵀ."""

    A: float
    G: np.ndarray


def gls_ones(K: np.ndarray) -> GLSOnes:
    """GLS weights for a correlation matrix ``K`` (overwritten); the diagonal is set to 1."""
    K[np.diag_indices_from(K)] = 1.0 + 1e-10
    c = linalg.cho_factor(K, lower=True, overwrite_a=True, check_finite=False)
    k1 = linalg.cho_solve(c, np.ones(K.shape[0]), check_finite=False)
    a = 1.0 / float(k1.sum())
    return GLSOnes(A=a, G=a * k1)


def corrected(
    x_b: np.ndarray,
    variance: np.ndarray,
    g: GLSOnes,
    z: np.ndarray,
    cross_b: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """(shift, sd) of the corrected block estimate: shift x_B β̂, variance
    V_B + x_B² A − 2 x_B Σ_i G_i c_B,i (the last term only with ``cross_b``)."""
    added = x_b**2 * g.A
    if cross_b is not None:
        added = added - 2.0 * x_b * cross_b
    return x_b * float(g.G @ z), np.sqrt(np.maximum(variance + added, 0.0))


def ppi(nb: int, blk: np.ndarray, resid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean residual of the points in each block and its SE; NaN below ``N_MIN`` points."""
    cnt = np.bincount(blk, minlength=nb)
    s1 = np.bincount(blk, weights=resid, minlength=nb)
    s2 = np.bincount(blk, weights=resid**2, minlength=nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = (s2 - cnt * mean**2) / (cnt - 1)
        sd = np.sqrt(np.maximum(var, 0.0) / cnt)
    ok = cnt >= N_MIN
    return np.where(ok, mean, np.nan), np.where(ok, sd, np.nan)


def olofsson(
    weights: np.ndarray, blk: np.ndarray, stratum: np.ndarray, is_k: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Post-stratified estimate of a class's area share per block, and its SE.

    Strata are the mapped classes with weights ``weights`` (blocks × strata). Strata with
    fewer than 2 points (including empty ones) are merged into the block's largest stratum
    that has 2 or more. NaN below ``N_MIN`` points.
    """
    nb, h = weights.shape
    idx = blk * h + stratum
    nh = np.bincount(idx, minlength=nb * h).reshape(nb, h).astype(np.float64)
    sh = np.bincount(idx, weights=is_k.astype(np.float64), minlength=nb * h).reshape(nb, h)
    w = weights.copy()
    present = w > 0
    small = present & (nh < 2)
    big = present & (nh >= 2)
    tgt = np.argmax(np.where(big, w, -1.0), axis=1)
    rows = np.arange(nb)
    for arr in (w, nh, sh):
        arr[rows, tgt] += np.where(small, arr, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(big, sh / nh, 0.0)
        est = np.where(big, w * p, 0.0).sum(axis=1)
        var = np.where(big, w**2 * p * (1 - p) / (nh - 1), 0.0).sum(axis=1)
    ok = (nh.sum(axis=1) >= N_MIN) & big.any(axis=1)
    return np.where(ok, est, np.nan), np.where(ok, np.sqrt(var), np.nan)


def attempt(fn: Callable[..., Any], *args: Any) -> Any:
    """fn(*args), or None if it fails (the arm is then NaN for this realization)."""
    try:
        return fn(*args)
    except Exception:
        return None


def fitted_correlogram(z: np.ndarray, x: np.ndarray, y: np.ndarray, mean: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return psu.fit_likelihood(None, z, x, y, mean=mean).model  # type: ignore[arg-type]


def describe(prefix: str, m: Any) -> dict[str, Any]:
    main = [s for s in m.structures if s.model != "Nug"] if m is not None else []
    return {
        f"{prefix}_model": (main[0].model if main else "Nug") if m is not None else None,
        f"{prefix}_range": main[0].range if main else math.nan,
        f"{prefix}_nugget_share": m.nugget / m.sill if m is not None and m.sill > 0 else math.nan,
        f"{prefix}_sill": m.sill if m is not None else math.nan,
    }


# ---------------------------------------------------------------------------
# One realization
# ---------------------------------------------------------------------------


def realization(
    mode: str,
    r: int,
    n_side: int = N_SIDE,
    sides: tuple[int, ...] = SIDES,
    sizes: tuple[int, ...] = SIZES,
) -> tuple[dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """All arms for every (map, design, holdout size) on one draw of the truth.

    Returns {(calibration, design, n, arm, class): (truth − estimate, claimed SD)} per
    block, and one row of fitted parameters per (calibration, design, n, class).
    """
    ls = landscape(mode, n_side, sides)
    blocks: Blocks = ls["blocks"]
    bmat = blocks.bmat
    nb = len(blocks.groups)
    mi = MODES.index(mode)
    rng = np.random.default_rng([SEED4, mi, r])
    s = ls["sampler"].sample(rng, 1)[0].ravel()  # structured part of the error / latent noise
    n_max = max(sizes)
    samples = {d: sample_cells(d, n_side, n_max, rng) for d in DESIGNS}
    eps = {d: rng.standard_normal(n_max) for d in DESIGNS}
    perms = {(d, n): rng.permutation(n)[:N_FIT] for d in DESIGNS for n in sizes}
    sd_n = math.sqrt(ls["nugget"])
    out: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    diag: list[dict[str, Any]] = []
    if mode == "continuous":
        err = {c: np.asarray(bmat @ (ls["sd_true"] * s + ls["bias"][c])) for c in CALS}
        classes: tuple[str, ...] = ("",)
    else:
        lim = ls["sc"].limits
        y_struct = ls["mu"] + s
        frac = np.stack(
            [
                interval_probability((lim[j] - y_struct) / sd_n, (lim[j + 1] - y_struct) / sd_n)
                for j in range(len(CLASSES))
            ]
        )
        truth = np.stack([np.asarray(bmat @ frac[j]) for j in range(len(CLASSES))])
        err = {c: truth - ls["pred"][c] for c in CALS}
        classes = CLASSES
    for di, design in enumerate(DESIGNS):
        for ni, n in enumerate(sizes):
            cells = samples[design][:n]
            x, y = ls["cx"][cells], ls["cy"][cells]
            sub = perms[(design, n)]
            pts, blk = blocks.membership(cells)
            D = cand.distance_matrix(x, y)
            # per class: z for every calibration, and the oracle GLS (shared by calibrations)
            zs: dict[tuple[str, int], np.ndarray] = {}
            if mode == "continuous":
                e_true = ls["sd_true"][cells] * (s[cells] + sd_n * eps[design][:n])
                resid = {c: e_true + ls["bias"][c][cells] for c in CALS}
                for c in CALS:
                    zs[(c, 0)] = resid[c] / ls["sd_user"][c][cells]
                kernels = [ls["oracle_kernel"]]
                fns = [ls["oracle_fn"]]
            else:
                y_pt = ls["mu"][cells] + s[cells] + sd_n * eps[design][:n]
                label = np.searchsorted(ls["sc"].thresholds, y_pt, side="left")
                for c in CALS:
                    for j in range(len(CLASSES)):
                        pk = ls["p_user"][c][j, cells]
                        zs[(c, j)] = ((label == j) - pk) / np.sqrt(pk * (1 - pk))
                kernels = ls["oracle_kernel"]
                fns = ls["oracle_fn"]
            g_or, v_or = [], []
            for j in range(len(classes)):
                g = attempt(gls_ones, fns[j](D))
                g_or.append(g)
                v_or.append(cross_field(kernels[j], cells, g.G) if g is not None else None)
            for c in CALS:
                fits = {}
                integrands, ikeys = [], []
                for j in range(len(classes)):
                    z = zs[(c, j)]
                    sd_c = ls["sd_user"][c] if mode == "continuous" else ls["sd_user"][c][j]
                    for key, mean in (("m0", "zero"), ("m1", "constant")):
                        m = attempt(fitted_correlogram, z[sub], x[sub], y[sub], mean)
                        fits[(j, key)] = m
                        if m is not None:
                            integrands.append(psu.Integrand(sd_c, cand.correlation(m)))
                            ikeys.append((j, key))
                var: dict[tuple[int, str], np.ndarray] = {}
                if integrands:  # the same point pairs for every calibration
                    mc_seed = np.random.SeedSequence([SEED4, mi, r, di, ni])
                    mc = psu.mc_block_variance(blocks.resolved, integrands, n_mc=N_MC, seed=mc_seed)
                    var = dict(zip(ikeys, mc.variance, strict=True))
                for j, cls in enumerate(classes):
                    z = zs[(c, j)]
                    if mode == "continuous":
                        sd_c, x_b, e_b = ls["sd_user"][c], ls["x_b"][c], err[c]
                        v_true, v_user = ls["v_true"], ls["v_user"][c]
                    else:
                        sd_c, x_b, e_b = ls["sd_user"][c][j], ls["x_b"][c][j], err[c][j]
                        v_true, v_user = ls["v_true"][j], ls["v_user"][c][j]
                    nan = (np.full(nb, np.nan), np.full(nb, np.nan))
                    arms: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                    m0, m1 = fits[(j, "m0")], fits[(j, "m1")]
                    arms["current"] = (
                        (np.zeros(nb), np.sqrt(np.maximum(var[(j, "m0")], 0.0)))
                        if m0 is not None
                        else nan
                    )
                    g1 = (
                        attempt(gls_ones, cand.structured_correlation(m1, D))
                        if m1 is not None
                        else None
                    )
                    if mode == "continuous":
                        if g1 is not None:
                            k1 = Kernel(n_side, lambda d, m1=m1: cand.structured_correlation(m1, d))
                            cross = np.asarray(bmat @ (sd_c * cross_field(k1, cells, g1.G)))
                            arms["ok"] = corrected(x_b, var[(j, "m1")], g1, z, cross)
                        else:
                            arms["ok"] = nan
                        arms["ppi"] = ppi(nb, blk, resid[c][pts])
                    else:
                        arms["ok_nc"] = (
                            corrected(x_b, var[(j, "m1")], g1, z, None) if g1 is not None else nan
                        )
                        arms["olofsson"] = olofsson(
                            ls["weights"][c],
                            blk,
                            ls["map_class"][c][cells][pts],
                            (label == j)[pts],
                        )
                        sh, sd_o = arms["olofsson"]
                        arms["olofsson"] = (sh - ls["pred"][c][j], sd_o)
                    arms["oracle"] = (np.zeros(nb), np.sqrt(np.maximum(v_true, 0.0)))
                    go = g_or[j]
                    if go is not None:
                        cross_o = np.asarray(bmat @ (sd_c * v_or[j]))
                        arms["oracle_ok"] = corrected(x_b, v_user, go, z, cross_o)
                    else:
                        arms["oracle_ok"] = nan
                    for arm, (shift, sd) in arms.items():
                        out[(c, design, str(n), arm, cls)] = (e_b - shift, sd)
                    diag.append(
                        {
                            "realization": r,
                            "calibration": c,
                            "design": design,
                            "n": n,
                            "class": cls,
                            **describe("m0", m0),
                            **describe("m1", m1),
                            "beta_ok": float(g1.G @ z) if g1 is not None else math.nan,
                            "se_beta_ok": math.sqrt(g1.A) if g1 is not None else math.nan,
                            "beta_oracle": float(go.G @ z) if go is not None else math.nan,
                            "se_beta_oracle": math.sqrt(go.A) if go is not None else math.nan,
                        }
                    )
    return out, diag


def worker(task: dict[str, Any]) -> dict[str, Any]:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    t0 = time.time()
    res, diag = realization(task["mode"], task["r"])
    stats = {}
    for key, (resid, sd) in res.items():
        valid = np.isfinite(resid) & np.isfinite(sd)
        r0 = np.where(valid, resid, 0.0)
        s0 = np.where(valid, sd, 1.0)
        logsd = np.log(np.maximum(s0, 1e-12))
        stats[key] = np.stack(
            [
                valid,
                valid & (np.abs(r0) <= Z * s0),
                r0,
                r0**2,
                np.where(valid, sd, 0.0) ** 2,
                np.where(valid, logsd, 0.0),
                np.where(valid, logsd, 0.0) ** 2,
            ]
        ).astype(np.float64)
    return {"stats": stats, "diag": diag, "seconds": time.time() - t0}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_chunk(mode: str, start: int, stop: int, suffix: str, n_workers: int) -> None:
    t0 = time.time()
    out = run_parallel(worker, [{"mode": mode, "r": r} for r in range(start, stop)], n_workers)
    acc: dict[tuple[str, ...], np.ndarray] = {}
    for o in out:
        for key, st in o["stats"].items():
            acc[key] = acc[key] + st if key in acc else st.copy()
    groups = landscape_groups()
    nb = len(groups)
    frames = []
    for (cal, design, n, arm, cls), st in acc.items():
        frames.append(
            pl.DataFrame(
                {
                    "group": groups,
                    "block": np.arange(nb, dtype=np.int32),
                    "n_real": np.full(nb, stop - start, dtype=np.int16),
                    **{
                        name: st[i].astype(np.int16 if i < 2 else np.float32)
                        for i, name in enumerate(STATS)
                    },
                }
            ).with_columns(
                pl.lit(mode).alias("mode"),
                pl.lit(cal).alias("calibration"),
                pl.lit(design).alias("design"),
                pl.lit(int(n)).alias("n"),
                pl.lit(arm).alias("arm"),
                pl.lit(cls).alias("class"),
            )
        )
    stem = RESULTS4 / f"{mode}_r{start:03d}_{stop:03d}"
    pl.concat(frames).select([*KEYS, "group", "block", "n_real", *STATS]).write_parquet(
        f"{stem}{suffix}.parquet", compression="zstd"
    )
    pl.DataFrame([row | {"mode": mode} for o in out for row in o["diag"]]).write_parquet(
        f"{stem}_fits{suffix}.parquet", compression="zstd"
    )
    per = [o["seconds"] for o in out]
    print(
        f"[round4] {mode} r{start}-{stop}: {time.time() - t0:.0f} s wall, "
        f"{np.mean(per):.0f} s per realization",
        flush=True,
    )


def landscape_groups() -> tuple[str, ...]:
    """Block group labels in the order of `make_blocks`."""
    return tuple(str(s) for s in SIDES for _ in range(min((N_SIDE // s) ** 2, N_SMALL)))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def load(suffix: str = "") -> pl.DataFrame:
    files = sorted(f for m in MODES for f in RESULTS4.glob(f"{m}_r???_???{suffix}.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files])


def block_metrics(raw: pl.DataFrame) -> pl.DataFrame:
    """Per-block metrics over all realizations: the true SD of an arm's block error is its
    RMSE over realizations."""
    s = raw.group_by([*KEYS, "group", "block"]).agg(
        pl.col("n_real").cast(pl.Int64).sum(),
        *(pl.col(c).cast(pl.Float64).sum() for c in STATS),
    )
    nv = pl.col("n_valid")
    log_rmse = (pl.col("s_res2") / nv).sqrt().log()
    return s.with_columns(  # NaN for blocks where an arm was never available
        (pl.col("s_res2") / nv).sqrt().alias("rmse"),
        (pl.col("s_res") / nv).alias("bias"),
        ((pl.col("s_sd2") / nv).sqrt() / (pl.col("s_res2") / nv).sqrt()).alias("ratio_rms"),
        (pl.col("s_logsd") / nv - log_rmse).alias("lr_mean"),
        (pl.col("s_logsd2") / nv - 2 * log_rmse * pl.col("s_logsd") / nv + log_rmse**2).alias(
            "lr_ms"
        ),
    )


def group_metrics(blk: pl.DataFrame) -> pl.DataFrame:
    """Per (map, design, n, arm, class, block side): coverage pooled over block-realizations,
    RMS over blocks of the true RMSE and bias, median SD ratio, RMS log error of the claimed
    SD, and the share of block-realizations where the arm was available."""
    on = pl.col("n_valid") > 0
    g = blk.group_by([*KEYS, "group"]).agg(
        (pl.col("n_cov").sum() / pl.col("n_valid").sum()).alias("coverage"),
        (pl.col("rmse").filter(on) ** 2).mean().sqrt().alias("rmse"),
        (pl.col("bias").filter(on) ** 2).mean().sqrt().alias("rms_bias"),
        pl.col("ratio_rms").filter(on).median().alias("sd_ratio"),
        (
            (pl.col("n_valid") * pl.col("lr_ms")).filter(on).sum()
            / pl.col("n_valid").filter(on).sum()
        )
        .clip(lower_bound=0)
        .sqrt()
        .alias("log_sd_err"),
        (pl.col("n_valid").sum() / pl.col("n_real").sum()).alias("available"),
        pl.col("n_real").max().alias("n_real"),
    )
    ref = g.filter(pl.col("arm") == "current").select(
        [k for k in KEYS if k != "arm"] + ["group", pl.col("rmse").alias("rmse_current")]
    )
    return g.join(ref, on=[k for k in KEYS if k != "arm"] + ["group"], how="left").with_columns(
        (pl.col("rmse") / pl.col("rmse_current")).alias("rel_rmse")
    )


ARM_ORDER = {
    "continuous": ("current", "ok", "ppi", "oracle", "oracle_ok"),
    "categorical": ("current", "ok_nc", "olofsson", "oracle", "oracle_ok"),
}


def summarize(suffix: str = "") -> None:
    grp = group_metrics(block_metrics(load(suffix)))
    grp = grp.with_columns(pl.col("group").cast(pl.Int64)).sort([*KEYS, "group"])
    grp.write_csv(RESULTS4 / f"summary{suffix}.csv", float_precision=4)

    def fmt(v: float | None, d: int = 2) -> str:
        return "–" if v is None or not math.isfinite(v) else f"{v:.{d}f}"

    lines = [
        "# Round 4: 1024 × 1024 landscape, scorecard",
        "",
        "Generated by `experiments/improvements/round4.py`. Each cell is coverage of the 90 % "
        "interval / median ratio of claimed SD to true RMSE / RMSE relative to `current`; "
        "a trailing `[a]` gives the share of (block, realization) pairs where a design-based "
        "arm could be computed (below 1 only). Columns are tile sides in cells.",
        "",
    ]
    sides = sorted(grp["group"].unique().to_list())
    for (mode, cls, cal), sub in grp.group_by("mode", "class", "calibration", maintain_order=True):
        name = "continuous map" if mode == "continuous" else f"class map, class {cls}"
        lines += [f"## {name}: {cal}", ""]
        lines += ["| design | n | arm | " + " | ".join(str(s) for s in sides) + " |"]
        lines += ["|---|---|---|" + "---|" * len(sides)]
        for design in DESIGNS:
            for n in sorted(sub["n"].unique().to_list()):
                for arm in ARM_ORDER[mode]:
                    cells = []
                    for side in sides:
                        row = sub.filter(
                            (pl.col("design") == design)
                            & (pl.col("n") == n)
                            & (pl.col("arm") == arm)
                            & (pl.col("group") == side)
                        )
                        if row.height == 0:
                            cells.append("–")
                            continue
                        r = row.row(0, named=True)
                        txt = f"{fmt(r['coverage'])} / {fmt(r['sd_ratio'])} / {fmt(r['rel_rmse'])}"
                        if r["available"] is not None and r["available"] < 0.995:
                            txt += f" [{fmt(r['available'])}]"
                        cells.append(txt)
                    lines.append(f"| {design} | {n} | {arm} | " + " | ".join(cells) + " |")
        lines.append("")
    out = RESULTS4 / f"scorecard{suffix}.md"
    out.write_text("\n".join(lines) + "\n")
    print(out, len(lines), "lines")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, action="append")
    ap.add_argument("--realizations", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--max-chunks", type=int, default=None, help="stop after this many chunks")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--quick", action="store_true", help="2 realizations, *_quick files")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, "1")
    suffix = "_quick" if args.quick else ""
    if args.summarize:
        summarize(suffix)
        return
    RESULTS4.mkdir(parents=True, exist_ok=True)
    n_real, size = (2, 2) if args.quick else (args.realizations, args.chunk)
    done = 0
    for mode in args.mode or MODES:
        for start, stop in chunks(n_real, size):
            stem = RESULTS4 / f"{mode}_r{start:03d}_{stop:03d}{suffix}.parquet"
            if args.skip_existing and stem.exists():
                continue
            if args.max_chunks is not None and done >= args.max_chunks:
                return
            run_chunk(mode, start, stop, suffix, args.workers)
            done += 1


if __name__ == "__main__":
    main()
