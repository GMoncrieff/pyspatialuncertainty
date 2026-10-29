"""Shared pieces of the round-5 experiments: landscape, samples, maps, calibrators and exact
block integrals by FFT.

The class landscape, blocks, samplers, Olofsson estimator, miscalibrations and calibrators
are those of the Landscape Check simulation (Parts 1 and 2), with the same seeds, so that
round 5 is paired with it realization by realization. The FFT kernels follow round 4
(``experiments/improvements/round4.py`` on branch ``claude/unknown-mean-experiments``).

Two kinds of kernel are used:

* ``support="point"``: the correlation between two points drawn uniformly in the two cells,
  as in the package's Monte Carlo integral over continuous blocks. Offsets of up to
  ``NEAR`` cells are averaged by quadrature over the triangular density of the difference
  of two uniform points; further offsets use ρ at the centre distance. Coincident points
  have probability zero, so the nugget never enters V_B.
* ``support="centre"``: ρ at cell-centre distances with the structured limit 1 − η at zero
  distance, as the package's cross term (validation points sit at cell centres).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import xarray as xr
from scipy import fft, optimize, sparse, special

import pyspatialuncertainty as psu
from pyspatialuncertainty.blocks import Block
from pyspatialuncertainty.synthetic import LatentClassScenario, SubGridLayout
from pyspatialuncertainty.synthetic.categorical import interval_probability
from pyspatialuncertainty.variogram.models import VariogramModel

SEED = 20261019  # the Landscape Check's seed (round 4's landscape and blocks)
N = 1024
CORRELOGRAM = psu.vgm(0.8, "Exp", 32.0, nugget=0.2)
SIGNAL_RANGE = 64.0
PREVALENCE = (0.67, 0.30, 0.03)
CLASSES = ("a", "b", "c")
SIDES = (4, 64, 256, 1024)
N_SMALL = 256
SIZES = (50, 500, 5000)
DESIGNS = ("random", "clustered")
CLUSTER_SIZE, CLUSTER_RADIUS = 10, 16.0
N_FIT = 500
P_CLIP = 1e-3
LEVEL = 0.90
Z = 1.6448536269514722  # 90 % normal quantile
NEAR = 3  # offsets (in cells) averaged by quadrature for point support
RARE_SHIFT, RARE_SLOPE, TEMPERATURE = -0.7, 1.3, 0.7  # round 4's class miscalibrations


# ---------------------------------------------------------------------------
# Correlograms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corr:
    """A correlogram of standardized errors: ρ(h) for h > 0 and its limit at h → 0."""

    rho: Callable[[np.ndarray], np.ndarray]
    rho0: float
    label: str
    model: VariogramModel | None = None

    def struct(self, d: np.ndarray) -> np.ndarray:
        """ρ between distinct points: ρ(d) for d > 0, the structured limit at d = 0."""
        d = np.asarray(d, dtype=np.float64)
        r = np.asarray(self.rho(np.where(d > 0, d, 1.0)), dtype=np.float64)
        return np.where(d > 0, r, self.rho0)


def family(model: VariogramModel) -> str:
    main = [s.model for s in model.structures if s.model != "Nug" and s.psill > 0]
    return main[0] if main else "Nug"


def main_range(model: VariogramModel) -> float:
    main = [s for s in model.structures if s.model != "Nug" and s.psill > 0]
    return float(main[0].range) if main else math.nan


def practical_range(model: VariogramModel) -> float:
    factor = {"Exp": 3.0, "Sph": 1.0, "Gau": math.sqrt(3.0)}
    fam = family(model)
    return main_range(model) * factor[fam] if fam in factor else 0.0


def nugget_share(model: VariogramModel) -> float:
    return model.nugget / model.sill if model.sill > 0 else 1.0


def corr_of_model(model: VariogramModel, label: str | None = None) -> Corr:
    if not model.sill > 0 or family(model) == "Nug":
        return Corr(
            lambda d: np.zeros_like(np.asarray(d, dtype=np.float64)), 0.0, label or "Nug", model
        )
    return Corr(
        lambda d, m=model: np.asarray(m.correlation(d), dtype=np.float64),
        1.0 - model.nugget / model.sill,
        label or family(model),
        model,
    )


def corr_of_callable(fn: Callable[[np.ndarray], np.ndarray], label: str) -> Corr:
    rho0 = float(np.asarray(fn(np.array([1e-3])), dtype=np.float64)[0])
    return Corr(lambda d: np.asarray(fn(d), dtype=np.float64), rho0, label)


NUGGET_ONLY = psu.vgm(1.0, "Nug")


# ---------------------------------------------------------------------------
# Layout: tiles, block-average matrix, exact kernels
# ---------------------------------------------------------------------------


@dataclass
class Layout:
    """Square grid of n × n unit cells with tiles of the given sides as blocks."""

    n: int
    sides: tuple[int, ...]
    tiles: dict[int, np.ndarray]  # side → chosen tile indices (row-major tile order)
    side_of_block: np.ndarray
    slices: dict[int, slice] = field(default_factory=dict)
    bmat: sparse.csr_matrix | None = None
    cx: np.ndarray | None = None
    cy: np.ndarray | None = None

    @property
    def n_blocks(self) -> int:
        return int(self.side_of_block.size)


def tile_cells(n: int, side: int, tile: int) -> np.ndarray:
    r0, c0 = (tile // (n // side)) * side, (tile % (n // side)) * side
    rr, cc = np.meshgrid(np.arange(r0, r0 + side), np.arange(c0, c0 + side), indexing="ij")
    return (rr * n + cc).ravel()


def make_layout(n: int, sides: tuple[int, ...], tiles: dict[int, np.ndarray]) -> Layout:
    side_of_block, rows, cols, vals, slices = [], [], [], [], {}
    b = 0
    for side in sides:
        chosen = tiles[side]
        slices[side] = slice(b, b + chosen.size)
        for t in chosen:
            cells = tile_cells(n, side, int(t))
            rows.append(np.full(cells.size, b))
            cols.append(cells)
            vals.append(np.full(cells.size, 1.0 / cells.size))
            side_of_block.append(side)
            b += 1
    bmat = sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(b, n * n)
    )
    # cell centres, row 0 at the top (as GridSpec.cell_centres on the landscape)
    idx = np.arange(n * n)
    cx = (idx % n) + 0.5
    cy = n - (idx // n) - 0.5
    return Layout(n, sides, tiles, np.array(side_of_block), slices, bmat, cx, cy)


def _offsets(side: int) -> np.ndarray:
    p = 2 * side
    i = np.arange(p)
    return np.where(i < side, i, i - p).astype(np.float64)


def cell_average(corr: Corr, dx: float, dy: float, m: int = 24) -> float:
    """E ρ(|(dx, dy) + U − V|) for U, V uniform on the unit cell (point support)."""
    a = (np.arange(2 * m) + 0.5) / m - 1.0  # midpoints on [−1, 1], never 0
    w = (1.0 - np.abs(a)) / m  # triangular density of U − V per axis
    A, B = np.meshgrid(a, a, indexing="ij")
    return float(np.sum(np.outer(w, w) * corr.rho(np.hypot(dx + A, dy + B))))


class Kernel:
    """ρ on the cell offsets of a side × side window, for exact sums by FFT."""

    def __init__(self, side: int, corr: Corr, support: str, base: np.ndarray | None = None):
        self.side = side
        off = _offsets(side)
        if base is None:
            base = corr.struct(np.hypot(off[:, None], off[None, :]))
        k = base.copy()
        if support == "point":
            near = int(min(NEAR, side - 1))
            for dx in range(-near, near + 1):
                for dy in range(-near, near + 1):
                    k[dy % (2 * side), dx % (2 * side)] = cell_average(corr, dx, dy)
        elif support != "centre":
            raise ValueError(support)
        self.hat = fft.rfft2(k)

    def conv(self, arr: np.ndarray) -> np.ndarray:
        """Σ_p arr(p) ρ(q − p) for every cell q of the window(s) (last two axes)."""
        p = 2 * self.side
        out = fft.irfft2(fft.rfft2(arr, s=(p, p)) * self.hat, s=(p, p))
        return out[..., : self.side, : self.side]


@dataclass
class Kernels:
    """Point-support kernels for every tile side, and the centre kernel of the grid."""

    point: dict[int, Kernel]
    centre: Kernel


def make_kernels(layout: Layout, corr: Corr) -> Kernels:
    point = {}
    for side in layout.sides:
        off = _offsets(side)
        base = corr.struct(np.hypot(off[:, None], off[None, :]))
        point[side] = Kernel(side, corr, "point", base)
        if side == layout.n:
            centre = Kernel(side, corr, "centre", base)
    if layout.n not in point:
        centre = Kernel(layout.n, corr, "centre")
    return Kernels(point, centre)


def exact_vb(layout: Layout, sd: np.ndarray, kernels: Kernels) -> np.ndarray:
    """(1/|B|²) ∫_B∫_B σ(s)σ(u)ρ(|s − u|) for every block (point support, σ per cell)."""
    out = np.empty(layout.n_blocks)
    n = layout.n
    sd2 = sd.reshape(n, n)
    for side in layout.sides:
        per = n // side
        patches = sd2.reshape(per, side, per, side).transpose(0, 2, 1, 3).reshape(-1, side, side)
        patches = patches[layout.tiles[side]]
        conv = kernels.point[side].conv(patches)
        out[layout.slices[side]] = (patches * conv).sum(axis=(1, 2)) / side**4
    return out


def cross_c(
    layout: Layout, kernel: Kernel, sd: np.ndarray, cells: np.ndarray, G: np.ndarray
) -> np.ndarray:
    """C[q, B] = block mean of σ(s)·Σ_i G[q, i] ρ(|s − x_i|) (validation points at cells)."""
    n = layout.n
    G = np.atleast_2d(G)
    out = np.empty((G.shape[0], layout.n_blocks))
    for q in range(G.shape[0]):
        raster = np.bincount(cells, weights=G[q], minlength=n * n).reshape(n, n)
        v = kernel.conv(raster).ravel()
        out[q] = layout.bmat @ (sd * v)
    return out


# ---------------------------------------------------------------------------
# Class landscape (the Landscape Check), samples and truth
# ---------------------------------------------------------------------------


def clip_norm(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return p / p.sum(axis=0, keepdims=True)


@dataclass
class ClassLandscape:
    sc: LatentClassScenario
    layout: Layout
    p_true: np.ndarray  # (K, n²) calibrated map, clipped and renormalised
    p_raw: np.ndarray  # (K, n²) unclipped true probabilities
    probs: xr.DataArray  # calibrated map as a DataArray (package baselines)
    bsets: dict[str, psu.BlockSet]
    grid: psu.GridSpec


def build_class_landscape() -> ClassLandscape:
    lay = SubGridLayout(n=N, f=1)
    sc = LatentClassScenario(
        layout=lay,
        latent=CORRELOGRAM,
        signal_range=SIGNAL_RANGE,
        prevalence=PREVALENCE,
        classes=CLASSES,
    )
    raw = sc.p_true.reshape(len(CLASSES), -1)
    p = clip_norm(raw)
    probs = xr.DataArray(
        p.reshape(len(CLASSES), N, N),
        coords={"class": list(CLASSES), "y": lay.cell_y, "x": lay.cell_x},
        dims=("class", "y", "x"),
    )
    rng = np.random.default_rng(SEED)  # round 4's block choice
    tiles: dict[int, np.ndarray] = {}
    for side in SIDES:
        n_tiles = (N // side) ** 2
        tiles[side] = (
            np.arange(n_tiles)
            if n_tiles <= N_SMALL
            else np.sort(rng.choice(n_tiles, N_SMALL, replace=False))
        )
    layout = make_layout(N, SIDES, tiles)
    grid = psu.GridSpec.from_dataarray(probs.isel({"class": 0}))
    gx, gy = grid.cell_centres(np.arange(N * N))
    assert np.allclose(gx, layout.cx)
    assert np.allclose(gy, layout.cy)
    blocks = [
        Block(id=f"{side}_{t}", cells=tile_cells(N, side, int(t)))
        for side in SIDES
        for t in tiles[side]
    ]
    small = layout.side_of_block <= 64
    bsets = {
        "small": psu.BlockSet(
            grid=grid,
            blocks=tuple(b for b, k in zip(blocks, small, strict=True) if k),
            kind="custom",
        ),
        "large": psu.BlockSet(
            grid=grid,
            blocks=tuple(b for b, k in zip(blocks, small, strict=True) if not k),
            kind="custom",
        ),
    }
    return ClassLandscape(sc, layout, p, raw, probs, bsets, grid)


def sample_cells(design: str, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Distinct cells and their cluster ids; every prefix is a sample of the same design.
    The random draws are those of the Landscape Check sampler."""
    if design == "random":
        return rng.choice(N * N, n, replace=False), np.arange(n)
    taken = np.zeros(N * N, dtype=bool)
    out, ids, got, r, k = [], [], 0, math.ceil(CLUSTER_RADIUS), 0
    while got < n:
        cy, cx = rng.uniform(0, N, 2)
        rows = np.arange(max(0, int(cy) - r), min(N, int(cy) + r + 1))
        cols = np.arange(max(0, int(cx) - r), min(N, int(cx) + r + 1))
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        cells = (rr * N + cc)[np.hypot(rr + 0.5 - cy, cc + 0.5 - cx) <= CLUSTER_RADIUS]
        cells = cells[~taken[cells]]
        pick = rng.choice(cells, min(CLUSTER_SIZE, cells.size, n - got), replace=False)
        taken[pick] = True
        out.append(pick)
        ids.append(np.full(pick.size, k))
        got += pick.size
        k += 1
    return np.concatenate(out), np.concatenate(ids)


@dataclass
class Draws:
    """One realization: the truth and every sample of the Landscape Check (Parts 1-2)."""

    r: int
    s: np.ndarray
    truth: np.ndarray  # (K, n_blocks)
    cells: dict[str, np.ndarray]
    clusters: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    perms: dict[tuple[str, int], np.ndarray]
    cal_cells: dict[str, np.ndarray]
    cal_clusters: dict[str, np.ndarray]
    cal_labels: dict[str, np.ndarray]


def class_draws(ls: ClassLandscape, r: int) -> Draws:
    sc = ls.sc
    rng = np.random.default_rng([SEED, 7, r])
    s = sc.sampler.sample(rng, 1)[0].ravel()
    sd_n = math.sqrt(sc.nugget)
    mu = sc.mu.ravel()
    lim = sc.limits
    y_struct = mu + s
    frac = np.stack(
        [
            interval_probability((lim[j] - y_struct) / sd_n, (lim[j + 1] - y_struct) / sd_n)
            for j in range(len(CLASSES))
        ]
    )
    truth = np.stack([ls.layout.bmat @ frac[j] for j in range(len(CLASSES))])
    n_max = max(SIZES)
    cells, clusters = {}, {}
    for d in DESIGNS:
        cells[d], clusters[d] = sample_cells(d, n_max, rng)
    eps = {d: rng.standard_normal(n_max) for d in DESIGNS}
    perms = {(d, n): rng.permutation(n)[:N_FIT] for d in DESIGNS for n in SIZES}
    labels = {
        d: np.searchsorted(sc.thresholds, mu[cells[d]] + s[cells[d]] + sd_n * eps[d], side="left")
        for d in DESIGNS
    }
    rng_c = np.random.default_rng([SEED, 8, r])  # the Part 2 calibration samples
    cal_cells, cal_clusters = {}, {}
    for d in DESIGNS:
        cal_cells[d], cal_clusters[d] = sample_cells(d, n_max, rng_c)
    cal_eps = {d: rng_c.standard_normal(n_max) for d in DESIGNS}
    cal_labels = {
        d: np.searchsorted(
            sc.thresholds, mu[cal_cells[d]] + s[cal_cells[d]] + sd_n * cal_eps[d], side="left"
        )
        for d in DESIGNS
    }
    return Draws(r, s, truth, cells, clusters, labels, perms, cal_cells, cal_clusters, cal_labels)


def olofsson_landscape(
    strata_share: np.ndarray, stratum: np.ndarray, is_k: np.ndarray
) -> tuple[float, float]:
    """Post-stratified share of a class over the landscape and its SE (strata = mapped class).
    Strata with fewer than 2 points are merged into the largest stratum that has 2 or more."""
    w = strata_share.astype(float).copy()
    nh = np.bincount(stratum, minlength=w.size).astype(float)
    sh = np.bincount(stratum, weights=is_k.astype(float), minlength=w.size)
    big = nh >= 2
    if not big.any() or nh.sum() < 2:
        return math.nan, math.nan
    tgt = int(np.argmax(np.where(big, w, -1.0)))
    for arr in (w, nh, sh):
        arr[tgt] += arr[~big].sum()
    p = np.where(big, sh / np.where(big, nh, 1), 0.0)
    est = float(np.sum(np.where(big, w * p, 0.0)))
    var = float(np.sum(np.where(big, w**2 * p * (1 - p) / np.where(big, nh - 1, 1), 0.0)))
    return est, math.sqrt(var)


# ---------------------------------------------------------------------------
# Miscalibrated maps and calibrators
# ---------------------------------------------------------------------------


def transform_rare(p: np.ndarray, shift: float = 0.0, slope: float = 1.0) -> np.ndarray:
    """Miscalibrate the last (rarest) class on the logit scale; rescale the others (round 3)."""
    pc = np.clip(p[-1], 1e-12, 1 - 1e-12)
    new = special.expit(slope * special.logit(pc) + shift)
    out = p.copy()
    out[:-1] = p[:-1] * ((1.0 - new) / np.maximum(1.0 - pc, 1e-12))
    out[-1] = new
    return out


def miscalibrate(p_raw: np.ndarray, kind: str) -> np.ndarray:
    """The map users get: round 4's miscalibrations, or Part 2's combination of two."""
    if kind == "combined":
        p = psu.temperature_scale(transform_rare(p_raw, shift=RARE_SHIFT), TEMPERATURE, class_dim=0)
    elif kind == "offset":
        p = transform_rare(p_raw, shift=RARE_SHIFT)
    elif kind == "slope":
        p = transform_rare(p_raw, slope=RARE_SLOPE)
    elif kind == "scale":
        p = psu.temperature_scale(p_raw, TEMPERATURE, class_dim=0)
    else:
        raise ValueError(kind)
    return clip_norm(np.asarray(p))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=0, keepdims=True)


def fit_temperature(lp: np.ndarray, y: np.ndarray) -> dict:
    idx = np.arange(y.size)

    def nll(log_t: float) -> float:
        z = lp / math.exp(log_t)
        return float(-(z[y, idx] - special.logsumexp(z, axis=0)).sum())

    res = optimize.minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
    return {"T": math.exp(res.x)}


def apply_temperature(par: dict, lp: np.ndarray) -> np.ndarray:
    return softmax(lp / par["T"])


VEC_RIDGE = 1.0


def fit_vector(lp: np.ndarray, y: np.ndarray) -> dict:
    k, _ = lp.shape
    Y = np.eye(k)[y].T

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        a, b = theta[:k], theta[k:]
        z = a[:, None] * lp + b[:, None]
        lse = special.logsumexp(z, axis=0)
        P = np.exp(z - lse)
        val = float(
            -(z * Y).sum() + lse.sum() + 0.5 * VEC_RIDGE * (((a - 1) ** 2).sum() + (b**2).sum())
        )
        G = P - Y
        grad = np.concatenate(
            [(G * lp).sum(axis=1) + VEC_RIDGE * (a - 1), G.sum(axis=1) + VEC_RIDGE * b]
        )
        return val, grad

    res = optimize.minimize(
        f, np.concatenate([np.ones(k), np.zeros(k)]), jac=True, method="L-BFGS-B"
    )
    return {"a": res.x[:k].tolist(), "b": res.x[k:].tolist()}


def apply_vector(par: dict, lp: np.ndarray) -> np.ndarray:
    return softmax(np.asarray(par["a"])[:, None] * lp + np.asarray(par["b"])[:, None])


CALIBRATORS: dict[str, tuple[Callable[..., dict], Callable[..., np.ndarray]]] = {
    "temperature": (fit_temperature, apply_temperature),
    "vector": (fit_vector, apply_vector),
}

# map name → (miscalibration or None, calibrator or None)
CLASS_MAPS: dict[str, tuple[str | None, str | None]] = {
    "M1": (None, None),
    "M2": ("combined", "vector"),
    "M3": ("combined", "temperature"),
    "M4": ("offset", "vector"),
    "M5": ("slope", "vector"),
    "M6": ("scale", "vector"),
}
