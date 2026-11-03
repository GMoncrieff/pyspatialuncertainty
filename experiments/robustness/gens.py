"""Landscape generators of the robustness study.

Every generator gives a world (a fixed part: the landscape side, blocks and, for the latent
family, the map) and, per realization, the truth and every sample (``sh.Draws``) together
with the probability map that the user gets before any miscalibration.

Thresholded latent family (``kind = "latent"``). A latent Y = μ + η is cut into classes, as
in round 6; the map p_k = Φ(t_k − μ) − Φ(t_{k−1} − μ) is exactly calibrated whatever the
correlation of η, as long as η is N(0, 1) at every cell. η is a structured field (sill 0.8)
plus a point-scale Gaussian nugget (0.2). Truth and labels come from
``round6.world.class_draws``, so ``base`` reproduces round 6 exactly. The variants change one
thing each: the structured correlogram (``smooth``, ``nested``, ``aniso``, ``nonstat``), the
classes (``rare6``) or the signal-to-noise ratio of the map (``sharp``, ``blurry``). μ uses
the same seed in every variant, so the maps differ only by their scale.

Classifier worlds (``kind = "classifier"``). Four fixed covariate fields C1–C4 and fixed
nonlinear class utilities u_k(C) with intercepts tuned to the prevalences. The structured
class S(s) is

* ``classifier``: argmax_k(u_k + G_k), with G_k Gumbel noise obtained from independent
  exponential Gaussian fields (range 32) through Φ, so P(S = k) = softmax(u)_k exactly;
* ``potts``: a Potts Markov field P(S) ∝ exp(Σ_s u_S(s)(s) + β Σ_{s~t} 1{S(s) = S(t)})
  on the 4-neighbour grid (β = 0.8, below the critical 1.005 of three states), drawn by 200
  checkerboard Gibbs sweeps from an independent draw of softmax(u).

A point's label is S(cell), or with probability 0.1 a fresh draw from softmax(u) (point-scale
salt-and-pepper noise), so the area fraction of class k in a cell is 0.9·1{S = k} +
0.1·softmax(u)_k. The map is a multinomial logistic regression linear in [1, C1, ..., C4],
fitted to labels at a separate simple random training sample of the same realization (the
first ``n_train`` of 5,000 cells), so it is never exactly calibrated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from scipy import fft, optimize, special

import pyspatialuncertainty as psu
from experiments.round5 import shared as sh
from experiments.round6 import world as wd
from pyspatialuncertainty.synthetic import LatentClassScenario, SubGridLayout
from pyspatialuncertainty.synthetic.grf import CirculantEmbedding
from pyspatialuncertainty.variogram.models import VariogramModel

NUGGET_MIX = 0.1  # classifier worlds: probability that a point label is salt-and-pepper
BETA = 0.8  # Potts interaction
SWEEPS = 200  # Gibbs sweeps per Potts realization
COV_RANGES = (128.0, 64.0, 32.0, 16.0)  # covariate fields C1-C4 (exponential ranges)
NOISE_RANGE = 32.0  # Gumbel noise fields of ``classifier``
N_TRAIN_MAX = 5000
N_TRAIN = 1000  # core training-sample size of the classifier maps
RIDGE = 1e-3  # tiny ridge of the logistic map (exists even when a class has few labels)


@dataclass(frozen=True)
class Gen:
    name: str
    gid: int
    kind: str  # "latent" or "classifier"
    distance: str  # "anchor", "near", "middle" or "far" from round 6's simulator
    prevalence: tuple[float, ...] = sh.PREVALENCE
    latent: VariogramModel = field(default_factory=lambda: sh.CORRELOGRAM)
    sampler: str = "iso"  # "iso", "aniso" or "nonstat"
    snr: float = 1.0
    truth: str = ""  # classifier worlds: "gumbel" or "potts"

    @property
    def classes(self) -> tuple[str, ...]:
        return tuple("abcdefghij"[: len(self.prevalence)])

    @property
    def roles(self) -> dict[str, str]:
        """Class role by prevalence: the commonest, the rarest and the others."""
        order = np.argsort(self.prevalence)
        out = dict.fromkeys(self.classes, "mid")
        out[self.classes[int(order[-1])]] = "common"
        out[self.classes[int(order[0])]] = "rare"
        return out


GENS: dict[str, Gen] = {
    g.name: g
    for g in (
        Gen("base", 0, "latent", "anchor"),
        Gen(
            "smooth", 1, "latent", "middle",
            latent=psu.vgm(0.8, "Gau", 96.0 / math.sqrt(3.0), nugget=0.2),
        ),
        Gen(
            "nested", 2, "latent", "middle",
            latent=psu.vgm(0.4, "Exp", 4.0, nugget=0.2) + psu.vgm(0.4, "Exp", 128.0),
        ),
        Gen("aniso", 3, "latent", "middle", sampler="aniso"),
        Gen("nonstat", 4, "latent", "middle", sampler="nonstat"),
        Gen("rare6", 5, "latent", "near", prevalence=(0.40, 0.25, 0.15, 0.10, 0.09, 0.01)),
        Gen("sharp", 6, "latent", "near", snr=4.0),
        Gen("blurry", 7, "latent", "near", snr=0.25),
        Gen("classifier", 8, "classifier", "far", truth="gumbel"),
        Gen("potts", 9, "classifier", "far", truth="potts"),
    )
}  # fmt: skip


# ---------------------------------------------------------------------------
# Samplers of the structured latent noise
# ---------------------------------------------------------------------------


class AnisoEmbedding:
    """Circulant-embedding sampler of a stationary anisotropic exponential field on n × n
    cells: C(d) = psill·exp(−|R d| / ranges), ranges (major, minor) along ``angle`` degrees."""

    def __init__(self, n: int, psill: float, major: float, minor: float, angle: float):
        self.n = n
        a = math.radians(angle)
        m = fft.next_fast_len(2 * n)
        while True:
            i = np.arange(m)
            off = np.where(i <= m // 2, i, i - m).astype(np.float64)
            dy, dx = off[:, None], off[None, :]
            u = (dx * math.cos(a) + dy * math.sin(a)) / major
            v = (-dx * math.sin(a) + dy * math.cos(a)) / minor
            base = psill * np.exp(-np.hypot(u, v))
            eig = fft.fft2(base).real
            if eig.min() >= -1e-10 * eig.max():
                break
            m = fft.next_fast_len(2 * m)
        self.m = m
        self._sqrt_eig = np.sqrt(np.clip(eig, 0.0, None) / (m * m))

    def sample(self, rng: np.random.Generator, k: int = 1) -> np.ndarray:
        out = np.empty((k, self.n, self.n))
        j = 0
        while j < k:
            z = rng.standard_normal((self.m, self.m)) + 1j * rng.standard_normal((self.m, self.m))
            y = fft.fft2(self._sqrt_eig * z)
            out[j] = y.real[: self.n, : self.n]
            if j + 1 < k:
                out[j + 1] = y.imag[: self.n, : self.n]
            j += 2
        return out


class BlendEmbedding:
    """cos θ(x)·S₁ + sin θ(x)·S₂ with θ = (π/2)·x/n: exponential fields of ranges r₁ (west)
    and r₂ (east), the same variance everywhere, a range that changes across the landscape."""

    def __init__(self, n: int, psill: float, r1: float, r2: float):
        self.e1 = CirculantEmbedding((n, n), 1.0, lambda h: psill * np.exp(-h / r1))
        self.e2 = CirculantEmbedding((n, n), 1.0, lambda h: psill * np.exp(-h / r2))
        th = 0.5 * math.pi * (np.arange(n) + 0.5) / n
        self.c, self.s = np.cos(th)[None, :], np.sin(th)[None, :]

    def sample(self, rng: np.random.Generator, k: int = 1) -> np.ndarray:
        return self.c * self.e1.sample(rng, k) + self.s * self.e2.sample(rng, k)


@dataclass(frozen=True)
class CustomLatent(LatentClassScenario):
    """``LatentClassScenario`` with an anisotropic or non-stationary structured noise (sill
    0.8); ``latent`` only provides the nugget share."""

    kind: str = "aniso"

    @cached_property
    def sampler(self) -> AnisoEmbedding | BlendEmbedding:  # type: ignore[override]
        n = self.layout.n
        psill = 1.0 - self.nugget
        if self.kind == "aniso":
            return AnisoEmbedding(n, psill, 64.0, 16.0, 30.0)
        if self.kind == "nonstat":
            return BlendEmbedding(n, psill, 8.0, 128.0)
        raise ValueError(self.kind)


# ---------------------------------------------------------------------------
# Classifier worlds: covariates, utilities, Potts sampler, logistic map
# ---------------------------------------------------------------------------


def covariates(n: int) -> np.ndarray:
    """Four fixed standardized exponential fields (4, n²)."""
    out = []
    for j, r in enumerate(COV_RANGES):
        emb = CirculantEmbedding((n, n), 1.0, lambda h, r=r: np.exp(-h / r))
        f = emb.sample(np.random.default_rng([sh.SEED, 40, j, n]), 1)[0].ravel()
        out.append((f - f.mean()) / f.std())
    return np.stack(out)


def utilities(C: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Nonlinear class utilities (3, n²): saturation, a quadratic, an interaction and a
    threshold niche for the rare class; b are the intercepts of classes b and c."""
    c1, c2, c3, c4 = C
    ub = 1.5 * np.tanh(2.0 * c1) - 0.8 * c2**2 + 0.6 * c3 + b[0]
    uc = 3.0 * special.expit(4.0 * (c2 - 1.0)) + c3 * c4 - 0.5 * c1**2 + b[1]
    return np.stack([np.zeros_like(c1), ub, uc])


def softmax(u: np.ndarray) -> np.ndarray:
    return sh.softmax(u)


def gibbs_potts(
    u: np.ndarray,
    n: int,
    rng: np.random.Generator,
    *,
    beta: float = BETA,
    sweeps: int = SWEEPS,
    trace: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Checkerboard Gibbs sampler of the Potts field with external field u (K, n²), started
    from an independent draw of softmax(u). Returns S (n²,) and, with ``trace``, the share of
    equal neighbour pairs after every sweep."""
    k_cls = u.shape[0]
    S = np.argmax(u + rng.gumbel(size=u.shape), axis=0).astype(np.int8).reshape(n, n)
    flat = S.reshape(-1)
    ii, jj = np.indices((n, n))
    sites = [np.flatnonzero(((ii + jj) % 2 == p).ravel()) for p in (0, 1)]
    us = [u[:, s] for s in sites]
    agree: list[float] = []
    cnt = np.empty((k_cls, n, n), dtype=np.float32)
    for _ in range(sweeps):
        for p in (0, 1):
            cnt.fill(0.0)
            for k in range(k_cls):
                eq = k == S
                c = cnt[k]
                c[1:] += eq[:-1]
                c[:-1] += eq[1:]
                c[:, 1:] += eq[:, :-1]
                c[:, :-1] += eq[:, 1:]
            lg = us[p] + beta * cnt.reshape(k_cls, -1)[:, sites[p]]
            flat[sites[p]] = np.argmax(lg + rng.gumbel(size=lg.shape), axis=0)
        if trace:
            same = (S[1:] == S[:-1]).sum() + (S[:, 1:] == S[:, :-1]).sum()
            agree.append(float(same) / (2 * n * (n - 1)))
    return flat.astype(np.int64), agree


def gumbel_fields(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k independent Gumbel fields (k, n²) from unit exponential Gaussian fields through Φ."""
    emb = CirculantEmbedding((n, n), 1.0, lambda h: np.exp(-h / NOISE_RANGE))
    w = emb.sample(rng, k).reshape(k, -1)
    return -np.log(-special.log_ndtr(w))


def structured(g: Gen, n: int, u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if g.truth == "gumbel":
        return np.argmax(u + gumbel_fields(n, u.shape[0], rng), axis=0)
    return gibbs_potts(u, n, rng)[0]


def tune_intercepts(g: Gen, n: int, C: np.ndarray) -> np.ndarray:
    """Intercepts of classes b and c giving the target prevalences: of softmax(u) for
    ``classifier``; of the Potts field (fixed-seed Gibbs runs at this side) for ``potts``."""
    target = np.asarray(g.prevalence)
    b = np.zeros(2)
    for _ in range(200):
        m = softmax(utilities(C, b)).mean(axis=1)
        step = np.log(target[1:] / m[1:]) - math.log(target[0] / m[0])
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    if g.truth == "potts":
        rng = np.random.default_rng([sh.SEED, 41, n])
        for _ in range(5):
            S = gibbs_potts(utilities(C, b), n, rng, sweeps=120)[0]
            m = np.bincount(S, minlength=3) / S.size
            b += 0.8 * (np.log(target[1:] / m[1:]) - math.log(target[0] / m[0]))
    return b


def fit_logistic(F: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Multinomial logistic regression (features F with an intercept column) with a tiny
    ridge; returns the coefficients (q, k)."""
    q = F.shape[1]
    Y = np.eye(k)[y]

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        W = w.reshape(q, k)
        z = F @ W
        lse = special.logsumexp(z, axis=1)
        P = np.exp(z - lse[:, None])
        val = float(lse.sum() - (z * Y).sum() + 0.5 * RIDGE * (W**2).sum())
        return val, (F.T @ (P - Y) + RIDGE * W).ravel()

    res = optimize.minimize(f, np.zeros(q * k), jac=True, method="L-BFGS-B")
    return res.x.reshape(q, k)


# ---------------------------------------------------------------------------
# Worlds and draws
# ---------------------------------------------------------------------------


@dataclass
class World:
    """What ``round6.run.setting`` needs (``n``, ``layout``) and the fixed parts of a world."""

    gen: Gen
    n: int
    layout: sh.Layout
    sc: LatentClassScenario | None = None  # latent family
    p_raw: np.ndarray | None = None  # latent family: the calibrated map (K, n²), unclipped
    C: np.ndarray | None = None  # classifier worlds: covariates (4, n²)
    b: np.ndarray | None = None  # classifier worlds: tuned intercepts
    p_star: np.ndarray | None = None  # classifier worlds: softmax(u) (K, n²)
    u: np.ndarray | None = None


def prepare(g: Gen, n: int) -> dict[str, list[float]]:
    """Parameters computed once per side (the tuned intercepts of the classifier worlds)."""
    if g.kind != "classifier":
        return {}
    return {"b": tune_intercepts(g, n, covariates(n)).tolist()}


def world(g: Gen, n: int, params: dict[str, list[float]] | None = None) -> World:
    layout = wd.make_layout(n)
    if g.kind == "latent":
        kw = {
            "layout": SubGridLayout(n=n, f=1),
            "latent": g.latent,
            "signal_range": sh.SIGNAL_RANGE,
            "prevalence": g.prevalence,
            "classes": g.classes,
            "snr": g.snr,
        }
        sc = LatentClassScenario(**kw) if g.sampler == "iso" else CustomLatent(**kw, kind=g.sampler)
        return World(g, n, layout, sc=sc, p_raw=sc.p_true.reshape(len(g.classes), -1))
    C = covariates(n)
    b = np.asarray(params["b"]) if params else tune_intercepts(g, n, C)
    u = utilities(C, b)
    return World(g, n, layout, C=C, b=b, p_star=softmax(u), u=u)


def _point_labels(
    S: np.ndarray, p_star: np.ndarray, cells: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """S at the cells, or with probability NUGGET_MIX a fresh draw from softmax(u)."""
    mix = rng.random(cells.size) < NUGGET_MIX
    cum = np.cumsum(p_star[:, cells], axis=0)
    fresh = np.minimum((rng.random(cells.size)[None, :] > cum).sum(axis=0), p_star.shape[0] - 1)
    return np.where(mix, fresh, S[cells])


def draws(w: World, r: int, n_train: int = N_TRAIN) -> tuple[sh.Draws, np.ndarray]:
    """One realization: the truth and samples, and the map before miscalibration (K, n²)."""
    g = w.gen
    if g.kind == "latent":
        assert w.p_raw is not None
        return wd.class_draws(w, r), w.p_raw  # type: ignore[arg-type]
    assert w.u is not None
    assert w.p_star is not None
    assert w.C is not None
    n, k_cls = w.n, len(g.classes)
    key = 100 + 10 * g.gid
    rng = np.random.default_rng(wd.seed(n, key, r))
    S = structured(g, n, w.u, rng)
    frac = (1.0 - NUGGET_MIX) * (S[None, :] == np.arange(k_cls)[:, None]) + NUGGET_MIX * w.p_star
    truth = np.stack([w.layout.bmat @ frac[j] for j in range(k_cls)])
    n_max = max(sh.SIZES)
    cells, clusters, labels = {}, {}, {}
    for d in sh.DESIGNS:
        cells[d], clusters[d] = wd.sample_cells(n, d, n_max, rng)
    for d in sh.DESIGNS:
        labels[d] = _point_labels(S, w.p_star, cells[d], rng)
    perms = {(d, k): rng.permutation(k)[: sh.N_FIT] for d in sh.DESIGNS for k in sh.SIZES}
    rng_c = np.random.default_rng(wd.seed(n, key + 1, r))
    cal_cells, cal_clusters, cal_labels = {}, {}, {}
    for d in sh.DESIGNS:
        cal_cells[d], cal_clusters[d] = wd.sample_cells(n, d, n_max, rng_c)
    for d in sh.DESIGNS:
        cal_labels[d] = _point_labels(S, w.p_star, cal_cells[d], rng_c)
    rng_t = np.random.default_rng(wd.seed(n, key + 2, r))
    train = rng_t.choice(n * n, N_TRAIN_MAX, replace=False)
    y_train = _point_labels(S, w.p_star, train, rng_t)[:n_train]
    F = np.column_stack([np.ones(n * n), w.C.T])
    W = fit_logistic(F[train[:n_train]], y_train, k_cls)
    p_map = softmax((F @ W).T)
    dr = sh.Draws(r, S, truth, cells, clusters, labels, perms, cal_cells, cal_clusters, cal_labels)
    return dr, p_map


def map_logits(p_raw: np.ndarray, state: str) -> np.ndarray:
    """Log-probabilities of the map users get: ``cal`` (as is), ``offset`` or ``scale``."""
    if state == "cal":
        return np.log(sh.clip_norm(p_raw))
    return np.log(sh.miscalibrate(p_raw, state))


STATES = ("cal", "offset", "scale")
