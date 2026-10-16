"""Full mapping pipelines for the round-3 workflow experiments (Stage 1).

A study area of 100 × 100 unit cells has three fixed covariates (exponential GRFs with
ranges 10, 25 and 50 cells) and a nonlinear signal of them that explains 60 % of the
variance. Each realization adds a residual field (exponential, range 15 cells, 20 %
nugget) and draws a labelled sample of n cells with one of three designs. A learner is
trained on the covariates (``covariates``) or on the covariates and the coordinates
(``location``):

* continuous: y = signal + residual; a ranger-style quantile regression forest gives the
  prediction and σ (the R script's QRF, reused from ``PipelineScenario``);
* categorical: a latent variable signal + residual is thresholded into classes with
  prevalences 0.67/0.30/0.03; a random-forest classifier gives class probabilities
  (clipped to [0.005, 0.995] and renormalised).

Four workflows turn the labelled sample into (a) out-of-sample residuals and (b) the map:

* ``holdout``: 70/30 random split; residuals and map from the 70 % model;
* ``holdout_refit``: the same residuals; map from a model refit on all points;
* ``cv``: 10-fold random cross-validation residuals; map refit on all points;
* ``cv_nndm``: 10-fold distance-matched cross-validation (a light version of kNNDM,
  Linnenbrink et al. 2024) residuals; map refit on all points.

Optional one-parameter recalibration is fitted on the workflow's residual set: a σ scale
factor (continuous) or a temperature (categorical).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy import optimize, stats
from scipy.spatial import cKDTree

import pyspatialuncertainty as psu
from pyspatialuncertainty.synthetic import PipelineScenario
from pyspatialuncertainty.synthetic.grf import CirculantEmbedding

N = 100
TILES = (10, 25, 50)
SIGNAL_SHARE = 0.6
RESID_RANGE = 15.0
NUGGET_SHARE = 0.2
PREVALENCE = (0.67, 0.30, 0.03)
N_SAMPLE = 300
N_FOLDS = 10
P_CLIP = 0.005
P_CLIP_CAL = 0.001
WORKFLOWS = ("holdout", "holdout_refit", "cv", "cv_nndm")
DESIGNS = ("random", "clustered", "roads")
LEARNERS = ("covariates", "location")


@dataclass(frozen=True)
class World:
    """Fixed covariates, signal, grid and blocks of the study area."""

    grid: psu.GridSpec
    template: xr.DataArray
    cx: np.ndarray
    cy: np.ndarray
    covariates: np.ndarray  # (cells, 3)
    signal: np.ndarray  # (cells,)
    resolved: list[psu.ResolvedBlock]
    groups: tuple[str, ...]
    sampler: CirculantEmbedding  # structured residual, variance (1 - share) * (1 - nugget)
    resid_struct_var: float
    nugget_var: float
    thresholds: np.ndarray  # categorical class thresholds of the latent variable


def make_world(seed: int = 20261016) -> World:
    rng = np.random.default_rng(seed)
    c = np.arange(N) + 0.5
    template = xr.DataArray(np.zeros((N, N)), coords={"y": c[::-1], "x": c}, dims=("y", "x"))
    grid = psu.GridSpec.from_dataarray(template)
    cx, cy = grid.cell_centres(np.arange(grid.size))
    covs = []
    for rng_ in (10.0, 25.0, 50.0):
        ce = CirculantEmbedding((N, N), 1.0, lambda h, r=rng_: np.exp(-h / r))
        f = ce.sample(rng, 1)[0].ravel()
        covs.append((f - f.mean()) / f.std())
    C = np.column_stack(covs)
    s = C[:, 0] + 0.8 * np.sin(1.5 * C[:, 1]) + 0.6 * (C[:, 2] ** 2 - 1.0) / math.sqrt(2.0)
    s = (s - s.mean()) / s.std() * math.sqrt(SIGNAL_SHARE)
    resid_var = 1.0 - SIGNAL_SHARE
    struct_var = resid_var * (1.0 - NUGGET_SHARE)
    sampler = CirculantEmbedding((N, N), 1.0, lambda h: struct_var * np.exp(-h / RESID_RANGE))
    # class thresholds of the latent variable L = signal + residual (marginal N(s_c, resid_var))
    sd = math.sqrt(resid_var)

    def share_below(t: float) -> float:
        return float(np.mean(stats.norm.cdf((t - s) / sd)))

    cum = np.cumsum(PREVALENCE)[:-1]
    thresholds = np.array(
        [optimize.brentq(lambda t, q=q: share_below(t) - q, -10, 10) for q in cum]
    )
    blocks: list[psu.blocks.Block] = []
    groups: list[str] = []
    for t in TILES:
        for b in psu.blocks.tiles(grid, t, drop_partial=True).blocks:
            blocks.append(psu.blocks.Block(id=f"tile{t}_{b.id}", cells=b.cells))
            groups.append(f"tile_{t}")
    blocks.append(psu.blocks.domain(grid).blocks[0])
    groups.append("domain")
    bs = psu.blocks.BlockSet(grid, tuple(blocks), "custom")
    resolved = psu.resolve_blocks(bs, np.ones(grid.size, dtype=bool))
    return World(
        grid,
        template,
        cx,
        cy,
        C,
        s,
        resolved,
        tuple(groups),
        sampler,
        struct_var,
        resid_var * NUGGET_SHARE,
        thresholds,
    )


# ---------------------------------------------------------------------------
# Truth and labelled samples
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Realization:
    truth: np.ndarray  # continuous: value per cell; categorical: class index per cell
    sample: np.ndarray  # sampled cells
    observed: np.ndarray  # continuous: noisy observation; categorical: class index


def simulate(world: World, mode: str, design: str, rng: np.random.Generator) -> Realization:
    struct = world.sampler.sample(rng, 1)[0].ravel()
    sample = draw_sample(world, design, N_SAMPLE, rng)
    if mode == "continuous":
        truth = world.signal + struct
        obs = truth[sample] + math.sqrt(world.nugget_var) * rng.standard_normal(sample.size)
        return Realization(truth, sample, obs)
    latent = world.signal + struct + math.sqrt(world.nugget_var) * rng.standard_normal(struct.size)
    cls = np.searchsorted(world.thresholds, latent, side="left")
    return Realization(cls, sample, cls[sample])


def draw_sample(world: World, design: str, n: int, rng: np.random.Generator) -> np.ndarray:
    """Distinct sampled cells for the ``random``, ``clustered`` or ``roads`` design."""
    n_cells = world.grid.size
    if design == "random":
        return rng.choice(n_cells, n, replace=False)
    chosen: list[int] = []
    taken = np.zeros(n_cells, dtype=bool)
    if design == "clustered":
        per = n // 30
        while len(chosen) < n:
            cxy = rng.uniform(0, N, 2)
            near = np.flatnonzero((np.hypot(world.cx - cxy[0], world.cy - cxy[1]) <= 4.0) & ~taken)
            k = min(per, near.size, n - len(chosen))
            pick = rng.choice(near, k, replace=False)
            taken[pick] = True
            chosen.extend(pick.tolist())
        return np.array(chosen)
    if design == "roads":
        cand = np.zeros(n_cells, dtype=bool)
        for _ in range(3):
            p0 = rng.uniform(0, N, 2)
            th = rng.uniform(0, math.pi)
            dist = np.abs((world.cx - p0[0]) * math.sin(th) - (world.cy - p0[1]) * math.cos(th))
            cand |= dist <= 2.0
        return rng.choice(np.flatnonzero(cand), n, replace=False)
    raise ValueError(f"unknown design {design!r}")


# ---------------------------------------------------------------------------
# Learners
# ---------------------------------------------------------------------------


_QRF = PipelineScenario(n_trees=250, max_features=2, min_node_size=5)


def features(world: World, learner: str) -> np.ndarray:
    if learner == "covariates":
        return world.covariates
    return np.column_stack([world.covariates, world.cx, world.cy])


def fit_predict(
    mode: str, X_tr: np.ndarray, y_tr: np.ndarray, X_new: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous: (mean, σ). Categorical: (probabilities (n_new × K), empty)."""
    rng = np.random.default_rng(seed)
    if mode == "continuous":
        mean, sd = _QRF._fit_predict(X_tr, y_tr, X_new, rng, cv=False)
        return mean, np.maximum(sd, 1e-6)
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=250, max_features=2, min_samples_leaf=3, random_state=seed, n_jobs=1
    )
    rf.fit(X_tr, y_tr)
    pk = rf.predict_proba(X_new)
    p = np.zeros((X_new.shape[0], len(PREVALENCE)))
    p[:, rf.classes_.astype(int)] = pk
    return clip_probabilities(p, P_CLIP), np.empty(0)


def clip_probabilities(p: np.ndarray, eps: float) -> np.ndarray:
    q = np.clip(p, eps, 1.0 - eps)
    return q / q.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------


def random_folds(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n) % k


def nndm_folds(
    xy: np.ndarray, xy_pred: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, str]:
    """Distance-matched k-fold assignment (a light version of kNNDM).

    Among random folds and k-means spatial folds with a range of cluster counts, pick the
    configuration whose test-to-training nearest-neighbour distances best match (in
    Wasserstein-1 distance) the prediction-to-sample nearest-neighbour distances.
    """
    from sklearn.cluster import KMeans

    n = xy.shape[0]
    pred = xy_pred[rng.choice(xy_pred.shape[0], min(4000, xy_pred.shape[0]), replace=False)]
    d_pred = cKDTree(xy).query(pred)[0]
    candidates: list[tuple[str, np.ndarray]] = [("random", random_folds(n, k, rng))]
    for q in sorted({round(float(v)) for v in np.geomspace(k, n / 3, 7)}):
        lab = KMeans(n_clusters=q, n_init=1, random_state=int(rng.integers(2**31))).fit_predict(xy)
        sizes = np.bincount(lab, minlength=q)
        folds_of_cluster = np.empty(q, dtype=int)
        load = np.zeros(k)
        for cl in np.argsort(-sizes):
            f = int(np.argmin(load))
            folds_of_cluster[cl] = f
            load[f] += sizes[cl]
        candidates.append((f"kmeans{q}", folds_of_cluster[lab]))
    best: tuple[float, str, np.ndarray] | None = None
    for name, folds in candidates:
        d_cv = np.empty(n)
        for f in range(k):
            test = folds == f
            if test.all() or not test.any():
                d_cv[test] = np.nan
                continue
            d_cv[test] = cKDTree(xy[~test]).query(xy[test])[0]
        w = stats.wasserstein_distance(d_cv[np.isfinite(d_cv)], d_pred)
        if best is None or w < best[0]:
            best = (w, name, folds)
    assert best is not None
    return best[2], best[1]


# ---------------------------------------------------------------------------
# Workflows and calibration
# ---------------------------------------------------------------------------


@dataclass
class Residuals:
    """Out-of-sample predictions at the validation (or all, under CV) sample points."""

    cells: np.ndarray
    observed: np.ndarray
    pred: np.ndarray  # continuous: mean; categorical: probabilities (n × K)
    sd: np.ndarray  # continuous σ (empty for categorical)


@dataclass
class MapOutput:
    pred: np.ndarray  # continuous: mean per cell; categorical: probabilities (cells × K)
    sd: np.ndarray  # continuous σ per cell (empty for categorical)


def run_workflows(
    world: World,
    mode: str,
    learner: str,
    real: Realization,
    rng: np.random.Generator,
    seed: int,
) -> tuple[dict[str, tuple[Residuals, MapOutput]], dict[str, Any]]:
    """Residual sets and maps for the four workflows (folds shared across learners)."""
    X_all = features(world, learner)
    xy = np.column_stack([world.cx, world.cy])
    s, obs = real.sample, real.observed
    n = s.size
    perm = rng.permutation(n)
    tr, va = np.sort(perm[: int(0.7 * n)]), np.sort(perm[int(0.7 * n) :])
    f_random = random_folds(n, N_FOLDS, rng)
    f_nndm, nndm_choice = nndm_folds(xy[s], xy, N_FOLDS, rng)
    seeds = iter(np.random.default_rng(seed).integers(0, 2**31 - 1, 64))

    def fit(train_idx: np.ndarray, new_cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return fit_predict(
            mode, X_all[s[train_idx]], obs[train_idx], X_all[new_cells], int(next(seeds))
        )

    # holdout model: predictions at the validation points and the map
    both = np.concatenate([s[va], np.arange(world.grid.size)])
    pr, sd = fit(tr, both)
    nv = va.size
    hold_res = Residuals(s[va], obs[va], pr[:nv], sd[:nv] if sd.size else sd)
    hold_map = MapOutput(pr[nv:], sd[nv:] if sd.size else sd)
    full_pr, full_sd = fit(np.arange(n), np.arange(world.grid.size))
    full_map = MapOutput(full_pr, full_sd)

    def oof(folds: np.ndarray) -> Residuals:
        pred = np.empty((n, len(PREVALENCE))) if mode == "categorical" else np.empty(n)
        sdv = np.empty(n) if mode == "continuous" else np.empty(0)
        for f in range(N_FOLDS):
            test = np.flatnonzero(folds == f)
            if test.size == 0:
                continue
            p_, s_ = fit(np.flatnonzero(folds != f), s[test])
            pred[test] = p_
            if mode == "continuous":
                sdv[test] = s_
        return Residuals(s, obs, pred, sdv)

    out = {
        "holdout": (hold_res, hold_map),
        "holdout_refit": (hold_res, full_map),
        "cv": (oof(f_random), full_map),
        "cv_nndm": (oof(f_nndm), full_map),
    }
    return out, {"nndm_choice": nndm_choice}


def calibrate(mode: str, res: Residuals, mp: MapOutput) -> tuple[Residuals, MapOutput, float]:
    """One-parameter recalibration fitted on the residual set, applied to residuals and map.

    Continuous: σ is multiplied by √mean(z²). Categorical: temperature scaling with T
    minimising the log-loss of the residual set.
    """
    if mode == "continuous":
        z = (res.observed - res.pred) / res.sd
        c = float(np.sqrt(np.mean(z**2)))
        return (
            Residuals(res.cells, res.observed, res.pred, res.sd * c),
            MapOutput(mp.pred, mp.sd * c),
            c,
        )
    lab = res.observed.astype(int)

    def nll(log_t: float) -> float:
        q = psu.temperature_scale(res.pred, math.exp(log_t), class_dim=1)
        return -float(np.mean(np.log(q[np.arange(lab.size), lab])))

    log_t = optimize.minimize_scalar(nll, bounds=(math.log(0.2), math.log(5.0)), method="bounded").x
    t = math.exp(log_t)
    p_res = clip_probabilities(psu.temperature_scale(res.pred, t, class_dim=1), P_CLIP_CAL)
    p_map = clip_probabilities(psu.temperature_scale(mp.pred, t, class_dim=1), P_CLIP_CAL)
    return Residuals(res.cells, res.observed, p_res, res.sd), MapOutput(p_map, mp.sd), t
