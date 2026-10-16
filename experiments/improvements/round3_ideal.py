"""Round 3, Stages 2-3: the unknown-mean correction on idealized inputs.

Stage 2 (``--stage core``) runs the existing round-1/2 scenarios plus new ones with a
simple random validation sample:

* continuous: baseline, n100, range64, nugget0.9, sd_scale0.75, spherical, pipeline
  (the R-script pipeline) and ``bias`` (the map over-predicts by 0.3 σ);
* categorical: baseline, n800, temperature0.7, range64 and three scenarios with a 3 %
  class: ``rare3`` (calibrated), ``rare3_shift`` (logit of the rare class − 0.7) and
  ``rare3_omission`` (logit slope 1.3 for the rare class). Their probabilities are
  clipped to [0.001, 0.999] as a user would.

Stage 3 (``--stage designs``) draws the validation sample with other designs:
``stratified`` (mapped class, equal allocation), ``clustered`` (clusters of 10 points
within 4 cells), ``roads`` (within 2 cells of 3 random lines), ``half`` (western half),
``confident`` (∝ (max p)⁴, or σ⁻² for continuous maps) and ``informative`` (×5 inclusion
where the true class is the rare one; categorical only).

Validation sizes 50/100/200/400 (``SIZES``, 100 realizations, common random numbers
across sizes) are added for the scenarios in ``SIZED_CORE`` and the (map, design) pairs in
``SIZED_DESIGNS``; the other cells use n = 300.

Arms, on shared realizations and Monte Carlo pairs: ``current`` (zero-mean ML, known
mean), ``ok`` (REML + unknown constant mean), ``ok_strata`` (mapped-class means),
``ok_ml0`` (the correction on the zero-mean ML fit), ``oracle`` (true σ and true
correlogram, known mean), ``oracle_ok`` (true correlogram, the method's σ, unknown
mean), ``ppi`` and ``olofsson``. The existing scenarios keep the round-2 seeds, so
``current`` reproduces round 2's ``ml_zero_mean``.

Usage::

    uv run --group experiments python -m experiments.improvements.round3_ideal --stage core
    uv run --group experiments python -m experiments.improvements.round3_ideal --stage designs
    uv run python -m experiments.improvements.round3_ideal --summarize core
"""

from __future__ import annotations

import argparse
import functools
import math
import os
import time
from typing import Any

import numpy as np
import polars as pl
from scipy import linalg, special

import pyspatialuncertainty as psu
from experiments.common import chunks, run_parallel
from experiments.improvements import candidates as cand
from experiments.improvements.round3 import RESULTS3, metrics, nan_pair, safe
from experiments.improvements.run import BASE_CAT, BASE_CONT, SEED, TILES, latent_model
from pyspatialuncertainty.continuous import block_means
from pyspatialuncertainty.synthetic import (
    ErrorModelScenario,
    LatentClassScenario,
    SubGridLayout,
    evaluation_blocks,
)
from pyspatialuncertainty.synthetic.categorical import ClassRealization, interval_probability
from pyspatialuncertainty.synthetic.continuous import ErrorRealization

N_MC = 10_000
N_WORKERS = 4
P_CLIP = 1e-3
RARE3 = {"prevalence": (0.67, 0.30, 0.03), "clip": True}
CORE: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {
    # name: (round-2 scenario index for the seeds, parameters)
    "continuous": {
        "baseline": (0, BASE_CONT),
        "n100": (1, BASE_CONT | {"n_validation": 100}),
        "range64": (2, BASE_CONT | {"range": 64.0}),
        "nugget0.9": (3, BASE_CONT | {"nugget": 0.9}),
        "sd_scale0.75": (4, BASE_CONT | {"sd_scale": 0.75}),
        "spherical": (5, BASE_CONT | {"model": "Sph", "range": 24.0}),
        "pipeline": (6, {"pipeline": True}),
        "bias": (7, BASE_CONT | {"bias": 0.3}),
    },
    "categorical": {
        "baseline": (100, BASE_CAT),
        "n800": (101, BASE_CAT | {"n_validation": 800}),
        "temperature0.7": (102, BASE_CAT | {"temperature": 0.7}),
        "range64": (103, BASE_CAT | {"range": 64.0}),
        "rare3": (104, BASE_CAT | RARE3),
        "rare3_shift": (105, BASE_CAT | RARE3 | {"shift": -0.7}),
        "rare3_omission": (106, BASE_CAT | RARE3 | {"slope": 1.3}),
    },
}
DESIGNS = {
    "continuous": ("clustered", "roads", "half", "confident"),
    "categorical": ("stratified", "clustered", "roads", "half", "confident", "informative"),
}
DESIGN_MAPS = {"continuous": ("baseline",), "categorical": ("rare3", "rare3_shift")}
CLUSTER_SIZE = 10  # clustered design: clusters of 10 points within 4 cells
N_DEFAULT = 300  # validation size of the round-2 scenarios (seeds reproduce round 2)
SIZES = (50, 100, 200, 400)  # validation-size axis (common random numbers across sizes)
SIZED_CORE = {
    "continuous": ("baseline", "bias"),
    "categorical": ("baseline", "rare3", "rare3_shift", "rare3_omission"),
}
SIZED_DESIGNS = {  # (map, design) pairs of Stage 3 that get the size axis
    "continuous": tuple(("baseline", d) for d in DESIGNS["continuous"]),
    "categorical": tuple(
        ("rare3_shift", d) for d in ("stratified", "clustered", "roads", "confident")
    ),
}
N_REAL_SIZED = 100  # realizations of the cells with n != N_DEFAULT


# ---------------------------------------------------------------------------
# Scenario set-up
# ---------------------------------------------------------------------------


def transform_rare(p: np.ndarray, shift: float = 0.0, slope: float = 1.0) -> np.ndarray:
    """Miscalibrate the last (rarest) class on the logit scale; rescale the others."""
    pc = np.clip(p[-1], 1e-12, 1 - 1e-12)
    new = special.expit(slope * special.logit(pc) + shift)
    out = p.copy()
    out[:-1] = p[:-1] * ((1.0 - new) / np.maximum(1.0 - pc, 1e-12))
    out[-1] = new
    return out


@functools.cache
def setup(mode: str, name: str, n: int | None = None) -> dict[str, Any]:
    _, params = CORE[mode][name]
    if n is not None:
        params = params | {"n_validation": n}
    if params.get("pipeline"):
        from experiments.run_pipeline import make_blocks, make_scenario

        sc: Any = make_scenario({"trend_r2": 0.6756756756756757, "n_train": 200})
        bs, groups = make_blocks(sc, [10, 25, 50])
        resolved = psu.resolve_blocks(bs, np.ones(bs.grid.size, dtype=bool))
        cx, cy = bs.grid.cell_centres(np.arange(bs.grid.size))
        return {
            "sc": sc,
            "bs": bs,
            "resolved": resolved,
            "groups": tuple(groups),
            "cx": cx,
            "cy": cy,
            "n_cells": bs.grid.size,
            "pipeline": True,
        }
    layout = SubGridLayout(n=128, f=4)
    eb = evaluation_blocks(layout, TILES, max_per_size=32, n_polygons=6)
    resolved = psu.resolve_blocks(eb.blockset, np.ones(layout.grid.size, dtype=bool))
    cx, cy = layout.grid.cell_centres(np.arange(layout.grid.size))
    out: dict[str, Any] = {
        "eb": eb,
        "resolved": resolved,
        "groups": eb.group,
        "cx": cx,
        "cy": cy,
        "n_cells": layout.grid.size,
        "layout": layout,
    }
    if mode == "continuous":
        sc = ErrorModelScenario(
            layout=layout, correlation=latent_model(params), n_validation=params["n_validation"]
        )
        out |= {
            "sc": sc,
            "sd_user": layout.grid.values(sc.sigma_map(params["sd_scale"])),
            "sd_true": layout.grid.values(sc.sigma_map()),
            "true_model": sc.correlogram,
            "bias": float(params.get("bias", 0.0)),
        }
        return out
    sc = LatentClassScenario(
        layout=layout,
        latent=latent_model(params),
        n_validation=params["n_validation"],
        prevalence=params.get("prevalence", (0.6, 0.3, 0.1)),
    )
    k = len(sc.classes)
    p_true = sc.p_true.reshape(k, -1)
    if "shift" in params or "slope" in params:
        p_user = transform_rare(p_true, params.get("shift", 0.0), params.get("slope", 1.0))
    else:
        p_user = sc.probabilities(params["temperature"]).values.reshape(k, -1)
    if params.get("clip"):
        p_user = np.clip(p_user, P_CLIP, 1 - P_CLIP)
        p_user = p_user / p_user.sum(axis=0, keepdims=True)
    out |= {
        "sc": sc,
        "p_user": p_user,
        "sd_user": np.sqrt(p_user * (1 - p_user)),
        "sd_true": np.sqrt(p_true * (1 - p_true)),
        "pred": np.stack([block_means(resolved, p_user[j]) for j in range(k)]),
        "map_class": np.argmax(p_user, axis=0),
    }
    return out


@functools.cache
def oracle_corr(name: str, k: int) -> tuple[psu.CorrelationFunction, float]:
    sc = setup("categorical", name)["sc"]
    corr = sc.oracle_correlation(k)
    return corr, float(np.asarray(corr(np.array([1e-9])))[0])


# ---------------------------------------------------------------------------
# Validation sampling designs (Stage 3)
# ---------------------------------------------------------------------------


@functools.cache
def node_coords(n_sub: int, f: int) -> tuple[np.ndarray, np.ndarray]:
    lay = SubGridLayout(n=n_sub // f, f=f)
    return lay.sub_coords(np.arange(n_sub * n_sub))


def sample_nodes(
    design: str,
    layout: SubGridLayout,
    n: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
    strata: np.ndarray | None = None,
) -> np.ndarray:
    """Sub-grid node indices for a validation design."""
    n_nodes = layout.n_sub * layout.n_sub
    x, y = node_coords(layout.n_sub, layout.f)
    if design == "random":
        return rng.choice(n_nodes, n, replace=False)
    if weights is not None:
        return rng.choice(n_nodes, n, replace=False, p=weights / weights.sum())
    if design == "stratified":
        assert strata is not None
        codes = np.unique(strata)
        per = n // codes.size
        picks = [
            rng.choice(
                np.flatnonzero(strata == c), min(per, int((strata == c).sum())), replace=False
            )
            for c in codes
        ]
        idx = np.concatenate(picks)
        if idx.size < n:
            rest = np.setdiff1d(np.arange(n_nodes), idx)
            idx = np.concatenate([idx, rng.choice(rest, n - idx.size, replace=False)])
        return idx
    if design == "clustered":
        chosen: list[int] = []
        taken = np.zeros(n_nodes, dtype=bool)
        while len(chosen) < n:
            cxy = rng.uniform(0, layout.extent, 2)
            near = np.flatnonzero((np.hypot(x - cxy[0], y - cxy[1]) <= 4.0) & ~taken)
            pick = rng.choice(near, min(CLUSTER_SIZE, near.size, n - len(chosen)), replace=False)
            taken[pick] = True
            chosen.extend(pick.tolist())
        return np.array(chosen)
    if design == "roads":
        cand_ = np.zeros(n_nodes, dtype=bool)
        for _ in range(3):
            p0 = rng.uniform(0, layout.extent, 2)
            th = rng.uniform(0, math.pi)
            cand_ |= np.abs((x - p0[0]) * math.sin(th) - (y - p0[1]) * math.cos(th)) <= 2.0
        return rng.choice(np.flatnonzero(cand_), n, replace=False)
    if design == "half":
        return rng.choice(np.flatnonzero(x < layout.extent / 2), n, replace=False)
    raise ValueError(f"unknown design {design!r}")


def simulate_categorical(
    st: dict[str, Any], rng: np.random.Generator, design: str
) -> ClassRealization:
    sc: LatentClassScenario = st["sc"]
    lay = sc.layout
    s = sc.sampler.sample(rng, 1)[0]
    y_struct = lay.upsample(sc.mu) + s
    lim = sc.limits
    k = len(sc.classes)
    sd_n = math.sqrt(sc.nugget)
    frac = np.stack(
        [
            interval_probability((lim[j] - y_struct) / sd_n, (lim[j + 1] - y_struct) / sd_n)
            for j in range(k)
        ]
    )
    n = sc.n_validation
    node_cell = lay.cell_of_sub(np.arange(lay.n_sub * lay.n_sub))
    if design == "informative":
        y_all = y_struct.ravel() + sd_n * rng.standard_normal(y_struct.size)
        lab_all = np.searchsorted(sc.thresholds, y_all, side="left")
        idx = sample_nodes(design, lay, n, rng, weights=1.0 + 4.0 * (lab_all == k - 1))
        label_idx = lab_all[idx]
    else:
        weights = None
        if design == "confident":
            weights = np.max(st["p_user"], axis=0)[node_cell] ** 4
        idx = sample_nodes(
            design,
            lay,
            n,
            rng,
            weights=weights,
            strata=st["map_class"][node_cell] if design == "stratified" else None,
        )
        y_point = y_struct.ravel()[idx] + sd_n * rng.standard_normal(idx.size)
        label_idx = np.searchsorted(sc.thresholds, y_point, side="left")
    x, y = lay.sub_coords(idx)
    val = pl.DataFrame(
        {
            "x": x,
            "y": y,
            "cell": lay.cell_of_sub(idx),
            "label": pl.Series([sc.classes[i] for i in label_idx]),
        }
    )
    return ClassRealization(fraction_sub=frac, validation=val)


def simulate_continuous(
    st: dict[str, Any], rng: np.random.Generator, design: str
) -> ErrorRealization:
    sc: ErrorModelScenario = st["sc"]
    lay = sc.layout
    s = sc.sampler.sample(rng, 1)[0]
    error_sub = lay.upsample(sc.sigma_true) * s
    node_cell = lay.cell_of_sub(np.arange(lay.n_sub * lay.n_sub))
    weights = st["sd_user"][node_cell] ** -2.0 if design == "confident" else None
    idx = sample_nodes(design, lay, sc.n_validation, rng, weights=weights)
    x, y = lay.sub_coords(idx)
    cell = lay.cell_of_sub(idx)
    err = sc.sigma_true.ravel()[cell] * (
        s.ravel()[idx] + math.sqrt(sc.nugget) * rng.standard_normal(idx.size)
    )
    return ErrorRealization(
        error_sub=error_sub, validation=pl.DataFrame({"x": x, "y": y, "cell": cell, "error": err})
    )


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def gls_callable(
    rho: psu.CorrelationFunction,
    rho0: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    X: np.ndarray,
) -> cand.GLS:
    D = cand.distance_matrix(x, y)
    K = np.where(D > 0, np.asarray(rho(D)), rho0)
    K[np.diag_indices_from(K)] = 1.0 + 1e-10
    c = linalg.cho_factor(K, lower=True, check_finite=False)
    kx = linalg.cho_solve(c, X, check_finite=False)
    A = np.linalg.inv(X.T @ kx)
    G = A @ kx.T
    return cand.GLS(beta=G @ z, A=A, G=G)


def terms_callable(
    bmat: Any,
    sd: np.ndarray,
    xc: np.ndarray,
    rho: psu.CorrelationFunction,
    rho0: float,
    d_cells: np.ndarray,
    g: cand.GLS,
) -> tuple[np.ndarray, np.ndarray]:
    xb = np.asarray(bmat @ (sd[:, None] * xc))
    cb = np.asarray(bmat @ (sd[:, None] * np.where(d_cells > 0, np.asarray(rho(d_cells)), rho0)))
    shift = xb @ g.beta
    added = np.einsum("bi,ij,bj->b", xb, g.A, xb) - 2.0 * np.einsum("bq,qn,bn->b", xb, g.G, cb)
    return shift, added


def corrected(
    bmat: Any,
    sd: np.ndarray,
    xc: np.ndarray,
    model: Any,
    d_cells: np.ndarray,
    g: Any,
    variance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    shift, added = cand.unknown_mean_terms(bmat, sd, xc, model, d_cells, g)
    return shift, np.sqrt(np.maximum(variance + added, 0.0))


def class_arms(
    *,
    st: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    sd_map: np.ndarray,
    sd_true: np.ndarray,
    d_cells: np.ndarray,
    bmat: Any,
    Xs: np.ndarray | None,
    x_cells_strata: np.ndarray | None,
    oracle: tuple[Any, float, bool],
    mc_seed: Any,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Arms for one variable (continuous map or one class). ``oracle`` is (correlation,
    limit at 0, is_variogram_model)."""
    nb = len(st["resolved"])
    n = z.size
    ones_v, ones_c = np.ones((n, 1)), np.ones((sd_map.size, 1))
    m0 = safe(lambda: cand.likelihood_fit(x, y, z, mean="zero"))
    m1 = safe(lambda: cand.likelihood_fit(x, y, z, mean="constant"))
    m2 = None
    if Xs is not None:
        m2 = safe(lambda: cand.likelihood_fit(x, y, z, X=Xs)) if Xs.shape[1] > 1 else m1
    g0 = safe(lambda: cand.gls_terms(m0, x, y, z, ones_v)) if m0 is not None else None
    g1 = safe(lambda: cand.gls_terms(m1, x, y, z, ones_v)) if m1 is not None else None
    g2 = (
        safe(lambda: cand.gls_terms(m2, x, y, z, Xs)) if m2 is not None and Xs is not None else None
    )
    ocorr, orho0, o_is_model = oracle
    integrands: list[psu.Integrand] = []
    keys: list[str] = []
    for key, sd_, corr in (("m0", sd_map, m0), ("m1", sd_map, m1), ("m2", sd_map, m2)):
        if corr is not None:
            integrands.append(psu.Integrand(sd_, cand.correlation(corr)))
            keys.append(key)
    o_corr_fn = cand.correlation(ocorr) if o_is_model else ocorr
    integrands += [psu.Integrand(sd_true, o_corr_fn), psu.Integrand(sd_map, o_corr_fn)]
    keys += ["oracle", "oracle_user"]
    mc = psu.mc_block_variance(st["resolved"], integrands, n_mc=N_MC, seed=mc_seed)
    var = dict(zip(keys, mc.variance, strict=True))
    zero = np.zeros(nb)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["current"] = (zero, np.sqrt(np.maximum(var["m0"], 0))) if m0 is not None else nan_pair(nb)
    out["ok"] = corrected(bmat, sd_map, ones_c, m1, d_cells, g1, var["m1"]) if g1 else nan_pair(nb)
    out["ok_ml0"] = (
        corrected(bmat, sd_map, ones_c, m0, d_cells, g0, var["m0"]) if g0 else nan_pair(nb)
    )
    if x_cells_strata is not None:
        out["ok_strata"] = (
            corrected(bmat, sd_map, x_cells_strata, m2, d_cells, g2, var["m2"])
            if g2
            else nan_pair(nb)
        )
    out["oracle"] = (zero, np.sqrt(np.maximum(var["oracle"], 0)))
    if o_is_model:
        go = cand.gls_terms(ocorr, x, y, z, ones_v)
        out["oracle_ok"] = corrected(bmat, sd_map, ones_c, ocorr, d_cells, go, var["oracle_user"])
    else:
        go = gls_callable(ocorr, orho0, x, y, z, ones_v)
        shift, added = terms_callable(bmat, sd_map, ones_c, ocorr, orho0, d_cells, go)
        out["oracle_ok"] = (shift, np.sqrt(np.maximum(var["oracle_user"] + added, 0)))
    return out


def realization(
    mode: str,
    name: str,
    design: str,
    rng: np.random.Generator,
    n: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    st = setup(mode, name, n)
    bmat = block_matrix(mode, name)
    cx, cy = st["cx"], st["cy"]
    arms: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    if mode == "continuous":
        if st.get("pipeline"):
            real = st["sc"].simulate(rng)
            v = real.validation
            cells = real.sample_index
            e = (v["observed"] - v["predicted"]).to_numpy()
            s = v["sd"].to_numpy()
            grid = st["bs"].grid
            sd_map = grid.values(real.sd)
            sd_true = sd_map  # no oracle for the pipeline
            err = block_means(st["resolved"], grid.values(real.truth)) - block_means(
                st["resolved"], grid.values(real.prediction)
            )
            oracle = None
        else:
            real = (
                st["sc"].simulate(rng)
                if design == "random"
                else simulate_continuous(st, rng, design)
            )
            v = real.validation
            cells = v["cell"].to_numpy()
            sd_map, sd_true = st["sd_user"], st["sd_true"]
            bias_sub = st["bias"] * st["layout"].upsample(st["sc"].sigma_true)
            e = v["error"].to_numpy() - st["bias"] * st["sc"].sigma_true.ravel()[cells]
            s = sd_map[cells]
            err = st["eb"].block_means(real.error_sub - bias_sub)
            oracle = (st["true_model"], 0.0, True)
        x, y = v["x"].to_numpy(), v["y"].to_numpy()
        d_cells = np.hypot(cx[:, None] - x[None, :], cy[:, None] - y[None, :])
        z = e / s
        if oracle is None:  # pipeline: use the zero-mean fit as a stand-in, flagged in output
            oracle = (
                safe(lambda: cand.likelihood_fit(x, y, z, mean="zero")) or psu.Nugget(1.0),
                0.0,
                True,
            )
        res = class_arms(
            st=st,
            x=x,
            y=y,
            z=z,
            sd_map=sd_map,
            sd_true=sd_true,
            d_cells=d_cells,
            bmat=bmat,
            Xs=None,
            x_cells_strata=None,
            oracle=oracle,
            mc_seed=rng,
        )
        if st.get("pipeline"):
            res.pop("oracle")
            res.pop("oracle_ok")
        for arm, val in res.items():
            arms[(arm, "")] = val
        arms[("ppi", "")] = cand.ppi_blocks(st["resolved"], cells, e)
        return {"": err}, arms
    sc = st["sc"]
    real = sc.simulate(rng) if design == "random" else simulate_categorical(st, rng, design)
    v = real.validation
    x, y = v["x"].to_numpy(), v["y"].to_numpy()
    cells = v["cell"].to_numpy()
    labels = np.asarray(v["label"].to_list(), dtype=object)
    d_cells = np.hypot(cx[:, None] - x[None, :], cy[:, None] - y[None, :])
    kept, Xs = cand.strata_design(st["map_class"][cells], 5)
    x_cells_strata = cand.design_rows(st["map_class"], kept)
    k_classes = len(sc.classes)
    mc_root = np.random.SeedSequence(int(rng.integers(2**63 - 1)))
    errs: dict[str, np.ndarray] = {}
    for k, cls in enumerate(sc.classes):
        pk = st["p_user"][k, cells]
        sk = np.sqrt(pk * (1 - pk))
        ek = (labels == cls).astype(np.float64) - pk
        oc, rho0 = oracle_corr(name, k)
        res = class_arms(
            st=st,
            x=x,
            y=y,
            z=ek / sk,
            sd_map=st["sd_user"][k],
            sd_true=st["sd_true"][k],
            d_cells=d_cells,
            bmat=bmat,
            Xs=Xs,
            x_cells_strata=x_cells_strata,
            oracle=(oc, rho0, False),
            mc_seed=mc_root.spawn(1)[0],
        )
        for arm, val in res.items():
            arms[(arm, cls)] = val
        arms[("ppi", cls)] = cand.ppi_blocks(st["resolved"], cells, ek)
        arms[("olofsson", cls)] = cand.olofsson_blocks(
            st["resolved"], st["map_class"], st["p_user"][k], cells, labels == cls, k_classes
        )
        truth = st["eb"].block_means(real.fraction_sub[k])
        errs[cls] = truth - st["pred"][k]
    return errs, arms


@functools.cache
def block_matrix(mode: str, name: str) -> Any:
    st = setup(mode, name)
    return cand.block_average_matrix(st["resolved"], st["n_cells"])


def worker(task: dict[str, Any]) -> dict[str, Any]:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    mode, name, design = task["mode"], task["name"], task["design"]
    si = CORE[mode][name][0]
    shifts: dict[tuple[str, str], list[np.ndarray]] = {}
    sds: dict[tuple[str, str], list[np.ndarray]] = {}
    errs: dict[str, list[np.ndarray]] = {}
    for i in range(task["start"], task["stop"]):
        seed = [SEED, si, i] if design == "random" else [SEED, si, i, 1000 + task["design_index"]]
        rng = np.random.default_rng(seed)
        err, arms = realization(mode, name, design, rng, task["n"])
        for cls, e in err.items():
            errs.setdefault(cls, []).append(e)
        for key, (shift, sd) in arms.items():
            shifts.setdefault(key, []).append(shift)
            sds.setdefault(key, []).append(sd)
    return {
        "shifts": {k: np.array(v) for k, v in shifts.items()},
        "sds": {k: np.array(v) for k, v in sds.items()},
        "errs": {k: np.array(v) for k, v in errs.items()},
    }


def cell_stem(mode: str, name: str, design: str, n: int) -> str:
    return (
        f"{mode}_{name}"
        + ("" if design == "random" else f"_{design}")
        + ("" if n == N_DEFAULT else f"_n{n}")
    )


def run_cells(
    stage: str,
    cells: list[tuple[str, str, str, int]],
    n_real: int,
    quick: bool,
    skip_existing: bool = False,
) -> None:
    out_dir = RESULTS3 / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""
    for mode, name, design, n in cells:
        stem = cell_stem(mode, name, design, n)
        if skip_existing and (out_dir / f"{stem}_realizations{suffix}.parquet").exists():
            continue
        t0 = time.time()
        di = DESIGNS[mode].index(design) if design != "random" else -1
        tasks = [
            {
                "mode": mode,
                "name": name,
                "design": design,
                "design_index": di,
                "n": None if n == N_DEFAULT else n,
                "start": a,
                "stop": b,
            }
            for a, b in chunks(n_real, max(1, n_real // (3 * N_WORKERS)))
        ]
        out = run_parallel(worker, tasks, N_WORKERS)
        groups = setup(mode, name)["groups"]
        shifts = {k: np.concatenate([o["shifts"][k] for o in out]) for k in out[0]["shifts"]}
        sds = {k: np.concatenate([o["sds"][k] for o in out]) for k in out[0]["sds"]}
        errs = {k: np.concatenate([o["errs"][k] for o in out]) for k in out[0]["errs"]}
        blocks, reals = [], []
        for (arm, cls), shift in shifts.items():
            keys = {
                "mode": mode,
                "scenario": name,
                "design": design,
                "n": n,
                "arm": arm,
                "class": cls or None,
            }
            b, r = metrics(errs[cls] - shift, sds[(arm, cls)], groups, keys)
            blocks.append(b)
            reals.append(r)
        pl.concat(blocks, how="diagonal_relaxed").write_parquet(
            out_dir / f"{stem}_blocks{suffix}.parquet", compression="zstd"
        )
        pl.concat(reals, how="diagonal_relaxed").write_parquet(
            out_dir / f"{stem}_realizations{suffix}.parquet", compression="zstd"
        )
        print(f"[{stage}] {stem}: {n_real} realizations in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

CELL = ["mode", "scenario", "design", "n", "arm", "class", "group"]


def summarize(stage: str, quick: bool = False) -> None:
    suffix = "_quick" if quick else ""
    files_b = sorted((RESULTS3 / stage).glob(f"*_blocks{suffix}.parquet"))
    files_r = sorted((RESULTS3 / stage).glob(f"*_realizations{suffix}.parquet"))

    def load(fs):
        return pl.concat(
            [pl.read_parquet(f).with_columns(pl.col("class").cast(pl.String)) for f in fs],
            how="diagonal_relaxed",
        )

    blocks, reals = load(files_b), load(files_r)
    c = (
        reals.group_by(CELL)
        .agg(
            (pl.col("n_cov").sum() / pl.col("n_blocks").sum()).alias("coverage"),
            (pl.col("sum_log_ratio_sq").sum() / pl.col("n_blocks").sum()).sqrt().alias("lr_rms"),
            pl.col("n_blocks").sum().alias("n_obs"),
        )
        .filter(pl.col("n_obs") > 0)
    )
    b = blocks.group_by(CELL).agg(
        pl.col("width").mean(),
        pl.col("sd_true").mean().alias("rmse"),
        pl.col("ratio_rms").median().alias("ratio_med"),
    )
    cell = c.join(b, on=CELL, how="left", nulls_equal=True).with_columns(
        pl.col(pl.Float64).fill_nan(None)
    )
    ref = cell.filter(pl.col("arm") == "current").select(
        [k for k in CELL if k != "arm"] + [pl.col("width").alias("w0"), pl.col("rmse").alias("r0")]
    )
    cell = cell.join(
        ref, on=[k for k in CELL if k != "arm"], how="left", nulls_equal=True
    ).with_columns(
        (pl.col("width") / pl.col("w0")).alias("rel_width"),
        (pl.col("rmse") / pl.col("r0")).alias("rel_rmse"),
    )
    score = (
        cell.group_by("mode", "scenario", "design", "n", "arm")
        .agg(
            pl.col("coverage").mean(),
            (pl.col("coverage") - 0.9).abs().mean().alias("cov_err"),
            pl.col("coverage").min().alias("worst"),
            pl.col("lr_rms").mean(),
            pl.col("rel_width").mean(),
            pl.col("rel_rmse").mean(),
        )
        .sort("mode", "scenario", "design", "n", "cov_err")
    )

    def table(df: pl.DataFrame) -> list[str]:
        fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
        lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
        return lines + ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.iter_rows()]

    title = "2 (core change)" if stage == "core" else "3 (sampling designs)"
    lines = [
        f"# Round 3, Stage {title}: scorecard",
        "",
        "Generated by `experiments/improvements/round3_ideal.py`. Cells are (class, block "
        "group); `rel_width` and `rel_rmse` are relative to `current` in the same cell.",
        "",
    ]
    for (mode, scen, design, n), sub in score.group_by(
        "mode", "scenario", "design", "n", maintain_order=True
    ):
        title = f"## {mode}: {scen}, n = {n}" + ("" if design == "random" else f", {design} sample")
        lines += [title, ""]
        lines += table(
            sub.drop("mode", "scenario", "design", "n").with_columns(pl.col(pl.Float64).round(3))
        )
        lines += [""]
    rare = cell.filter(
        (pl.col("mode") == "categorical")
        & pl.col("group").is_in(["tile_64", "tile_128"])
        & (pl.col("class") == pl.col("class").max().over("scenario", "design", "n"))
    )
    if rare.height:
        lines += ["## Rarest class at 64- and 128-cell tiles: coverage (median SD ratio)", ""]
        piv = (
            rare.with_columns(
                pl.format(
                    "{} ({})", pl.col("coverage").round(3), pl.col("ratio_med").round(2)
                ).alias("v")
            )
            .pivot(on="arm", index=["scenario", "design", "n", "group"], values="v")
            .sort("scenario", "design", "n", "group")
        )
        lines += table(piv)
    out = RESULTS3 / stage / f"scorecard{suffix}.md"
    out.write_text("\n".join(lines) + "\n")
    print(out.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["core", "designs"], default=None)
    ap.add_argument("--summarize", choices=["core", "designs"], default=None)
    ap.add_argument("--realizations", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cells", default=None, help="comma-separated mode:scenario[:design[:n]]")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, "1")
    if args.stage:
        if args.cells:
            cells = []
            for c in args.cells.split(","):
                parts = [*c.split(":"), "random", str(N_DEFAULT)]
                cells.append((parts[0], parts[1], parts[2], int(parts[3])))
        elif args.stage == "core":
            cells = [(m, s, "random", N_DEFAULT) for m in CORE for s in CORE[m]]
            cells += [(m, s, "random", n) for m in SIZED_CORE for s in SIZED_CORE[m] for n in SIZES]
        else:
            cells = [
                (m, s, d, N_DEFAULT) for m in DESIGNS for s in DESIGN_MAPS[m] for d in DESIGNS[m]
            ]
            cells += [
                (m, s, d, n) for m in SIZED_DESIGNS for s, d in SIZED_DESIGNS[m] for n in SIZES
            ]
        for cell in cells:
            if args.quick:
                n_real = 4
            elif args.realizations:
                n_real = args.realizations
            elif cell[3] != N_DEFAULT:
                n_real = N_REAL_SIZED
            else:
                n_real = 200 if args.stage == "core" else 150
            run_cells(args.stage, [cell], n_real, args.quick, args.skip_existing)
    if args.summarize:
        summarize(args.summarize, args.quick)


if __name__ == "__main__":
    main()
