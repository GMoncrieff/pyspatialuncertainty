"""Round 3, Stage 1b: a separate holdout sample for the area estimates.

The user trains the map on a labelled training sample and never refits it; a separate
holdout sample of ``N_HOLD`` points gives the residuals for the area estimates. Each
realization of a cell (mode × learner × training design) shares the truth, the training
sample and the map across:

* holdout sources:

  - ``split``: ``N_HOLD`` points of the same campaign, split off at random before
    modelling (the campaign has ``N_TRAIN + N_HOLD`` points with the training design);
  - ``ind_random``: an independent simple random sample of non-training cells;
  - ``ind_stratified`` (categorical): independent, stratified by mapped class with equal
    allocation;
  - ``ind_clustered``: independent, clusters of 10 points within 4 cells;

* recalibration, fitted on distance-matched out-of-fold predictions within the training
  sample, so the holdout is used only for the area estimates:

  - ``raw``;
  - ``cal_global``: σ scale factor (continuous) or temperature (categorical);
  - ``cal_class`` (categorical): temperature and per-class biases (bias-corrected
    temperature scaling, Alexandari et al. 2020);

* arms: those of Stage 1 (``round3.evaluate``).

Usage::

    uv run --group experiments python -m experiments.improvements.round3_holdout --stage
    uv run python -m experiments.improvements.round3_holdout --summarize
"""

from __future__ import annotations

import argparse
import math
import os
import time
from typing import Any

import numpy as np
import polars as pl
from scipy import optimize

from experiments.common import chunks, run_parallel
from experiments.improvements import pipelines as pipe
from experiments.improvements.round3 import (
    CLASSES,
    MODES,
    N_WORKERS,
    RESULTS3,
    SEED3,
    block_matrix,
    evaluate,
    metrics,
    world,
)

N_TRAIN = 300
N_HOLD = 100
CLUSTER_SIZE = 10
STAGE_ID = 31  # seed stream of this stage
SOURCES = {
    "continuous": ("split", "ind_random", "ind_clustered"),
    "categorical": ("split", "ind_random", "ind_stratified", "ind_clustered"),
}
CALS = {"continuous": ("raw", "cal_global"), "categorical": ("raw", "cal_global", "cal_class")}
OUT = RESULTS3 / "holdout"


# ---------------------------------------------------------------------------
# Holdout samples
# ---------------------------------------------------------------------------


def clustered_cells(
    w: pipe.World, n: int, rng: np.random.Generator, allowed: np.ndarray
) -> np.ndarray:
    chosen: list[int] = []
    taken = ~allowed
    while len(chosen) < n:
        cxy = rng.uniform(0, pipe.N, 2)
        near = np.flatnonzero((np.hypot(w.cx - cxy[0], w.cy - cxy[1]) <= 4.0) & ~taken)
        if near.size == 0:
            continue
        pick = rng.choice(near, min(CLUSTER_SIZE, near.size, n - len(chosen)), replace=False)
        taken[pick] = True
        chosen.extend(pick.tolist())
    return np.array(chosen)


def stratified_cells(
    strata: np.ndarray, n: int, rng: np.random.Generator, allowed: np.ndarray
) -> np.ndarray:
    codes = np.unique(strata[allowed])
    per = n // codes.size
    picks = []
    for c in codes:
        pool = np.flatnonzero(allowed & (strata == c))
        picks.append(rng.choice(pool, min(per, pool.size), replace=False))
    idx = np.concatenate(picks)
    if idx.size < n:
        rest = np.flatnonzero(allowed & ~np.isin(np.arange(strata.size), idx))
        idx = np.concatenate([idx, rng.choice(rest, n - idx.size, replace=False)])
    return idx


# ---------------------------------------------------------------------------
# Recalibration on training-data out-of-fold predictions
# ---------------------------------------------------------------------------


def oof_predictions(
    mode: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    folds: np.ndarray,
    seeds: Any,
) -> pipe.Residuals:
    n = y_tr.size
    pred = np.empty((n, len(pipe.PREVALENCE))) if mode == "categorical" else np.empty(n)
    sd = np.empty(n) if mode == "continuous" else np.empty(0)
    for f in range(pipe.N_FOLDS):
        test = np.flatnonzero(folds == f)
        if test.size == 0:
            continue
        train = np.flatnonzero(folds != f)
        p_, s_ = pipe.fit_predict(mode, X_tr[train], y_tr[train], X_tr[test], int(next(seeds)))
        pred[test] = p_
        if mode == "continuous":
            sd[test] = s_
    return pipe.Residuals(np.arange(n), y_tr, pred, sd)


def class_calibration(oof: pipe.Residuals, mp: pipe.MapOutput) -> tuple[pipe.MapOutput, np.ndarray]:
    """Temperature and per-class biases (first class 0) minimising the log-loss."""
    lab = oof.observed.astype(int)
    logp = np.log(oof.pred)
    k = logp.shape[1]

    def transform(lp: np.ndarray, theta: np.ndarray) -> np.ndarray:
        z = lp / math.exp(theta[0]) + np.concatenate([[0.0], theta[1:]])
        z -= z.max(axis=1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

    def nll(theta: np.ndarray) -> float:
        return -float(np.mean(transform(logp, theta)[np.arange(lab.size), lab]))

    bounds = [(math.log(0.2), math.log(5.0))] + [(-5.0, 5.0)] * (k - 1)
    theta = optimize.minimize(nll, np.zeros(k), method="L-BFGS-B", bounds=bounds).x
    p_map = pipe.clip_probabilities(np.exp(transform(np.log(mp.pred), theta)), pipe.P_CLIP_CAL)
    return pipe.MapOutput(p_map, mp.sd), theta


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def holdout_worker(task: dict[str, Any]) -> dict[str, Any]:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    mode, learner, design = task["mode"], task["learner"], task["design"]
    mi, di, li = MODES.index(mode), pipe.DESIGNS.index(design), pipe.LEARNERS.index(learner)
    w = world()
    bmat = block_matrix()
    n_cells = w.grid.size
    X_all = pipe.features(w, learner)
    xy = np.column_stack([w.cx, w.cy])
    shifts: dict[tuple[str, ...], list[np.ndarray]] = {}
    sds: dict[tuple[str, ...], list[np.ndarray]] = {}
    errs: dict[tuple[str, ...], list[np.ndarray]] = {}
    diag: list[dict[str, Any]] = []
    for r in range(task["start"], task["stop"]):
        # truth, campaign, split and the learner-independent holdouts: shared by learners
        rng = np.random.default_rng([SEED3, STAGE_ID, mi, di, r])
        struct = w.sampler.sample(rng, 1)[0].ravel()
        noise = math.sqrt(w.nugget_var) * rng.standard_normal(n_cells)
        if mode == "continuous":
            truth_cells = w.signal + struct
            obs_cells = truth_cells + noise  # a point observation includes the nugget
            truth = {"": np.asarray(bmat @ truth_cells)}
        else:
            cls_cells = np.searchsorted(w.thresholds, w.signal + struct + noise, side="left")
            obs_cells = cls_cells
            truth = {
                c: np.asarray(bmat @ (cls_cells == k).astype(np.float64))
                for k, c in enumerate(CLASSES)
            }
        campaign = pipe.draw_sample(w, design, N_TRAIN + N_HOLD, rng)
        perm = rng.permutation(campaign.size)
        train, split = campaign[perm[:N_TRAIN]], campaign[perm[N_TRAIN:]]
        allowed = np.ones(n_cells, dtype=bool)
        allowed[train] = False
        hold = {
            "split": split,
            "ind_random": rng.choice(np.flatnonzero(allowed), N_HOLD, replace=False),
            "ind_clustered": clustered_cells(w, N_HOLD, rng, allowed.copy()),
        }
        folds, nndm_choice = pipe.nndm_folds(xy[train], xy, pipe.N_FOLDS, rng)
        strat_rng = np.random.default_rng([SEED3, STAGE_ID, mi, di, r, li, 1])
        # learner: map from the training sample only, and out-of-fold training predictions
        seeds = iter(
            np.random.default_rng([SEED3, STAGE_ID, mi, di, r, li]).integers(0, 2**31 - 1, 32)
        )
        pr, sd = pipe.fit_predict(mode, X_all[train], obs_cells[train], X_all, int(next(seeds)))
        mp_raw = pipe.MapOutput(pr, sd)
        oof = oof_predictions(mode, X_all[train], obs_cells[train], folds, seeds)
        maps: dict[str, pipe.MapOutput] = {"raw": mp_raw}
        row: dict[str, Any] = {"realization": r, "nndm_choice": nndm_choice}
        _, maps["cal_global"], row["cal_global_param"] = pipe.calibrate(mode, oof, mp_raw)
        if mode == "categorical":
            maps["cal_class"], theta = class_calibration(oof, mp_raw)
            row["cal_class_T"] = math.exp(theta[0])
            for k, c in enumerate(CLASSES[1:], start=1):
                row[f"cal_class_bias_{c}"] = float(theta[k])
            hold["ind_stratified"] = stratified_cells(
                np.argmax(pr, axis=1), N_HOLD, strat_rng, allowed.copy()
            )
            row["train_n_c"] = int((obs_cells[train] == 2).sum())
        mc_root = np.random.SeedSequence([SEED3, STAGE_ID, mi, di, r, li, 7])
        for source in SOURCES[mode]:
            cells = hold[source]
            if mode == "categorical":
                row[f"{source}_n_c"] = int((obs_cells[cells] == 2).sum())
            d_cells = np.hypot(
                w.cx[:, None] - w.cx[cells][None, :], w.cy[:, None] - w.cy[cells][None, :]
            )
            for cal in CALS[mode]:
                mp = maps[cal]
                res = pipe.Residuals(
                    cells,
                    obs_cells[cells],
                    mp.pred[cells],
                    mp.sd[cells] if mode == "continuous" else np.empty(0),
                )
                cache: dict[str, Any] = {"d_cells": d_cells}
                arms, map_means = evaluate(mode, res, mp, cache, mc_root.spawn(1)[0], True)
                for (arm, cls), (shift, sdv) in arms.items():
                    key = (source, cal, arm, cls)
                    shifts.setdefault(key, []).append(shift)
                    sds.setdefault(key, []).append(sdv)
                for cls, mb in map_means.items():
                    errs.setdefault((source, cal, cls), []).append(truth[cls] - mb)
        diag.append(row)
    return {
        "shifts": {k: np.array(v) for k, v in shifts.items()},
        "sds": {k: np.array(v) for k, v in sds.items()},
        "errs": {k: np.array(v) for k, v in errs.items()},
        "diag": diag,
    }


def run_stage(
    n_real: int, quick: bool, cells: list[tuple[str, str, str]] | None, skip_existing: bool
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""
    todo = cells or [(m, lr, d) for m in MODES for d in pipe.DESIGNS for lr in pipe.LEARNERS]
    groups = world().groups
    for mode, learner, design in todo:
        stem = f"{mode}_{learner}_{design}"
        if skip_existing and (OUT / f"{stem}_realizations{suffix}.parquet").exists():
            continue
        t0 = time.time()
        tasks = [
            {"mode": mode, "learner": learner, "design": design, "start": a, "stop": b}
            for a, b in chunks(n_real, max(1, n_real // (3 * N_WORKERS)))
        ]
        out = run_parallel(holdout_worker, tasks, N_WORKERS)
        shifts = {k: np.concatenate([o["shifts"][k] for o in out]) for k in out[0]["shifts"]}
        sds = {k: np.concatenate([o["sds"][k] for o in out]) for k in out[0]["sds"]}
        errs = {k: np.concatenate([o["errs"][k] for o in out]) for k in out[0]["errs"]}
        blocks, reals = [], []
        for (source, cal, arm, cls), shift in shifts.items():
            keys = {
                "mode": mode,
                "learner": learner,
                "design": design,
                "workflow": source,
                "calibration": cal,
                "arm": arm,
                "class": cls or None,
            }
            b, r = metrics(
                errs[(source, cal, cls)] - shift, sds[(source, cal, arm, cls)], groups, keys
            )
            blocks.append(b)
            reals.append(r)
        pl.concat(blocks, how="diagonal_relaxed").write_parquet(
            OUT / f"{stem}_blocks{suffix}.parquet", compression="zstd"
        )
        pl.concat(reals, how="diagonal_relaxed").write_parquet(
            OUT / f"{stem}_realizations{suffix}.parquet", compression="zstd"
        )
        pl.DataFrame([d for o in out for d in o["diag"]]).write_parquet(
            OUT / f"{stem}_diag{suffix}.parquet", compression="zstd"
        )
        print(f"[holdout] {stem}: {n_real} realizations in {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--realizations", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--cells", default=None, help="comma-separated mode:learner:design")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, "1")
    cells = [tuple(c.split(":")) for c in args.cells.split(",")] if args.cells else None
    if args.stage:
        run_stage(4 if args.quick else args.realizations, args.quick, cells, args.skip_existing)  # type: ignore[arg-type]
    if args.summarize:
        from experiments.improvements.summarize3 import summarize_holdout

        summarize_holdout(quick=args.quick)


if __name__ == "__main__":
    main()
