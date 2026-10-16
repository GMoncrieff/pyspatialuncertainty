"""Round 3: the unknown-mean correction, tested in full mapping pipelines (Stage 1).

For every realization of a pipeline cell (mode × learner × sampling design), each of the
four workflows of ``pipelines.py`` produces a residual set and a map, with and without a
one-parameter recalibration. The arms are evaluated on the same realizations, residuals,
maps and Monte Carlo point pairs:

* ``current``: zero-mean ML correlogram, known mean (the package default);
* ``ok`` / ``ok_nc``: REML correlogram with an unknown constant mean, corrected block
  estimate, with / without the cross term of the variance;
* ``ok_strata`` / ``ok_strata_nc`` (categorical): the same with mapped-class means;
* ``ok_logit`` / ``ok_logit_nc`` (categorical): the same with the linearised
  logistic-recalibration mean [σ, σ·logit p], GLS on the ``ok`` correlogram;
* ``ppi``: block map mean + mean residual of the points in the block (≥ 20 points);
* ``olofsson`` (categorical, holdout workflows): post-stratified estimator per block.

Each arm's error is truth − (map block mean + shift); its true SD is the RMSE of that
error over realizations. Results go to ``results/round3/pipeline/``.

Usage::

    uv run --group experiments python -m experiments.improvements.round3 --stage pipeline
    uv run python -m experiments.improvements.round3 --summarize pipeline
"""

from __future__ import annotations

import argparse
import functools
import os
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import pyspatialuncertainty as psu
from experiments.common import chunks, run_parallel
from experiments.improvements import candidates as cand
from experiments.improvements import pipelines as pipe

RESULTS3 = Path(__file__).resolve().parent / "results" / "round3"
Z = 1.6448536269514722
SEED3 = 20261016
N_MC = 10_000
N_WORKERS = 4
MODES = ("continuous", "categorical")
CALS = ("raw", "cal")
CLASSES = ("a", "b", "c")


@functools.cache
def world() -> pipe.World:
    return pipe.make_world()


@functools.cache
def block_matrix() -> Any:
    w = world()
    return cand.block_average_matrix(w.resolved, w.grid.size)


def safe(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:  # a failed fit makes the arm NaN for this realization
        return None


# ---------------------------------------------------------------------------
# Arms for one residual set and map
# ---------------------------------------------------------------------------


def logit_design(p: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Linearised logistic recalibration of a class probability: design [σ, σ·logit p]
    with σ = √(p(1-p)) (see ``round3_ideal.logit_design``)."""
    q = np.clip(p, eps, 1.0 - eps)
    s = np.sqrt(q * (1.0 - q))
    return np.column_stack([s, s * np.log(q / (1.0 - q))])


def fit_set(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    Xs: np.ndarray | None,
    Xl: np.ndarray | None = None,
) -> dict[str, Any]:
    """The correlogram fits and GLS terms that depend only on the residual set."""
    ones = np.ones((z.size, 1))
    m0 = safe(lambda: cand.likelihood_fit(x, y, z, mean="zero"))
    m1 = safe(lambda: cand.likelihood_fit(x, y, z, mean="constant"))
    g1 = safe(lambda: cand.gls_terms(m1, x, y, z, ones)) if m1 is not None else None
    out = {"m0": m0, "m1": m1, "g1": g1}
    if Xs is not None:
        m2 = safe(lambda: cand.likelihood_fit(x, y, z, X=Xs)) if Xs.shape[1] > 1 else m1
        g2 = safe(lambda: cand.gls_terms(m2, x, y, z, Xs)) if m2 is not None else None
        out |= {"m2": m2, "g2": g2}
    if Xl is not None:  # ok_logit: GLS on the constant-mean correlogram (stable; see round3_ideal)
        g3 = safe(lambda: cand.gls_terms(m1, x, y, z, Xl)) if m1 is not None else None
        out |= {"m3": m1, "g3": g3}
    return out


def nan_pair(n_blocks: int) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n_blocks, np.nan), np.full(n_blocks, np.nan)


def arms_for_class(
    *,
    fits: dict[str, Any],
    sd_map: np.ndarray,
    x_cells_strata: np.ndarray | None,
    d_cells: np.ndarray,
    variances: dict[str, np.ndarray],
    x_cells_logit: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(shift, sd) per block for current / ok / ok_nc (/ ok_strata / ok_logit and their
    ``_nc`` variants)."""
    bmat = block_matrix()
    nb = bmat.shape[0]
    zero = np.zeros(nb)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    out["current"] = (
        (zero, np.sqrt(np.maximum(variances["m0"], 0.0)))
        if fits["m0"] is not None
        else nan_pair(nb)
    )
    ones = np.ones((sd_map.size, 1))
    for arm, mkey, gkey, xc in (
        ("ok", "m1", "g1", ones),
        ("ok_strata", "m2", "g2", x_cells_strata),
        ("ok_logit", "m3", "g3", x_cells_logit),
    ):
        if xc is None:
            continue
        if fits.get(mkey) is None or fits.get(gkey) is None:
            out[arm] = out[f"{arm}_nc"] = nan_pair(nb)
            continue
        for suffix, cross in (("", True), ("_nc", False)):
            shift, added = cand.unknown_mean_terms(
                bmat, sd_map, xc, fits[mkey], d_cells, fits[gkey], cross=cross
            )
            out[arm + suffix] = (shift, np.sqrt(np.maximum(variances[mkey] + added, 0.0)))
    return out


def evaluate(
    mode: str,
    res: pipe.Residuals,
    mp: pipe.MapOutput,
    fits_cache: dict[str, Any],
    mc_seed: np.random.SeedSequence,
    holdout: bool,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    """Arms per class for one (workflow, calibration): returns {(arm, class): (shift, sd)}
    and the map block means per class."""
    w = world()
    bmat = block_matrix()
    x, y = w.cx[res.cells], w.cy[res.cells]
    if "d_cells" not in fits_cache:
        fits_cache["d_cells"] = np.hypot(w.cx[:, None] - x[None, :], w.cy[:, None] - y[None, :])
    d_cells = fits_cache["d_cells"]
    arms: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    map_means: dict[str, np.ndarray] = {}
    if mode == "continuous":
        z = (res.observed - res.pred) / res.sd
        if "fits" not in fits_cache:
            fits_cache["fits"] = fit_set(x, y, z, None)
        fits = fits_cache["fits"]
        integrands, keys = [], []
        for key in ("m0", "m1"):
            if fits[key] is not None:
                integrands.append(psu.Integrand(mp.sd, cand.correlation(fits[key])))
                keys.append(key)
        variances = {}
        if integrands:
            mc = psu.mc_block_variance(w.resolved, integrands, n_mc=N_MC, seed=mc_seed)
            variances = dict(zip(keys, mc.variance, strict=True))
        for arm, val in arms_for_class(
            fits=fits, sd_map=mp.sd, x_cells_strata=None, d_cells=d_cells, variances=variances
        ).items():
            arms[(arm, "")] = val
        arms[("ppi", "")] = cand.ppi_blocks(w.resolved, res.cells, res.observed - res.pred)
        map_means[""] = np.asarray(bmat @ mp.pred)
        return arms, map_means
    k_classes = len(CLASSES)
    val_strata = np.argmax(res.pred, axis=1)
    map_class = np.argmax(mp.pred, axis=1)
    kept, Xs = cand.strata_design(val_strata, 5)
    x_cells_strata = cand.design_rows(map_class, kept)
    per_class = []
    integrands, keys = [], []
    for k, cls in enumerate(CLASSES):
        pk = res.pred[:, k]
        sk = np.sqrt(pk * (1.0 - pk))
        ek = (res.observed == k).astype(np.float64) - pk
        zk = ek / sk
        if f"fits_{k}" not in fits_cache:
            fits_cache[f"fits_{k}"] = fit_set(x, y, zk, Xs, logit_design(pk))
        fits = fits_cache[f"fits_{k}"]
        map_pk = mp.pred[:, k]
        map_sk = np.sqrt(map_pk * (1.0 - map_pk))
        per_class.append((cls, fits, map_sk, ek, logit_design(map_pk)))
        for key in ("m0", "m1", "m2"):
            if fits.get(key) is not None:
                integrands.append(psu.Integrand(map_sk, cand.correlation(fits[key])))
                keys.append((k, key))
        map_means[cls] = np.asarray(bmat @ map_pk)
    mc = psu.mc_block_variance(w.resolved, integrands, n_mc=N_MC, seed=mc_seed)
    var_by = dict(zip(keys, mc.variance, strict=True))
    for k, (cls, fits, map_sk, ek, x_cells_logit) in enumerate(per_class):
        variances = {key: var_by[(k, key)] for key in ("m0", "m1", "m2") if (k, key) in var_by}
        if "m1" in variances:
            variances["m3"] = variances["m1"]  # ok_logit shares the ok correlogram
        for arm, val in arms_for_class(
            fits=fits,
            sd_map=map_sk,
            x_cells_strata=x_cells_strata,
            d_cells=d_cells,
            variances=variances,
            x_cells_logit=x_cells_logit,
        ).items():
            arms[(arm, cls)] = val
        arms[("ppi", cls)] = cand.ppi_blocks(w.resolved, res.cells, ek)
        if holdout:
            arms[("olofsson", cls)] = cand.olofsson_blocks(
                w.resolved,
                map_class,
                mp.pred[:, k],
                res.cells,
                res.observed == k,
                k_classes,
            )
    return arms, map_means


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def pipeline_worker(task: dict[str, Any]) -> dict[str, Any]:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    mode, learner, design = task["mode"], task["learner"], task["design"]
    mi, di, li = MODES.index(mode), pipe.DESIGNS.index(design), pipe.LEARNERS.index(learner)
    w = world()
    bmat = block_matrix()
    shifts: dict[tuple[str, ...], list[np.ndarray]] = {}
    sds: dict[tuple[str, ...], list[np.ndarray]] = {}
    errs: dict[tuple[str, ...], list[np.ndarray]] = {}
    diag: list[dict[str, Any]] = []
    for r in range(task["start"], task["stop"]):
        rng = np.random.default_rng([SEED3, mi, di, r])  # truth, sample, splits: shared by learners
        real = pipe.simulate(w, mode, design, rng)
        wf_out, info = pipe.run_workflows(
            w,
            mode,
            learner,
            real,
            rng,
            seed=int(np.random.default_rng([SEED3, mi, di, r, li]).integers(2**31)),
        )
        if mode == "continuous":
            truth = {"": np.asarray(bmat @ real.truth)}
        else:
            truth = {
                c: np.asarray(bmat @ (real.truth == k).astype(np.float64))
                for k, c in enumerate(CLASSES)
            }
        mc_root = np.random.SeedSequence([SEED3, mi, di, r, li, 7])
        caches: dict[tuple[str, str], dict[str, Any]] = {}
        row = {"realization": r, "nndm_choice": info["nndm_choice"]}
        for wf in pipe.WORKFLOWS:
            res, mp = wf_out[wf]
            set_name = "holdout" if wf.startswith("holdout") else wf
            for cal in CALS:
                if cal == "raw":
                    res_c, mp_c, param = res, mp, float("nan")
                else:
                    res_c, mp_c, param = pipe.calibrate(mode, res, mp)
                row[f"{wf}_{cal}_param"] = param
                cache = caches.setdefault((set_name, cal), {})
                arms, map_means = evaluate(
                    mode, res_c, mp_c, cache, mc_root.spawn(1)[0], wf.startswith("holdout")
                )
                for (arm, cls), (shift, sd) in arms.items():
                    key = (wf, cal, arm, cls)
                    shifts.setdefault(key, []).append(shift)
                    sds.setdefault(key, []).append(sd)
                for cls, mb in map_means.items():
                    errs.setdefault((wf, cal, cls), []).append(truth[cls] - mb)
        diag.append(row)
    return {
        "shifts": {k: np.array(v) for k, v in shifts.items()},
        "sds": {k: np.array(v) for k, v in sds.items()},
        "errs": {k: np.array(v) for k, v in errs.items()},
        "diag": diag,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(
    err: np.ndarray, sd: np.ndarray, groups: tuple[str, ...], keys: dict[str, Any]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Block-level and (realization, group)-level tables; NaN (unavailable) cases excluded."""
    valid = np.isfinite(err) & np.isfinite(sd)
    n_valid = valid.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # blocks without valid values
        covered = np.where(valid, np.abs(err) <= Z * sd, False)
        rmse = np.sqrt(np.nanmean(np.where(valid, err**2, np.nan), axis=0))
        ratio = sd / rmse
        logr = np.where(valid, np.log(np.maximum(ratio, 1e-3)), np.nan)
        block = pl.DataFrame(
            {
                "block": np.arange(err.shape[1]),
                "group": list(groups),
                "n_valid": n_valid,
                "cov_90": np.where(
                    n_valid > 0, covered.sum(axis=0) / np.maximum(n_valid, 1), np.nan
                ),
                "sd_true": rmse,
                "width": np.nanmean(np.where(valid, sd, np.nan), axis=0),
                "bias": np.nanmean(np.where(valid, err, np.nan), axis=0),
                "ratio_rms": np.sqrt(np.nanmean(np.where(valid, sd**2, np.nan), axis=0)) / rmse,
                "log_ratio_mean": np.nanmean(logr, axis=0),
                "log_ratio_rms": np.sqrt(np.nanmean(logr**2, axis=0)),
            }
        ).with_columns(*(pl.lit(v).alias(k) for k, v in keys.items()))
    g = np.asarray(groups)
    real_rows = []
    for grp in dict.fromkeys(groups):
        sel = g == grp
        real_rows.append(
            pl.DataFrame(
                {
                    "realization": np.arange(err.shape[0]),
                    "group": grp,
                    "n_blocks": valid[:, sel].sum(axis=1),
                    "n_cov": covered[:, sel].sum(axis=1),
                    "sum_log_ratio_sq": np.nansum(logr[:, sel] ** 2, axis=1),
                }
            ).with_columns(*(pl.lit(v).alias(k) for k, v in keys.items()))
        )
    return block, pl.concat(real_rows)


def run_pipeline_stage(n_real: int, quick: bool, cells: list[tuple[str, str, str]] | None) -> None:
    out_dir = RESULTS3 / "pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_quick" if quick else ""
    todo = cells or [(m, lr, d) for m in MODES for d in pipe.DESIGNS for lr in pipe.LEARNERS]
    groups = world().groups
    for mode, learner, design in todo:
        t0 = time.time()
        tasks = [
            {"mode": mode, "learner": learner, "design": design, "start": a, "stop": b}
            for a, b in chunks(n_real, max(1, n_real // (3 * N_WORKERS)))
        ]
        out = run_parallel(pipeline_worker, tasks, N_WORKERS)
        shifts = {k: np.concatenate([o["shifts"][k] for o in out]) for k in out[0]["shifts"]}
        sds = {k: np.concatenate([o["sds"][k] for o in out]) for k in out[0]["sds"]}
        errs = {k: np.concatenate([o["errs"][k] for o in out]) for k in out[0]["errs"]}
        blocks, reals = [], []
        for (wf, cal, arm, cls), shift in shifts.items():
            err = errs[(wf, cal, cls)] - shift
            keys = {
                "mode": mode,
                "learner": learner,
                "design": design,
                "workflow": wf,
                "calibration": cal,
                "arm": arm,
                "class": cls or None,
            }
            b, r = metrics(err, sds[(wf, cal, arm, cls)], groups, keys)
            blocks.append(b)
            reals.append(r)
        stem = f"{mode}_{learner}_{design}"
        pl.concat(blocks, how="diagonal_relaxed").write_parquet(
            out_dir / f"{stem}_blocks{suffix}.parquet", compression="zstd"
        )
        pl.concat(reals, how="diagonal_relaxed").write_parquet(
            out_dir / f"{stem}_realizations{suffix}.parquet", compression="zstd"
        )
        pl.DataFrame([d for o in out for d in o["diag"]]).write_parquet(
            out_dir / f"{stem}_diag{suffix}.parquet", compression="zstd"
        )
        print(f"[pipeline] {stem}: {n_real} realizations in {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pipeline"], default=None)
    ap.add_argument("--summarize", choices=["pipeline"], default=None)
    ap.add_argument("--realizations", type=int, default=100)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cells", default=None, help="comma-separated mode:learner:design")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, "1")
    cells = [tuple(c.split(":")) for c in args.cells.split(",")] if args.cells else None
    if args.stage == "pipeline":
        run_pipeline_stage(4 if args.quick else args.realizations, args.quick, cells)  # type: ignore[arg-type]
    if args.summarize == "pipeline":
        from experiments.improvements.summarize3 import summarize_pipeline

        summarize_pipeline(quick=args.quick)


if __name__ == "__main__":
    main()
