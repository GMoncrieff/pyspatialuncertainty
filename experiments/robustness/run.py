"""Driver of the robustness study: the frozen arms on one generator × map state × realization.

Arms are those of round 6 (``experiments.round6.run``), run unchanged through its
``setting``: ``cur`` (the paper), ``ok`` and R6 = ``thr+x2+vp+disp+cg``, plus ``olofsson``
as an unranked reference. Every map, the calibrated one included, is recalibrated by vector
scaling fitted on the calibration sample (same design; ``n_cal`` points).

Settings per task: ``--grid diag`` runs n_cal = n_val ∈ {50, 500, 5000} for both designs
(seeds as round 6, so ``base`` × ``offset`` reproduces round 6's M4); ``--grid offdiag``
runs the six n_cal ≠ n_val pairs. ``--ntrain`` sets the training-sample size of the
classifier maps (classifier worlds only).

Usage::

    uv run python -m experiments.robustness.run <out_dir> --side 256 --gens base,potts
        --states cal,offset,scale --start 0 --stop 30 [--grid diag] [--ntrain 1000]
        [--workers 4]

Writes ``<out_dir>/<side>/<gen>/<state>/r###.parquet`` (block estimates and variances per
arm, and the truth) and ``d###.parquet`` (fits and diagnostics), or ``o###`` / ``e###`` for
the off-diagonal grid; existing files are skipped.
With ``--ntrain`` other than 1000 the generator directory is ``<gen>-t<ntrain>``.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from experiments.robustness import gens as G  # noqa: N812
from experiments.round5 import shared as sh
from experiments.round6 import run as r6
from experiments.round6 import world as wd

ARMS = ("cur", "ok", "thr+x2+vp+disp+cg", "olofsson")
R6 = "thr+x2+vp+disp+cg"
PREFIX = {"diag": ("r", "d"), "offdiag": ("o", "e")}  # result and diagnostics file prefixes


def settings(grid: str) -> list[tuple[int, str, int, int, list[int]]]:
    """(design index, design, n_val, n_cal, seed key) of a grid."""
    out = []
    for di, d in enumerate(sh.DESIGNS):
        for vi, nv in enumerate(sh.SIZES):
            for ci, nc in enumerate(sh.SIZES):
                if grid == "diag" and vi == ci:
                    out.append((di, d, nv, nc, [9, di, vi]))  # round 6's seeds
                elif grid == "offdiag" and vi != ci:
                    out.append((di, d, nv, nc, [19, di, vi, ci]))
    return out


def gen_dir(gen: str, n_train: int) -> str:
    return gen if n_train == G.N_TRAIN else f"{gen}-t{n_train}"


def realization(
    side: int,
    gen: str,
    state: str,
    r: int,
    *,
    grid: str = "diag",
    n_train: int = G.N_TRAIN,
    params: dict | None = None,
    arms: tuple[str, ...] = ARMS,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    g = G.GENS[gen]
    w = G.world(g, side, params)
    dr, p_raw = G.draws(w, r, n_train)
    lp = G.map_logits(p_raw, state)
    nb = w.layout.n_blocks
    bside = w.layout.side_of_block
    fit_cal, apply_cal = sh.CALIBRATORS["vector"]
    frames, diag_rows = [], []
    for j, c in enumerate(g.classes):
        frames.append(pl.DataFrame({
            "design": "-", "n": 0, "n_cal": 0, "arm": "truth", "cls": c,
            "block": np.arange(nb), "bside": bside, "est": dr.truth[j], "var": np.zeros(nb),
        }))  # fmt: skip
    for _di, design, nv, nc, key in settings(grid):
        t0 = time.time()
        par = fit_cal(lp[:, dr.cal_cells[design][:nc]], dr.cal_labels[design][:nc])
        pf = sh.clip_norm(apply_cal(par, lp))
        rng = np.random.default_rng(wd.seed(side, key[0], r, *key[1:]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out, diags = r6.setting(
                w,  # type: ignore[arg-type]
                dr, design, nv, pf, arms, calib=("vector", lp, par), rng=rng, n_cal=nc,
                classes=g.classes,
            )  # fmt: skip
        for (arm, c), (est, var) in out.items():
            keep = np.flatnonzero(np.isfinite(est))
            frames.append(pl.DataFrame({
                "design": design, "n": nv, "n_cal": nc, "arm": arm, "cls": c, "block": keep,
                "bside": bside[keep], "est": est[keep], "var": var[keep],
            }))  # fmt: skip
        for d in diags:
            d |= {
                "n_cal": nc,
                "calibrator": repr(par),
                "warnings": len(caught),
                "setting_seconds": time.time() - t0,
            }
        diag_rows += diags
    meta = [
        pl.lit(r).cast(pl.Int16).alias("r"),
        pl.lit(gen).alias("gen"),
        pl.lit(state).alias("state"),
        pl.lit(side).cast(pl.Int16).alias("side"),
        pl.lit(n_train if g.kind == "classifier" else 0).cast(pl.Int16).alias("n_train"),
    ]
    df = pl.concat(frames).with_columns(
        *meta,
        pl.col("n").cast(pl.Int16),
        pl.col("n_cal").cast(pl.Int16),
        pl.col("block").cast(pl.Int16),
        pl.col("bside").cast(pl.Int16),
        pl.col("est").cast(pl.Float64),
        pl.col("var").cast(pl.Float64),
    )
    dg = pl.from_dicts(diag_rows, infer_schema_length=None).with_columns(*meta)
    return df, dg


def task(args: tuple) -> tuple[str, float, float]:
    side, gen, state, r, out_dir, grid, n_train, params = args
    t0 = time.time()
    df, dg = realization(side, gen, state, r, grid=grid, n_train=n_train, params=params)
    d = Path(out_dir) / str(side) / gen_dir(gen, n_train) / state
    res, diag = PREFIX[grid]
    dg.write_parquet(d / f"{diag}{r:03d}.parquet")
    df.write_parquet(d / f"{res}{r:03d}.parquet")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return f"{side} {gen_dir(gen, n_train)} {state} r{r}", time.time() - t0, peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--side", type=int, default=256)
    ap.add_argument("--gens", default=",".join(G.GENS))
    ap.add_argument("--states", default=",".join(G.STATES))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=30)
    ap.add_argument("--grid", choices=("diag", "offdiag"), default="diag")
    ap.add_argument("--ntrain", type=int, default=G.N_TRAIN)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out_dir)
    gens = a.gens.split(",")
    states = a.states.split(",")
    params = {}
    for gen in gens:
        if a.ntrain != G.N_TRAIN and G.GENS[gen].kind != "classifier":
            raise ValueError(f"--ntrain applies to classifier worlds only, not {gen}")
        cache = out / "_params" / f"{gen}_{a.side}.json"
        if cache.exists():
            params[gen] = json.loads(cache.read_text())
        else:
            params[gen] = G.prepare(G.GENS[gen], a.side)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(params[gen]))
        for s in states:
            (out / str(a.side) / gen_dir(gen, a.ntrain) / s).mkdir(parents=True, exist_ok=True)
    todo = [
        (a.side, gen, s, r, str(out), a.grid, a.ntrain, params[gen])
        for r in range(a.start, a.stop)
        for gen in gens
        for s in states
        if not (
            out / str(a.side) / gen_dir(gen, a.ntrain) / s / f"{PREFIX[a.grid][0]}{r:03d}.parquet"
        ).exists()
    ]
    print(f"{len(todo)} tasks", flush=True)
    with ProcessPoolExecutor(
        a.workers, mp_context=get_context("spawn"), max_tasks_per_child=1
    ) as ex:  # a fresh process per task keeps memory from creeping up
        for name, sec, peak in ex.map(task, todo):
            print(f"{name}: {sec:.0f}s, peak {peak:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
