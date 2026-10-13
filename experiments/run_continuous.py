"""E1/E2 for continuous maps: known spatially correlated errors (ErrorModelScenario).

For each configuration (baseline and one-factor ablations) and realization:
the true block-average errors come from the sub-grid truth, and four standard deviations
are evaluated per block:

* ``method``     -- σ handed to the method and a correlogram fitted to the standardized
                    validation errors (steps 3-8 as in the paper);
* ``oracle``     -- the true σ and the true correlation function (MC integration only);
* ``naive_avg``  -- block mean of σ (assumes perfectly correlated errors);
* ``naive_indep``-- sqrt(Σσ²)/n (assumes independent errors).

Usage: ``uv run --group experiments python experiments/run_continuous.py [--quick]``
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import polars as pl

import pyspatialuncertainty as psu
from experiments.common import (
    block_metrics,
    chunks,
    configurations,
    fit_correlation,
    fit_summary,
    load_config,
    run_parallel,
    write,
)
from pyspatialuncertainty.synthetic import ErrorModelScenario, SubGridLayout, evaluation_blocks


def make_scenario(params: dict[str, Any], sec: dict[str, Any]) -> ErrorModelScenario:
    nug = float(params["nugget"])
    corr = (
        psu.vgm(1.0 - nug, "Exp", float(params["range"]), nugget=nug)
        if nug < 1
        else psu.Nugget(1.0)
    )
    return ErrorModelScenario(
        layout=SubGridLayout(n=sec["grid_cells"], f=sec["subgrid"]),
        correlation=corr,
        n_validation=int(params["n_validation"]),
    )


def make_blocks(sc: ErrorModelScenario, sec: dict[str, Any]):  # type: ignore[no-untyped-def]
    return evaluation_blocks(
        sc.layout,
        tuple(sec["tile_sizes"]),
        max_per_size=sec["max_tiles_per_size"],
        n_polygons=sec["n_polygons"],
    )


def worker(task: dict[str, Any]) -> dict[str, Any]:
    params, sec, n_mc = task["params"], task["section"], task["n_mc"]
    sc = make_scenario(params, sec)
    eb = make_blocks(sc, sec)
    g = sc.layout.grid
    sd_user = g.values(sc.sigma_map(params["sd_scale"], params["sd_shape"]))
    resolved = psu.resolve_blocks(eb.blockset, np.ones(g.size, dtype=bool))
    errs, sds, fits = [], [], []
    for i in range(task["start"], task["stop"]):
        rng = np.random.default_rng([task["seed"], task["config_index"], i])
        real = sc.simulate(rng)
        val = real.validation.with_columns(
            (pl.col("error") / pl.Series(sd_user[real.validation["cell"].to_numpy()])).alias(
                "std_error"
            )
        )
        fit, corr = fit_correlation(val)
        mc = psu.mc_block_variance(resolved, [psu.Integrand(sd_user, corr)], n_mc=n_mc, seed=rng)
        errs.append(eb.block_means(real.error_sub))
        sds.append(np.sqrt(np.maximum(mc.variance[0], 0)))
        fits.append({"realization": i, **fit_summary(fit)})
    return {"err": np.array(errs), "sd": np.array(sds), "fits": fits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="few realizations, for testing")
    args = ap.parse_args()
    cfg = load_config()
    sec = cfg["continuous"]
    if args.quick:
        sec = {**sec, "realizations": 8, "baseline_realizations": 8}
    levels = cfg["levels"]
    block_tabs, real_tabs, fit_tabs = [], [], []
    for ci, (factor, level, params, is_base) in enumerate(configurations(sec)):
        t0 = time.time()
        n_real = sec["baseline_realizations"] if is_base else sec["realizations"]
        sc = make_scenario(params, sec)
        eb = make_blocks(sc, sec)
        g = sc.layout.grid
        resolved = psu.resolve_blocks(eb.blockset, np.ones(g.size, dtype=bool))
        sd_user = g.values(sc.sigma_map(params["sd_scale"], params["sd_shape"]))
        sd_true = g.values(sc.sigma_map())
        oracle = psu.mc_block_variance(
            resolved,
            [psu.Integrand(sd_true, sc.true_correlation())],
            n_mc=100_000,
            seed=[cfg["seed"], ci, 999],
        )
        sd_oracle = np.sqrt(oracle.variance[0])
        sd_exact = sc.exact_block_sd(eb)
        tasks = [
            {
                "params": params,
                "section": sec,
                "n_mc": cfg["n_mc"],
                "seed": cfg["seed"],
                "config_index": ci,
                "start": a,
                "stop": b,
            }
            for a, b in chunks(n_real, max(1, n_real // (4 * cfg["n_workers"])))
        ]
        out = run_parallel(worker, tasks, cfg["n_workers"])
        err = np.concatenate([o["err"] for o in out])
        sd_method = np.concatenate([o["sd"] for o in out])
        keys = {"experiment": "continuous", "factor": factor, "level": str(level)}
        common = {
            "keys": keys,
            "block_ids": [str(i) for i in eb.blockset.ids],
            "groups": eb.group,
            "sizes": eb.size_cells,
            "sd_exact": sd_exact,
        }
        for name, sd in (
            ("method", sd_method),
            ("oracle", sd_oracle),
            ("naive_avg", psu.sd_average(resolved, sd_user)),
            ("naive_indep", psu.sd_independent(resolved, sd_user)),
        ):
            bt, rt = block_metrics(err, sd, levels, estimator=name, **common)
            block_tabs.append(bt)
            real_tabs.append(rt)
        fit_tabs.append(
            pl.DataFrame([f for o in out for f in o["fits"]]).with_columns(
                pl.lit("continuous").alias("experiment"),
                pl.lit(factor).alias("factor"),
                pl.lit(str(level)).alias("level"),
            )
        )
        print(
            f"[continuous] {factor}={level}: {n_real} realizations in {time.time() - t0:.0f} s",
            flush=True,
        )
    suffix = "_quick" if args.quick else ""
    write(pl.concat(block_tabs, how="vertical"), f"continuous_blocks{suffix}.parquet")
    write(pl.concat(real_tabs, how="vertical"), f"continuous_realizations{suffix}.parquet")
    write(pl.concat(fit_tabs, how="vertical"), f"continuous_fits{suffix}.parquet")


if __name__ == "__main__":
    main()
