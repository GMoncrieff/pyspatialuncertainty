"""E1/E2 for categorical maps: thresholded latent Gaussian fields (LatentClassScenario).

Per class, configuration and realization, the true block fraction comes from the sub-grid
truth and four standard deviations of the block-fraction error are evaluated:

* ``method``      -- probabilities handed to the method (optionally temperature-scaled),
                     σ_k = c·sqrt(p_k(1 − p_k)) and a correlogram fitted per class to the
                     standardized indicator errors at the validation points;
* ``oracle``      -- calibrated probabilities and the exact domain-averaged correlogram of
                     the standardized indicator errors (best stationary correlation);
* ``naive_avg``   -- block mean of σ_k;
* ``naive_indep`` -- sqrt(Σσ_k²)/n.

The block prediction is always the block mean of the probabilities handed to the method.

Usage: ``uv run --group experiments python -m experiments.run_categorical [--quick]``
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
from pyspatialuncertainty.continuous import block_means
from pyspatialuncertainty.synthetic import LatentClassScenario, SubGridLayout, evaluation_blocks


def make_scenario(params: dict[str, Any], sec: dict[str, Any]) -> LatentClassScenario:
    nug = float(params["nugget"])
    latent = psu.vgm(1.0 - nug, "Exp", float(params["range"]), nugget=nug)
    return LatentClassScenario(
        layout=SubGridLayout(n=sec["grid_cells"], f=sec["subgrid"]),
        latent=latent,
        snr=float(params["snr"]),
        n_validation=int(params["n_validation"]),
    )


def make_blocks(sc: LatentClassScenario, sec: dict[str, Any]):  # type: ignore[no-untyped-def]
    return evaluation_blocks(
        sc.layout,
        tuple(sec["tile_sizes"]),
        max_per_size=sec["max_tiles_per_size"],
        n_polygons=sec["n_polygons"],
    )


def user_maps(sc: LatentClassScenario, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Flat probabilities (K, cells) handed to the method and the matching σ_k."""
    p = sc.probabilities(float(params["temperature"])).values.reshape(len(sc.classes), -1)
    sd = float(params["sd_scale"]) * np.sqrt(p * (1.0 - p))
    return p, sd


def worker(task: dict[str, Any]) -> dict[str, Any]:
    params, sec, n_mc = task["params"], task["section"], task["n_mc"]
    sc = make_scenario(params, sec)
    eb = make_blocks(sc, sec)
    g = sc.layout.grid
    resolved = psu.resolve_blocks(eb.blockset, np.ones(g.size, dtype=bool))
    p_user, sd_user = user_maps(sc, params)
    pred = np.stack([block_means(resolved, p_user[k]) for k in range(len(sc.classes))])
    cols = {c: f"p_{c}" for c in sc.classes}
    errs, sds, fits = [], [], []
    for i in range(task["start"], task["stop"]):
        rng = np.random.default_rng([task["seed"], task["config_index"], i])
        real = sc.simulate(rng)
        cell = real.validation["cell"].to_numpy()
        val = real.validation.with_columns(
            [pl.Series(f"p_{c}", p_user[k, cell]) for k, c in enumerate(sc.classes)]
        )
        ie = psu.indicator_errors(val, cols, "label")
        ie = ie.with_columns((pl.col("std_error") / float(params["sd_scale"])).alias("std_error"))
        integrands = []
        for k, c in enumerate(sc.classes):
            fit, corr = fit_correlation(ie.filter(pl.col("class") == c))
            integrands.append(psu.Integrand(sd_user[k], corr))
            fits.append({"realization": i, "class": str(c), **fit_summary(fit)})
        mc = psu.mc_block_variance(resolved, integrands, n_mc=n_mc, seed=rng)
        truth = np.stack([eb.block_means(real.fraction_sub[k]) for k in range(len(sc.classes))])
        errs.append(truth - pred)
        sds.append(np.sqrt(np.maximum(mc.variance, 0.0)))
    return {"err": np.array(errs), "sd": np.array(sds), "fits": fits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    sec = cfg["categorical"]
    if args.quick:
        sec = {**sec, "realizations": 8, "baseline_realizations": 8, "exact_max_blocks": 2}
    levels = cfg["levels"]
    block_tabs, real_tabs, fit_tabs = [], [], []
    for ci, (factor, level, params, is_base) in enumerate(configurations(sec)):
        t0 = time.time()
        n_real = sec["baseline_realizations"] if is_base else sec["realizations"]
        sc = make_scenario(params, sec)
        eb = make_blocks(sc, sec)
        g = sc.layout.grid
        resolved = psu.resolve_blocks(eb.blockset, np.ones(g.size, dtype=bool))
        _, sd_user = user_maps(sc, params)
        p_true = sc.p_true.reshape(len(sc.classes), -1)
        sd_true = np.sqrt(p_true * (1 - p_true))
        oracle = psu.mc_block_variance(
            resolved,
            [psu.Integrand(sd_true[k], sc.oracle_correlation(k)) for k in range(len(sc.classes))],
            n_mc=100_000,
            seed=[cfg["seed"], ci, 999],
        )
        # exact SD for a few small tiles (bivariate normal formula)
        exact = np.full((len(sc.classes), len(resolved)), np.nan)
        for t in sec["exact_tile_sizes"]:
            idx = [b for b, grp in enumerate(eb.group) if grp == f"tile_{t}"][
                : sec["exact_max_blocks"]
            ]
            for b in idx:
                for k in range(len(sc.classes)):
                    exact[k, b] = sc.exact_block_sd(k, eb.sub_index[b])
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
        err = np.concatenate([o["err"] for o in out])  # (R, K, B)
        sd_method = np.concatenate([o["sd"] for o in out])
        for k, c in enumerate(sc.classes):
            keys = {
                "experiment": "categorical",
                "factor": factor,
                "level": str(level),
                "class": str(c),
            }
            common = {
                "keys": keys,
                "block_ids": [str(i) for i in eb.blockset.ids],
                "groups": eb.group,
                "sizes": eb.size_cells,
                "sd_exact": exact[k],
            }
            for name, sd in (
                ("method", sd_method[:, k, :]),
                ("oracle", np.sqrt(oracle.variance[k])),
                ("naive_avg", psu.sd_average(resolved, sd_user[k])),
                ("naive_indep", psu.sd_independent(resolved, sd_user[k])),
            ):
                bt, rt = block_metrics(err[:, k, :], sd, levels, estimator=name, **common)
                block_tabs.append(bt)
                real_tabs.append(rt)
        fit_tabs.append(
            pl.DataFrame([f for o in out for f in o["fits"]]).with_columns(
                pl.lit("categorical").alias("experiment"),
                pl.lit(factor).alias("factor"),
                pl.lit(str(level)).alias("level"),
            )
        )
        print(
            f"[categorical] {factor}={level}: {n_real} realizations in {time.time() - t0:.0f} s",
            flush=True,
        )
    suffix = "_quick" if args.quick else ""
    write(pl.concat(block_tabs, how="vertical"), f"categorical_blocks{suffix}.parquet")
    write(pl.concat(real_tabs, how="vertical"), f"categorical_realizations{suffix}.parquet")
    write(pl.concat(fit_tabs, how="vertical"), f"categorical_fits{suffix}.parquet")


if __name__ == "__main__":
    main()
