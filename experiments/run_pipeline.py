"""E1/E2 for the full pipeline of the authors' R example (PipelineScenario).

Every realization draws a new field and sample, cross-validates a ranger-style quantile
regression forest and fits a variogram to the standardized CV errors in two ways:

* ``method``: automatic choice of Nug + Exp/Sph/Gau (bounded weighted least squares);
* ``method_nested``: the R script's nested model, ranges fixed and sills fitted.

It then integrates over blocks: the whole domain (the box [0, 100]² as in the script) and
tiles of 10, 25 and 50 nodes. Baselines: ``naive_avg`` and ``naive_indep``. There is no
oracle, since the true conditional error SD of a QRF is not known.

Usage: ``uv run --group experiments python -m experiments.run_pipeline [--quick]``
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import polars as pl
import xarray as xr

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
from pyspatialuncertainty.blocks import Block, BlockSet
from pyspatialuncertainty.continuous import block_means
from pyspatialuncertainty.synthetic import PipelineScenario


def make_scenario(params: dict[str, Any]) -> PipelineScenario:
    return PipelineScenario.with_trend_r2(float(params["trend_r2"]), n_train=int(params["n_train"]))


def make_blocks(sc: PipelineScenario, tile_sizes: list[int]) -> tuple[BlockSet, list[str]]:
    c = sc.coords
    template = psu.GridSpec.from_dataarray(
        xr.DataArray(np.zeros((c.size, c.size)), coords={"y": c, "x": c}, dims=("y", "x"))
    )
    blocks: list[Block] = []
    groups: list[str] = []
    dom = psu.blocks.box(0.0, 0.0, sc.extent, sc.extent, template, id="domain")
    blocks.extend(dom.blocks)
    groups.append("domain")
    for t in tile_sizes:
        ts = psu.blocks.tiles(template, t, drop_partial=True)
        for b in ts.blocks:
            blocks.append(Block(id=f"tile{t}_{b.id}", cells=b.cells))
            groups.append(f"tile_{t}")
    return BlockSet(template, tuple(blocks), "custom"), groups


def worker(task: dict[str, Any]) -> dict[str, Any]:
    params, tile_sizes, n_mc = task["params"], task["tile_sizes"], task["n_mc"]
    sc = make_scenario(params)
    bs, _ = make_blocks(sc, tile_sizes)
    resolved = psu.resolve_blocks(bs, np.ones(bs.grid.size, dtype=bool))
    rows: dict[str, list[np.ndarray]] = {
        "err": [],
        "method": [],
        "method_nested": [],
        "naive_avg": [],
        "naive_indep": [],
    }
    nested_start = (
        psu.vgm(0.46, "Exp", 2, nugget=0.05) + psu.vgm(0.08, "Sph", 15) + psu.vgm(0.06, "Exp", 60)
    )
    fits = []
    for i in range(task["start"], task["stop"]):
        rng = np.random.default_rng([task["seed"], task["config_index"], i])
        real = sc.simulate(rng)
        val = psu.standardized_errors(real.validation)
        fit, corr = fit_correlation(val)
        sd = bs.grid.values(real.sd)
        ev = psu.empirical_variogram(val, "std_error", "x", "y")
        nested = psu.fit_variogram(ev, nested_start, fit_ranges=False)
        corr_nested = psu.as_correlation(nested.model, check=False)
        mc = psu.mc_block_variance(
            resolved, [psu.Integrand(sd, corr), psu.Integrand(sd, corr_nested)], n_mc=n_mc, seed=rng
        )
        truth = block_means(resolved, bs.grid.values(real.truth))
        pred = block_means(resolved, bs.grid.values(real.prediction))
        rows["err"].append(truth - pred)
        rows["method"].append(np.sqrt(np.maximum(mc.variance[0], 0.0)))
        rows["method_nested"].append(np.sqrt(np.maximum(mc.variance[1], 0.0)))
        rows["naive_avg"].append(psu.sd_average(resolved, sd))
        rows["naive_indep"].append(psu.sd_independent(resolved, sd))
        fits.append(
            {
                "realization": i,
                "std_error_var": float(val["std_error"].var()),
                "cv_rmse": float(
                    np.sqrt(
                        np.mean(
                            (real.validation["observed"] - real.validation["predicted"]).to_numpy()
                            ** 2
                        )
                    )
                ),
                **fit_summary(fit),
                "nested_sill": nested.model.sill,
            }
        )
    return {k: np.array(v) for k, v in rows.items()} | {"fits": fits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    sec = cfg["pipeline"]
    if args.quick:
        sec = {**sec, "realizations": 4, "baseline_realizations": 4}
    levels = cfg["levels"]
    block_tabs, real_tabs, fit_tabs = [], [], []
    for ci, (factor, level, params, is_base) in enumerate(configurations(sec)):
        t0 = time.time()
        n_real = sec["baseline_realizations"] if is_base else sec["realizations"]
        sc = make_scenario(params)
        bs, groups = make_blocks(sc, sec["tile_sizes"])
        resolved = psu.resolve_blocks(bs, np.ones(bs.grid.size, dtype=bool))
        sizes = np.array([b.area / bs.grid.cell_area for b in resolved])
        tasks = [
            {
                "params": params,
                "tile_sizes": sec["tile_sizes"],
                "n_mc": cfg["n_mc"],
                "seed": cfg["seed"],
                "config_index": ci,
                "start": a,
                "stop": b,
            }
            for a, b in chunks(n_real, max(1, n_real // (2 * cfg["n_workers"])))
        ]
        out = run_parallel(worker, tasks, cfg["n_workers"])
        err = np.concatenate([o["err"] for o in out])
        keys = {"experiment": "pipeline", "factor": factor, "level": str(level)}
        common = {
            "keys": keys,
            "block_ids": [str(i) for i in bs.ids],
            "groups": groups,
            "sizes": sizes,
        }
        for name in ("method", "method_nested", "naive_avg", "naive_indep"):
            sd = np.concatenate([o[name] for o in out])
            bt, rt = block_metrics(err, sd, levels, estimator=name, **common)
            block_tabs.append(bt)
            real_tabs.append(rt)
        fit_tabs.append(
            pl.DataFrame([f for o in out for f in o["fits"]]).with_columns(
                pl.lit("pipeline").alias("experiment"),
                pl.lit(factor).alias("factor"),
                pl.lit(str(level)).alias("level"),
            )
        )
        print(
            f"[pipeline] {factor}={level}: {n_real} realizations in {time.time() - t0:.0f} s",
            flush=True,
        )
    suffix = "_quick" if args.quick else ""
    write(pl.concat(block_tabs, how="vertical"), f"pipeline_blocks{suffix}.parquet")
    write(pl.concat(real_tabs, how="vertical"), f"pipeline_realizations{suffix}.parquet")
    write(pl.concat(fit_tabs, how="vertical"), f"pipeline_fits{suffix}.parquet")


if __name__ == "__main__":
    main()
