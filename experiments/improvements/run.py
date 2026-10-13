"""Improvement experiments: candidate variants against the current method on known truth.

All variants of a scenario see the same realizations, the same validation data and the
same Monte Carlo point pairs, so their differences are paired. For each scenario, variant,
class and block, the script records:

* coverage of the 90 % interval;
* the RMS ratio and the log-ratio of the estimated SD to the true SD;
* the share of realizations with a badly wrong SD (ratio < 0.5 or > 2).

It also records coverage per realization and block group, which gives paired standard
errors. The true SD comes from the exact block SD (continuous Gaussian scenarios) or the
RMSE of the block error over realizations (pipeline and categorical scenarios).

Usage (from the repository root)::

    uv run --group experiments python -m experiments.improvements.run --mode continuous
    uv run --group experiments python -m experiments.improvements.run --mode categorical
    # --quick for a smoke test, --scenarios a,b to select scenarios
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import pyspatialuncertainty as psu
from experiments.common import chunks, run_parallel
from experiments.improvements import candidates as C
from pyspatialuncertainty.continuous import block_means
from pyspatialuncertainty.synthetic import (
    ErrorModelScenario,
    LatentClassScenario,
    SubGridLayout,
    evaluation_blocks,
)

RESULTS = Path(__file__).resolve().parent / "results"
LEVEL = 0.90
Z = 1.6448536269514722
SEED = 20261013
N_MC = 10_000
N_BOOT = 40
N_WORKERS = 4
TILES = (2, 4, 8, 16, 32, 64, 128)

BASE_CONT = {"range": 8.0, "nugget": 0.25, "n_validation": 300, "sd_scale": 1.0}
BASE_CAT = {"range": 8.0, "nugget": 0.25, "n_validation": 300, "temperature": 1.0}
SCENARIOS: dict[str, dict[str, dict[str, Any]]] = {
    "continuous": {
        "baseline": BASE_CONT,
        "n100": BASE_CONT | {"n_validation": 100},
        "range64": BASE_CONT | {"range": 64.0},
        "nugget0.9": BASE_CONT | {"nugget": 0.9},
        "sd_scale0.75": BASE_CONT | {"sd_scale": 0.75},
        "pipeline": {"pipeline": True},
    },
    "categorical": {
        "baseline": BASE_CAT,
        "n800": BASE_CAT | {"n_validation": 800},
        "temperature0.7": BASE_CAT | {"temperature": 0.7},
        "range64": BASE_CAT | {"range": 64.0},
    },
}
FIT_VARIANTS = ["current", "wls_npairs", "wls_cressie", "exp_only", "sigma_weighted", "reml"]
CONT_VARIANTS = [*FIT_VARIANTS, "rescale", "boot_spread", "boot_bc"]
CAT_VARIANTS = [*FIT_VARIANTS, "pooled", "rescale", "boot_spread", "boot_bc", "beta"]


# ---------------------------------------------------------------------------
# Scenario set-up
# ---------------------------------------------------------------------------


def latent_model(params: dict[str, Any]) -> psu.VariogramModel:
    nug = float(params["nugget"])
    return psu.vgm(1.0 - nug, "Exp", float(params["range"]), nugget=nug)


def grid_setup(mode: str, params: dict[str, Any]) -> dict[str, Any]:
    """Scenario object, blocks and the fixed inputs handed to the method."""
    if params.get("pipeline"):
        from experiments.run_pipeline import make_blocks, make_scenario

        sc = make_scenario({"trend_r2": 0.6756756756756757, "n_train": 200})
        bs, groups = make_blocks(sc, [10, 25, 50])
        resolved = psu.resolve_blocks(bs, np.ones(bs.grid.size, dtype=bool))
        return {"sc": sc, "bs": bs, "resolved": resolved, "groups": tuple(groups)}
    layout = SubGridLayout(n=128, f=4)
    if mode == "continuous":
        sc: Any = ErrorModelScenario(
            layout=layout, correlation=latent_model(params), n_validation=params["n_validation"]
        )
    else:
        sc = LatentClassScenario(
            layout=layout, latent=latent_model(params), n_validation=params["n_validation"]
        )
    eb = evaluation_blocks(layout, TILES, max_per_size=32, n_polygons=6)
    resolved = psu.resolve_blocks(eb.blockset, np.ones(layout.grid.size, dtype=bool))
    out = {"sc": sc, "eb": eb, "resolved": resolved, "groups": eb.group}
    if mode == "continuous":
        out["sd_user"] = layout.grid.values(sc.sigma_map(params["sd_scale"]))
    else:
        p = sc.probabilities(params["temperature"]).values.reshape(len(sc.classes), -1)
        out["p_user"] = p
        out["sd_user"] = np.sqrt(p * (1.0 - p))
        out["pred"] = np.stack([block_means(resolved, p[k]) for k in range(p.shape[0])])
    return out


# ---------------------------------------------------------------------------
# Variants for one realization
# ---------------------------------------------------------------------------


def fit_models(
    p: C.Pairs, x: np.ndarray, y: np.ndarray, e: np.ndarray, s: np.ndarray
) -> dict[str, psu.VariogramModel]:
    z = e / s
    ev = C.matheron(p, z)
    return {
        "current": C.wls(ev),
        "wls_npairs": C.wls(ev, fit_method=1),
        "wls_cressie": C.wls(ev, fit_method=2),
        "exp_only": C.wls(ev, models=("Exp",)),
        "sigma_weighted": C.wls(C.sigma_weighted(p, e, s)),
        "reml": C.reml(x, y, z),
    }


def assemble(
    variances: dict[str, np.ndarray], boot: np.ndarray, rescale: float
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(sd, lower, upper) on the error scale for every variant from the MC variances."""
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, v in variances.items():
        sd = np.sqrt(np.maximum(v, 0.0))
        out[name] = (sd, -Z * sd, Z * sd)
    cur = np.maximum(variances["current"], 0.0)
    sd = np.sqrt(cur * rescale)
    out["rescale"] = (sd, -Z * sd, Z * sd)
    boot = np.maximum(boot, 0.0)
    for name, power in (("boot_spread", 1.0), ("boot_bc", 2.0)):
        vb = C.recentre(boot, cur, power)
        q = C.mixture_halfwidth(vb, LEVEL)
        out[name] = (np.sqrt(vb.mean(axis=0)), -q, q)
    return out


def realization_continuous(
    setup: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, ...]]]:
    sc, resolved = setup["sc"], setup["resolved"]
    if "bs" in setup:  # pipeline
        real = sc.simulate(rng)
        v = real.validation
        x, y = v["x"].to_numpy(), v["y"].to_numpy()
        e = (v["observed"] - v["predicted"]).to_numpy()
        s = v["sd"].to_numpy()
        grid = setup["bs"].grid
        sd_map = grid.values(real.sd)
        err = block_means(resolved, grid.values(real.truth)) - block_means(
            resolved, grid.values(real.prediction)
        )
    else:
        real = sc.simulate(rng)
        v = real.validation
        x, y = v["x"].to_numpy(), v["y"].to_numpy()
        e = v["error"].to_numpy()
        sd_map = setup["sd_user"]
        s = sd_map[v["cell"].to_numpy()]
        err = setup["eb"].block_means(real.error_sub)
    p = C.pairs(x, y)
    models = fit_models(p, x, y, e, s)
    D = C.distance_matrix(x, y)
    draws = C.gaussian_draws(models["current"], D, N_BOOT, rng)
    boot_models = [C.wls(C.matheron(p, draws[:, b])) for b in range(N_BOOT)]
    integrands = [psu.Integrand(sd_map, C.correlation(m)) for m in models.values()]
    integrands += [psu.Integrand(sd_map, C.correlation(m)) for m in boot_models]
    mc = psu.mc_block_variance(resolved, integrands, n_mc=N_MC, seed=rng)
    nv = len(models)
    variances = dict(zip(models, mc.variance[:nv], strict=True))
    res = assemble(variances, mc.variance[nv:], float(np.mean((e / s) ** 2)))
    return err[None, :], {k: tuple(a[None, :] for a in val) for k, val in res.items()}


def realization_categorical(
    setup: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, ...]]]:
    sc, resolved, eb = setup["sc"], setup["resolved"], setup["eb"]
    p_user, sd_user, pred = setup["p_user"], setup["sd_user"], setup["pred"]
    real = sc.simulate(rng)
    v = real.validation
    x, y = v["x"].to_numpy(), v["y"].to_numpy()
    cell = v["cell"].to_numpy()
    labels = np.asarray(v["label"].to_list(), dtype=object)
    p = C.pairs(x, y)
    D = C.distance_matrix(x, y)
    K = len(sc.classes)
    per_class: list[dict[str, psu.VariogramModel]] = []
    zs, rescale, boot_models = [], [], []
    for k, cls in enumerate(sc.classes):
        pk = p_user[k, cell]
        sk = np.sqrt(pk * (1.0 - pk))
        ek = (labels == cls).astype(np.float64) - pk
        per_class.append(fit_models(p, x, y, ek, sk))
        zs.append(ek / sk)
        rescale.append(float(np.mean((ek / sk) ** 2)))
        latent = C.gaussian_draws(per_class[k]["current"], D, N_BOOT, rng)
        zb = C.indicator_draws(latent, pk)
        boot_models.append([C.wls(C.matheron(p, zb[:, b])) for b in range(N_BOOT)])
    pooled_model = C.wls(C.pooled(p, zs))
    names = [*FIT_VARIANTS, "pooled"]
    integrands = []
    for k in range(K):
        models = per_class[k] | {"pooled": pooled_model}
        integrands += [psu.Integrand(sd_user[k], C.correlation(models[n])) for n in names]
        integrands += [psu.Integrand(sd_user[k], C.correlation(m)) for m in boot_models[k]]
    mc = psu.mc_block_variance(resolved, integrands, n_mc=N_MC, seed=rng)
    var = mc.variance.reshape(K, len(names) + N_BOOT, -1)
    truth = np.stack([eb.block_means(real.fraction_sub[k]) for k in range(K)])
    err = truth - pred
    out: dict[str, list[tuple[np.ndarray, ...]]] = {}
    for k in range(K):
        variances = dict(zip(names, var[k, : len(names)], strict=True))
        res = assemble(variances, var[k, len(names) :], rescale[k])
        lo, hi = C.beta_bounds(pred[k], np.maximum(variances["current"], 0.0), LEVEL)
        res["beta"] = (res["current"][0], lo, hi)
        for name, val in res.items():
            out.setdefault(name, []).append(val)
    stacked = {
        n: tuple(np.stack([t[i] for t in vals]) for i in range(3)) for n, vals in out.items()
    }
    return err, stacked


def worker(task: dict[str, Any]) -> dict[str, Any]:
    setup = grid_setup(task["mode"], task["params"])
    fn = realization_continuous if task["mode"] == "continuous" else realization_categorical
    errs: list[np.ndarray] = []
    res: dict[str, list[list[np.ndarray]]] = {}
    for i in range(task["start"], task["stop"]):
        rng = np.random.default_rng([SEED, task["scenario_index"], i])
        err, out = fn(setup, rng)
        errs.append(err)
        for name, arrs in out.items():
            res.setdefault(name, [[], [], []])
            for j in range(3):
                res[name][j].append(arrs[j])
    return {"err": np.array(errs), "res": {n: [np.array(a) for a in v] for n, v in res.items()}}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(
    err: np.ndarray,
    res: dict[str, list[np.ndarray]],
    sd_true: np.ndarray,
    groups: tuple[str, ...],
    keys: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Block-level and (realization, group)-level tables. err: (R, B); sd_true: (B,)."""
    block_rows, real_rows = [], []
    g = np.asarray(groups)
    for name, (sd, lo, hi) in res.items():
        covered = (err >= lo) & (err <= hi)
        ratio = sd / sd_true
        logr = np.log(np.maximum(ratio, 1e-3))
        block_rows.append(
            pl.DataFrame(
                {
                    "block": np.arange(err.shape[1]),
                    "group": list(groups),
                    "sd_true": sd_true,
                    "cov_90": covered.mean(axis=0),
                    "ratio_rms": np.sqrt(np.mean(sd**2, axis=0)) / sd_true,
                    "log_ratio_mean": logr.mean(axis=0),
                    "log_ratio_rms": np.sqrt(np.mean(logr**2, axis=0)),
                    "frac_bad": np.mean((ratio < 0.5) | (ratio > 2.0), axis=0),
                    "frac_zero": np.mean(sd <= 1e-12 * sd_true, axis=0),
                }
            ).with_columns(*(pl.lit(v).alias(k) for k, v in (keys | {"variant": name}).items()))
        )
        for grp in dict.fromkeys(groups):
            sel = g == grp
            real_rows.append(
                pl.DataFrame(
                    {
                        "realization": np.arange(err.shape[0]),
                        "group": grp,
                        "n_blocks": int(sel.sum()),
                        "n_cov": covered[:, sel].sum(axis=1),
                        "sum_log_ratio_sq": (logr[:, sel] ** 2).sum(axis=1),
                    }
                ).with_columns(*(pl.lit(v).alias(k) for k, v in (keys | {"variant": name}).items()))
            )
    return pl.concat(block_rows), pl.concat(real_rows)


def true_sd(mode: str, setup: dict[str, Any], err: np.ndarray, k: int | None) -> np.ndarray:
    if mode == "continuous" and "eb" in setup:
        return np.asarray(setup["sc"].exact_block_sd(setup["eb"]))
    e = err if k is None else err[:, k, :]
    return np.sqrt(np.mean(e**2, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["continuous", "categorical"], required=True)
    ap.add_argument("--scenarios", default=None, help="comma-separated subset")
    ap.add_argument("--realizations", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, "1")
    scen = SCENARIOS[args.mode]
    names = args.scenarios.split(",") if args.scenarios else list(scen)
    n_real = 4 if args.quick else args.realizations
    suffix = "_quick" if args.quick else ""
    RESULTS.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        params = scen[name]
        si = list(SCENARIOS[args.mode]).index(name) + (100 if args.mode == "categorical" else 0)
        tasks = [
            {
                "mode": args.mode,
                "params": params,
                "scenario_index": si,
                "start": a,
                "stop": b,
            }
            for a, b in chunks(n_real, max(1, n_real // (3 * N_WORKERS)))
        ]
        out = run_parallel(worker, tasks, N_WORKERS)
        err = np.concatenate([o["err"] for o in out])  # (R, K, B)
        res = {
            v: [np.concatenate([o["res"][v][j] for o in out]) for j in range(3)]
            for v in out[0]["res"]
        }
        setup = grid_setup(args.mode, params)
        blocks, reals = [], []
        for k in range(err.shape[1]):
            cls = None if args.mode == "continuous" else str(setup["sc"].classes[k])
            sd_t = true_sd(args.mode, setup, err, k)
            keys = {"mode": args.mode, "scenario": name, "class": cls}
            b, r = metrics(
                err[:, k, :],
                {v: [a[:, k, :] for a in arrs] for v, arrs in res.items()},
                sd_t,
                setup["groups"],
                keys,
            )
            blocks.append(b)
            reals.append(r)
        pl.concat(blocks, how="diagonal_relaxed").write_parquet(
            RESULTS / f"{args.mode}_{name}_blocks{suffix}.parquet", compression="zstd"
        )
        pl.concat(reals, how="diagonal_relaxed").write_parquet(
            RESULTS / f"{args.mode}_{name}_realizations{suffix}.parquet", compression="zstd"
        )
        print(
            f"[{args.mode}] {name}: {n_real} realizations in {time.time() - t0:.0f} s", flush=True
        )


if __name__ == "__main__":
    main()
