"""Improvement experiments: candidate variants against the current method on known truth.

All variants of a scenario see the same realizations, the same validation data and the
same Monte Carlo point pairs, so their differences are paired. For each scenario, variant,
class and block, the script records:

* coverage of the 90 % normal interval, prediction ± 1.645·SD;
* the RMS ratio and the log-ratio of the estimated SD to the true SD;
* the share of realizations with a badly wrong SD (ratio < 0.5 or > 2).

It also records coverage per realization and block group, which gives paired standard
errors. The true SD comes from the exact block SD (continuous Gaussian scenarios) or the
RMSE of the block error over realizations (pipeline and categorical scenarios).

Round 1 compares single changes to the correlogram step. Round 2 adds a likelihood fit
with the error mean fixed at zero and keeps the round-1 leaders. Results go to
``results/round<n>/``.

Usage (from the repository root)::

    uv run --group experiments python -m experiments.improvements.run --mode continuous --round 2
    uv run --group experiments python -m experiments.improvements.run --mode categorical --round 2
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
from experiments.improvements import candidates as cand
from pyspatialuncertainty.continuous import block_means
from pyspatialuncertainty.synthetic import (
    ErrorModelScenario,
    LatentClassScenario,
    SubGridLayout,
    evaluation_blocks,
)

RESULTS = Path(__file__).resolve().parent / "results"
Z = 1.6448536269514722  # two-sided 90 % normal quantile
SEED = 20261013
N_MC = 10_000
N_WORKERS = 4
TILES = (2, 4, 8, 16, 32, 64, 128)

BASE_CONT = {"model": "Exp", "range": 8.0, "nugget": 0.25, "n_validation": 300, "sd_scale": 1.0}
BASE_CAT = {"model": "Exp", "range": 8.0, "nugget": 0.25, "n_validation": 300, "temperature": 1.0}
SCENARIOS: dict[str, dict[str, dict[str, Any]]] = {
    "continuous": {
        "baseline": BASE_CONT,
        "n100": BASE_CONT | {"n_validation": 100},
        "range64": BASE_CONT | {"range": 64.0},
        "nugget0.9": BASE_CONT | {"nugget": 0.9},
        "sd_scale0.75": BASE_CONT | {"sd_scale": 0.75},
        # same practical range as the baseline, but not the exponential model
        "spherical": BASE_CONT | {"model": "Sph", "range": 24.0},
        "pipeline": {"pipeline": True},
    },
    "categorical": {
        "baseline": BASE_CAT,
        "n800": BASE_CAT | {"n_validation": 800},
        "temperature0.7": BASE_CAT | {"temperature": 0.7},
        "range64": BASE_CAT | {"range": 64.0},
    },
}
# Correlogram fits per round, plus derived variants: "rescale" (both modes) and "pooled"
# (categorical only).
ROUNDS: dict[int, dict[str, list[str]]] = {
    1: {
        "fits": ["current", "wls_npairs", "wls_cressie", "exp_only", "sigma_weighted", "reml"],
        "extras": ["rescale", "pooled"],
    },
    2: {"fits": ["current", "wls_npairs", "reml", "ml_zero_mean"], "extras": []},
}


# ---------------------------------------------------------------------------
# Scenario set-up
# ---------------------------------------------------------------------------


def latent_model(params: dict[str, Any]) -> psu.VariogramModel:
    nug = float(params["nugget"])
    return psu.vgm(1.0 - nug, params["model"], float(params["range"]), nugget=nug)


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
    names: list[str], p: cand.Pairs, x: np.ndarray, y: np.ndarray, e: np.ndarray, s: np.ndarray
) -> dict[str, psu.VariogramModel]:
    z = e / s
    ev = cand.matheron(p, z)
    fits = {
        "current": lambda: cand.wls(ev),
        "wls_npairs": lambda: cand.wls(ev, fit_method=1),
        "wls_cressie": lambda: cand.wls(ev, fit_method=2),
        "exp_only": lambda: cand.wls(ev, models=("Exp",)),
        "sigma_weighted": lambda: cand.wls(cand.sigma_weighted(p, e, s)),
        "reml": lambda: cand.likelihood_fit(x, y, z),
        "ml_zero_mean": lambda: cand.likelihood_fit(x, y, z, mean="zero"),
    }
    return {name: fits[name]() for name in names}


def block_sds(
    variances: dict[str, np.ndarray], rescale: float, extras: list[str]
) -> dict[str, np.ndarray]:
    """Block SDs of every variant from the MC variances, plus the rescaled current SD.

    ``rescale`` is the mean squared standardized validation error, which estimates
    (true σ / σ handed to the method)².
    """
    out = {name: np.sqrt(np.maximum(v, 0.0)) for name, v in variances.items()}
    if "rescale" in extras:
        out["rescale"] = out["current"] * np.sqrt(rescale)
    return out


def realization_continuous(
    setup: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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
    models = fit_models(setup["fits"], cand.pairs(x, y), x, y, e, s)
    integrands = [psu.Integrand(sd_map, cand.correlation(m)) for m in models.values()]
    mc = psu.mc_block_variance(resolved, integrands, n_mc=N_MC, seed=rng)
    variances = dict(zip(models, mc.variance, strict=True))
    sds = block_sds(variances, float(np.mean((e / s) ** 2)), setup["extras"])
    return err[None, :], {k: v[None, :] for k, v in sds.items()}


def realization_categorical(
    setup: dict[str, Any], rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    sc, resolved, eb = setup["sc"], setup["resolved"], setup["eb"]
    p_user, sd_user, pred = setup["p_user"], setup["sd_user"], setup["pred"]
    real = sc.simulate(rng)
    v = real.validation
    x, y = v["x"].to_numpy(), v["y"].to_numpy()
    cell = v["cell"].to_numpy()
    labels = np.asarray(v["label"].to_list(), dtype=object)
    p = cand.pairs(x, y)
    K = len(sc.classes)
    per_class: list[dict[str, psu.VariogramModel]] = []
    zs, rescale = [], []
    for k, cls in enumerate(sc.classes):
        pk = p_user[k, cell]
        sk = np.sqrt(pk * (1.0 - pk))
        ek = (labels == cls).astype(np.float64) - pk
        per_class.append(fit_models(setup["fits"], p, x, y, ek, sk))
        zs.append(ek / sk)
        rescale.append(float(np.mean((ek / sk) ** 2)))
    names = list(setup["fits"])
    if "pooled" in setup["extras"]:
        names.append("pooled")
        pooled_model = cand.wls(cand.pooled(p, zs))
        per_class = [m | {"pooled": pooled_model} for m in per_class]
    integrands = []
    for k in range(K):
        models = per_class[k]
        integrands += [psu.Integrand(sd_user[k], cand.correlation(models[n])) for n in names]
    mc = psu.mc_block_variance(resolved, integrands, n_mc=N_MC, seed=rng)
    var = mc.variance.reshape(K, len(names), -1)
    truth = np.stack([eb.block_means(real.fraction_sub[k]) for k in range(K)])
    per_k = [
        block_sds(dict(zip(names, var[k], strict=True)), rescale[k], setup["extras"])
        for k in range(K)
    ]
    return truth - pred, {n: np.stack([sds[n] for sds in per_k]) for n in per_k[0]}


def worker(task: dict[str, Any]) -> dict[str, Any]:
    setup = grid_setup(task["mode"], task["params"]) | ROUNDS[task["round"]]
    fn = realization_continuous if task["mode"] == "continuous" else realization_categorical
    errs: list[np.ndarray] = []
    res: dict[str, list[np.ndarray]] = {}
    for i in range(task["start"], task["stop"]):
        rng = np.random.default_rng([SEED, task["scenario_index"], i])
        err, out = fn(setup, rng)
        errs.append(err)
        for name, sd in out.items():
            res.setdefault(name, []).append(sd)
    return {"err": np.array(errs), "res": {n: np.array(v) for n, v in res.items()}}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metrics(
    err: np.ndarray,
    res: dict[str, np.ndarray],
    sd_true: np.ndarray,
    groups: tuple[str, ...],
    keys: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Block-level and (realization, group)-level tables.

    err and every SD array in ``res`` have shape (R, B); sd_true has shape (B,).
    """
    block_rows, real_rows = [], []
    g = np.asarray(groups)
    for name, sd in res.items():
        covered = np.abs(err) <= Z * sd
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
    ap.add_argument("--round", type=int, choices=sorted(ROUNDS), default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
        os.environ.setdefault(var, "1")
    scen = SCENARIOS[args.mode]
    names = args.scenarios.split(",") if args.scenarios else list(scen)
    n_real = 4 if args.quick else args.realizations
    suffix = "_quick" if args.quick else ""
    out_dir = RESULTS / f"round{args.round}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        t0 = time.time()
        params = scen[name]
        si = list(SCENARIOS[args.mode]).index(name) + (100 if args.mode == "categorical" else 0)
        tasks = [
            {
                "mode": args.mode,
                "round": args.round,
                "params": params,
                "scenario_index": si,
                "start": a,
                "stop": b,
            }
            for a, b in chunks(n_real, max(1, n_real // (3 * N_WORKERS)))
        ]
        out = run_parallel(worker, tasks, N_WORKERS)
        err = np.concatenate([o["err"] for o in out])  # (R, K, B)
        res = {v: np.concatenate([o["res"][v] for o in out]) for v in out[0]["res"]}
        setup = grid_setup(args.mode, params)
        blocks, reals = [], []
        for k in range(err.shape[1]):
            cls = None if args.mode == "continuous" else str(setup["sc"].classes[k])
            sd_t = true_sd(args.mode, setup, err, k)
            keys = {"mode": args.mode, "scenario": name, "class": cls}
            b, r = metrics(
                err[:, k, :],
                {v: sd[:, k, :] for v, sd in res.items()},
                sd_t,
                setup["groups"],
                keys,
            )
            blocks.append(b)
            reals.append(r)
        pl.concat(blocks, how="diagonal_relaxed").write_parquet(
            out_dir / f"{args.mode}_{name}_blocks{suffix}.parquet", compression="zstd"
        )
        pl.concat(reals, how="diagonal_relaxed").write_parquet(
            out_dir / f"{args.mode}_{name}_realizations{suffix}.parquet", compression="zstd"
        )
        print(
            f"[{args.mode}] {name}: {n_real} realizations in {time.time() - t0:.0f} s", flush=True
        )


if __name__ == "__main__":
    main()
