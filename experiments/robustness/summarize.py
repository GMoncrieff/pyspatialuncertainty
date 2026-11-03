"""Pre-registered scorecards of the robustness study.

A cell is one side × generator × map state × design × n_val × n_cal × n_train × class ×
block size. Coverage of normal 90 % intervals (``normal``) is the scored rule; ``clip``
(the same interval around the estimate clipped to [0, 1]) is a diagnostic only. The universal
error of an arm is the mean |coverage − 0.90| over cells, every cell weighted equally
(``experiments.round6.summarize.universal``).

Scorecards (each its own set of cells):

* ``core``: 256², all generators, n_cal = n_val, n_train = 1,000, every realization;
* ``sizes``: ``base``, ``nonstat``, ``classifier``, ``potts`` at every side run, r < 20;
* ``cross``: the same four generators at 256², n_val × n_cal, r < 20;
* ``training``: ``classifier`` and ``potts`` at 256² with n_train 200 / 1,000 / 5,000, r < 20;
* ``all``: every cell of every run.

For each: the universal error, worst cell and marginals per arm, overall and per generator;
the ranking tests d1 = err(R6) − err(ok) and d2 = err(ok) − err(cur) with paired bootstrap
90 % CIs over realizations, per generator and per level of the satellite's own dimension;
estimates outside [0, 1].

Usage: ``uv run python -m experiments.robustness.summarize <run_dir> <out.json>
[--scorecard scorecard.md]``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from experiments.robustness import gens as G  # noqa: N812
from experiments.robustness.run import R6
from experiments.round6 import summarize as s6

LEVEL = s6.LEVEL
Z = s6.Z
RULES = ("normal", "clip")
DIMS = ("side", "gen", "state", "design", "n", "n_cal", "n_train", "cls", "role", "bside")
ARMS = ("cur", "ok", R6)
SHORT = {"cur": "cur", "ok": "ok", R6: "R6", "olofsson": "olofsson"}
SIZE_GENS = ("base", "nonstat", "classifier", "potts")
TRAIN_GENS = ("classifier", "potts")
R_SAT = 20


def load_dir(d: Path, gen: str, prefixes: tuple[str, ...]) -> pl.DataFrame | None:
    files = sorted(f for p in prefixes for f in d.glob(f"{p}*.parquet"))
    if not files:
        return None
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    truth = (
        df.filter(pl.col("arm") == "truth")
        .unique(["r", "cls", "block"])
        .select("r", "cls", "block", pl.col("est").alias("truth"))
    )
    roles = G.GENS[gen].roles
    df = df.filter(pl.col("arm") != "truth").join(truth, on=["r", "cls", "block"])
    df = df.with_columns(
        sd=pl.col("var").clip(lower_bound=0.0).sqrt(),
        role=pl.col("cls").replace_strict(roles),
    )
    lo, hi = pl.col("est") - Z * pl.col("sd"), pl.col("est") + Z * pl.col("sd")
    ec = pl.col("est").clip(0.0, 1.0)
    t = pl.col("truth")
    return df.with_columns(
        cov_normal=(t >= lo) & (t <= hi),
        cov_clip=(t >= ec - Z * pl.col("sd")) & (t <= ec + Z * pl.col("sd")),
        resid=t - pl.col("est"),
        out=(pl.col("est") < 0) | (pl.col("est") > 1),
    ).with_columns(az=(pl.col("resid") / pl.col("sd")).abs())


def collect(run_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Cells and per-realization coverage of every run, all realizations and r < R_SAT."""
    cls_all, pr_all, cls_sat, pr_sat = [], [], [], []
    for side_dir in sorted(p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        for gdir in sorted(p for p in side_dir.iterdir() if p.is_dir()):
            gen = gdir.name.split("-t")[0]
            for sdir in sorted(p for p in gdir.iterdir() if p.is_dir()):
                df = load_dir(sdir, gen, ("r", "o"))
                if df is None:
                    continue
                cls_all.append(s6.cells(df, RULES, DIMS))
                pr_all.append(s6.per_r(df, RULES, DIMS))
                sub = df.filter(pl.col("r") < R_SAT)
                cls_sat.append(s6.cells(sub, RULES, DIMS))
                pr_sat.append(s6.per_r(sub, RULES, DIMS))
    cat = lambda f: pl.concat(f, how="vertical_relaxed")  # noqa: E731
    return cat(cls_all), cat(pr_all), cat(cls_sat), cat(pr_sat)


def ranking(cl: pl.DataFrame, pr: pl.DataFrame) -> dict[str, object]:
    """Errors of the three arms, d1 = R6 − ok and d2 = ok − cur with paired 90 % CIs."""
    err = {
        SHORT[a]: float(
            cl.filter(pl.col("arm") == a).select((pl.col("cov_normal") - LEVEL).abs().mean()).item()
        )
        for a in ARMS
        if cl.filter(pl.col("arm") == a).height
    }
    if len(err) < 3:
        return {"err": err}
    d1 = s6.paired_ci(pr, R6, "normal", "ok", "normal", dims=DIMS)
    d2 = s6.paired_ci(pr, "ok", "normal", "cur", "normal", dims=DIMS)
    e1, e2 = err["R6"] - err["ok"], err["ok"] - err["cur"]

    def verdict() -> str:
        if d1[0] > 0 or d2[0] > 0:
            return "reversed"
        if d1[1] < 0 and d2[1] < 0:
            return "clear"
        if e1 < 0 and e2 < 0:
            return "holds"
        return "tie"

    return {
        "err": err,
        "d1": e1,
        "d1_ci": list(d1),
        "d2": e2,
        "d2_ci": list(d2),
        "verdict": verdict(),
        "cells": int(cl.filter(pl.col("arm") == R6).height),
    }


def worst(cl: pl.DataFrame, arm: str) -> dict[str, object]:
    a = cl.filter(pl.col("arm") == arm).sort("cov_normal")
    if a.height == 0:
        return {}
    row = a.row(0, named=True)
    return {k: row[k] for k in (*DIMS, "cov_normal", "n_real")}


def out_of_range(cl: pl.DataFrame) -> list[dict[str, object]]:
    """Estimates outside [0, 1] per arm and generator, at the landscape and at all blocks."""
    land = pl.col("bside") == pl.col("side")
    g = cl.group_by("arm", "gen").agg(
        pl.col("n_out").sum().alias("out_all"),
        pl.col("n_rows").sum().alias("rows_all"),
        pl.col("n_out").filter(land).sum().alias("out_land"),
        pl.col("n_rows").filter(land).sum().alias("rows_land"),
    )
    return g.sort("gen", "arm").to_dicts()


def scorecard(cl: pl.DataFrame, pr: pl.DataFrame, level_dim: str | None) -> dict[str, object]:
    arms = [a for a in (*ARMS, "olofsson") if cl.filter(pl.col("arm") == a).height]
    score = [
        row
        for a in arms
        for rule in RULES
        if (row := s6.universal(cl, a, rule, dims=DIMS))
        and not (a == "olofsson" and rule != "normal")
    ]
    gens = sorted(cl["gen"].unique().to_list(), key=lambda g: G.GENS[g].gid)
    per_gen = {}
    for g in gens:
        c, p = cl.filter(pl.col("gen") == g), pr.filter(pl.col("gen") == g)
        per_gen[g] = ranking(c, p) | {
            "worst": {SHORT[a]: worst(c, a) for a in ARMS},
            "distance": G.GENS[g].distance,
        }
    levels = {}
    if level_dim:
        for v in sorted(cl[level_dim].unique().to_list()):
            c, p = cl.filter(pl.col(level_dim) == v), pr.filter(pl.col(level_dim) == v)
            levels[str(v)] = ranking(c, p)
    n_real = int(cl["n_real"].max())
    return {
        "n_real": n_real,
        "floor_land": s6.noise_floor(n_real),
        "overall": ranking(cl, pr),
        "gen_mean": {
            SHORT[a]: float(np.mean([per_gen[g]["err"][SHORT[a]] for g in gens]))
            for a in ARMS
            if all(SHORT[a] in per_gen[g]["err"] for g in gens)
        },
        "score": score,
        "per_gen": per_gen,
        "level_dim": level_dim,
        "levels": levels,
        "worst": {SHORT[a]: worst(cl, a) for a in ARMS},
        "out_of_range": out_of_range(cl),
    }


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _f(v: object, nd: int = 3) -> str:
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def _table(rows: list[list[object]], head: list[str]) -> list[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(_f(v) for v in r) + " |" for r in rows]
    return out


def _rank_row(name: str, rk: dict) -> list[object]:
    e = rk["err"]
    if "d1" not in rk:
        return [name, *(e.get(k, math.nan) for k in ("cur", "ok", "R6")), "", "", ""]
    ci = lambda c: f"[{c[0]:+.3f}, {c[1]:+.3f}]"  # noqa: E731
    return [
        name, e["cur"], e["ok"], e["R6"],
        f"{rk['d1']:+.3f} {ci(rk['d1_ci'])}", f"{rk['d2']:+.3f} {ci(rk['d2_ci'])}", rk["verdict"],
    ]  # fmt: skip


def markdown(name: str, sc: dict) -> list[str]:
    head = ["", "cur", "ok", "R6", "d1 = R6 − ok [90 % CI]", "d2 = ok − cur [90 % CI]", "ranking"]
    lines = [
        f"## {name}",
        "",
        f"{sc['overall'].get('cells', 0)} cells per arm; up to {sc['n_real']} realizations "
        f"per cell; noise floor of a landscape cell {sc['floor_land']:.3f}.",
        "",
        *_table([_rank_row("all cells", sc["overall"])], head),
        "",
        "Per generator (mean of per-generator errors: "
        + ", ".join(f"{k} {v:.3f}" for k, v in sc["gen_mean"].items())
        + "):",
        "",
        *_table([_rank_row(f"{g} ({v['distance']})", v) for g, v in sc["per_gen"].items()], head),
    ]
    if sc["levels"]:
        lines += [
            "",
            f"Per {sc['level_dim']}:",
            "",
            *_table([_rank_row(k, v) for k, v in sc["levels"].items()], head),
        ]
    dims = ("bside", "n", "n_cal", "n_train", "design", "side", "state", "gen", "role")
    marg = []
    for row in sc["score"]:
        if row["rule"] != "normal" or row["arm"] == "olofsson":
            continue
        for d in dims:
            by = row[f"by_{d}"]
            if len(by) > 1:
                marg.append([SHORT[row["arm"]], d, *[f"{k}: {v:.3f}" for k, v in by.items()]])
    width = max(len(m) for m in marg) if marg else 2
    marg = [m + [""] * (width - len(m)) for m in marg]
    lines += [
        "",
        "Marginals (normal intervals):",
        "",
        *_table(marg, ["arm", "dim"] + [""] * (width - 2)),
    ]
    lines += [
        "",
        "Universal scorecard (`clip` = interval around the estimate clipped to [0, 1], "
        "diagnostic only):",
        "",
        *_table(
            [
                [SHORT[r["arm"]], r["rule"], r["err"], r["worst"], r["within"], r["acc"],
                 r["out_land"], r["wild_land"]]
                for r in sc["score"]
            ],
            ["arm", "rule", "err", "worst", "within", "acc", "out_land", "wild_land"],
        ),
    ]  # fmt: skip
    lines += ["", "Worst cell per arm:", ""]
    wrows = []
    for a, w in sc["worst"].items():
        if w:
            wrows.append([a, w["cov_normal"], *(w[k] for k in DIMS)])
    lines += _table(wrows, ["arm", "coverage", *DIMS])
    oor = [
        [SHORT.get(o["arm"], o["arm"]), o["gen"], o["out_land"], o["rows_land"], o["out_all"],
         o["rows_all"]]
        for o in sc["out_of_range"]
        if o["arm"] in ARMS and o["out_all"] > 0
    ]  # fmt: skip
    if oor:
        lines += [
            "",
            "Estimates outside [0, 1] (arms and generators with any):",
            "",
            *_table(oor, ["arm", "gen", "landscape", "of", "all blocks", "of"]),
        ]
    return [*lines, ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_json")
    ap.add_argument("--scorecard", default=None)
    a = ap.parse_args()
    cl_all, pr_all, cl_sat, pr_sat = collect(Path(a.run_dir))
    diag = (pl.col("n") == pl.col("n_cal")) & pl.col("n_train").is_in([0, G.N_TRAIN])
    sets = {
        "core": (cl_all, pr_all, (pl.col("side") == 256) & diag, None),
        "sizes": (cl_sat, pr_sat, pl.col("gen").is_in(SIZE_GENS) & diag, "side"),
        "cross": (
            cl_sat, pr_sat,
            (pl.col("side") == 256) & pl.col("gen").is_in(SIZE_GENS)
            & pl.col("n_train").is_in([0, G.N_TRAIN]),
            "n_cal",
        ),
        "training": (
            cl_sat, pr_sat,
            (pl.col("side") == 256) & pl.col("gen").is_in(TRAIN_GENS)
            & (pl.col("n") == pl.col("n_cal")),
            "n_train",
        ),
        "all": (cl_all, pr_all, pl.lit(True), None),
    }  # fmt: skip
    out: dict[str, object] = {}
    for name, (cl, pr, cond, level) in sets.items():
        c, p = cl.filter(cond), pr.filter(cond)
        if c.height == 0:
            continue
        if name in ("sizes", "cross", "training") and c[level].n_unique() < 2:
            continue
        out[name] = scorecard(c, p, level)
    out["cells"] = cl_all.to_dicts()
    Path(a.out_json).write_text(json.dumps(out, allow_nan=True, default=str))
    if a.scorecard:
        text = ["# Robustness scorecard", ""]
        for name in ("core", "sizes", "cross", "training", "all"):
            if name in out:
                text += markdown(name, out[name])  # type: ignore[arg-type]
        Path(a.scorecard).write_text("\n".join(text))


if __name__ == "__main__":
    main()
