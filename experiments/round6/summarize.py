"""Interval rules, cells and the universal scorecard of round-6 runs.

Interval rules for a block estimate and its SD (90 %); continuous maps use ``normal`` only:

* ``normal``: est ± z·sd;
* ``logit``: expit(logit e ± z·sd / (e(1 − e))), e = est clipped to [1e-6, 1 − 1e-6];
* ``hybrid``: the union of the normal interval clipped to [0, 1] and the logit interval
  (normal bound on the side of the nearer boundary, logit bound on the far side);
* ``arcsine``: sin²(asin √e ± z·sd / (2√(e(1 − e)))), the angle kept in [0, π/2].

A cell is one landscape side × map × design × n × class × block size. Per cell and rule:
coverage (over blocks and realizations), the SD ratio 0.674 / median |error ÷ SD|, median
|error|, and counts of out-of-range (outside [0, 1], class maps) and wild (|error ÷ SD| > 4)
estimates. The universal error of an arm is the mean |coverage − 0.90| over all cells.

Usage: ``uv run python -m experiments.round6.summarize <run_dir> <out.json> --kind class
--sides 1024 --maps M1,M2,M3,M7 [--scorecard scorecard.md]``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from experiments.round5 import shared as sh

LEVEL = sh.LEVEL
Z = sh.Z
ROB = 0.6744897501960817
EPS = 1e-6
RULES_CLASS = ("normal", "logit", "hybrid", "arcsine")
RULES_CONT = ("normal",)
DIMS = ("side", "map", "design", "n", "cls", "bside")


def intervals(est: np.ndarray, sd: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Lower and upper bounds of every rule (numpy version of the expressions in `load`)."""
    est = np.asarray(est, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    e = np.clip(est, EPS, 1 - EPS)
    lg = np.log(e / (1 - e))
    hw = Z * sd / (e * (1 - e))
    out = {"normal": (est - Z * sd, est + Z * sd)}
    with np.errstate(over="ignore"):
        out["logit"] = (1 / (1 + np.exp(-(lg - hw))), 1 / (1 + np.exp(-(lg + hw))))
    nlo, nhi = np.clip(out["normal"][0], 0, 1), np.clip(out["normal"][1], 0, 1)
    out["hybrid"] = (np.minimum(nlo, out["logit"][0]), np.maximum(nhi, out["logit"][1]))
    th = np.arcsin(np.sqrt(e))
    hw_a = Z * sd / (2 * np.sqrt(e * (1 - e)))
    out["arcsine"] = (
        np.sin(np.clip(th - hw_a, 0, math.pi / 2)) ** 2,
        np.sin(np.clip(th + hw_a, 0, math.pi / 2)) ** 2,
    )
    return out


def _bounds(rule: str) -> tuple[pl.Expr, pl.Expr]:
    est, sd = pl.col("est"), pl.col("sd")
    e = est.clip(EPS, 1 - EPS)
    lg = (e / (1 - e)).log()
    hw = Z * sd / (e * (1 - e))
    if rule == "normal":
        return est - Z * sd, est + Z * sd
    llo, lhi = 1 / (1 + (-(lg - hw)).exp()), 1 / (1 + (-(lg + hw)).exp())
    if rule == "logit":
        return llo, lhi
    if rule == "hybrid":
        nlo, nhi = (est - Z * sd).clip(0, 1), (est + Z * sd).clip(0, 1)
        return pl.min_horizontal(nlo, llo), pl.max_horizontal(nhi, lhi)
    th = e.sqrt().arcsin()
    hw_a = Z * sd / (2 * (e * (1 - e)).sqrt())
    return (th - hw_a).clip(0, math.pi / 2).sin() ** 2, (th + hw_a).clip(0, math.pi / 2).sin() ** 2


def load(run_dir: Path, side: int, map_name: str, rules: tuple[str, ...]) -> pl.DataFrame:
    files = sorted((run_dir / str(side) / map_name).glob("r*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    truth = df.filter(pl.col("arm") == "truth")
    on = (
        ["r", "cls", "block"]
        if (truth["design"] == "-").all()
        else ["r", "design", "n", "cls", "block"]
    )
    truth = truth.select(*on, pl.col("est").alias("truth"))
    df = df.filter(pl.col("arm") != "truth").join(truth, on=on)
    df = df.with_columns(sd=pl.col("var").clip(lower_bound=0.0).sqrt())
    covers = []
    for rule in rules:
        lo, hi = _bounds(rule)
        covers.append(((pl.col("truth") >= lo) & (pl.col("truth") <= hi)).alias(f"cov_{rule}"))
    return df.with_columns(
        *covers,
        resid=pl.col("truth") - pl.col("est"),
        out=(pl.col("est") < 0) | (pl.col("est") > 1),
    ).with_columns(az=(pl.col("resid") / pl.col("sd")).abs())


def cells(df: pl.DataFrame, rules: tuple[str, ...]) -> pl.DataFrame:
    keys = ["arm", *DIMS]
    return df.group_by(keys).agg(
        pl.len().alias("n_rows"),
        pl.col("r").n_unique().alias("n_real"),
        *[pl.col(f"cov_{rule}").mean().alias(f"cov_{rule}") for rule in rules],
        (ROB / pl.col("az").median()).alias("rob"),
        pl.col("resid").abs().median().alias("mae"),
        pl.col("out").sum().alias("n_out"),
        (pl.col("az") > 4).sum().alias("n_wild"),
    )


def per_r(df: pl.DataFrame, rules: tuple[str, ...]) -> pl.DataFrame:
    """Coverage per cell and realization (mean over the cell's blocks), for the bootstrap."""
    return df.group_by(["arm", *DIMS, "r"]).agg(
        *[pl.col(f"cov_{rule}").mean().alias(f"cov_{rule}") for rule in rules]
    )


def noise_floor(n_real: int) -> float:
    """Mean |coverage − 0.90| of a perfect method on a single-block cell with n_real draws."""
    k = np.arange(n_real + 1)
    return float(np.sum(stats.binom.pmf(k, n_real, LEVEL) * np.abs(k / n_real - LEVEL)))


def universal(cl: pl.DataFrame, arm: str, rule: str, ref: str = "ok") -> dict[str, object]:
    a = cl.filter(pl.col("arm") == arm)
    if a.height == 0:
        return {}
    err = (pl.col(f"cov_{rule}") - LEVEL).abs()
    keys = list(DIMS)
    o = cl.filter(pl.col("arm") == ref).select(*keys, pl.col("mae").alias("mae_ref"))
    acc = a.join(o, on=keys).select((pl.col("mae") / pl.col("mae_ref")).median()).item()
    land = a.filter(pl.col("bside") == pl.col("side"))
    row: dict[str, object] = {
        "arm": arm,
        "rule": rule,
        "cells": a.height,
        "err": float(a.select(err.mean()).item()),
        "worst": float(a[f"cov_{rule}"].min()),
        "within": float(a.select(((pl.col(f"cov_{rule}") - LEVEL).abs() <= 0.05).mean()).item()),
        "acc": float(acc) if acc is not None else math.nan,
        "out_land": int(land["n_out"].sum()),
        "wild_land": int(land["n_wild"].sum()),
    }
    for dim in DIMS:
        g = a.group_by(dim).agg(err.mean().alias("e")).sort(dim)
        row[f"by_{dim}"] = {str(k): float(v) for k, v in zip(g[dim], g["e"], strict=True)}
    return row


def paired_ci(
    pr: pl.DataFrame, arm: str, rule: str, base: str, base_rule: str, n_boot: int = 400
) -> tuple[float, float]:
    """Paired bootstrap 90 % interval (over realizations) of err(arm, rule) − err(base)."""
    keys = list(DIMS)

    def matrix(name: str, rl: str) -> tuple[np.ndarray, list[int]]:
        w = (
            pr.filter(pl.col("arm") == name)
            .select(*keys, "r", pl.col(f"cov_{rl}").alias("c"))
            .pivot(on="r", index=keys, values="c")
            .sort(keys)
        )
        rs = sorted(int(c) for c in w.columns if c not in keys)
        return w.select([str(r) for r in rs]).to_numpy(), rs

    A, ra = matrix(arm, rule)
    B, rb = matrix(base, base_rule)
    if A.shape != B.shape or ra != rb:
        return math.nan, math.nan
    rng = np.random.default_rng(0)
    R = A.shape[1]
    d = np.empty(n_boot)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
        wa = np.nansum(A * cnt, axis=1) / np.sum(np.where(np.isnan(A), 0, cnt), axis=1)
        wb = np.nansum(B * cnt, axis=1) / np.sum(np.where(np.isnan(B), 0, cnt), axis=1)
        d[b] = np.nanmean(np.abs(wa - LEVEL)) - np.nanmean(np.abs(wb - LEVEL))
    return float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))


def advance(row: dict, base: dict, ci: tuple[float, float]) -> dict[str, object]:
    worse = max(
        (row[f"by_{dim}"][k] - base[f"by_{dim}"][k])
        for dim in DIMS
        for k in row[f"by_{dim}"]
        if k in base[f"by_{dim}"]
    )
    passes = (
        row["err"] < base["err"]
        and worse <= 0.02
        and row["acc"] <= 1.10
        and row["out_land"] <= base["out_land"]
        and row["wild_land"] <= base["wild_land"]
    )
    return {
        "arm": row["arm"],
        "rule": row["rule"],
        "base": f"{base['arm']}|{base['rule']}",
        "d_err": row["err"] - base["err"],
        "d_lo": ci[0],
        "d_hi": ci[1],
        "worst_marginal": worse,
        "acc": row["acc"],
        "out": row["out_land"] - base["out_land"],
        "wild": row["wild_land"] - base["wild_land"],
        "advances": passes,
    }


def markdown(rows: list[dict], keys: list[str]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return "nan" if math.isnan(v) else f"{v:.3f}"
        return str(v)

    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out_json")
    ap.add_argument("--kind", choices=("class", "cont"), default="class")
    ap.add_argument("--sides", default="1024")
    ap.add_argument("--maps", default="M1,M2,M3,M7")
    ap.add_argument("--scorecard", default=None)
    ap.add_argument(
        "--compare",
        default="",
        help="comma-separated arm|rule:base|rule pairs for the advance rule",
    )
    a = ap.parse_args()
    rules = RULES_CLASS if a.kind == "class" else RULES_CONT
    run_dir = Path(a.run_dir)
    frames = []
    for side in (int(s) for s in a.sides.split(",")):
        for m in a.maps.split(","):
            if (run_dir / str(side) / m).exists():
                frames.append(load(run_dir, side, m, rules))
    df = pl.concat(frames, how="vertical_relaxed")
    cl = cells(df, rules)
    pr = per_r(df, rules)
    arms = sorted(cl["arm"].unique().to_list())
    score = [
        row
        for arm in arms
        for rule in rules
        if (row := universal(cl, arm, rule)) and not (arm == "olofsson" and rule != "normal")
    ]
    index = {(r["arm"], r["rule"]): r for r in score}
    adv = []
    for pair in filter(None, a.compare.split(",")):
        left, right = pair.split(":")
        key, bkey = tuple(left.split("|")), tuple(right.split("|"))
        if key in index and bkey in index:
            ci = paired_ci(pr, key[0], key[1], bkey[0], bkey[1])
            adv.append(advance(index[key], index[bkey], ci))
    n_real = int(cl["n_real"].max())
    out = {
        "kind": a.kind,
        "n_real": n_real,
        "floor_land": noise_floor(n_real),
        "score": score,
        "advance": adv,
        "cells": cl.to_dicts(),
    }
    Path(a.out_json).write_text(json.dumps(out, allow_nan=True))
    if a.scorecard:
        flat = [
            {
                **{k: v for k, v in r.items() if not k.startswith("by_")},
                **{
                    f"{d}:{k}": v for d in ("bside", "n", "design") for k, v in r[f"by_{d}"].items()
                },
            }
            for r in score
        ]
        keys = ["arm", "rule", "err", "worst", "within", "acc", "out_land", "wild_land"]
        keys += sorted({k for r in flat for k in r if ":" in k}, key=lambda k: (k.split(":")[0], k))
        n_cells = score[0]["cells"] if score else 0
        text = [
            f"Universal error = mean |coverage − 0.90| over {n_cells} cells; "
            f"noise floor of a landscape cell with {n_real} realizations: "
            f"{noise_floor(n_real):.3f}.",
            "",
            markdown(flat, keys),
        ]
        if adv:
            text += [
                "",
                "### Advance rule",
                "",
                markdown(
                    adv,
                    ["arm", "rule", "base", "d_err", "d_lo", "d_hi", "worst_marginal", "acc",
                     "out", "wild", "advances"],
                ),
            ]  # fmt: skip
        Path(a.scorecard).write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
