"""Summaries and scorecard of a round-5 run.

Per block (over realizations): coverage of the normal 90 % interval and of the logit-scale
interval, the robust SD ratio 0.674 / median |error ÷ SD|, RMSE, median |error| (of the
estimate and of the estimate clipped to [0, 1]), and counts of wild (|error ÷ SD| > 4) and
out-of-range estimates. Per block size: the mean and spread of these over blocks.

Scorecard per map and arm, over the 18 settings (3 classes × 3 sizes × 2 designs):

* ``land_err`` / ``t256_err``: mean |coverage − 0.90| at the landscape / 256-cell tiles;
* ``cov4`` / ``cov64``: mean coverage at 4- and 64-cell tiles;
* ``worst_land``: the lowest landscape coverage;
* ``acc``: median over settings of the landscape median |error| relative to ``ok``;
* ``wild`` / ``out``: counts at the landscape;
* paired bootstrap 90 % intervals (over realizations) of the differences from ``ok``.

Usage: ``uv run python -m experiments.round5.summarize <run_dir> <out.json> [--maps M1,M2,M3]
[--scorecard scorecard.md]``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from experiments.round5 import shared as sh

KEYS = ["arm", "design", "n", "cls"]
ROB = 0.6744897501960817


def load(run_dir: Path, map_name: str) -> pl.DataFrame:
    files = sorted((run_dir / map_name).glob("r*.parquet"))
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    truth = df.filter(pl.col("arm") == "truth")
    # class maps: one truth per realization; continuous maps: one per setting
    on = (
        ["r", "cls", "block"]
        if (truth["design"] == "-").all()
        else ["r", "design", "n", "cls", "block"]
    )
    truth = truth.select(*on, pl.col("est").alias("truth"))
    sides = pl.DataFrame({"block": np.arange(529, dtype=np.int16), "side": build_sides()})
    df = df.filter(pl.col("arm") != "truth").join(truth, on=on).join(sides, on="block")
    sd = pl.col("var").cast(pl.Float64).clip(lower_bound=0.0).sqrt()
    est_c = pl.col("est").clip(1e-6, 1 - 1e-6)
    hw = sh.Z * sd / (est_c * (1 - est_c))
    logit = (est_c / (1 - est_c)).log()
    lo = 1 / (1 + (-(logit - hw)).exp())
    hi = 1 / (1 + (-(logit + hw)).exp())
    return (
        df.with_columns(sd=sd)
        .with_columns(
            resid=pl.col("truth") - pl.col("est"),
            resid_clip=pl.col("truth") - pl.col("est").clip(0.0, 1.0),
            cover=((pl.col("truth") - pl.col("est")).abs() <= sh.Z * pl.col("sd")),
            cover_lgt=(pl.col("truth") >= lo) & (pl.col("truth") <= hi),
            out=(pl.col("est") < 0) | (pl.col("est") > 1),
        )
        .with_columns(az=(pl.col("resid") / pl.col("sd")).abs())
    )


def build_sides() -> np.ndarray:
    return np.repeat(sh.SIDES, [256, 256, 16, 1]).astype(np.int64)


def per_block(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by([*KEYS, "side", "block"]).agg(
        n_valid=pl.len(),
        cover=pl.col("cover").mean(),
        cover_lgt=pl.col("cover_lgt").mean(),
        rmse=(pl.col("resid") ** 2).mean().sqrt(),
        mae=pl.col("resid").abs().median(),
        mae_clip=pl.col("resid_clip").abs().median(),
        rob=ROB / pl.col("az").median(),
        n_out=pl.col("out").sum(),
        n_wild=(pl.col("az") > 4).sum(),
    )


def q(col: str, p: float) -> pl.Expr:
    return pl.col(col).quantile(p, interpolation="linear")


def per_side(pb: pl.DataFrame) -> pl.DataFrame:
    return (
        pb.group_by([*KEYS, "side"])
        .agg(
            n_blocks=pl.len(),
            n_valid=pl.col("n_valid").min(),
            cov_mean=pl.col("cover").mean(),
            cov_min=pl.col("cover").min(),
            cov_max=pl.col("cover").max(),
            cov_p10=q("cover", 0.1),
            cov_p90=q("cover", 0.9),
            lgt_mean=pl.col("cover_lgt").mean(),
            lgt_min=pl.col("cover_lgt").min(),
            lgt_max=pl.col("cover_lgt").max(),
            lgt_p10=q("cover_lgt", 0.1),
            lgt_p90=q("cover_lgt", 0.9),
            rob_mean=pl.col("rob").log().mean().exp(),
            rob_min=pl.col("rob").min(),
            rob_max=pl.col("rob").max(),
            rob_p10=q("rob", 0.1),
            rob_p90=q("rob", 0.9),
            rmse_mean=pl.col("rmse").mean(),
            mae_mean=pl.col("mae").mean(),
            mae_clip_mean=pl.col("mae_clip").mean(),
            n_out=pl.col("n_out").sum(),
            n_wild=pl.col("n_wild").sum(),
        )
        .sort([*KEYS, "side"])
    )


def scorecard(ps: pl.DataFrame, df: pl.DataFrame, n_boot: int = 300, seed: int = 0) -> pl.DataFrame:
    """One row per arm (the 18 settings pooled), with paired bootstrap CIs against ``ok``."""
    land = ps.filter(pl.col("side") == 1024)
    ok_mae = land.filter(pl.col("arm") == "ok").select(
        "design", "n", "cls", pl.col("mae_mean").alias("ok_mae")
    )
    rows = []
    for arm in sorted(ps["arm"].unique().to_list()):
        a = ps.filter(pl.col("arm") == arm)
        la = a.filter(pl.col("side") == 1024)
        t2 = a.filter(pl.col("side") == 256)
        acc = (
            la.join(ok_mae, on=["design", "n", "cls"])
            .select((pl.col("mae_mean") / pl.col("ok_mae")).median())
            .item()
        )
        rows.append({
            "arm": arm,
            "settings": la.height,
            "land_err": float((la["cov_mean"] - sh.LEVEL).abs().mean()) if la.height else math.nan,
            "land_err_lgt": (
                float((la["lgt_mean"] - sh.LEVEL).abs().mean()) if la.height else math.nan
            ),
            "t256_err": float((t2["cov_mean"] - sh.LEVEL).abs().mean()) if t2.height else math.nan,
            "t256_err_lgt": (
                float((t2["lgt_mean"] - sh.LEVEL).abs().mean()) if t2.height else math.nan
            ),
            "cov4": float(a.filter(pl.col("side") == 4)["cov_mean"].mean() or math.nan),
            "cov64": float(a.filter(pl.col("side") == 64)["cov_mean"].mean() or math.nan),
            "worst_land": float(la["cov_mean"].min()) if la.height else math.nan,
            "acc": float(acc) if acc is not None else math.nan,
            "wild": int(la["n_wild"].sum()),
            "out": int(la["n_out"].sum()),
        })  # fmt: skip
    sc = pl.DataFrame(rows)
    # paired bootstrap over realizations of the differences in land_err and t256_err from ok
    big = df.filter(pl.col("side") >= 256).select("r", *KEYS, "side", "block", "cover")
    reals = np.sort(big["r"].unique().to_numpy())
    rng = np.random.default_rng(seed)
    draws = [rng.choice(reals, reals.size, replace=True) for _ in range(n_boot)]
    cov = (
        big.group_by(["r", *KEYS, "side"])
        .agg(c=pl.col("cover").mean())
        .pivot(on="r", index=[*KEYS, "side"], values="c")
    )
    rcols = [str(r) for r in reals]
    mat = cov.select(rcols).to_numpy()  # (rows, realizations): block-mean coverage per realization
    keys = cov.select(*KEYS, "side")
    out = {}
    for side, name in ((1024, "land"), (256, "t256")):
        sel = (keys["side"] == side).to_numpy()
        arms = keys["arm"].to_numpy()[sel]
        m = mat[sel]
        okm = m[arms == "ok"]
        okkeys = (
            keys.filter(pl.col("side") == side)
            .filter(pl.col("arm") == "ok")
            .select("design", "n", "cls")
        )
        for arm in np.unique(arms):
            am = m[arms == arm]
            akeys = (
                keys.filter(pl.col("side") == side)
                .filter(pl.col("arm") == arm)
                .select("design", "n", "cls")
            )
            # align settings with ok
            idx_ok = {tuple(r): i for i, r in enumerate(okkeys.iter_rows())}
            pairs = [
                (i, idx_ok[tuple(r)]) for i, r in enumerate(akeys.iter_rows()) if tuple(r) in idx_ok
            ]
            if not pairs:
                continue
            ia, io = np.array(pairs).T
            diffs = []
            for d in draws:
                cols = np.searchsorted(reals, d)
                ca = np.nanmean(am[ia][:, cols], axis=1)
                co = np.nanmean(okm[io][:, cols], axis=1)
                diffs.append(np.mean(np.abs(ca - sh.LEVEL)) - np.mean(np.abs(co - sh.LEVEL)))
            out[(arm, name)] = np.percentile(diffs, [5, 95])
    sc = sc.with_columns(
        pl.Series("d_land_lo", [float(out.get((a, "land"), [math.nan] * 2)[0]) for a in sc["arm"]]),
        pl.Series("d_land_hi", [float(out.get((a, "land"), [math.nan] * 2)[1]) for a in sc["arm"]]),
        pl.Series("d_t256_lo", [float(out.get((a, "t256"), [math.nan] * 2)[0]) for a in sc["arm"]]),
        pl.Series("d_t256_hi", [float(out.get((a, "t256"), [math.nan] * 2)[1]) for a in sc["arm"]]),
    )
    return sc


def advance(scores: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """The pre-registered advance rule, against ok, across the main maps."""
    arms = sorted(set.intersection(*[set(s["arm"].to_list()) for s in scores.values()]))
    rows = []
    for arm in arms:
        if arm == "ok":
            continue
        d_err, d_small, d_acc, more_bad = [], [], [], False
        for s in scores.values():
            a = s.filter(pl.col("arm") == arm).row(0, named=True)
            o = s.filter(pl.col("arm") == "ok").row(0, named=True)
            d_err.append(((a["land_err"] - o["land_err"]) + (a["t256_err"] - o["t256_err"])) / 2)
            d_small.append(min(a["cov4"] - o["cov4"], a["cov64"] - o["cov64"]))
            d_acc.append(a["acc"])
            more_bad |= (a["wild"] > o["wild"]) or (a["out"] > o["out"])
        better = sum(d < 0 for d in d_err)
        rows.append({
            "arm": arm,
            "maps_better": better,
            "worst_increase": max(d_err),
            "small_cost": -min(d_small),
            "acc_max": max(d_acc),
            "more_wild_or_out": more_bad,
            "advances": (
                better >= 2 and max(d_err) <= 0.03 and -min(d_small) <= 0.02
                and max(d_acc) <= 1.10 and not more_bad
            ),
        })  # fmt: skip
    return pl.DataFrame(rows)


def markdown(df: pl.DataFrame) -> str:
    def fmt(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    head = "| " + " | ".join(df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join(
        [head, rule] + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.iter_rows()]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("out")
    ap.add_argument("--maps", default="M1,M2,M3")
    ap.add_argument("--scorecard")
    a = ap.parse_args()
    run_dir = Path(a.run_dir)
    res: dict[str, object] = {}
    scores = {}
    for m in a.maps.split(","):
        df = load(run_dir, m)
        pb = per_block(df)
        ps = per_side(pb)
        sc = scorecard(ps, df)
        scores[m] = sc
        land = df.filter(pl.col("side") == 1024).select("r", *KEYS, "est", "sd", "truth")
        diag = pl.concat(
            [pl.read_parquet(f) for f in sorted((run_dir / m).glob("d*.parquet"))],
            how="diagonal_relaxed",
        )
        res[m] = {
            "n_real": int(df["r"].n_unique()),
            "summary": ps.to_dicts(),
            "scorecard": sc.to_dicts(),
            "land": land.to_dicts(),
            "diag": diag.drop([c for c in diag.columns if diag[c].dtype == pl.Object]).to_dicts(),
        }
        with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=220, float_precision=3):
            print(f"== {m}: {res[m]['n_real']} realizations")  # type: ignore[index]
            print(
                sc.select(
                    "arm",
                    "land_err",
                    "land_err_lgt",
                    "t256_err",
                    "cov4",
                    "cov64",
                    "worst_land",
                    "acc",
                    "wild",
                    "out",
                    "d_land_lo",
                    "d_land_hi",
                    "d_t256_lo",
                    "d_t256_hi",
                )
            )
    adv = advance(scores)
    res["advance"] = adv.to_dicts()
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=3):
        print(adv)
    Path(a.out).write_text(json.dumps(res, default=float))
    if a.scorecard:
        lines = []
        for m, sc in scores.items():
            lines.append(f"### {m}\n")
            lines.append(
                markdown(
                    sc.select(
                        "arm",
                        "land_err",
                        "t256_err",
                        "cov4",
                        "cov64",
                        "worst_land",
                        "acc",
                        "wild",
                        "out",
                    )
                )
            )
            lines.append("")
        lines.append("### Advance rule\n")
        lines.append(markdown(adv))
        Path(a.scorecard).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
