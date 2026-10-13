"""Scorecard of the improvement experiments.

Each (scenario, class, block group) cell counts equally. For every scenario and variant:

* ``cov``: mean coverage of the 90 % interval over the cells;
* ``cov_err``: mean |coverage − 0.90| over the cells;
* ``cov_min``: worst cell;
* ``lr_rms``: RMS of log(estimated SD / true SD), the SD accuracy. It combines bias and
  realization-to-realization noise; 0.10 is about 10 % error;
* ``ratio_med``: median over blocks of RMS estimated SD / true SD (the bias alone);
* ``frac_bad``: share of SDs off by more than a factor of two.

Paired differences against ``current`` are taken per realization, with standard errors
over realizations: ``d_cov`` (coverage) and ``d_msl`` (mean squared log ratio).

Usage: ``uv run python -m experiments.improvements.summarize --round 2 [--quick]``
"""

from __future__ import annotations

import argparse

import polars as pl

from experiments.improvements.run import RESULTS

CELL = ["mode", "scenario", "variant", "class", "group"]


def load(kind: str, round_: int = 2, suffix: str = "") -> pl.DataFrame:
    files = sorted((RESULTS / f"round{round_}").glob(f"*_{kind}{suffix}.parquet"))
    return pl.concat(
        [pl.read_parquet(f).with_columns(pl.col("class").cast(pl.String)) for f in files],
        how="diagonal_relaxed",
    )


def cells(blocks: pl.DataFrame, reals: pl.DataFrame) -> pl.DataFrame:
    """One row per (mode, scenario, variant, class, group)."""
    c = reals.group_by(CELL).agg(
        (pl.col("n_cov").sum() / pl.col("n_blocks").sum()).alias("cov_90"),
        (pl.col("sum_log_ratio_sq").sum() / pl.col("n_blocks").sum()).sqrt().alias("lr_rms"),
    )
    b = blocks.group_by(CELL).agg(
        pl.col("ratio_rms").median().alias("ratio_med"),
        pl.col("frac_bad").mean(),
        pl.col("frac_zero").mean(),
    )
    return c.join(b, on=CELL, how="left", nulls_equal=True)


def paired(reals: pl.DataFrame) -> pl.DataFrame:
    """Per-realization coverage and mean squared log ratio, differenced against current."""
    per_r = (
        reals.with_columns(
            (pl.col("n_cov") / pl.col("n_blocks")).alias("c"),
            (pl.col("sum_log_ratio_sq") / pl.col("n_blocks")).alias("m"),
        )
        .group_by("mode", "scenario", "variant", "realization")
        .agg(pl.col("c").mean(), pl.col("m").mean())
    )
    cur = per_r.filter(pl.col("variant") == "current").select(
        "mode", "scenario", "realization", pl.col("c").alias("c0"), pl.col("m").alias("m0")
    )
    d = per_r.join(cur, on=["mode", "scenario", "realization"]).with_columns(
        (pl.col("c") - pl.col("c0")).alias("dc"), (pl.col("m") - pl.col("m0")).alias("dm")
    )
    return d.group_by("mode", "scenario", "variant").agg(
        pl.col("dc").mean().alias("d_cov"),
        (pl.col("dc").std() / pl.len().sqrt()).alias("d_cov_se"),
        pl.col("dm").mean().alias("d_msl"),
        (pl.col("dm").std() / pl.len().sqrt()).alias("d_msl_se"),
    )


def scorecard(cell: pl.DataFrame, pair: pl.DataFrame) -> pl.DataFrame:
    s = cell.group_by("mode", "scenario", "variant").agg(
        pl.col("cov_90").mean().alias("cov"),
        (pl.col("cov_90") - 0.9).abs().mean().alias("cov_err"),
        pl.col("cov_90").min().alias("cov_min"),
        pl.col("lr_rms").mean().alias("lr_rms"),
        pl.col("ratio_med").median().alias("ratio_med"),
        pl.col("frac_bad").mean().alias("frac_bad"),
    )
    return s.join(pair, on=["mode", "scenario", "variant"], how="left")


def overall(score: pl.DataFrame) -> pl.DataFrame:
    """Average over scenarios, and the number of scenarios a variant improves on current."""
    cur = score.filter(pl.col("variant") == "current").select(
        "mode", "scenario", pl.col("cov_err").alias("ce0"), pl.col("lr_rms").alias("lr0")
    )
    s = score.join(cur, on=["mode", "scenario"])
    return (
        s.group_by("mode", "variant")
        .agg(
            pl.col("cov_err").mean(),
            pl.col("cov_min").min(),
            pl.col("lr_rms").mean(),
            (pl.col("cov_err") < pl.col("ce0") - 0.005).sum().alias("n_cov_better"),
            (pl.col("cov_err") > pl.col("ce0") + 0.005).sum().alias("n_cov_worse"),
            (pl.col("lr_rms") < pl.col("lr0") - 0.01).sum().alias("n_sd_better"),
            (pl.col("lr_rms") > pl.col("lr0") + 0.01).sum().alias("n_sd_worse"),
            pl.len().alias("n_scenarios"),
        )
        .sort("mode", "cov_err")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="summarise the smoke-test outputs")
    args = ap.parse_args()
    suffix = "_quick" if args.quick else ""
    blocks = load("blocks", args.round, suffix)
    reals = load("realizations", args.round, suffix)
    cell = cells(blocks, reals)
    score = scorecard(cell, paired(reals))
    over = overall(score)
    out = RESULTS / f"round{args.round}"
    cell.write_parquet(out / f"summary_cells{suffix}.parquet")
    score.write_parquet(out / f"scorecard{suffix}.parquet")
    over.write_parquet(out / f"overall{suffix}.parquet")
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_tbl_width_chars(200)
    num = [c for c in score.columns if score[c].dtype == pl.Float64]
    print(score.with_columns(pl.col(num).round(3)).sort("mode", "scenario", "cov_err"))
    print(over.with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
