"""Summarise the raw experiment outputs into one table per (configuration, estimator, block group).

Coverage confidence intervals come from a cluster bootstrap over realizations (blocks of
the same realization are dependent). SD ratios are summarised over the blocks of a group.

Usage: ``uv run python -m experiments.summarize``
"""

from __future__ import annotations

import numpy as np
import polars as pl

from experiments.common import RESULTS, write

LEVEL_COLS = ["50", "80", "90", "95"]
N_BOOT = 1000


def bootstrap_coverage(real: pl.DataFrame, keys: list[str], seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, sub in real.group_by(keys, maintain_order=True):
        sub = sub.sort("realization")
        n_blocks = sub["n_blocks"].to_numpy().astype(float)
        r = n_blocks.size
        idx = rng.integers(0, r, size=(N_BOOT, r))
        row = dict(zip(keys, key, strict=True))
        row["n_real"] = r
        row["z_rms_pooled"] = float(np.sqrt(sub["sum_z2"].sum() / n_blocks.sum()))
        for lv in LEVEL_COLS:
            cov = sub[f"n_cov_{lv}"].to_numpy().astype(float)
            row[f"cov_{lv}"] = float(cov.sum() / n_blocks.sum())
            boot = cov[idx].sum(axis=1) / n_blocks[idx].sum(axis=1)
            row[f"cov_{lv}_lo"], row[f"cov_{lv}_hi"] = (
                float(v) for v in np.quantile(boot, [0.025, 0.975])
            )
        rows.append(row)
    return pl.DataFrame(rows)


def block_summary(blocks: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return blocks.group_by(keys, maintain_order=True).agg(
        pl.len().alias("n_blocks_group"),
        pl.col("size_cells").median().alias("size_cells"),
        pl.col("ratio_brute").median().alias("ratio_median"),
        pl.col("ratio_brute").quantile(0.25).alias("ratio_q25"),
        pl.col("ratio_brute").quantile(0.75).alias("ratio_q75"),
        (pl.col("sd_est_rms") / pl.col("sd_exact"))
        .fill_nan(None)
        .median()
        .alias("ratio_exact_median"),
        (pl.col("rmse_true") / pl.col("sd_exact"))
        .fill_nan(None)
        .median()
        .alias("brute_over_exact"),
        (pl.col("bias") / pl.col("rmse_true")).median().alias("bias_over_rmse"),
        pl.col("rmse_true").median().alias("rmse_true_median"),
        pl.col("sd_est_rms").median().alias("sd_est_median"),
    )


def main() -> None:
    summaries = []
    for exp in ("continuous", "categorical", "pipeline"):
        bpath = RESULTS / f"{exp}_blocks.parquet"
        rpath = RESULTS / f"{exp}_realizations.parquet"
        if not bpath.exists():
            print(f"skipping {exp}: no results")
            continue
        blocks = pl.read_parquet(bpath)
        real = pl.read_parquet(rpath)
        keys = ["experiment", "factor", "level", "estimator", "group"]
        if "class" in blocks.columns:
            keys.insert(3, "class")
        cov = bootstrap_coverage(real, keys)
        summ = block_summary(blocks, keys).join(cov, on=keys, how="left")
        if "class" not in summ.columns:
            summ = summ.with_columns(pl.lit(None, dtype=pl.String).alias("class"))
        summaries.append(summ.select(sorted(summ.columns)))
    out = pl.concat(summaries, how="diagonal_relaxed")
    path = write(out, "summary.parquet")
    print(f"wrote {path} ({out.height} rows)")


if __name__ == "__main__":
    main()
