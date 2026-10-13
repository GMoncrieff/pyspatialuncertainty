"""Data loading and figure helpers for report.qmd."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.plotting import (  # noqa: E402
    BLUES,
    ESTIMATOR_COLOR,
    ESTIMATOR_LABEL,
    ESTIMATOR_MARKER,
    SERIES,
    TEXT_2,
    plt,
)

__all__ = [
    "BLUES",
    "ESTIMATOR_COLOR",
    "ESTIMATOR_LABEL",
    "ESTIMATOR_MARKER",
    "GROUP_ORDER",
    "REF",
    "RESULTS",
    "SERIES",
    "TEXT_2",
    "coverage_ratio_panels",
    "level_sort_key",
    "plt",
    "ref",
    "summary",
    "tile_side",
]

RESULTS = ROOT / "experiments" / "results"
REF = ROOT / "tests" / "data" / "r_reference"
GROUP_ORDER = ["tile_2", "tile_4", "tile_8", "tile_16", "tile_32", "tile_64", "tile_128"]
ESTIMATOR_COLOR = {**ESTIMATOR_COLOR, "method_nested": SERIES[1]}
ESTIMATOR_MARKER = {**ESTIMATOR_MARKER, "method_nested": "D"}
ESTIMATOR_LABEL = {**ESTIMATOR_LABEL, "method_nested": "method, nested model (R-script ranges)"}


def ref(name: str) -> pl.DataFrame:
    return pl.read_csv(REF / name, infer_schema_length=None)


def summary() -> pl.DataFrame:
    return pl.read_parquet(RESULTS / "summary.parquet")


def tile_side(group: str) -> int:
    return int(group.split("_")[1])


def coverage_ratio_panels(
    df: pl.DataFrame, estimators: list[str], title: str, level: str = "90"
) -> plt.Figure:
    """Two panels against block size: coverage of the PI and ratio of estimated to true SD."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.4), constrained_layout=True)
    tiles = df.filter(pl.col("group").str.starts_with("tile_"))
    for est in estimators:
        d = (
            tiles.filter(pl.col("estimator") == est)
            .with_columns(
                pl.col("group").map_elements(tile_side, return_dtype=pl.Int64).alias("side")
            )
            .sort("side")
        )
        if d.height == 0:
            continue
        x = d["side"].to_numpy()
        kw = {
            "color": ESTIMATOR_COLOR[est],
            "marker": ESTIMATOR_MARKER[est],
            "label": ESTIMATOR_LABEL[est],
        }
        cov = d[f"cov_{level}"].to_numpy()
        ax1.plot(x, cov, **kw)
        ax1.fill_between(
            x,
            d[f"cov_{level}_lo"].to_numpy(),
            d[f"cov_{level}_hi"].to_numpy(),
            color=ESTIMATOR_COLOR[est],
            alpha=0.15,
            linewidth=0,
        )
        ax2.plot(x, d["ratio_median"].to_numpy(), **kw)
    ax1.axhline(int(level) / 100, color=TEXT_2, lw=1, ls="--")
    ax1.set_xscale("log", base=2)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel("block side (cells)")
    ax1.set_ylabel(f"coverage of the {level}% PI")
    ax2.axhline(1.0, color=TEXT_2, lw=1, ls="--")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("block side (cells)")
    ax2.set_ylabel("estimated / true SD (median over blocks)")
    for ax in (ax1, ax2):
        ax.set_xticks([tile_side(g) for g in GROUP_ORDER])
        ax.set_xticklabels([str(tile_side(g)) for g in GROUP_ORDER])
    ax1.legend(loc="lower left", fontsize=7.5)
    fig.suptitle(title, fontsize=11, fontweight="bold", x=0.01, ha="left")
    return fig


def level_sort_key(level: str) -> float:
    try:
        return float(level)
    except ValueError:
        return 0.0
