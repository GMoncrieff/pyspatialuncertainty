"""Shared utilities for the simulation experiments."""

from __future__ import annotations

import multiprocessing
import tomllib
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy import stats

import pyspatialuncertainty as psu

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load_config(path: Path | None = None) -> dict[str, Any]:
    with open(path or ROOT / "config.toml", "rb") as fh:
        return tomllib.load(fh)


def configurations(section: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any], bool]]:
    """(factor, level, params, is_baseline) for the baseline and each one-factor change."""
    base = dict(section["baseline"])
    out: list[tuple[str, Any, dict[str, Any], bool]] = [("baseline", "baseline", base, True)]
    for factor, levels in section.get("ablations", {}).items():
        for level in levels:
            params = dict(base)
            params[factor] = level
            out.append((factor, level, params, False))
    return out


def fit_correlation(
    data: pl.DataFrame, value: str = "std_error"
) -> tuple[psu.VariogramFit, psu.CorrelationFunction]:
    """Step 4 with gstat defaults and automatic model choice (Nug + Exp/Sph/Gau)."""
    ev = psu.empirical_variogram(data, value, "x", "y")
    fit = psu.fit_variogram_auto(ev)
    return fit, psu.as_correlation(fit.model, check=False)


def fit_summary(fit: psu.VariogramFit) -> dict[str, Any]:
    m = fit.model
    main = [s for s in m.structures if s.model != "Nug"]
    return {
        "fit_model": main[0].model if main else "Nug",
        "fit_sill": m.sill,
        "fit_nugget_share": m.nugget / m.sill if m.sill > 0 else np.nan,
        "fit_range": main[0].range if main else np.nan,
    }


def run_parallel(func: Callable[..., Any], tasks: Sequence[Any], n_workers: int) -> list[Any]:
    if n_workers <= 1:
        return [func(t) for t in tasks]
    # spawn: forking a process that runs Polars/OpenBLAS thread pools can deadlock
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        return list(ex.map(func, tasks))


def chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, size):
        yield start, min(start + size, n)


def block_metrics(
    err: np.ndarray,
    sd_est: np.ndarray,
    levels: Sequence[float],
    *,
    keys: dict[str, Any],
    block_ids: Sequence[str],
    groups: Sequence[str],
    sizes: np.ndarray,
    sd_exact: np.ndarray | None = None,
    estimator: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Per-block and per-(realization, group) coverage summaries.

    ``err`` has shape (R, B): true minus predicted block value. ``sd_est`` has shape (R, B)
    or (B,) for estimators that do not vary between realizations.
    """
    r, b = err.shape
    sd = np.broadcast_to(sd_est, (r, b))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.abs(err) / sd
    cover = {lv: (z <= stats.norm.ppf(0.5 + lv / 2)) for lv in levels}
    rmse = np.sqrt(np.mean(err**2, axis=0))
    sd_rms = np.sqrt(np.mean(sd**2, axis=0))
    blocks = {
        **{k: [v] * b for k, v in keys.items()},
        "estimator": [estimator] * b,
        "block_id": list(block_ids),
        "group": list(groups),
        "size_cells": sizes,
        "n_real": np.full(b, r),
        "bias": err.mean(axis=0),
        "sd_err": err.std(axis=0, ddof=1),
        "rmse_true": rmse,
        "sd_exact": sd_exact if sd_exact is not None else np.full(b, np.nan),
        "sd_est_mean": sd.mean(axis=0),
        "sd_est_rms": sd_rms,
        "sd_est_q05": np.quantile(sd, 0.05, axis=0),
        "sd_est_q95": np.quantile(sd, 0.95, axis=0),
        "ratio_brute": sd_rms / rmse,
        "z_rms": np.sqrt(np.mean(z**2, axis=0)),
    }
    for lv, c in cover.items():
        blocks[f"cov_{round(lv * 100)}"] = c.mean(axis=0)
    block_df = pl.DataFrame(blocks)

    grp = np.asarray(groups)
    rows: list[dict[str, Any]] = []
    for g in dict.fromkeys(groups):
        sel = grp == g
        for i in range(r):
            row = {
                **keys,
                "estimator": estimator,
                "group": g,
                "realization": i,
                "n_blocks": int(sel.sum()),
            }
            for lv, c in cover.items():
                row[f"n_cov_{round(lv * 100)}"] = int(c[i, sel].sum())
            row["sum_z2"] = float(np.sum(z[i, sel] ** 2))
            rows.append(row)
    return block_df, pl.DataFrame(rows)


def write(df: pl.DataFrame, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    df.write_parquet(path, compression="zstd")
    return path


def level_label(level: Any) -> str:
    return str(level)
