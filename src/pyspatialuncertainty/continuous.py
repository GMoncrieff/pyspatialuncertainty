"""Uncertainty of block averages and totals of continuous maps (steps 5-8 of the method).

Given the point-support prediction-error standard deviation σ(s) on a grid, a correlation
function ρ(h) of the standardized prediction errors, and a set of blocks, the variance of
the block-averaged prediction error is estimated by Monte Carlo integration of
σ(s)σ(u)ρ(|s − u|) over B × B (steps 5-6). The variance of the block total is |B|² times
that of the block average (step 7), and normal prediction intervals follow from the
standard deviations (step 8).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import xarray as xr

from pyspatialuncertainty._results import ids_to_series, to_xarray_cells, to_xarray_tiles
from pyspatialuncertainty._types import FloatArray, IntArray, SeedLike
from pyspatialuncertainty.blocks import BlockSet, ResolvedBlock, resolve_blocks
from pyspatialuncertainty.correlation import CorrelationFunction, as_correlation
from pyspatialuncertainty.integrate import Integrand, mc_block_variance
from pyspatialuncertainty.intervals import normal_interval, z_value
from pyspatialuncertainty.variogram.models import VariogramModel


@dataclass(frozen=True)
class BlockUncertainty:
    """Block-level predictions and their uncertainty for a continuous map.

    Attributes
    ----------
    table
        One row per block with columns ``block_id``, ``area``, ``n_cells``, ``mean_pred``
        (block-average prediction), ``var_mean`` and ``mcse_var`` (variance of the
        block-average error and its Monte Carlo standard error), ``sd_mean``, ``mcse_sd``,
        ``lower`` and ``upper`` (prediction interval of the block average), ``rel_error``
        (``sd_mean / |mean_pred|``), ``total_pred``, ``sd_total``, ``total_lower`` and
        ``total_upper`` (the same for the block total, i.e. |B| times the block average).
    blockset
        The blocks.
    n_mc
        Number of Monte Carlo point pairs per block.
    level
        Nominal coverage of the prediction intervals.
    z
        Normal quantile used for the intervals.
    """

    table: pl.DataFrame
    blockset: BlockSet
    n_mc: int
    level: float
    z: float
    cells: tuple[IntArray, ...] = field(repr=False, default=())

    def to_xarray(
        self, columns: Sequence[str] | None = None, *, coarse: bool = False
    ) -> xr.Dataset:
        """Block-level results as an xarray Dataset.

        Parameters
        ----------
        columns
            Columns of :attr:`table` to include (default: predictions, SDs and intervals).
        coarse
            For tiles, return one value per tile on the coarse grid of tile centres instead
            of mapping block values back onto the cells of the original grid.

        Returns
        -------
        xarray.Dataset
            One variable per column.
        """
        cols = (
            list(columns)
            if columns is not None
            else [
                "mean_pred",
                "sd_mean",
                "lower",
                "upper",
                "rel_error",
                "total_pred",
                "sd_total",
            ]
        )
        if coarse:
            return to_xarray_tiles(self.blockset, self.table, cols)
        return to_xarray_cells(self.blockset, self.cells, self.table, cols)


def _flat(blockset: BlockSet, da: xr.DataArray | None) -> FloatArray | None:
    return None if da is None else blockset.grid.values(da)


def block_means(blocks: Sequence[ResolvedBlock], values: FloatArray) -> FloatArray:
    """Mean of cell values over the valid cells of each block (NaN for blocks without cells)."""
    return np.array([float(np.mean(values[b.cells])) if b.cells.size else np.nan for b in blocks])


def aggregate_continuous(
    sd: xr.DataArray,
    correlation: CorrelationFunction | VariogramModel,
    blocks: BlockSet,
    *,
    prediction: xr.DataArray | None = None,
    n_mc: int = 10_000,
    level: float = 0.90,
    z: float | None = None,
    seed: SeedLike = None,
    area_scale: float = 1.0,
) -> BlockUncertainty:
    """Uncertainty of block averages and totals of a continuous map (steps 5-8).

    Parameters
    ----------
    sd
        Point-support prediction-error standard deviation σ(s) on the grid of ``blocks``
        (step 2). Cells with NaN are treated as outside the mapped area.
    correlation
        Correlation function of the standardized prediction errors (step 4), either a
        callable or a variogram model fitted to the standardized errors (converted with
        ρ(h) = (sill − γ(h)) / sill).
    blocks
        Blocks to aggregate over (see :mod:`pyspatialuncertainty.blocks`).
    prediction
        Point predictions on the same grid; needed for block predictions, totals and
        intervals.
    n_mc
        Number of pairs of uniform points per block for the Monte Carlo integration
        (the paper uses 10,000).
    level
        Nominal coverage of the normal prediction intervals.
    z
        Explicit normal multiplier for the intervals (overrides ``level``).
    seed
        Random seed for reproducibility.
    area_scale
        Factor converting squared CRS units to the area unit of totals, e.g. ``1e-4`` for
        hectares when the CRS unit is metres and the variable is a density per hectare.

    Returns
    -------
    BlockUncertainty
        Block-level predictions, standard deviations and prediction intervals.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> import pyspatialuncertainty as psu
    >>> x = np.arange(50) + 0.5
    >>> sd = xr.DataArray(np.ones((50, 50)), coords={"y": x[::-1], "x": x}, dims=("y", "x"))
    >>> res = psu.aggregate_continuous(
    ...     sd, psu.vgm(1.0, "Exp", 5.0), psu.blocks.tiles(sd, 25), prediction=sd * 10, seed=1
    ... )
    >>> res.table.height
    4
    """
    corr = as_correlation(correlation)
    sd_flat = _flat(blocks, sd)
    assert sd_flat is not None
    pred_flat = _flat(blocks, prediction)
    valid = np.isfinite(sd_flat) & (sd_flat >= 0)
    if pred_flat is not None:
        valid &= np.isfinite(pred_flat)
    resolved = resolve_blocks(blocks, valid)
    mc = mc_block_variance(resolved, [Integrand(sd_flat, corr)], n_mc=n_mc, seed=seed)
    mult = z_value(level) if z is None else float(z)

    area = np.array([b.area for b in resolved]) * area_scale
    var_mean = mc.variance[0]
    mcse_var = mc.mcse[0]
    sd_mean = np.sqrt(np.maximum(var_mean, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        mcse_sd = np.where(sd_mean > 0, mcse_var / (2.0 * sd_mean), np.nan)
    mean_pred = (
        block_means(resolved, pred_flat)
        if pred_flat is not None
        else np.full(len(resolved), np.nan)
    )
    lower, upper = normal_interval(mean_pred, sd_mean, z=mult)
    total_pred = area * mean_pred
    sd_total = area * sd_mean
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = sd_mean / np.abs(mean_pred)

    table = pl.DataFrame(
        [
            ids_to_series([b.id for b in resolved]),
            pl.Series("area", area),
            pl.Series("n_cells", [b.n_cells for b in resolved], dtype=pl.Int64),
            pl.Series("mean_pred", mean_pred),
            pl.Series("var_mean", var_mean),
            pl.Series("mcse_var", mcse_var),
            pl.Series("sd_mean", sd_mean),
            pl.Series("mcse_sd", mcse_sd),
            pl.Series("lower", lower),
            pl.Series("upper", upper),
            pl.Series("rel_error", rel_error),
            pl.Series("total_pred", total_pred),
            pl.Series("var_total", area**2 * var_mean),
            pl.Series("sd_total", sd_total),
            pl.Series("total_lower", area * lower),
            pl.Series("total_upper", area * upper),
        ]
    )
    return BlockUncertainty(
        table=table,
        blockset=blocks,
        n_mc=n_mc,
        level=level,
        z=mult,
        cells=tuple(b.cells for b in resolved),
    )
