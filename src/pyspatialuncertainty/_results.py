"""Helpers shared by the result objects of the continuous and categorical modes."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, SupportsInt, cast

import numpy as np
import polars as pl
import xarray as xr

from pyspatialuncertainty._types import IntArray
from pyspatialuncertainty.blocks import BlockSet


def ids_to_series(ids: Sequence[Hashable], name: str = "block_id") -> pl.Series:
    """Block identifiers as a Polars series (integers when possible, strings otherwise)."""
    if all(isinstance(i, int | np.integer) and not isinstance(i, bool) for i in ids):
        return pl.Series(name, [int(cast(SupportsInt, i)) for i in ids], dtype=pl.Int64)
    return pl.Series(name, [str(i) for i in ids], dtype=pl.String)


def _pivot(
    table: pl.DataFrame, column: str, n_blocks: int, group_col: str | None
) -> tuple[np.ndarray, list[object] | None]:
    """Values of ``column`` as an array (n_groups, n_blocks) or (n_blocks,)."""
    if group_col is None:
        return table[column].cast(pl.Float64).to_numpy(), None
    groups = table[group_col].unique(maintain_order=True).to_list()
    out = np.full((len(groups), n_blocks), np.nan)
    for gi, gval in enumerate(groups):
        out[gi] = table.filter(pl.col(group_col) == gval)[column].cast(pl.Float64).to_numpy()
    return out, groups


def to_xarray_cells(
    blockset: BlockSet,
    cells: Sequence[IntArray],
    table: pl.DataFrame,
    columns: Sequence[str],
    group_col: str | None = None,
) -> xr.Dataset:
    """Map block-level values back onto the cells of the original grid."""
    g = blockset.grid
    n_blocks = len(cells)
    owner = np.full(g.size, -1, dtype=np.int64)
    for b, c in enumerate(cells):
        owner[c] = b
    has = owner >= 0
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, Any]]] = {}
    groups = None
    for col in columns:
        vals, groups = _pivot(table, col, n_blocks, group_col)
        if group_col is None:
            flat = np.full(g.size, np.nan)
            flat[has] = vals[owner[has]]
            data_vars[col] = ((g.y_dim, g.x_dim), flat.reshape(g.shape))
        else:
            arr = np.full((vals.shape[0], g.size), np.nan)
            arr[:, has] = vals[:, owner[has]]
            data_vars[col] = ((group_col, g.y_dim, g.x_dim), arr.reshape((vals.shape[0], *g.shape)))
    coords: dict[str, object] = {g.y_dim: g.y, g.x_dim: g.x}
    if group_col is not None and groups is not None:
        coords[group_col] = groups
    ds = xr.Dataset(data_vars, coords=coords)
    if g.crs is not None:
        ds = ds.rio.write_crs(g.crs)
    return ds


def to_xarray_tiles(
    blockset: BlockSet,
    table: pl.DataFrame,
    columns: Sequence[str],
    group_col: str | None = None,
) -> xr.Dataset:
    """Block-level values of tiles on the coarse grid of tile centres."""
    if blockset.kind != "tiles" or blockset.tile_shape is None or blockset.tile_size is None:
        raise ValueError("a coarse grid is only available for tiles")
    g = blockset.grid
    n_rows, n_cols = blockset.tile_shape
    ty, tx = blockset.tile_size
    xc = g.x_edge + (np.arange(n_cols) * tx + tx / 2.0) * g.dx
    yc = g.y_edge + (np.arange(n_rows) * ty + ty / 2.0) * g.dy
    n_blocks = n_rows * n_cols
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray[Any, Any]]] = {}
    groups = None
    for col in columns:
        vals, groups = _pivot(table, col, n_blocks, group_col)
        if group_col is None:
            data_vars[col] = ((g.y_dim, g.x_dim), vals.reshape(n_rows, n_cols))
        else:
            data_vars[col] = (
                (group_col, g.y_dim, g.x_dim),
                vals.reshape(vals.shape[0], n_rows, n_cols),
            )
    coords: dict[str, object] = {g.y_dim: yc, g.x_dim: xc}
    if group_col is not None and groups is not None:
        coords[group_col] = groups
    ds = xr.Dataset(data_vars, coords=coords)
    if g.crs is not None:
        ds = ds.rio.write_crs(g.crs)
    return ds
