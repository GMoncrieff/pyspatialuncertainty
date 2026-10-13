"""Grid layouts and evaluation blocks for the synthetic scenarios.

The map (predictions, σ, probabilities) lives on a grid of n × n square cells. To make the
synthetic truth consistent with the estimand of the method -- the *continuous* block
average (Wadoux & Heuvelink, 2023, Eq. 2) -- the true field is simulated on a finer
sub-grid with f × f nodes per cell, and the true block average is the mean over the
sub-grid nodes inside the block. Point-scale variation (a nugget) only enters point
observations and averages out within blocks, as it does for a continuous block average.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import shapely
import xarray as xr

from pyspatialuncertainty._types import FloatArray, IntArray
from pyspatialuncertainty.blocks import Block, BlockSet
from pyspatialuncertainty.grid import GridSpec


@dataclass(frozen=True)
class SubGridLayout:
    """An n × n grid of square cells with an f × f sub-grid of nodes per cell.

    Coordinates run from 0 to ``n * cell_size`` in both directions; arrays are stored
    north-up (row 0 at the top), for both the cell grid and the sub-grid.
    """

    n: int = 128
    cell_size: float = 1.0
    f: int = 4

    @property
    def n_sub(self) -> int:
        """Number of sub-grid nodes per side."""
        return self.n * self.f

    @property
    def sub_spacing(self) -> float:
        """Spacing of the sub-grid nodes."""
        return self.cell_size / self.f

    @property
    def extent(self) -> float:
        """Side length of the square domain."""
        return self.n * self.cell_size

    @property
    def cell_x(self) -> FloatArray:
        """Cell-centre x coordinates."""
        return (np.arange(self.n, dtype=np.float64) + 0.5) * float(self.cell_size)

    @property
    def cell_y(self) -> FloatArray:
        """Cell-centre y coordinates (north-up, decreasing)."""
        return self.cell_x[::-1].copy()

    @property
    def sub_x(self) -> FloatArray:
        """Sub-grid node x coordinates."""
        return (np.arange(self.n_sub, dtype=np.float64) + 0.5) * float(self.sub_spacing)

    @property
    def sub_y(self) -> FloatArray:
        """Sub-grid node y coordinates (north-up, decreasing)."""
        return self.sub_x[::-1].copy()

    def cell_dataarray(self, values: FloatArray, name: str | None = None) -> xr.DataArray:
        """Wrap an (n, n) array as a DataArray on the cell grid."""
        return xr.DataArray(
            np.asarray(values, dtype=np.float64),
            coords={"y": self.cell_y, "x": self.cell_x},
            dims=("y", "x"),
            name=name,
        )

    @property
    def grid(self) -> GridSpec:
        """The cell grid."""
        return GridSpec.from_dataarray(self.cell_dataarray(np.zeros((self.n, self.n))))

    def upsample(self, cell_values: FloatArray) -> FloatArray:
        """Cell values repeated on the sub-grid."""
        return np.repeat(np.repeat(cell_values, self.f, axis=0), self.f, axis=1)

    def sub_coords(self, flat_index: IntArray) -> tuple[FloatArray, FloatArray]:
        """Coordinates of sub-grid nodes given by flat (row-major) indices."""
        return self.sub_x[flat_index % self.n_sub], self.sub_y[flat_index // self.n_sub]

    def cell_of_sub(self, flat_index: IntArray) -> IntArray:
        """Flat cell index of sub-grid nodes."""
        row = (flat_index // self.n_sub) // self.f
        col = (flat_index % self.n_sub) // self.f
        out: IntArray = (row * self.n + col).astype(np.int64)
        return out


@dataclass(frozen=True)
class EvaluationBlocks:
    """Blocks for a simulation study, with the sub-grid nodes that define their truth.

    Attributes
    ----------
    blockset
        The blocks on the cell grid, as used by the method.
    sub_index
        For each block, flat indices of the sub-grid nodes inside it.
    group
        For each block, a label such as ``"tile_8"`` (tiles of 8 × 8 cells) or ``"polygon"``.
    size_cells
        For each block, its area in cells.
    """

    blockset: BlockSet
    sub_index: tuple[IntArray, ...]
    group: tuple[str, ...]
    size_cells: FloatArray

    def block_means(self, sub_field: FloatArray) -> FloatArray:
        """Mean of a sub-grid field over each block."""
        flat = sub_field.ravel()
        return np.array([float(flat[idx].mean()) for idx in self.sub_index])


def evaluation_blocks(
    layout: SubGridLayout,
    tile_sizes: Sequence[int] = (2, 4, 8, 16, 32, 64, 128),
    *,
    max_per_size: int = 64,
    n_polygons: int = 6,
    seed: int = 20230327,
) -> EvaluationBlocks:
    """Square tiles of several sizes (a fixed random subset per size) and Voronoi polygons.

    Parameters
    ----------
    layout
        The grid layout.
    tile_sizes
        Tile side lengths in cells; sizes larger than the grid are skipped.
    max_per_size
        Maximum number of tiles kept per size (chosen at random with ``seed``).
    n_polygons
        Number of Voronoi polygons covering the domain (0 for none).
    seed
        Seed for the tile subset and the Voronoi generators.

    Returns
    -------
    EvaluationBlocks
        Tiles use cell-based sampling; polygons use geometry-based sampling.
    """
    rng = np.random.default_rng(seed)
    g = layout.grid
    n, f, ns = layout.n, layout.f, layout.n_sub
    blocks: list[Block] = []
    subs: list[IntArray] = []
    groups: list[str] = []
    sizes: list[float] = []
    for t in tile_sizes:
        if t > n:
            continue
        k = n // t
        all_tiles = [(i, j) for i in range(k) for j in range(k)]
        pick = rng.permutation(len(all_tiles))[:max_per_size]
        for p in sorted(pick):
            i, j = all_tiles[p]
            rows = np.arange(i * t, (i + 1) * t)
            cols = np.arange(j * t, (j + 1) * t)
            cells = (rows[:, None] * n + cols[None, :]).ravel().astype(np.int64)
            srows = np.arange(i * t * f, (i + 1) * t * f)
            scols = np.arange(j * t * f, (j + 1) * t * f)
            sidx = (srows[:, None] * ns + scols[None, :]).ravel().astype(np.int64)
            bid: Hashable = f"tile{t}_{i}_{j}"
            blocks.append(Block(id=bid, cells=cells))
            subs.append(sidx)
            groups.append(f"tile_{t}")
            sizes.append(float(t * t))
    if n_polygons > 0:
        ext = layout.extent
        pts = shapely.MultiPoint(rng.uniform(0.1 * ext, 0.9 * ext, size=(n_polygons, 2)))
        domain_box = shapely.box(0.0, 0.0, ext, ext)
        cells_poly = shapely.voronoi_polygons(pts, extend_to=domain_box)
        sx, sy = np.meshgrid(layout.sub_x, layout.sub_y)
        for q, poly in enumerate(shapely.get_parts(cells_poly)):
            poly = shapely.intersection(poly, domain_box)
            shapely.prepare(poly)
            cx, cy = np.meshgrid(g.x, g.y)
            cin = shapely.intersects_xy(poly, cx.ravel(), cy.ravel())
            sin = shapely.contains_xy(poly, sx.ravel(), sy.ravel())
            blocks.append(
                Block(id=f"polygon_{q}", cells=np.flatnonzero(cin).astype(np.int64), geometry=poly)
            )
            subs.append(np.flatnonzero(sin).astype(np.int64))
            groups.append("polygon")
            sizes.append(float(poly.area / layout.cell_size**2))
    bs = BlockSet(g, tuple(blocks), "custom")
    return EvaluationBlocks(bs, tuple(subs), tuple(groups), np.array(sizes))
