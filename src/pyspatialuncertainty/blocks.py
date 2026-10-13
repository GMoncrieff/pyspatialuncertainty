"""Blocks: the spatial supports over which predictions are averaged or summed.

A :class:`BlockSet` describes one or more blocks on a :class:`~pyspatialuncertainty.grid.GridSpec`:

* :func:`tiles` -- regular square or rectangular tiles of grid cells (e.g. 10 km × 10 km);
* :func:`zones` -- blocks defined by a zone raster aligned with the grid;
* :func:`polygons` -- arbitrary (multi)polygons, e.g. administrative units;
* :func:`box` -- a single rectangle;
* :func:`domain` -- the whole mapped area.

Blocks are purely geometric. Cells without a valid prediction or standard deviation (NaN)
are removed when the blocks are *resolved* against a validity mask
(:func:`resolve_blocks`), so that a block B always means "the part of the block that is
mapped". The resolved blocks know their area |B| and can draw points uniformly in B, which
is what the Monte Carlo integration of the method needs (step 6).

Two sampling schemes are used:

* cell-based blocks (tiles, zones, domain, and polygons with ``sampling="cells"``) are the
  union of their cells: a cell is drawn uniformly and then a point uniformly inside it;
* geometry-based blocks (polygons and boxes with ``sampling="geometry"``, the default)
  draw points uniformly inside the exact geometry, as ``spatstat::runifpoint()`` does in
  the authors' R script, and look up σ in the cell containing each point.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import numpy as np
import shapely
import xarray as xr

from pyspatialuncertainty._types import BoolArray, FloatArray, IntArray
from pyspatialuncertainty.grid import GridSpec

BlockKind = Literal["tiles", "zones", "polygons", "domain", "custom"]
Sampling = Literal["geometry", "cells"]


@dataclass(frozen=True)
class Block:
    """A single block.

    Attributes
    ----------
    id
        Block identifier.
    cells
        Flat (row-major) indices of the grid cells whose centres lie in the block.
    geometry
        The exact geometry for geometry-based sampling, or ``None`` when the block is the
        union of its cells.
    """

    id: Hashable
    cells: IntArray
    geometry: Any = None


@dataclass(frozen=True)
class BlockSet:
    """A collection of blocks on a grid.

    Attributes
    ----------
    grid
        The grid the blocks refer to.
    blocks
        The blocks.
    kind
        How the blocks were created.
    tile_shape
        For tiles: number of tile rows and columns.
    tile_size
        For tiles: number of grid rows and columns per tile.
    """

    grid: GridSpec
    blocks: tuple[Block, ...]
    kind: BlockKind
    tile_shape: tuple[int, int] | None = None
    tile_size: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.blocks)

    @property
    def ids(self) -> list[Hashable]:
        """Block identifiers."""
        return [b.id for b in self.blocks]


def _grid_of(obj: GridSpec | xr.DataArray) -> GridSpec:
    return obj if isinstance(obj, GridSpec) else GridSpec.from_dataarray(obj)


def tiles(
    grid: GridSpec | xr.DataArray,
    size: int | tuple[int, int],
    *,
    units: Literal["cells", "map"] = "cells",
    drop_partial: bool = False,
) -> BlockSet:
    """Regular tiles of grid cells.

    Parameters
    ----------
    grid
        Grid (or a DataArray on the grid).
    size
        Tile size as a number of cells (``units="cells"``) or in map units
        (``units="map"``), either one value for square tiles or ``(rows, cols)``.
    units
        Unit of ``size``.
    drop_partial
        Drop the incomplete tiles at the right and bottom edges of the grid.

    Returns
    -------
    BlockSet
        One block per tile, numbered row by row (``id = tile_row * n_tile_cols + tile_col``).
    """
    g = _grid_of(grid)
    ty, tx = (size, size) if isinstance(size, int | float) else size
    if units == "map":
        ty = round(float(ty) / abs(g.dy))
        tx = round(float(tx) / abs(g.dx))
    ty, tx = int(ty), int(tx)
    if ty < 1 or tx < 1:
        raise ValueError("tiles must be at least one cell in each direction")
    n_rows = g.ny // ty if drop_partial else math.ceil(g.ny / ty)
    n_cols = g.nx // tx if drop_partial else math.ceil(g.nx / tx)
    rows = np.arange(g.ny)
    cols = np.arange(g.nx)
    blocks = []
    for i in range(n_rows):
        r = rows[i * ty : (i + 1) * ty]
        for j in range(n_cols):
            c = cols[j * tx : (j + 1) * tx]
            cells = (r[:, None] * g.nx + c[None, :]).ravel().astype(np.int64)
            blocks.append(Block(id=i * n_cols + j, cells=cells))
    return BlockSet(g, tuple(blocks), "tiles", tile_shape=(n_rows, n_cols), tile_size=(ty, tx))


def zones(
    zone: xr.DataArray,
    grid: GridSpec | None = None,
    *,
    ignore: Iterable[float] = (),
) -> BlockSet:
    """Blocks given by the labels of a zone raster aligned with the grid.

    Parameters
    ----------
    zone
        DataArray of zone labels (integers, or floats with NaN for "no zone").
    grid
        The grid; defaults to the grid of ``zone``.
    ignore
        Label values that do not define a block (e.g. a nodata value).

    Returns
    -------
    BlockSet
        One block per distinct label, sorted by label.
    """
    g = grid if grid is not None else GridSpec.from_dataarray(zone)
    labels = g.values(zone)
    keep = np.isfinite(labels)
    for v in ignore:
        keep &= labels != v
    idx = np.flatnonzero(keep)
    lab = labels[idx]
    order = np.argsort(lab, kind="stable")
    idx, lab = idx[order], lab[order]
    uniq, starts = np.unique(lab, return_index=True)
    groups = np.split(idx, starts[1:])
    blocks = []
    for u, cells in zip(uniq, groups, strict=True):
        label: Hashable = int(u) if float(u).is_integer() else float(u)
        blocks.append(Block(id=label, cells=cells.astype(np.int64)))
    return BlockSet(g, tuple(blocks), "zones")


def domain(grid: GridSpec | xr.DataArray, id: Hashable = "domain") -> BlockSet:
    """A single block covering the whole grid (invalid cells are removed when resolving)."""
    g = _grid_of(grid)
    return BlockSet(g, (Block(id=id, cells=np.arange(g.size, dtype=np.int64)),), "domain")


def _cells_with_centres_in(g: GridSpec, geom: Any) -> IntArray:
    xmin, ymin, xmax, ymax = geom.bounds
    cx = np.flatnonzero((g.x >= xmin) & (g.x <= xmax))
    cy = np.flatnonzero((g.y >= ymin) & (g.y <= ymax))
    if cx.size == 0 or cy.size == 0:
        return np.empty(0, dtype=np.int64)
    xx, yy = np.meshgrid(g.x[cx], g.y[cy])
    inside = shapely.intersects_xy(geom, xx.ravel(), yy.ravel())
    flat = (cy[:, None] * g.nx + cx[None, :]).ravel()
    out: IntArray = np.sort(flat[inside]).astype(np.int64)
    return out


def polygons(
    geometries: Any,
    grid: GridSpec | xr.DataArray,
    *,
    ids: Sequence[Hashable] | None = None,
    sampling: Sampling = "geometry",
) -> BlockSet:
    """Blocks from (multi)polygons.

    Parameters
    ----------
    geometries
        Shapely polygons, or a GeoSeries/GeoDataFrame (its ``geometry`` column is used and
        its CRS must match the grid's CRS).
    grid
        Grid (or a DataArray on the grid).
    ids
        Block identifiers; defaults to 0, 1, ...
    sampling
        ``"geometry"`` (default) draws Monte Carlo points uniformly inside the exact
        polygons; ``"cells"`` treats each polygon as the union of the cells whose centres
        fall inside it.

    Returns
    -------
    BlockSet
        One block per polygon. The block prediction is the mean over the cells whose centres
        fall inside the polygon.
    """
    g = _grid_of(grid)
    geo_crs = getattr(geometries, "crs", None)
    if hasattr(geometries, "geometry"):
        geometries = geometries.geometry
    if geo_crs is not None and g.crs is not None:
        from pyproj import CRS

        if not CRS.from_user_input(geo_crs).equals(CRS.from_user_input(g.crs)):
            raise ValueError("the polygons and the grid have different CRSs; reproject one of them")
    geoms = list(geometries)
    if ids is None:
        ids = list(range(len(geoms)))
    if len(ids) != len(geoms):
        raise ValueError("ids must have one entry per geometry")
    if sampling not in ("geometry", "cells"):
        raise ValueError("sampling must be 'geometry' or 'cells'")
    blocks = []
    for bid, geom in zip(ids, geoms, strict=True):
        if geom is None or geom.is_empty:
            raise ValueError(f"geometry of block {bid!r} is empty")
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise ValueError(f"block {bid!r}: expected a (multi)polygon, got {geom.geom_type}")
        shapely.prepare(geom)
        cells = _cells_with_centres_in(g, geom)
        blocks.append(Block(id=bid, cells=cells, geometry=geom if sampling == "geometry" else None))
    return BlockSet(g, tuple(blocks), "polygons")


def box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    grid: GridSpec | xr.DataArray,
    *,
    id: Hashable = "box",
) -> BlockSet:
    """A single rectangular block with geometry-based sampling."""
    return polygons([shapely.box(xmin, ymin, xmax, ymax)], grid, ids=[id])


# ---------------------------------------------------------------------------
# Resolution against a validity mask, and samplers
# ---------------------------------------------------------------------------


class PointSampler(Protocol):
    """Draws points uniformly in a resolved block."""

    def __call__(
        self, n: int, rng: np.random.Generator, /
    ) -> tuple[FloatArray, FloatArray, IntArray]:
        """Return ``n`` points (x, y) and the flat indices of their cells."""
        ...


@dataclass(frozen=True)
class CellSampler:
    """Uniform points in a union of equal-area cells."""

    grid: GridSpec
    cells: IntArray

    def __call__(
        self, n: int, rng: np.random.Generator, /
    ) -> tuple[FloatArray, FloatArray, IntArray]:
        pick = self.cells[rng.integers(0, self.cells.size, size=n)]
        x, y = self.grid.jitter_in_cells(pick, rng)
        return x, y, pick


@dataclass(frozen=True)
class GeometrySampler:
    """Uniform points in a geometry, restricted to valid grid cells (rejection sampling)."""

    grid: GridSpec
    geometry: Any
    valid: BoolArray
    acceptance: float = 1.0

    def __call__(
        self, n: int, rng: np.random.Generator, /
    ) -> tuple[FloatArray, FloatArray, IntArray]:
        xmin, ymin, xmax, ymax = self.geometry.bounds
        xs_out: list[FloatArray] = []
        ys_out: list[FloatArray] = []
        cs_out: list[IntArray] = []
        need = n
        rate = max(self.acceptance, 1e-3)
        while need > 0:
            m = math.ceil(need / rate * 1.1) + 8
            xs = rng.uniform(xmin, xmax, size=m)
            ys = rng.uniform(ymin, ymax, size=m)
            cells = self.grid.flat_index_of(xs, ys)
            ok = cells >= 0
            ok[ok] = self.valid[cells[ok]]
            ok[ok] = shapely.contains_xy(self.geometry, xs[ok], ys[ok])
            sel = np.flatnonzero(ok)[:need]
            xs_out.append(xs[sel])
            ys_out.append(ys[sel])
            cs_out.append(cells[sel])
            need -= sel.size
            rate = max(min(1.0, float(ok.mean())), 1e-3) if ok.size else rate
        return np.concatenate(xs_out), np.concatenate(ys_out), np.concatenate(cs_out)


@dataclass(frozen=True)
class ResolvedBlock:
    """A block restricted to valid cells, with its area and a point sampler.

    Attributes
    ----------
    id
        Block identifier.
    cells
        Valid cells whose centres lie in the block (used for block means of cell values).
    area
        Area |B| of the valid part of the block, in squared CRS units.
    sampler
        Draws points uniformly in the valid part of the block.
    """

    id: Hashable
    cells: IntArray
    area: float
    sampler: PointSampler = field(repr=False)

    @property
    def n_cells(self) -> int:
        """Number of valid cells with their centre in the block."""
        return int(self.cells.size)


def _geometry_valid_area(g: GridSpec, geom: Any, valid: BoolArray) -> float:
    """Area of the part of ``geom`` covered by valid cells."""
    xmin, ymin, xmax, ymax = geom.bounds
    gx0, gy0, gx1, gy1 = g.bounds
    rows, cols = g.cell_of(
        np.clip([xmin, xmax], gx0, gx1 - 1e-12 * abs(g.dx)),
        np.clip([ymin, ymax], gy0, gy1 - 1e-12 * abs(g.dy)),
    )
    r0, r1 = int(min(rows)), int(max(rows))
    c0, c1 = int(min(cols)), int(max(cols))
    rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
    flat = (rr * g.nx + cc).ravel()
    x_lo = g.x_edge + cc.ravel() * g.dx
    y_lo = g.y_edge + rr.ravel() * g.dy
    boxes = shapely.box(
        np.minimum(x_lo, x_lo + g.dx),
        np.minimum(y_lo, y_lo + g.dy),
        np.maximum(x_lo, x_lo + g.dx),
        np.maximum(y_lo, y_lo + g.dy),
    )
    hit = shapely.intersects(geom, boxes)
    within_grid = shapely.box(gx0, gy0, gx1, gy1).contains(geom)
    if within_grid and np.all(valid[flat[hit]]):
        return float(geom.area)
    use = hit & valid[flat]
    full = shapely.contains(geom, boxes[use])
    area = float(full.sum()) * g.cell_area
    partial = boxes[use][~full]
    if partial.size:
        area += float(np.sum(shapely.area(shapely.intersection(partial, geom))))
    return area


def resolve_blocks(blockset: BlockSet, valid: BoolArray) -> list[ResolvedBlock]:
    """Restrict blocks to valid cells and attach areas and samplers.

    Parameters
    ----------
    blockset
        The blocks.
    valid
        Flat (row-major) boolean mask of cells with valid inputs.

    Returns
    -------
    list of ResolvedBlock
        One entry per block. Blocks without any valid area raise ``ValueError``.
    """
    g = blockset.grid
    valid = np.asarray(valid, dtype=bool).ravel()
    if valid.size != g.size:
        raise ValueError("the validity mask does not match the grid")
    out = []
    for b in blockset.blocks:
        cells = b.cells[valid[b.cells]] if b.cells.size else b.cells
        if b.geometry is None:
            if cells.size == 0:
                raise ValueError(f"block {b.id!r} contains no valid cells")
            area = cells.size * g.cell_area
            sampler: PointSampler = CellSampler(g, cells)
        else:
            area = _geometry_valid_area(g, b.geometry, valid)
            if area <= 0:
                raise ValueError(f"block {b.id!r} does not overlap any valid cell")
            bx0, by0, bx1, by1 = b.geometry.bounds
            bbox_area = max((bx1 - bx0) * (by1 - by0), np.finfo(float).tiny)
            sampler = GeometrySampler(g, b.geometry, valid, acceptance=min(1.0, area / bbox_area))
        out.append(ResolvedBlock(id=b.id, cells=cells, area=float(area), sampler=sampler))
    return out
