"""Regular raster grids described by xarray/rioxarray DataArrays.

The method works with point-support predictions and standard deviations given on a
regular grid of cells (a raster). `GridSpec` captures the geometry of such a
grid: cell-centre coordinates, signed cell sizes, the coordinate reference system and
the mapping from map coordinates to cells. Looking up the cell that contains a point is
equivalent to ``raster::extract()`` with the default ``"simple"`` method in R.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike

from pyspatialuncertainty._types import FloatArray, IntArray


class GridError(ValueError):
    """Raised when a DataArray does not describe a supported regular grid."""


def _spatial_dims(da: xr.DataArray) -> tuple[str, str]:
    """Return the names of the (y, x) dimensions of a DataArray."""
    try:
        x_dim = da.rio.x_dim
        y_dim = da.rio.y_dim
    except Exception:  # rioxarray raises its own exceptions when dims are not found
        x_dim = y_dim = None
    if x_dim is None or y_dim is None or x_dim not in da.dims or y_dim not in da.dims:
        for xd, yd in (("x", "y"), ("lon", "lat"), ("longitude", "latitude")):
            if xd in da.dims and yd in da.dims:
                return yd, xd
        raise GridError(
            "could not identify the spatial dimensions; name them 'x' and 'y' or set them "
            "with da.rio.set_spatial_dims()"
        )
    return str(y_dim), str(x_dim)


def _regular_spacing(coord: FloatArray, name: str) -> float:
    if coord.size == 1:
        raise GridError(f"cannot infer the cell size from a single {name} coordinate")
    steps = np.diff(coord)
    step = float((coord[-1] - coord[0]) / (coord.size - 1))
    if step == 0 or not np.allclose(steps, step, rtol=1e-6, atol=0):
        raise GridError(f"the {name} coordinates are not regularly spaced")
    return step


@dataclass(frozen=True)
class GridSpec:
    """Geometry of a regular grid of cells.

    Attributes
    ----------
    x, y
        Cell-centre coordinates along the x (column) and y (row) axes, in array order.
    dx, dy
        Signed cell sizes (``dy`` is negative for north-up rasters).
    crs
        Coordinate reference system (a ``pyproj.CRS``) or ``None`` for plain Cartesian units.
    x_dim, y_dim
        Names of the spatial dimensions of the source DataArray.
    """

    x: FloatArray
    y: FloatArray
    dx: float
    dy: float
    crs: Any = None
    x_dim: str = "x"
    y_dim: str = "y"

    @classmethod
    def from_dataarray(cls, da: xr.DataArray, *, allow_geographic: bool = False) -> GridSpec:
        """Describe the grid of a DataArray with two spatial dimensions.

        Parameters
        ----------
        da
            DataArray with regularly spaced cell-centre coordinates on its spatial dims.
        allow_geographic
            Geographic (longitude/latitude) coordinates are rejected by default because the
            method needs Euclidean distances and equal-area cells. Reproject the data with
            ``da.rio.reproject(...)`` to a projected, preferably equal-area, CRS.

        Returns
        -------
        GridSpec
            The grid description.
        """
        y_dim, x_dim = _spatial_dims(da)
        x = np.asarray(da[x_dim].values, dtype=np.float64)
        y = np.asarray(da[y_dim].values, dtype=np.float64)
        dx = _regular_spacing(x, x_dim)
        dy = _regular_spacing(y, y_dim)
        crs = None
        try:
            crs = da.rio.crs
        except Exception:
            crs = None
        if crs is not None and getattr(crs, "is_geographic", False) and not allow_geographic:
            raise GridError(
                "the grid has a geographic CRS; reproject it to a projected (preferably "
                "equal-area) CRS, e.g. with da.rio.reproject('EPSG:6933')"
            )
        return cls(x=x, y=y, dx=dx, dy=dy, crs=crs, x_dim=x_dim, y_dim=y_dim)

    @property
    def nx(self) -> int:
        """Number of columns."""
        return int(self.x.size)

    @property
    def ny(self) -> int:
        """Number of rows."""
        return int(self.y.size)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape (rows, columns)."""
        return (self.ny, self.nx)

    @property
    def size(self) -> int:
        """Number of cells."""
        return self.nx * self.ny

    @property
    def cell_area(self) -> float:
        """Area of one cell in squared CRS units."""
        return abs(self.dx * self.dy)

    @property
    def x_edge(self) -> float:
        """x coordinate of the outer edge of the first column."""
        return float(self.x[0] - 0.5 * self.dx)

    @property
    def y_edge(self) -> float:
        """y coordinate of the outer edge of the first row."""
        return float(self.y[0] - 0.5 * self.dy)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Extent of the grid as (xmin, ymin, xmax, ymax)."""
        xs = (self.x_edge, self.x_edge + self.nx * self.dx)
        ys = (self.y_edge, self.y_edge + self.ny * self.dy)
        return (min(xs), min(ys), max(xs), max(ys))

    def cell_of(self, x: ArrayLike, y: ArrayLike) -> tuple[IntArray, IntArray]:
        """Row and column of the cells containing the points (−1 outside the grid).

        A point on the far edge of the grid is assigned to the last row or column, as in
        ``raster::extract()``.
        """
        xa = np.asarray(x, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        col = np.floor((xa - self.x_edge) / self.dx).astype(np.int64)
        row = np.floor((ya - self.y_edge) / self.dy).astype(np.int64)
        col = np.where(col == self.nx, self.nx - 1, col)
        row = np.where(row == self.ny, self.ny - 1, row)
        outside = (col < 0) | (col >= self.nx) | (row < 0) | (row >= self.ny) | ~np.isfinite(xa)
        outside |= ~np.isfinite(ya)
        return np.where(outside, -1, row), np.where(outside, -1, col)

    def flat_index_of(self, x: ArrayLike, y: ArrayLike) -> IntArray:
        """Flat (row-major) index of the cells containing the points (−1 outside)."""
        row, col = self.cell_of(x, y)
        return np.where(row < 0, -1, row * self.nx + col)

    def cell_centres(self, flat_index: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Centre coordinates of cells given by flat indices."""
        idx = np.asarray(flat_index, dtype=np.int64)
        return self.x[idx % self.nx], self.y[idx // self.nx]

    def jitter_in_cells(
        self, flat_index: IntArray, rng: np.random.Generator
    ) -> tuple[FloatArray, FloatArray]:
        """Uniformly distributed points inside the given cells."""
        col = flat_index % self.nx
        row = flat_index // self.nx
        u = rng.random((2, flat_index.size))
        xs = self.x_edge + (col + u[0]) * self.dx
        ys = self.y_edge + (row + u[1]) * self.dy
        return xs, ys

    def values(self, da: xr.DataArray) -> FloatArray:
        """Values of a DataArray on this grid as a flat (row-major) float array."""
        y_dim, x_dim = _spatial_dims(da)
        if da[x_dim].size != self.nx or da[y_dim].size != self.ny:
            raise GridError("the DataArray does not have the shape of the grid")
        if not (
            np.allclose(da[x_dim].values, self.x, rtol=0, atol=1e-9 * abs(self.dx))
            and np.allclose(da[y_dim].values, self.y, rtol=0, atol=1e-9 * abs(self.dy))
        ):
            raise GridError("the DataArray is not aligned with the grid")
        extra = [d for d in da.dims if d not in (x_dim, y_dim)]
        if extra:
            raise GridError(f"unexpected non-spatial dimensions {extra}; select a 2-D slice")
        arr = np.asarray(da.transpose(y_dim, x_dim).values, dtype=np.float64)
        return arr.ravel()

    def to_dataarray(self, flat_values: ArrayLike, name: str | None = None) -> xr.DataArray:
        """Wrap flat (row-major) values as a DataArray on this grid."""
        arr = np.asarray(flat_values, dtype=np.float64).reshape(self.shape)
        da = xr.DataArray(
            arr,
            coords={self.y_dim: self.y, self.x_dim: self.x},
            dims=(self.y_dim, self.x_dim),
            name=name,
        )
        if self.crs is not None:
            da = da.rio.write_crs(self.crs)
        return da
