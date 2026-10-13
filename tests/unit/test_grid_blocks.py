from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401  (registers the .rio accessor)
import shapely
import xarray as xr
from scipy import stats

from pyspatialuncertainty.blocks import box, domain, polygons, resolve_blocks, tiles, zones
from pyspatialuncertainty.grid import GridError, GridSpec


def make_da(nx: int = 10, ny: int = 8, dx: float = 2.0, north_up: bool = True) -> xr.DataArray:
    x = 100 + dx * (np.arange(nx) + 0.5)
    y = 50 + dx * (np.arange(ny) + 0.5)
    if north_up:
        y = y[::-1]
    data = np.arange(nx * ny, dtype=float).reshape(ny, nx)
    return xr.DataArray(data, coords={"y": y, "x": x}, dims=("y", "x"))


def test_grid_geometry_and_lookup() -> None:
    g = GridSpec.from_dataarray(make_da())
    assert g.shape == (8, 10)
    assert g.dx == pytest.approx(2.0)
    assert g.dy == pytest.approx(-2.0)
    assert g.cell_area == pytest.approx(4.0)
    assert g.bounds == pytest.approx((100.0, 50.0, 120.0, 66.0))
    rows, cols = g.cell_of([100.1, 119.9, 120.0, 99.0], [65.9, 50.1, 50.0, 60.0])
    assert rows.tolist() == [0, 7, 7, -1]
    assert cols.tolist() == [0, 9, 9, -1]
    xs, ys = g.cell_centres(g.flat_index_of([101.3], [64.2]))
    assert (xs[0], ys[0]) == pytest.approx((101.0, 65.0))


def test_grid_values_transposes_and_checks_alignment() -> None:
    da = make_da()
    g = GridSpec.from_dataarray(da)
    np.testing.assert_array_equal(g.values(da.transpose("x", "y")), da.values.ravel())
    with pytest.raises(GridError, match="aligned"):
        g.values(da.assign_coords(x=da.x + 0.5))
    back = g.to_dataarray(g.values(da))
    xr.testing.assert_allclose(back, da)


def test_grid_rejects_irregular_and_geographic() -> None:
    da = make_da()
    with pytest.raises(GridError, match="regularly"):
        GridSpec.from_dataarray(da.assign_coords(x=np.r_[da.x.values[:-1], 1000.0]))
    geo = da.rio.write_crs("EPSG:4326")
    with pytest.raises(GridError, match="geographic"):
        GridSpec.from_dataarray(geo)
    proj = da.rio.write_crs("EPSG:6933")
    assert GridSpec.from_dataarray(proj).crs is not None


def test_tiles() -> None:
    g = GridSpec.from_dataarray(make_da())
    t = tiles(g, 4)
    assert t.tile_shape == (2, 3)
    assert len(t) == 6
    assert t.blocks[0].cells.size == 16
    assert t.blocks[2].cells.size == 8  # partial tile: 4 rows x 2 cols
    assert len(tiles(g, 4, drop_partial=True)) == 4  # 2 x 2 complete tiles
    assert tiles(g, 8.0, units="map").tile_size == (4, 4)
    all_cells = np.sort(np.concatenate([b.cells for b in t.blocks]))
    np.testing.assert_array_equal(all_cells, np.arange(g.size))


def test_zones_and_domain() -> None:
    da = make_da()
    g = GridSpec.from_dataarray(da)
    z = xr.where(da.x < 110, 1.0, 2.0)
    z = z.where(da.y < 64)
    zs = zones(z, g)
    assert zs.ids == [1, 2]
    assert sum(b.cells.size for b in zs.blocks) == 7 * 10
    assert len(zones(z, g, ignore=[2.0])) == 1
    assert domain(g).blocks[0].cells.size == g.size


def test_polygon_cells_include_boundary_centres() -> None:
    g = GridSpec.from_dataarray(make_da())
    b = box(101.0, 51.0, 105.0, 55.0, g)  # boundary passes through cell centres
    assert b.blocks[0].cells.size == 9


def test_polygon_crs_mismatch() -> None:
    g = GridSpec.from_dataarray(make_da().rio.write_crs("EPSG:6933"))

    class FakeGeoSeries:
        crs = "EPSG:3857"
        geometry = (shapely.box(100, 50, 110, 60),)

    with pytest.raises(ValueError, match="different CRS"):
        polygons(FakeGeoSeries(), g)


def test_resolve_areas_and_validity() -> None:
    g = GridSpec.from_dataarray(make_da())
    valid = np.ones(g.size, dtype=bool)
    rb = resolve_blocks(tiles(g, 4), valid)
    assert rb[0].area == pytest.approx(64.0)
    assert rb[2].area == pytest.approx(32.0)
    poly = shapely.Polygon([(100, 50), (120, 50), (100, 66)])
    gb = resolve_blocks(polygons([poly], g), valid)[0]
    assert gb.area == pytest.approx(poly.area)
    # invalidate the top-left cell: area loses that cell's overlap with the triangle
    valid[0] = False
    gb2 = resolve_blocks(polygons([poly], g), valid)[0]
    cell = shapely.box(100, 64, 102, 66)
    assert gb2.area == pytest.approx(poly.area - cell.intersection(poly).area)
    assert 0 not in gb2.cells
    with pytest.raises(ValueError, match="no valid cells"):
        resolve_blocks(tiles(g, 1), np.zeros(g.size, dtype=bool))


def test_samplers_are_uniform_and_respect_validity() -> None:
    g = GridSpec.from_dataarray(make_da())
    valid = np.ones(g.size, dtype=bool)
    rng = np.random.default_rng(3)
    tile = resolve_blocks(tiles(g, 4), valid)[0]
    x, y, c = tile.sampler(20_000, rng)
    assert stats.kstest((x - 100) / 8, "uniform").pvalue > 0.01
    assert stats.kstest((y - 58) / 8, "uniform").pvalue > 0.01
    np.testing.assert_array_equal(g.flat_index_of(x, y), c)
    valid[g.flat_index_of([103.0], [63.0])] = False
    geo = resolve_blocks(box(100, 50, 110, 66, g), valid)[0]
    x, y, c = geo.sampler(20_000, rng)
    assert np.all(valid[c])
    assert np.all((x >= 100) & (x <= 110))
    np.testing.assert_array_equal(g.flat_index_of(x, y), c)
