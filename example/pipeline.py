import laspy
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.compute as pc

from example.catalog import Bounds, Catalog, Point
from pond import node, pipe
from pond.transforms.transform_pipe import TransformPipe


@node(Catalog, "file:cloud_files[:]", "clouds[:].points")
def parse_clouds(cloud_file: laspy.LasData) -> list[Point]:
    """Turn the cloud files into a pond-friendly format"""
    return [Point(x=point[0], y=point[1], z=point[2]) for point in cloud_file.xyz[::20]]


@node(Catalog, "table:clouds[:].points", "clouds[:].bounds")
def compute_cloud_bounds(cloud: pa.Table) -> Bounds:
    """Compute bounds for individual clouds"""
    min_max_x = pc.min_max(cloud["x"])
    min_max_y = pc.min_max(cloud["y"])
    return Bounds(
        minx=min_max_x["min"].as_py(),
        maxx=min_max_x["max"].as_py(),
        miny=min_max_y["min"].as_py(),
        maxy=min_max_y["max"].as_py(),
    )


@node(Catalog, "clouds[:].bounds", "bounds")
def compute_bounds(cloud_bounds: list[Bounds]) -> Bounds:
    """Combine individual bounds into global bounds"""
    bounds = Bounds(minx=np.inf, maxx=-np.inf, miny=np.inf, maxy=-np.inf)
    for cloud_bound in cloud_bounds:
        bounds.minx = min(cloud_bound.minx, bounds.minx)
        bounds.miny = min(cloud_bound.miny, bounds.miny)
        bounds.maxx = max(cloud_bound.maxx, bounds.maxx)
        bounds.maxy = max(cloud_bound.maxy, bounds.maxy)
    return bounds


@node(
    Catalog,
    ["params.res", "table:clouds[:].points", "bounds"],
    ["file:clouds[:].grid_sum", "file:clouds[:].grid_count"],
)
def compute_cloud_heightmap(
    res: float, cloud: pa.Table, bounds: Bounds
) -> tuple[np.ndarray, np.ndarray]:
    """Compute individual heightmaps from clouds and bounds"""
    xbins = np.arange(bounds.minx, bounds.maxx + 0.5 * res, res)
    ybins = np.arange(bounds.miny, bounds.maxy + 0.5 * res, res)
    sums, _, _ = np.histogram2d(
        cloud["x"].to_numpy(),
        cloud["y"].to_numpy(),
        bins=(xbins, ybins),
        density=False,
        weights=cloud["z"].to_numpy(),
    )
    counts, _, _ = np.histogram2d(
        cloud["x"].to_numpy(),
        cloud["y"].to_numpy(),
        bins=(xbins, ybins),
        density=False,
    )
    return sums, counts


@node(
    Catalog, ["file:clouds[:].grid_sum", "file:clouds[:].grid_count"], "file:heightmap"
)
def compute_heightmap(sums: list[np.ndarray], counts: list[np.ndarray]) -> np.ndarray:
    """Combine individual heightmaps into global heightmap"""
    count = sum(counts)
    count[count == 0] = 1.0
    return sum(sums) / count


@node(Catalog, ["params.res", "file:heightmap", "bounds"], "file:heightmap_plot")
def plot_heightmap(res: float, heightmap: np.ndarray, bounds: Bounds) -> go.Figure:
    """Plot final heightmap as png"""
    xbins = np.arange(bounds.minx + 0.5 * res, bounds.maxx, res)
    ybins = np.arange(bounds.miny + 0.5 * res, bounds.maxy, res)
    fig = px.imshow(
        heightmap,
        x=ybins,
        y=xbins,
        labels={"x": "Easting", "y": "Northing", "color": "Height"},
    )
    return fig


def heightmap_pipe() -> TransformPipe:
    return pipe(
        [
            parse_clouds,
            compute_cloud_bounds,
            compute_bounds,
            compute_cloud_heightmap,
            compute_heightmap,
            plot_heightmap,
        ],
        input=["cloud_files", "params"],
        output=["heightmap_plot", "bounds"],
    )
