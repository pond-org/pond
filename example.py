import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import plotly.graph_objects as go
import plotly.express as px
import laspy
from pyiceberg.catalog import load_catalog

from pydantic import BaseModel

from pond import State, File, Field, node, pipe, index_files
from pond.readers import read_npz, read_las
from pond.writers import write_npz, write_plotly_png

from pond.transform_pipe import TransformPipe
from pond.abstract_catalog import IcebergCatalog
from pond.ui import UIClient

# os.environ["PYICEBERG_HOME"] = os.getcwd()


class Parameters(BaseModel):
    res: float


class Point(BaseModel):
    x: float
    y: float
    z: float


class Bounds(BaseModel):
    minx: float
    maxx: float
    miny: float
    maxy: float


class Cloud(BaseModel):
    points: list[Point]


class Catalog(BaseModel):
    params: Parameters
    cloud_files: list[File[laspy.LasData]] = Field(reader=read_las, ext="laz")
    clouds: list[Cloud]
    cloud_bounds: list[Bounds]
    bounds: Bounds
    grid_sums: list[File[np.ndarray]] = Field(
        reader=read_npz, writer=write_npz, ext="npy"
    )
    grid_counts: list[File[np.ndarray]] = Field(
        reader=read_npz, writer=write_npz, ext="npy"
    )
    heightmap: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
    heightmap_plot: File[go.Figure] = Field(writer=write_plotly_png, ext="png")


@node(Catalog, "file:cloud_files[:]", "clouds[:]")
def parse_clouds(cloud_file: laspy.LasData) -> Cloud:
    points = [
        Point(x=point[0], y=point[1], z=point[2]) for point in cloud_file.xyz[::20]
    ]
    return Cloud(points=points)


@node(Catalog, "table:clouds[:].points", "cloud_bounds[:]")
def compute_cloud_bounds(cloud: pa.Table) -> Bounds:
    min_max_x = pc.min_max(cloud["x"])
    min_max_y = pc.min_max(cloud["y"])
    return Bounds(
        minx=min_max_x["min"].as_py(),
        maxx=min_max_x["max"].as_py(),
        miny=min_max_y["min"].as_py(),
        maxy=min_max_y["max"].as_py(),
    )


@node(Catalog, "cloud_bounds", "bounds")
def compute_bounds(cloud_bounds: list[Bounds]) -> Bounds:
    bounds = Bounds(minx=np.inf, maxx=-np.inf, miny=np.inf, maxy=-np.inf)
    for cloud_bound in cloud_bounds:
        bounds.minx = min(cloud_bound.minx, bounds.minx)
        bounds.miny = min(cloud_bound.miny, bounds.miny)
        bounds.maxx = max(cloud_bound.maxx, bounds.maxx)
        bounds.maxy = max(cloud_bound.maxy, bounds.maxy)
    print("Result: ", bounds)
    return bounds


@node(
    Catalog,
    ["params.res", "table:clouds[:].points", "bounds"],
    ["file:grid_sums[:]", "file:grid_counts[:]"],
)
def compute_cloud_heightmap(
    res: float, cloud: pa.Table, bounds: Bounds
) -> tuple[np.ndarray, np.ndarray]:
    xbins = np.arange(bounds.minx, bounds.maxx + 0.5 * res, res)
    ybins = np.arange(bounds.miny, bounds.maxy + 0.5 * res, res)
    print(type(cloud))
    print("CLOUD: ", cloud["x"])
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


@node(Catalog, ["file:grid_sums", "file:grid_counts"], "file:heightmap")
def compute_heightmap(sums: list[np.ndarray], counts: list[np.ndarray]) -> np.ndarray:
    count = sum(counts)
    count[count == 0] = 1.0
    return sum(sums) / count


@node(Catalog, ["params.res", "file:heightmap", "bounds"], "file:heightmap_plot")
def plot_heightmap(res: float, heightmap: np.ndarray, bounds: Bounds) -> go.Figure:
    xbins = np.arange(bounds.minx + 0.5 * res, bounds.maxx, res)
    ybins = np.arange(bounds.miny + 0.5 * res, bounds.maxy, res)
    fig = px.imshow(
        heightmap,
        x=ybins,
        y=xbins,
        labels={"x": "Easting", "y": "Northing", "color": "Height"},
    )
    return fig


def prepare_pipe() -> TransformPipe:
    return pipe(
        [
            index_files(Catalog, "cloud_files"),
        ],
        output="cloud_files",
    )


def heightmap_pipe() -> TransformPipe:
    return pipe(
        [
            prepare_pipe(),
            parse_clouds,
            compute_cloud_bounds,
            compute_bounds,
            compute_cloud_heightmap,
            compute_heightmap,
            plot_heightmap,
        ],
        input="params",
        output="heightmap_plot",
    )


def main():
    catalog = IcebergCatalog(load_catalog(name="default"))
    state = State(
        Catalog, catalog, storage_path="/home/nbore/Workspace/py/pypond/storage"
    )
    state["params.res"] = 4.0
    pipeline = heightmap_pipe()
    ui_client = UIClient(1, "nils", "pond", Catalog)
    transforms = pipeline.get_transforms()
    ui_client.post_graph_construct(transforms)
    ui_client.pre_graph_execute("test", transforms, [], [])
    error = None
    success = True
    result = None
    for transform in transforms:
        ui_client.pre_node_execute("test", transform)
        try:
            for unit in transform.get_execute_units(state):
                unit.execute_on(state)
        except Exception as e:
            error = e
            success = False

        ui_client.post_node_execute("test", transform, success, error, result)
        if error is not None:
            break
    ui_client.post_graph_execute("test", transforms, success, error)
    # for unit in plot_heightmap.get_execute_units(state):
    #     unit.execute_on(state)


if __name__ == "__main__":
    main()
