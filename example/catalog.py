import laspy
import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel

from pond import Field, File
from pond.io.readers import read_las, read_npz
from pond.io.writers import write_npz, write_plotly_png


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
    bounds: Bounds
    grid_sum: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
    grid_count: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")


class Catalog(BaseModel):
    params: Parameters
    cloud_files: list[File[laspy.LasData]] = Field(
        reader=read_las,
        ext="laz",
        protocol="github",
        path="liblas/LDR030828*",
        # path="raw_clouds",
    )
    clouds: list[Cloud]
    bounds: Bounds
    heightmap: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
    heightmap_plot: File[go.Figure] = Field(writer=write_plotly_png, ext="png")
