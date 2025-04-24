from pydantic import BaseModel
import numpy as np
import laspy
import plotly.graph_objects as go

from pond import File, Field
from pond.io.readers import read_npz, read_las
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


class Catalog(BaseModel):
    params: Parameters
    cloud_files: list[File[laspy.LasData]] = Field(
        reader=read_las, ext="laz", path="raw_clouds"
    )
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
