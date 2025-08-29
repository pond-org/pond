# Copyright 2025 Nils Bore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
