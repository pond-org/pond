from typing import Any
import pickle

import numpy as np
from PIL import Image
import laspy
import plotly.graph_objects as go


def write_pickle(object: Any, fs, path: str):
    with fs.open(path, mode="wb") as fs_file:
        pickle.dump(object, fs_file)


def write_image(im: Image.Image, fs, path: str):
    print(f"Trying to write {path}")
    with fs.open(path, mode="wb") as fs_file:
        im.save(fs_file)


def write_las(cloud: laspy.LasData, fs, path: str):
    print(f"Trying to write {path}")
    with fs.open(path, mode="wb") as fs_file:
        cloud.write(fs_file)


def write_npz(array: np.ndarray, fs, path: str):
    print(f"Trying to write {path}")
    with fs.open(path, mode="wb") as fs_file:
        np.save(fs_file, array)


def write_plotly_png(fig: go.Figure, fs, path: str):
    print(f"Trying to write {path}")
    with fs.open(path, mode="wb") as fs_file:
        fig.write_image(fs_file, format="png")
