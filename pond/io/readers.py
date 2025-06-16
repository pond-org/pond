import pickle
from typing import Any

import laspy
import numpy as np
from PIL import Image


def read_pickle(fs, path: str) -> Any:
    with fs.open(path, mode="rb") as fs_file:
        return pickle.load(fs_file)


def read_image(fs, path: str) -> Image.Image:
    with fs.open(path, mode="rb") as fs_file:
        return Image.open(fs_file).copy()


def read_las(fs, path: str) -> laspy.LasData:
    with fs.open(path, mode="rb") as fs_file:
        return laspy.read(fs_file)


def read_npz(fs, path: str) -> np.ndarray:
    with fs.open(path, mode="rb") as fs_file:
        return np.load(fs_file)
