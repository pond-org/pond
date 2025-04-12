import numpy as np
from typing import Any
import pickle

from PIL import Image
import laspy


def read_pickle(fs, path: str) -> Any:
    with fs.open(path, mode="rb") as fs_file:
        return pickle.load(fs_file)


def read_image(fs, path: str) -> Image.Image:
    print(f"Trying to read {path}")
    with fs.open(path, mode="rb") as fs_file:
        return Image.open(fs_file).copy()


def read_las(fs, path: str) -> laspy.LasData:
    print(f"Trying to read {path}")
    with fs.open(path, mode="rb") as fs_file:
        return laspy.read(fs_file)


def read_npz(fs, path: str) -> np.ndarray:
    print(f"Trying to read {path}")
    with fs.open(path, mode="rb") as fs_file:
        return np.load(fs_file)
