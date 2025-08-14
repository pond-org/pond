import pickle
from typing import Any

import laspy  # type: ignore
import numpy as np
from PIL import Image


def read_pickle(fs, path: str) -> Any:
    """Read a Python object from a pickle file.

    Args:
        fs: Filesystem interface (fsspec-compatible).
        path: Path to the pickle file to read.

    Returns:
        The unpickled Python object.

    Note:
        Uses Python's pickle module for deserialization.
        Compatible with any fsspec filesystem backend.
    """
    with fs.open(path, mode="rb") as fs_file:
        return pickle.load(fs_file)


def read_image(fs, path: str) -> Image.Image:
    """Read an image file using PIL/Pillow.

    Args:
        fs: Filesystem interface (fsspec-compatible).
        path: Path to the image file to read.

    Returns:
        PIL Image object loaded from the file.

    Note:
        Uses PIL/Pillow for image loading. Creates a copy to ensure
        the image data is fully loaded before the file is closed.
        Supports all image formats supported by PIL.
    """
    with fs.open(path, mode="rb") as fs_file:
        return Image.open(fs_file).copy()


def read_las(fs, path: str) -> laspy.LasData:
    """Read LiDAR point cloud data from LAS/LAZ files.

    Args:
        fs: Filesystem interface (fsspec-compatible).
        path: Path to the LAS or LAZ file to read.

    Returns:
        LasData object containing the point cloud data.

    Note:
        Uses the laspy library for LAS/LAZ file parsing.
        Supports both uncompressed LAS and compressed LAZ formats.
        Compatible with various LAS versions and point data formats.
    """
    with fs.open(path, mode="rb") as fs_file:
        return laspy.read(fs_file)


def read_npz(fs, path: str) -> np.ndarray:
    """Read a NumPy array from a .npy file.

    Args:
        fs: Filesystem interface (fsspec-compatible).
        path: Path to the .npy file to read.

    Returns:
        NumPy array loaded from the file.

    Note:
        Uses NumPy's load function for .npy file parsing.
        Preserves array dtype and shape information.
        For .npz files, use the appropriate key access after loading.
    """
    with fs.open(path, mode="rb") as fs_file:
        return np.load(fs_file)
