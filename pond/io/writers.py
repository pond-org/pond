import pickle
from typing import Any

import laspy  # type: ignore
import numpy as np
import plotly.graph_objects as go  # type: ignore
from PIL import Image


def write_pickle(object: Any, fs, path: str):
    """Write a Python object to a pickle file.

    Args:
        object: The Python object to serialize and write.
        fs: Filesystem interface (fsspec-compatible).
        path: Path where the pickle file will be written.

    Note:
        Uses Python's pickle module for serialization.
        Compatible with any fsspec filesystem backend.
    """
    with fs.open(path, mode="wb") as fs_file:
        pickle.dump(object, fs_file)


def write_image(im: Image.Image, fs, path: str):
    """Write a PIL Image to a file.

    Args:
        im: PIL Image object to write.
        fs: Filesystem interface (fsspec-compatible).
        path: Path where the image file will be written.

    Note:
        Uses PIL/Pillow for image saving. Format is determined by
        the file extension in the path. Supports all output formats
        supported by PIL.
    """
    with fs.open(path, mode="wb") as fs_file:
        im.save(fs_file)


def write_las(cloud: laspy.LasData, fs, path: str):
    """Write LiDAR point cloud data to LAS/LAZ file.

    Args:
        cloud: LasData object containing point cloud data to write.
        fs: Filesystem interface (fsspec-compatible).
        path: Path where the LAS/LAZ file will be written.

    Note:
        Uses the laspy library for LAS/LAZ file writing.
        Output format (LAS vs LAZ compression) is determined by
        the file extension. Preserves all point data and metadata.
    """
    with fs.open(path, mode="wb") as fs_file:
        cloud.write(fs_file)


def write_npz(array: np.ndarray, fs, path: str):
    """Write a NumPy array to a .npy file.

    Args:
        array: NumPy array to write.
        fs: Filesystem interface (fsspec-compatible).
        path: Path where the .npy file will be written.

    Note:
        Uses NumPy's save function for .npy file creation.
        Preserves array dtype and shape information.
        For multiple arrays, consider using np.savez instead.
    """
    with fs.open(path, mode="wb") as fs_file:
        np.save(fs_file, array)


def write_plotly_png(fig: go.Figure, fs, path: str):
    """Write a Plotly figure to a PNG image file.

    Args:
        fig: Plotly Figure object to write.
        fs: Filesystem interface (fsspec-compatible).
        path: Path where the PNG file will be written.

    Note:
        Uses Plotly's write_image function to export as PNG.
        Requires kaleido or orca to be installed for image export.
        Useful for saving visualizations created with Plotly.
    """
    with fs.open(path, mode="wb") as fs_file:
        fig.write_image(fs_file, format="png")
