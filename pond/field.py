from typing import Callable, Generic, TypeVar

import pydantic
from pydantic import BaseModel, ConfigDict

DataT = TypeVar("DataT")


class File(BaseModel, Generic[DataT]):
    """Generic file container for unstructured data storage.

    A pydantic model that represents a file on disk with optional lazy loading
    capabilities. Can store either a file path for lazy loading or the actual
    data object for immediate access.

    Attributes:
        path: File system path where the data is or will be stored.

    Note:
        Uses ConfigDict(extra="allow") to store additional fields like 'loader'
        and 'object' which are set dynamically but not part of the base schema.
    """

    # This is needed in order to store the "loader"
    model_config = ConfigDict(extra="allow")
    path: str

    @staticmethod
    def save(path: str, loader: Callable[[], DataT]) -> "File[DataT]":
        """Create a File instance with lazy loading capability.

        Args:
            path: File system path where data will be loaded from.
            loader: Function that returns the data when called, taking no arguments.

        Returns:
            File instance configured for lazy loading from the specified path.

        Note:
            The loader function is stored in the 'loader' extra field and can be
            called later via the load() method.
        """
        return File(path=path, loader=loader)  # type: ignore

    @staticmethod
    def set(object: DataT) -> "File[DataT]":
        """Create a File instance with immediate data access.

        Args:
            object: The actual data to be stored in the File instance.

        Returns:
            File instance with the data immediately available via get().

        Note:
            The object is stored in the 'object' extra field and path is set to
            empty string since no file loading is involved.
        """
        return File(path="", object=object)  # type: ignore

    def get(self) -> DataT:
        """Retrieve the stored data object.

        Returns:
            The data object that was previously set via the set() method.

        Raises:
            AttributeError: If no object was set (i.e., File was created for lazy loading).

        Note:
            This method assumes the 'object' extra field was set via set() method.
            For lazy-loaded files, use load() instead.
        """
        return self.object  # type: ignore

    def load(self) -> DataT:
        """Load data using the stored loader function.

        Returns:
            The data returned by calling the stored loader function.

        Raises:
            AttributeError: If no loader was set (i.e., File was created via set()).
            TypeError: If the stored loader is not callable.

        Note:
            This method assumes the 'loader' extra field was set via save() method.
            The loader function is expected to take no arguments and return DataT.
        """
        return self.loader()  # type: ignore


def Field(
    *args,
    reader=None,
    writer=None,
    ext="pickle",
    path=None,
    protocol=None,
    json_schema_extra={},
    **kwargs,
):
    """Create a pydantic Field with pond-specific metadata for file handling.

    A wrapper around pydantic.Field that adds pond-specific configuration for
    handling file I/O, storage protocols, and data serialization. The metadata
    is stored in json_schema_extra and used by the lens system for data access.

    Args:
        *args: Positional arguments passed through to pydantic.Field.
        reader: Function to read/deserialize data from storage. Should accept
            (filesystem, path) arguments and return the deserialized data.
        writer: Function to write/serialize data to storage. Should accept
            (data, filesystem, path) arguments.
        ext: File extension for storage, defaults to "pickle". Used to determine
            the full file path when reading/writing.
        path: Custom storage path override. If None, path is derived from the
            field's position in the data structure hierarchy.
        protocol: Storage protocol name (e.g., 's3', 'gcs', 'file'). If None,
            uses the default protocol from the State configuration.
        json_schema_extra: Additional metadata to merge into the field schema.
        **kwargs: Additional keyword arguments passed to pydantic.Field.

    Returns:
        A pydantic Field instance with pond metadata attached for file handling.

    Note:
        The metadata is accessible via the field's json_schema_extra attribute
        and used by Lens instances to determine how to read/write data.
        Default mutable argument {} is safe here as it's immediately copied.
    """
    json_schema_extra = {
        "reader": reader,
        "writer": writer,
        "ext": ext,
        "path": path,
        "protocol": protocol,
        **json_schema_extra,
    }
    return pydantic.Field(*args, json_schema_extra=json_schema_extra, **kwargs)
