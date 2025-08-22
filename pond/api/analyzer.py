"""Field analysis utilities for separating File[DataT] from regular pydantic fields."""

from typing import Any, Type, get_args, get_origin

from pydantic import BaseModel
from pydantic._internal import _generics

from pond.field import File
from pond.lens import LensInfo


def is_file_field(root_type: Type[BaseModel], path: str) -> bool:
    """Check if a path points to a File[DataT] field.

    Determines whether the field at the given path is a File type by
    inspecting the type annotations in the pydantic model schema.

    Args:
        root_type: Pydantic model class defining the schema structure.
        path: PyPond path string like "clouds[0].points" or "heightmap_plot".

    Returns:
        True if the field is a File[DataT] type, False otherwise.

    Note:
        Uses LensInfo to resolve the type at the path location and checks
        if the origin type is the File generic class.
    """
    lens_info = LensInfo.from_path(root_type, path)
    field_type = lens_info.get_type()

    # Check if it's a File[DataT] type (using pydantic's _generics like lens.py)
    if _generics.get_origin(field_type) is File:
        return True

    # Check if it's a list[File[DataT]] type (following lens.py pattern)
    if get_origin(field_type) is list:
        args = get_args(field_type)
        if args and _generics.get_origin(args[0]) is File:
            return True

    return False


def get_field_type(root_type: Type[BaseModel], path: str) -> Type[Any]:
    """Get the Python type for a field at the given path.

    Resolves the type annotation for the field using the pydantic schema.
    For File[DataT] fields, returns the inner data type DataT.

    Args:
        root_type: Pydantic model class defining the schema structure.
        path: PyPond path string to analyze.

    Returns:
        The Python type at the path location.
    """
    lens_info = LensInfo.from_path(root_type, path)
    field_type = lens_info.get_type()

    # For File[DataT] types, return the inner data type
    if get_origin(field_type) is File:
        args = get_args(field_type)
        if args:
            return args[0]

    return field_type


def get_field_metadata(root_type: Type[BaseModel], path: str) -> dict[str, Any]:
    """Get metadata for a field at the given path.

    Retrieves the json_schema_extra metadata from pydantic Field()
    definitions, which contains pond-specific configuration like
    readers, writers, extensions, and protocols.

    Args:
        root_type: Pydantic model class defining the schema structure.
        path: PyPond path string to get metadata for.

    Returns:
        Dictionary containing field metadata, empty if no metadata exists.

    Note:
        Metadata is used to determine file extensions, storage protocols,
        and custom reader/writer functions for file handling.
    """
    # Navigate to the field and get its metadata
    parts = path.split(".")
    current_type = root_type

    for part in parts:
        # Handle array indices by extracting field name
        if "[" in part:
            field_name = part.split("[")[0]
        else:
            field_name = part

        field_info = current_type.model_fields[field_name]

        # For the last part, return the metadata
        if part == parts[-1]:
            return field_info.json_schema_extra or {}

        # Navigate deeper into nested types
        field_type = field_info.annotation
        if get_origin(field_type) is list:
            field_type = get_args(field_type)[0]

        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            current_type = field_type
        else:
            break

    return {}


def get_file_extension(root_type: Type[BaseModel], path: str) -> str:
    """Get the file extension for a File[DataT] field.

    Retrieves the file extension from field metadata, defaulting to
    'pickle' if no extension is specified.

    Args:
        root_type: Pydantic model class defining the schema structure.
        path: PyPond path string for a File[DataT] field.

    Returns:
        File extension string without leading dot.

    Note:
        Only relevant for File[DataT] fields. Regular fields return 'pickle'
        as a fallback but shouldn't be used for file operations.
    """
    metadata = get_field_metadata(root_type, path)
    return metadata.get("ext", "pickle")


def separate_file_and_regular_fields(
    root_type: Type[BaseModel], paths: list[str]
) -> tuple[list[str], list[str]]:
    """Separate a list of paths into file fields and regular fields.

    Analyzes each path to determine if it points to a File[DataT] field
    or a regular pydantic field, returning separate lists for each type.

    Args:
        root_type: Pydantic model class defining the schema structure.
        paths: List of PyPond path strings to analyze.

    Returns:
        Tuple containing (file_paths, regular_paths) where:
        - file_paths: List of paths pointing to File[DataT] fields
        - regular_paths: List of paths pointing to regular fields

    Note:
        This separation is crucial for FastAPI endpoint generation since
        file fields require multipart form handling while regular fields
        use JSON serialization.
    """
    file_paths = []
    regular_paths = []

    for path in paths:
        if is_file_field(root_type, path):
            file_paths.append(path)
        else:
            regular_paths.append(path)

    return file_paths, regular_paths
