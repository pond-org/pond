"""Utility functions for FastAPI integration with PyPond paths."""

import re
from typing import Type

from pydantic import BaseModel

from pond.catalogs.abstract_catalog import LensPath
from pond.lens import LensInfo


def path_to_url(path: str) -> str:
    """Convert PyPond path string to URL-safe format.

    Transforms dotted path notation with array indices into URL path segments.
    Converts array wildcards to FastAPI path parameters for dynamic routing.

    Args:
        path: PyPond path string like "clouds[0].points" or "data[:].values".

    Returns:
        URL-safe path string with segments separated by forward slashes.

    Examples:
        >>> path_to_url("params")
        'params'
        >>> path_to_url("clouds[0].points")
        'clouds/0/points'
        >>> path_to_url("data[:].bounds")
        'data/{index}/bounds'
        >>> path_to_url("nested.field[5].subfield")
        'nested/field/5/subfield'

    Note:
        Array wildcard notation "[:] " is converted to FastAPI path parameter
        "{index}" for dynamic URL routing. Specific indices become literal
        path segments.
    """
    # Replace array wildcards [:] with FastAPI path parameter {index}
    url = re.sub(r"\[:\]", "/{index}", path)

    # Replace specific array indices [n] with /n
    url = re.sub(r"\[(\d+)\]", r"/\1", url)

    # Replace dots with forward slashes
    url = url.replace(".", "/")

    return url


def url_to_path(url: str, index_value: int | None = None) -> str:
    """Convert URL path back to PyPond path string.

    Reverses the path_to_url transformation, converting URL segments back
    to dotted PyPond path notation with array indices.

    Args:
        url: URL path string like "clouds/0/points" or "data/{index}/bounds".
        index_value: Value to substitute for {index} path parameters.
            If None, {index} segments are converted to [:] wildcards.

    Returns:
        PyPond path string in dotted notation with array indices.

    Examples:
        >>> url_to_path("params")
        'params'
        >>> url_to_path("clouds/0/points")
        'clouds[0].points'
        >>> url_to_path("data/{index}/bounds", index_value=2)
        'data[2].bounds'
        >>> url_to_path("data/{index}/bounds")
        'data[:].bounds'

    Note:
        If index_value is provided, {index} segments are replaced with [index_value].
        Otherwise, {index} segments become [:] wildcards.
    """
    # Handle {index} path parameters
    if index_value is not None:
        url = url.replace("{index}", str(index_value))
        # Convert numeric segments to array indices
        path = re.sub(r"/(\d+)/", r"[\1].", url)
        path = re.sub(r"/(\d+)$", r"[\1]", path)
    else:
        # Convert {index} to wildcard notation
        url = url.replace("/{index}", "[:]")
        # Convert any remaining numeric segments to array indices
        path = re.sub(r"/(\d+)/", r"[\1].", url)
        path = re.sub(r"/(\d+)$", r"[\1]", url)

    # Convert remaining forward slashes to dots
    path = url.replace("/", ".")

    return path


def validate_path_in_schema(path: str, root_type: Type[BaseModel]) -> bool:
    """Validate that a path string exists in the pydantic schema.

    Checks if the given path is valid according to the root type schema
    by attempting to create a LensPath and resolve its type.

    Args:
        path: PyPond path string to validate like "clouds[0].points".
        root_type: Pydantic model class defining the schema structure.

    Returns:
        True if the path is valid in the schema, False otherwise.

    Note:
        Uses LensPath.from_path() internally to validate path structure
        and type resolution. Invalid paths will raise exceptions that
        are caught and converted to False return values.
    """
    try:
        lens_path = LensPath.from_path(path, root_path="catalog")
        # Attempt to resolve the type - this will fail for invalid paths
        LensInfo(root_type, lens_path.to_path()).get_type()
        return True
    except (KeyError, AttributeError, IndexError, TypeError):
        return False


def extract_file_extension(field_metadata: dict) -> str:
    """Extract file extension from field metadata.

    Retrieves the file extension from pydantic field metadata,
    defaulting to 'pickle' if no extension is specified.

    Args:
        field_metadata: Dictionary containing field metadata from
            pydantic Field() json_schema_extra.

    Returns:
        File extension string without leading dot.

    Note:
        The extension is used for determining MIME types and
        file handling behavior in HTTP responses.
    """
    if field_metadata and isinstance(field_metadata, dict):
        return field_metadata.get("ext", "pickle")
    return "pickle"
