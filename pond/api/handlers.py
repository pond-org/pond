"""HTTP request and response handlers for FastAPI integration."""

import io
from typing import Any, Type

from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pond.api.analyzer import get_file_extension
from pond.field import File
from pond.lens import LensInfo
from pond.state import State


def handle_file_upload(
    root_type: Type[BaseModel], path: str, upload_file: UploadFile
) -> dict:
    """Handle FastAPI file upload by saving temporarily.

    Saves uploaded file to a temporary location and returns metadata
    that will be processed later by the lens system during execute_on.

    Args:
        root_type: Pydantic model class defining the schema structure.
        path: PyPond path string where the file will be stored.
        upload_file: FastAPI UploadFile instance from multipart form.

    Returns:
        Dictionary containing temporary file info for later processing.

    Note:
        The actual File objects will be created during execute_on using
        the State's lens system for proper fsspec and path integration.
    """
    import tempfile

    # Read the file content
    content = upload_file.file.read()

    # Save to temporary file, preserving original filename if possible
    original_filename = upload_file.filename or "uploaded_file"

    # Create temporary file with appropriate suffix
    _, temp_path = tempfile.mkstemp(
        suffix=f"_{original_filename}", prefix="pypond_upload_"
    )

    # Write content to temp file
    with open(temp_path, "wb") as f:
        f.write(content)

    # Return metadata for later processing
    return {
        "temp_path": temp_path,
        "original_filename": original_filename,
        "content_length": len(content),
    }


def handle_json_input(
    root_type: Type[BaseModel], path: str, data: dict[str, Any]
) -> Any:
    """Convert JSON data to appropriate Python object for the path.

    Validates and converts JSON input data to the expected type
    for the given path in the schema.

    Args:
        root_type: Pydantic model class defining the schema structure.
        path: PyPond path string where the data will be stored.
        data: Dictionary containing JSON-decoded data.

    Returns:
        Python object suitable for storage at the given path.

    Note:
        For complex nested types, this may need additional validation
        and type conversion based on the schema.
    """
    # Get the expected type for this path
    lens_info = LensInfo.from_path(root_type, path)
    expected_type = lens_info.get_type()

    # If the expected type is a pydantic model, create an instance
    from pydantic import BaseModel

    if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
        return expected_type(**data)

    # For primitive types, return as-is
    return data


def handle_file_download(
    root_type: Type[BaseModel], path: str, file_obj: File[Any]
) -> StreamingResponse:
    """Convert PyPond File[DataT] to HTTP file download response.

    Creates a streaming response for downloading file content with
    appropriate MIME type and filename headers.

    Args:
        root_type: Pydantic model class defining the schema structure.
        path: PyPond path string where the file is stored.
        file_obj: File[DataT] instance containing the file data.

    Returns:
        StreamingResponse configured for file download.

    Note:
        Uses file extension from field metadata to determine MIME type.
        Filename is derived from the path string.
    """
    # Get file extension for MIME type
    ext = get_file_extension(root_type, path)

    # Determine MIME type from extension
    mime_types = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "pdf": "application/pdf",
        "json": "application/json",
        "csv": "text/csv",
        "txt": "text/plain",
    }
    mime_type = mime_types.get(ext, "application/octet-stream")

    # Get the file content
    if hasattr(file_obj, "get"):
        content = file_obj.get()
    elif hasattr(file_obj, "load"):
        content = file_obj.load()
    else:
        # Fallback - read from path if available
        content = b""

    # Convert content to bytes if needed
    if isinstance(content, str):
        content = content.encode("utf-8")
    elif not isinstance(content, bytes):
        # For complex objects, this would need serialization
        # based on the field's writer configuration
        content = str(content).encode("utf-8")

    # Create filename from path
    filename = path.replace(".", "_").replace("[", "_").replace("]", "") + f".{ext}"

    # Create streaming response
    return StreamingResponse(
        io.BytesIO(content),
        media_type=mime_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def handle_json_output(
    root_type: Type[BaseModel], path: str, data: Any
) -> dict[str, Any]:
    """Convert Python object to JSON-serializable format.

    Serializes data from the given path to a format suitable for
    JSON HTTP responses.

    Args:
        root_type: Pydantic model class defining the schema structure.
        path: PyPond path string where the data is stored.
        data: Python object to serialize.

    Returns:
        Dictionary containing JSON-serializable data.

    Note:
        For complex objects like pydantic models, this uses model_dump()
        for proper serialization.
    """
    # Handle pydantic models
    if isinstance(data, BaseModel):
        return data.model_dump()

    # Handle lists of pydantic models
    if isinstance(data, list) and data and isinstance(data[0], BaseModel):
        return [item.model_dump() for item in data]

    # Handle basic types
    if isinstance(data, (str, int, float, bool, type(None))):
        return {"value": data}

    # Handle lists and dicts
    if isinstance(data, (list, dict)):
        return {"value": data}

    # Fallback for other types
    return {"value": str(data)}


def set_state_value(state: State, path: str, value: Any) -> None:
    """Set a value in the PyPond state at the specified path.

    Uses the state's dictionary-style access to store the value
    at the given path location.

    Args:
        state: PyPond State instance to modify.
        path: PyPond path string where to store the value.
        value: Value to store at the path.

    Note:
        The state object handles type validation and storage via
        the lens system and catalog backend.
    """
    state[path] = value


def get_state_value(state: State, path: str) -> Any:
    """Get a value from the PyPond state at the specified path.

    Uses the state's dictionary-style access to retrieve the value
    from the given path location.

    Args:
        state: PyPond State instance to read from.
        path: PyPond path string where to read the value.

    Returns:
        Value stored at the path location.

    Note:
        The state object handles data loading via the lens system
        and catalog backend, including file content loading.
    """
    return state[path]
