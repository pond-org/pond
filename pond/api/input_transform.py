"""FastAPI input transform for receiving HTTP data."""

import os
import threading
from typing import Any, Type, get_origin

import fsspec
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from pond.api.analyzer import separate_file_and_regular_fields, get_field_metadata
from pond.api.handlers import handle_file_upload, handle_json_input, set_state_value
from pond.api.shared import BaseExecuteUnit
from pond.api.utils import path_to_url
from pond.field import File
from pond.lens import LensPath, LensInfo, get_cleaned_path
from pond.state import State
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractExecuteUnit,
)


class FastAPIInputExecuteUnit(BaseExecuteUnit):
    """Execute unit that registers FastAPI input endpoints and waits for input data."""

    def __init__(self, root_type: Type[BaseModel], paths: list[str], app: FastAPI):
        """Initialize the FastAPI input execute unit.

        Args:
            root_type: Pydantic model class defining the schema structure.
            paths: List of PyPond path strings to expose as input endpoints.
            app: FastAPI application instance to register endpoints on.
        """
        super().__init__(inputs=[], outputs=paths)
        self.root_type = root_type
        self.paths = paths
        self.app = app
        self.received_data: dict[str, Any] = {}
        self.data_event = threading.Event()

    def load_inputs(self, state: State) -> list[Any]:
        """Load inputs - not used for input transforms."""
        return []

    def save_outputs(self, state: State, outputs: list[Any]) -> list[Any]:
        """Save outputs - not used for input transforms."""
        return []

    def commit(self, state: State, values: list[Any]) -> bool:
        """Commit - not used for input transforms."""
        return True

    def run(self, args: list[Any]) -> list[Any]:
        """Register FastAPI endpoints and wait for input data.

        Creates FastAPI endpoints for each input path and blocks until
        all required data has been received via HTTP requests.

        Args:
            args: Unused for input transforms.

        Returns:
            Empty list - data is stored directly in state.
        """
        # Separate file and regular fields
        file_paths, regular_paths = separate_file_and_regular_fields(
            self.root_type, self.paths
        )

        # Create endpoints for regular fields (JSON input)
        for path in regular_paths:
            url = path_to_url(path)
            self._create_json_endpoint(path, url)

        # Create endpoints for file fields (multipart input)
        for path in file_paths:
            url = path_to_url(path)
            self._create_file_endpoint(path, url)

        # Add status endpoint
        @self.app.get("/status")
        def get_status():
            missing = [path for path in self.paths if path not in self.received_data]
            return {
                "received": list(self.received_data.keys()),
                "missing": missing,
                "complete": len(missing) == 0,
            }

        # Wait for all data to be received
        print(f"Waiting for input data on paths: {self.paths}")
        self.data_event.wait()

        return []

    def _create_json_endpoint(self, path: str, url: str) -> None:
        """Create JSON input endpoint for a regular field."""

        def create_handler():
            def handler(data: dict):
                processed_data = handle_json_input(self.root_type, path, data)
                self.received_data[path] = processed_data
                if len(self.received_data) == len(self.paths):
                    self.data_event.set()
                return {"status": "received", "path": path}

            # Give each handler a unique name based on the path
            handler.__name__ = f"receive_json_{path.replace('.', '_').replace('[', '_').replace(']', '_').replace(':', '_')}"
            return handler

        self.app.post(f"/input/{url}")(create_handler())

    def _create_file_endpoint(self, path: str, url: str) -> None:
        """Create file upload endpoint for a File[DataT] field."""

        def create_handler():
            def handler(files: list[UploadFile]):
                # Check if this path expects a list of files
                lens_info = LensInfo.from_path(self.root_type, path)
                field_type = lens_info.get_type()

                if get_origin(field_type) is list:
                    # Process all files and store as list
                    processed_files = []
                    for file in files:
                        processed_file = handle_file_upload(self.root_type, path, file)
                        processed_files.append(processed_file)
                    self.received_data[path] = processed_files
                else:
                    # Single file - take the first one if multiple provided
                    if files:
                        processed_file = handle_file_upload(
                            self.root_type, path, files[0]
                        )
                        self.received_data[path] = processed_file

                if len(self.received_data) == len(self.paths):
                    self.data_event.set()
                return {
                    "status": "received",
                    "path": path,
                    "count": len(files),
                    "filenames": [f.filename for f in files],
                }

            # Give each handler a unique name based on the path
            handler.__name__ = f"receive_files_{path.replace('.', '_').replace('[', '_').replace(']', '_').replace(':', '_')}"
            return handler

        self.app.post(f"/input/{url}")(create_handler())

    def execute_on(self, state: State) -> None:
        """Execute the input transform and populate state with received data."""
        self.run([])

        # Process all received data and store in state
        for path, value in self.received_data.items():
            # Check if this path contains file upload metadata
            if self._is_file_upload_data(value):
                # Process file uploads - save files and write to catalog directly
                self._process_file_uploads_to_catalog(state, path, value)
            else:
                # Regular data - store directly
                set_state_value(state, path, value)

    def _is_file_upload_data(self, value) -> bool:
        """Check if value contains file upload metadata."""
        if isinstance(value, dict) and "temp_path" in value:
            return True
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "temp_path" in value[0]
        ):
            return True
        return False

    def _process_file_uploads_to_catalog(self, state: State, path: str, upload_data):
        """Process file upload data and write directly to catalog like index_files does."""
        from pydantic_to_pyarrow import get_pyarrow_schema
        import pyarrow as pa

        # Get field metadata for file extension and storage path
        metadata = get_field_metadata(self.root_type, path)
        ext = metadata.get("ext", "bin")

        # Use protocol from field metadata or default
        protocol = metadata.get("protocol") or state.default_volume_protocol
        print(f"DEBUG: protocol = {protocol}")
        print(f"DEBUG: volume_protocol_args = {state.volume_protocol_args}")
        print(f"DEBUG: protocol config = {state.volume_protocol_args.get(protocol)}")
        fs = fsspec.filesystem(**state.volume_protocol_args[protocol])

        # Determine the base file path using field metadata or lens system
        if metadata.get("path"):
            # Use explicit path from field metadata
            base_file_path = metadata["path"]
        else:
            # Use computed path from lens system
            lens = state.lens(path)
            base_file_path = lens.lens_path.to_volume_path()

        # Get lens path for catalog operations
        lens = state.lens(path)
        lens_path = lens.lens_path

        if isinstance(upload_data, list):
            # Handle list of files - create multiple File objects and write to catalog
            file_objects = []
            for i, file_info in enumerate(upload_data):
                file_obj = self._save_temp_file_via_fsspec(
                    fs, base_file_path, file_info, ext, index=i
                )
                file_objects.append(file_obj)

            # Write to catalog using same pattern as lens.py index_files
            # For list[File[T]], we need the schema for the File class, not the list
            from typing import get_args, get_origin

            if get_origin(lens.type) is list:
                # Extract File[T] from list[File[T]]
                file_type = get_args(lens.type)[0]
                schema = get_pyarrow_schema(file_type)
            else:
                schema = get_pyarrow_schema(lens.type)
            values = [file_obj.model_dump() for file_obj in file_objects]
            table = pa.Table.from_pylist(values, schema=schema)
            state.catalog.write_table(table, lens_path, schema, per_row=False)
        else:
            # Handle single file
            file_obj = self._save_temp_file_via_fsspec(
                fs, base_file_path, upload_data, ext
            )

            # Write to catalog using same pattern as lens.py index_files
            # For single File[T], extract the File class
            from typing import get_args, get_origin

            if get_origin(lens.type) is list:
                # Extract File[T] from list[File[T]] (shouldn't happen in single file case)
                file_type = get_args(lens.type)[0]
                schema = get_pyarrow_schema(file_type)
            else:
                schema = get_pyarrow_schema(lens.type)
            table = pa.Table.from_pylist([file_obj.model_dump()], schema=schema)
            state.catalog.write_table(table, lens_path, schema, per_row=False)

    def _save_temp_file_via_fsspec(
        self, fs, base_file_path: str, file_info: dict, ext: str, index: int = None
    ):
        """Save a temporary file using fsspec filesystem."""
        # Generate the final filename
        original_name = file_info["original_filename"]
        if original_name.endswith(f".{ext}"):
            base_name = original_name[: -len(f".{ext}")]
        else:
            base_name = original_name

        # For lists, add index to avoid name collisions
        if index is not None:
            final_base_name = f"{base_name}_{index}"
        else:
            final_base_name = base_name

        # Construct the file path that PyPond expects
        if "/" in base_file_path or "\\" in base_file_path:
            # base_file_path includes directory
            file_object_path = f"{base_file_path}/{final_base_name}"
        else:
            # base_file_path is just the name
            file_object_path = f"{base_file_path}/{final_base_name}"

        # Construct the full file path with extension for fsspec
        full_file_path = f"{file_object_path}.{ext}"

        # Copy from temp file to final location using fsspec
        with open(file_info["temp_path"], "rb") as src:
            with fs.open(full_file_path, "wb") as dst:
                dst.write(src.read())

        # Clean up temp file
        os.unlink(file_info["temp_path"])

        # Return File object with path that matches PyPond conventions
        return File(path=file_object_path)


class FastAPIInputTransform(AbstractExecuteTransform):
    """Transform that creates FastAPI input endpoints for specified paths."""

    def __init__(self, root_type: Type[BaseModel], paths: list[str], app: FastAPI):
        """Initialize the FastAPI input transform.

        Args:
            root_type: Pydantic model class defining the schema structure.
            paths: List of PyPond path strings to expose as input endpoints.
            app: FastAPI application instance to register endpoints on.
        """
        self.root_type = root_type
        self.paths = paths
        self.app = app

    def get_inputs(self) -> list[LensPath]:
        """Get input paths - none for input transforms."""
        return []

    def get_outputs(self) -> list[LensPath]:
        """Get output paths - the paths we populate."""
        return [get_cleaned_path(path, "catalog") for path in self.paths]

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        """Get transform list - just this transform."""
        return [self]

    def get_name(self) -> str:
        """Get transform name."""
        return f"fastapi_input({', '.join(self.paths)})"

    def get_docs(self) -> str:
        """Get transform documentation."""
        return f"FastAPI input endpoints for paths: {', '.join(self.paths)}"

    def get_fn(self) -> Any:
        """Get underlying function - not applicable for this transform."""
        return lambda: None

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        """Get execute units for this transform."""
        return [FastAPIInputExecuteUnit(self.root_type, self.paths, self.app)]
