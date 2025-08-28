"""FastAPI output transform for serving HTTP data."""

from typing import Any, Type

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pond.api.analyzer import separate_file_and_regular_fields
from pond.api.handlers import get_state_value, handle_json_output
from pond.api.shared import BaseExecuteUnit
from pond.api.utils import path_to_url
from pond.lens import LensPath, get_cleaned_path
from pond.state import State
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractExecuteUnit,
)


class FastAPIOutputExecuteUnit(BaseExecuteUnit):
    """Execute unit that registers FastAPI output endpoints to serve data."""

    def __init__(self, root_type: Type[BaseModel], paths: list[str], app: FastAPI):
        """Initialize the FastAPI output execute unit.

        Args:
            root_type: Pydantic model class defining the schema structure.
            paths: List of PyPond path strings to expose as output endpoints.
            app: FastAPI application instance to register endpoints on.
        """
        super().__init__(inputs=paths, outputs=[])
        self.root_type = root_type
        self.paths = paths
        self.app = app

    def load_inputs(self, state: State) -> list[Any]:
        """Load input data from state."""
        self.state_ref = state
        return [get_state_value(state, path) for path in self.paths]

    def save_outputs(self, state: State, outputs: list[Any]) -> list[Any]:
        """Save outputs - not used for output transforms."""
        return []

    def commit(self, state: State, values: list[Any]) -> bool:
        """Commit - not used for output transforms."""
        return True

    def run(self, args: list[Any]) -> list[Any]:
        """Register FastAPI endpoints to serve output data.

        Creates FastAPI endpoints for each output path to serve
        the data via HTTP responses.

        Args:
            args: Input data loaded from state.

        Returns:
            Empty list - endpoints are registered on the app.
        """
        # Separate file and regular fields
        file_paths, regular_paths = separate_file_and_regular_fields(
            self.root_type, self.paths
        )

        # Create endpoints for regular fields (JSON output)
        for path in regular_paths:
            url = path_to_url(path)
            self._create_json_endpoint(path, url)

        # Create endpoints for file fields (file download)
        for path in file_paths:
            url = path_to_url(path)
            self._create_file_endpoint(path, url)

        # Add/update status endpoint (may conflict with input status, but that's okay)
        @self.app.get("/output-status")
        def get_output_status():
            available = []
            for path in self.paths:
                try:
                    get_state_value(self.state_ref, path)
                    available.append(path)
                except Exception:
                    pass
            return {"available": available, "paths": self.paths}

        return []

    def _serve_raw_file(self, path: str):
        """Serve raw file content directly from filesystem without using reader."""
        import fsspec
        import io
        from fastapi import HTTPException
        from fastapi.responses import StreamingResponse
        from pond.api.analyzer import get_field_metadata, get_file_extension

        # Get field metadata for file path and protocol info
        metadata = get_field_metadata(self.root_type, path)
        ext = get_file_extension(self.root_type, path)

        # Use protocol from field metadata or default
        protocol = metadata.get("protocol") or self.state_ref.default_volume_protocol
        fs = fsspec.filesystem(**self.state_ref.volume_protocol_args[protocol])

        # Determine the file path
        if metadata.get("path"):
            # Use explicit path from field metadata
            base_file_path = metadata["path"]
        else:
            # Use computed path from lens system
            lens = self.state_ref.lens(path)
            base_file_path = lens.lens_path.to_volume_path()

        # Construct full file path with extension
        full_file_path = f"{base_file_path}.{ext}"

        # Determine MIME type from extension
        mime_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "pdf": "application/pdf",
            "json": "application/json",
            "csv": "text/csv",
            "txt": "text/plain",
            "npy": "application/octet-stream",
        }
        mime_type = mime_types.get(ext, "application/octet-stream")

        # Read raw file content
        try:
            with fs.open(full_file_path, "rb") as f:
                content = f.read()
        except Exception:
            raise HTTPException(
                status_code=404, detail=f"File not found: {full_file_path}"
            )

        # Create filename from path
        filename = path.replace(".", "_").replace("[", "_").replace("]", "") + f".{ext}"

        # Create streaming response
        return StreamingResponse(
            io.BytesIO(content),
            media_type=mime_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def _create_json_endpoint(self, path: str, url: str) -> None:
        """Create JSON output endpoint for a regular field."""

        @self.app.get(f"/output/{url}")
        def serve_json_data():
            if not self.state_ref:
                raise HTTPException(status_code=500, detail="State not available")
            data = get_state_value(self.state_ref, path)
            return handle_json_output(self.root_type, path, data)

    def _create_file_endpoint(self, path: str, url: str) -> None:
        """Create file download endpoint for a File[DataT] field."""

        @self.app.get(f"/output/{url}")
        def serve_file_data():
            if not self.state_ref:
                raise HTTPException(status_code=500, detail="State not available")
            return self._serve_raw_file(path)


class FastAPIOutputTransform(AbstractExecuteTransform):
    """Transform that creates FastAPI output endpoints for specified paths."""

    def __init__(self, root_type: Type[BaseModel], paths: list[str], app: FastAPI):
        """Initialize the FastAPI output transform.

        Args:
            root_type: Pydantic model class defining the schema structure.
            paths: List of PyPond path strings to expose as output endpoints.
            app: FastAPI application instance to register endpoints on.
        """
        self.root_type = root_type
        self.paths = paths
        self.app = app

    def get_inputs(self) -> list[LensPath]:
        """Get input paths - the paths we serve."""
        return [get_cleaned_path(path, "catalog") for path in self.paths]

    def get_outputs(self) -> list[LensPath]:
        """Get output paths - none for output transforms."""
        return []

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        """Get transform list - just this transform."""
        return [self]

    def get_name(self) -> str:
        """Get transform name."""
        return f"fastapi_output({', '.join(self.paths)})"

    def get_docs(self) -> str:
        """Get transform documentation."""
        return f"FastAPI output endpoints for paths: {', '.join(self.paths)}"

    def get_fn(self) -> Any:
        """Get underlying function - not applicable for this transform."""
        return lambda: None

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        """Get execute units for this transform."""
        return [FastAPIOutputExecuteUnit(self.root_type, self.paths, self.app)]
