# Copyright 2025 Nils Bore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Type

from pydantic import BaseModel

from pond.catalogs.abstract_catalog import AbstractCatalog
from pond.lens import Lens


class State:
    """Central state manager for pond data pipelines.

    Provides the main interface for accessing and managing data in pond pipelines.
    Acts as a factory for Lens instances and coordinates between the data catalog,
    file system protocols, and the hierarchical data model.

    Attributes:
        root_type: The root pydantic model class that defines the data structure.
        catalog: The data catalog implementation for structured data storage.
        root_path: Root path name used in data access expressions.
        volume_protocol_args: Configuration for different file system protocols.
        default_volume_protocol: Default protocol to use when none specified.
    """

    def __init__(
        self,
        root_type: Type[BaseModel],
        catalog: AbstractCatalog,
        root_path: str = "catalog",
        # storage_path: str = ".",
        volume_protocol_args: dict[str, Any] = {},
        default_volume_protocol: str = "dir",
    ):
        """Initialize a new State instance.

        Args:
            root_type: Pydantic model class that defines the root data structure.
                Must be a subclass of BaseModel with appropriate field annotations.
            catalog: Catalog implementation for managing structured data storage.
                Must implement the AbstractCatalog interface.
            root_path: Name to use as the root in data path expressions, defaults
                to "catalog". Affects how paths are resolved in lens expressions.
            volume_protocol_args: Dictionary mapping protocol names to fsspec
                configuration dictionaries. Empty dict uses defaults.
            default_volume_protocol: Protocol to use when none specified in field
                metadata, defaults to "dir" for local directory access.
        """
        self.root_type = root_type
        self.catalog = catalog
        self.root_path = root_path
        self.volume_protocol_args = volume_protocol_args
        self.default_volume_protocol = default_volume_protocol
        # self.storage_path = storage_path

    def lens(self, path: str) -> Lens:
        """Create a Lens for accessing data at the specified path.

        Args:
            path: Dot-separated path expression describing the data location.
                Can include array indices (e.g., "clouds[0].points") and variants
                (e.g., "table:clouds[:].bounds").

        Returns:
            Lens instance configured for the specified path with access to the
            catalog and file system protocols.

        Note:
            The returned Lens inherits this State's catalog and protocol configuration.
            Path expressions are resolved relative to the root_type schema.
        """
        lens = Lens(
            self.root_type,
            path,
            self.catalog,
            self.root_path,
            self.volume_protocol_args,
            self.default_volume_protocol,
        )
        return lens

    def index_files(self, paths: list[str] = [""]):
        """Index files into the catalog for the specified paths.

        Scans the file system for files matching the schema at each path and
        creates catalog entries. Useful for discovering and registering existing
        files before pipeline execution.

        Args:
            paths: List of path expressions to index. Empty string indexes the
                root. Defaults to indexing just the root path.

        Note:
            Default mutable argument [] is dangerous but used here for backward
            compatibility. Each path must be valid in the root_type schema.
            Indexing behavior depends on field metadata (extensions, protocols, etc.).
        """
        for path in paths:
            lens = self.lens(path)
            lens.index_files()

    def __getitem__(self, path: str) -> Any:
        """Get data at the specified path using dictionary-style access.

        Args:
            path: Dot-separated path expression describing the data location.

        Returns:
            The data stored at the path, deserialized according to the schema
            and any custom readers defined in field metadata.

        Raises:
            Various exceptions depending on catalog implementation and data availability.

        Note:
            This is a convenience method equivalent to self.lens(path).get().
            The return type depends on the path's position in the data schema.
        """
        return self.lens(path).get()

    def __setitem__(self, path: str, value: Any):
        """Set data at the specified path using dictionary-style access.

        Args:
            path: Dot-separated path expression describing the data location.
            value: The data to store, which must be compatible with the schema
                type at the specified path.

        Raises:
            Various exceptions depending on catalog implementation and type compatibility.

        Note:
            This is a convenience method equivalent to self.lens(path).set(value).
            The value is serialized according to field metadata and stored in the catalog.
        """
        self.lens(path).set(value)
