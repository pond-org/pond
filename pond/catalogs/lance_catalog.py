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
import os

import lance  # type: ignore
import pyarrow as pa  # type: ignore

from pond.catalogs.abstract_catalog import AbstractCatalog, LensPath


class LanceCatalog(AbstractCatalog):
    """Lance vector database catalog implementation for data storage.

    Provides a catalog interface using Lance datasets for high-performance
    storage and retrieval of structured data. Lance is optimized for analytical
    workloads with columnar storage and vector search capabilities.

    Attributes:
        db_path: Base filesystem path for storing Lance datasets.

    Note:
        Lance datasets are stored as directories containing columnar data files.
        Each table is stored as a separate .lance directory under db_path.
    """

    def __init__(self, db_path: os.PathLike):
        """Initialize a new LanceCatalog instance.

        Args:
            db_path: Base filesystem path for Lance datasets. Should be
                a directory path where Lance datasets will be stored.

        Note:
            The db_path directory will be created if it doesn't exist
            when the first dataset is written.
        """
        self.db_path = db_path

    def __getstate__(self):
        """Prepare instance state for pickling.

        Returns:
            The db_path needed to restore the catalog instance after unpickling.
        """
        return self.db_path

    def __setstate__(self, state):
        """Restore instance state after unpickling.

        Args:
            state: The db_path value from __getstate__.
        """
        self.db_path = state

    # TODO: make this more efficient
    def len(self, path: LensPath) -> int:
        """Get the number of rows in the dataset at the specified path.

        Args:
            path: LensPath specifying the location of the data to count.

        Returns:
            Number of rows in the dataset, or 0 if no dataset exists at the path.

        Note:
            This method loads the entire dataset to get the row count, which
            could be optimized using Lance dataset metadata in the future.
        """
        table, _ = self.load_table(path)
        return 0 if table is None else table.num_rows

    def write_table(
        self,
        table: pa.Table,
        path: LensPath,
        schema: pa.Schema,
        per_row: bool = False,
        append: bool = False,
    ) -> bool:
        """Write a PyArrow table to Lance dataset storage.

        Args:
            table: The PyArrow table containing data to write.
            path: LensPath specifying where to store the dataset.
            schema: Expected PyArrow schema for the dataset.
            per_row: If True, create the dataset with first row, then insert
                remaining rows individually for incremental processing.
            append: If True, append to existing dataset; otherwise overwrite.

        Returns:
            True if the write operation completed successfully.

        Note:
            Creates Lance datasets as .lance directories under db_path.
            Per-row writes use Lance's insert capability for incremental updates.
        """
        fs_path = path.to_fspath(level=len(path.path))
        mode = "append" if append else "overwrite"
        if per_row:
            ds = lance.write_dataset(
                table.take([0]),
                os.path.join(self.db_path, f"{fs_path}.lance"),
                schema=schema,
                mode=mode,
            )
            for row in range(1, table.num_rows):
                ds.insert(table.take([row]))
        else:
            lance.write_dataset(
                table,
                os.path.join(self.db_path, f"{fs_path}.lance"),
                schema=schema,
                mode=mode,
            )
        return True

    def exists_at_level(self, path: LensPath) -> bool:
        """Check if a Lance dataset exists at the exact path level.

        Args:
            path: LensPath to check for dataset existence.

        Returns:
            True if a Lance dataset directory exists at the specified path.

        Note:
            Checks for .lance directory existence in the filesystem.
            The last_index=True parameter includes array indices in the path.
        """
        # Not sure about the last index
        field_path = path.to_fspath(len(path.path), last_index=True)
        fs_path = os.path.join(self.db_path, f"{field_path}.lance")
        return os.path.exists(fs_path)

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        """Load a PyArrow table from Lance dataset storage.

        Searches for Lance datasets at different path levels, starting from the
        most specific path and working up the hierarchy. Handles both direct
        dataset access and query-based column extraction.

        Args:
            path: LensPath specifying the data location to load.

        Returns:
            Tuple containing:
            - PyArrow table with the requested data, or None if not found
            - Boolean indicating whether this was a query result (True) or
              direct dataset access (False)

        Note:
            For query results, extracts specific columns and applies row filtering
            based on array indices. Uses Lance's efficient columnar access patterns.
        """
        offset = None
        limit = None
        found = False
        for level in reversed(range(1, len(path.path) + 1)):
            field_path, query = path.path_and_query(level, last_index=True)
            fs_path = os.path.join(self.db_path, f"{field_path}.lance")
            if os.path.exists(fs_path):
                found = True
                break
            offset = path.path[level - 1].index
            if offset is None:
                continue
            field_path, query = path.path_and_query(level, last_index=False)
            fs_path = os.path.join(self.db_path, f"{field_path}.lance")
            if os.path.exists(fs_path):
                limit = 1
                found = True
                break
            offset = None
        if not found:
            return None, False
        ds = lance.dataset(fs_path)
        if query:
            table = ds.to_table(offset=offset, limit=limit, columns={"value": query})
            # return type.parse_obj(table.to_pylist()[0]["value"])
        else:
            table = ds.to_table(offset=offset, limit=limit)
        return table, bool(query)
