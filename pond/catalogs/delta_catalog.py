import os
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
from deltalake import DeltaTable, write_deltalake

from pond.catalogs.abstract_catalog import AbstractCatalog, LensPath


class DeltaCatalog(AbstractCatalog):
    """Delta Lake catalog implementation for hierarchical data storage.

    Provides a catalog interface using Delta Lake as the storage backend.
    Supports versioned table storage with ACID transactions and schema evolution.
    Organizes tables in filesystem paths derived from LensPath structures.

    Attributes:
        db_path: Base filesystem path for Delta tables storage.
        storage_options: Optional configuration for cloud storage backends.

    Note:
        Delta tables are stored as filesystem directories containing
        Parquet files and transaction logs. Path structure follows
        LensPath.to_fspath() conventions.
    """

    def __init__(
        self,
        db_path: Union[str, Path, os.PathLike[str]],
        storage_options: Optional[dict[str, str]] = None,
    ):
        """Initialize a new DeltaCatalog instance.

        Args:
            db_path: Base path for storing Delta tables. Can be local filesystem
                or cloud storage path (s3://, gcs://, etc.).
            storage_options: Optional dictionary of storage configuration options
                for cloud storage backends (credentials, region, etc.).

        Note:
            The db_path serves as the root directory for all Delta tables.
            Storage options are passed directly to Delta Lake operations.
        """
        self.db_path = db_path
        self.storage_options = storage_options

    def __getstate__(self):
        """Prepare instance state for pickling.

        Returns:
            Tuple containing (db_path, storage_options) needed to restore the
            catalog instance after unpickling.

        Note:
            This method is called automatically during pickling to capture
            the minimal state needed for serialization.
        """
        return self.db_path, self.storage_options

    def __setstate__(self, state):
        """Restore instance state after unpickling.

        Args:
            state: Tuple containing (db_path, storage_options) from __getstate__.

        Note:
            This method is called automatically during unpickling to restore
            the catalog's configuration state.
        """
        self.db_path, self.storage_options = state

    # TODO: make this more efficient
    def len(self, path: LensPath) -> int:
        """Get the number of rows in the table at the specified path.

        Args:
            path: LensPath specifying the location of the data to count.

        Returns:
            Number of rows in the table, or 0 if no table exists at the path.

        Note:
            This method loads the entire table to get the row count, which
            is inefficient for large tables. A future optimization could
            use Delta Lake metadata to get counts more efficiently.
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
        """Write a PyArrow table to Delta Lake storage.

        Args:
            table: The PyArrow table containing data to write.
            path: LensPath specifying where to store the table.
            schema: Expected PyArrow schema for the table.
            per_row: If True, write each row as a separate transaction.
                Currently disabled (set to False internally).
            append: If True, append to existing table; otherwise overwrite.

        Returns:
            True if the write operation completed successfully.

        Raises:
            Various exceptions from Delta Lake operations depending on
            storage configuration and data compatibility.

        Note:
            The per_row parameter is currently disabled due to implementation
            issues. All writes are performed as single transactions.
        """
        fs_path = path.to_fspath(level=len(path.path))
        mode = "append" if append else "overwrite"
        if False:  # per_row:
            write_deltalake(
                os.path.join(self.db_path, f"{fs_path}"),
                table.take([0]),
                schema=schema,
                mode=mode,
            )
            for row in range(1, table.num_rows):
                write_deltalake(
                    os.path.join(self.db_path, f"{fs_path}"),
                    table.take([0]),
                    schema=schema,
                    mode="append",
                )
        else:
            write_deltalake(
                os.path.join(self.db_path, f"{fs_path}"),
                table,
                schema=schema,
                mode=mode,
            )  # type: ignore
        return True

    def exists_at_level(self, path: LensPath) -> bool:
        """Check if a Delta table exists at the exact path level.

        Args:
            path: LensPath to check for table existence.

        Returns:
            True if a valid Delta table exists at the specified path.

        Note:
            Uses Delta Lake's is_deltatable() method to verify both
            directory existence and valid Delta table structure.
            The last_index=True parameter includes array indices in the path.
        """
        # Not sure about the last index
        field_path = path.to_fspath(len(path.path), last_index=True)
        fs_path = os.path.join(self.db_path, field_path)
        return DeltaTable.is_deltatable(fs_path, self.storage_options)

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        """Load a PyArrow table from Delta Lake storage.

        Searches for Delta tables at different path levels, starting from the
        most specific path and working up the hierarchy. Handles both direct
        table access and query-based column extraction.

        Args:
            path: LensPath specifying the data location to load.

        Returns:
            Tuple containing:
            - PyArrow table with the requested data, or None if not found
            - Boolean indicating whether this was a query result (True) or
              direct table access (False)

        Note:
            The method tries multiple path variations to find existing tables,
            including paths with and without array indices. Query results
            return data in a 'value' column structure.
        """
        offset = None
        # limit = None
        found = False
        for level in reversed(range(1, len(path.path) + 1)):
            field_path, query = path.path_and_query(
                level, last_index=True, dot_accessor=True
            )
            fs_path = os.path.join(self.db_path, field_path)
            if DeltaTable.is_deltatable(fs_path, self.storage_options):
                found = True
                break
            offset = path.path[level - 1].index
            if offset is None:
                continue
            field_path, query = path.path_and_query(
                level, last_index=False, dot_accessor=True
            )
            fs_path = os.path.join(self.db_path, field_path)
            # if os.path.exists(fs_path):
            #     limit = 1
            #     break
            if DeltaTable.is_deltatable(fs_path, self.storage_options):
                found = True
                break
            offset = None
            # offset = None
        if not found:
            return None, False
        # ds = lance.dataset(fs_path)
        delta_table = DeltaTable(fs_path, self.storage_options)  # type: ignore[arg-type]
        indices = [offset] if offset is not None else [0]
        if query:
            if offset is not None:
                table = delta_table.to_pyarrow_dataset().take(
                    indices=indices, columns={"value": pc.field(query)}
                )
            else:
                table = delta_table.to_pyarrow_table(columns={"value": pc.field(query)})  # type: ignore[arg-type]
        elif offset is not None:
            # table = delta_table.to_pyarrow_table(filters=[("index", "=", str(offset))])
            table = delta_table.to_pyarrow_dataset().take(
                indices=indices,
            )
        else:
            table = delta_table.to_pyarrow_table()
        return table, bool(query)
