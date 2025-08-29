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
from typing import Optional

import pyarrow as pa  # type: ignore
from pyiceberg.catalog import load_catalog

from pond.catalogs.abstract_catalog import AbstractCatalog, LensPath


class IcebergCatalog(AbstractCatalog):
    """Apache Iceberg catalog implementation for structured data storage.

    Provides a catalog interface using Apache Iceberg tables for versioned,
    ACID-compliant data storage. Supports schema evolution and time travel
    queries through Iceberg's metadata management.

    Attributes:
        name: Name of the Iceberg catalog configuration.
        properties: Additional properties passed to the Iceberg catalog.
        catalog: The underlying PyIceberg catalog instance.

    Note:
        Requires proper Iceberg catalog configuration (catalog.properties,
        environment variables, or configuration files) to connect to the
        metadata store (Hive, Glue, etc.).
    """

    def __init__(self, name: str, **properties: Optional[str]):
        """Initialize a new IcebergCatalog instance.

        Args:
            name: Name of the catalog configuration to load. Must match
                a configured catalog name in Iceberg settings.
            **properties: Additional catalog properties to override defaults.
                Common properties include warehouse location, metadata store URI, etc.

        Note:
            The catalog configuration must be properly set up in Iceberg
            configuration files or environment variables before initialization.
        """
        self.name = name
        self.properties = properties
        self.catalog = load_catalog(name=name, **properties)

    def __getstate__(self):
        """Prepare instance state for pickling.

        Returns:
            Tuple containing (name, properties) needed to restore the
            catalog instance after unpickling.

        Note:
            The PyIceberg catalog instance is not serialized as it contains
            non-serializable resources and will be recreated on unpickling.
        """
        return self.name, self.properties

    def __setstate__(self, state):
        """Restore instance state after unpickling.

        Args:
            state: Tuple containing (name, properties) from __getstate__.

        Note:
            Recreates the PyIceberg catalog instance using the stored
            configuration, as the catalog contains non-serializable resources.
        """
        (self.name, self.properties) = state
        self.catalog = load_catalog(name=self.name, **self.properties)

    # TODO: make this more efficient
    def len(self, path: LensPath) -> int:
        """Get the number of rows in the table at the specified path.

        Args:
            path: LensPath specifying the location of the data to count.

        Returns:
            Number of rows in the table, or 0 if no table exists at the path.

        Note:
            This method loads the entire table to get the row count, which
            is inefficient for large tables. Future optimization could use
            Iceberg metadata to get counts without reading data.
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
        """Write a PyArrow table to Iceberg storage.

        Args:
            table: The PyArrow table containing data to write.
            path: LensPath specifying where to store the table. Path components
                are mapped to Iceberg namespace and table name.
            schema: Expected PyArrow schema for the table.
            per_row: If True, write each row as a separate transaction.
                Useful for incremental processing but less efficient.
            append: If True, append to existing table; otherwise overwrite.

        Returns:
            True if the write operation completed successfully.

        Note:
            Creates Iceberg namespace and table if they don't exist.
            Single-component paths get a 'root' namespace prefix.
            Per-row writes create separate transactions for each row.
        """
        # names = [p.name for p in path.path]
        names = [
            p.name if p.index is None else f"{p.name}[{p.index}]" for p in path.path
        ]
        if len(names) == 1:
            names.insert(0, "root")
        namespace = "/".join(names[:-1])
        self.catalog.create_namespace_if_not_exists(namespace)
        iceberg_table = self.catalog.create_table_if_not_exists(
            identifier=namespace + "." + names[-1],
            schema=schema,
        )
        if append:
            # iceberg_table = self.catalog.load_table("/".join(names))
            iceberg_table.append(table)
        elif per_row:
            iceberg_table.overwrite(df=table.take([table.num_rows - 1]))
            for row in reversed(range(0, table.num_rows - 1)):
                iceberg_table.append(table.take([row]))
        else:
            iceberg_table.overwrite(df=table)
        return True

    def exists_at_level(self, path: LensPath) -> bool:
        """Check if an Iceberg table exists at the exact path level.

        Args:
            path: LensPath to check for table existence.

        Returns:
            True if a valid Iceberg table exists at the specified path.

        Note:
            Constructs Iceberg table identifier from path components,
            with array indices included in the name. Single-component
            paths get a 'root' namespace prefix.
        """
        names = [
            p.name if p.index is None else f"{p.name}[{p.index}]" for p in path.path
        ]
        if len(names) == 1:
            names.insert(0, "root")
        identifier = "/".join(names[:-1]) + "." + names[-1]
        return self.catalog.table_exists(identifier)

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        """Load a PyArrow table from Iceberg storage.

        Searches for Iceberg tables at different path levels, starting from the
        most specific path and working up the hierarchy. Handles both direct
        table access and nested column extraction with query processing.

        Args:
            path: LensPath specifying the data location to load.

        Returns:
            Tuple containing:
            - PyArrow table with the requested data, or None if not found
            - Boolean indicating whether this was a query result (True) or
              direct table access (False)

        Note:
            For nested queries, the method extracts specific columns/fields
            and wraps results in a 'value' column structure. Array indices
            are handled by taking specific rows from the result table.
        """
        # names = ["catalog"] + [p.name for p in path.path]
        names = [
            p.name if p.index is None else f"{p.name}[{p.index}]" for p in path.path
        ]
        index = None
        found = False
        for level in reversed(range(0, len(path.path))):
            # namespace = "/".join(names[: level - 1])
            levels = names[: level + 1]
            if len(levels) == 1:
                levels.insert(0, "root")
            identifier = "/".join(levels[:-1]) + "." + levels[-1]
            query = path.path[level + 1 :]  # if level > 1 else []
            if self.catalog.table_exists(identifier):
                found = True
                break
            index = path.path[level].index
            if index is None:
                continue

            # we want to see if x.example as well as x.example[0]
            levels = names[:level] + [path.path[level].name]
            if len(levels) == 1:
                levels.insert(0, "root")
            identifier = "/".join(levels[:-1]) + "." + levels[-1]
            if self.catalog.table_exists(identifier):
                found = True
                break
            index = None
        if not found:
            return None, False
        iceberg_table = self.catalog.load_table(identifier)
        if query:
            # iceberg queries can not be done
            # on index, so we need to get all entries
            field = ".".join(q.name for q in query)
            table = iceberg_table.scan(selected_fields=(field,)).to_arrow()
            if index is not None:
                table = table.take((index,))
            # if index is not None:
            #     table = table[index][0]

            for level, q in enumerate(query):
                if level < len(query) - 1 or q.index is not None:
                    table = table[q.name]
                if not isinstance(table, pa.Scalar):
                    table = table[0]
                if q.index is not None:
                    table = table[q.index]

            if query[-1].index is None:
                table = pa.table({"value": table})
            else:
                table = pa.table({"value": [table]})
        else:
            table = iceberg_table.scan().to_arrow()
            if index is not None:
                table = table.take((index,))
        return table, bool(query)
