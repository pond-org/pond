import os
from pathlib import Path
from typing import Optional, Union

import pyarrow as pa  # type: ignore
import pyarrow.compute as pc  # type: ignore
from deltalake import DeltaTable, write_deltalake

from pond.catalogs.abstract_catalog import AbstractCatalog, LensPath


class DeltaCatalog(AbstractCatalog):
    def __init__(
        self,
        db_path: Union[str, Path, os.PathLike[str]],
        storage_options: Optional[dict[str, str]] = None,
    ):
        self.db_path = db_path
        self.storage_options = storage_options

    def __getstate__(self):
        return self.db_path, self.storage_options

    def __setstate__(self, state):
        self.table_uir, self.storage_options = state

    # TODO: make this more efficient
    def len(self, path: LensPath) -> int:
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
        # Not sure about the last index
        field_path = path.to_fspath(len(path.path), last_index=True)
        fs_path = os.path.join(self.db_path, field_path)
        return DeltaTable.is_deltatable(fs_path, self.storage_options)

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
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
