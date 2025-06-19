import os

import lance  # type: ignore
import pyarrow as pa  # type: ignore

from pond.catalogs.abstract_catalog import AbstractCatalog, LensPath


class LanceCatalog(AbstractCatalog):
    def __init__(self, db_path: os.PathLike):
        self.db_path = db_path

    def __getstate__(self):
        return self.db_path

    def __setstate__(self, state):
        self.db_path = state

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
        # Not sure about the last index
        field_path = path.to_fspath(len(path.path), last_index=True)
        fs_path = os.path.join(self.db_path, f"{field_path}.lance")
        return os.path.exists(fs_path)

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
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
