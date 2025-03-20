import os
from typing import Self
from dataclasses import dataclass
from abc import ABC

from parse import parse
import pyarrow as pa

import lance
from pyiceberg.catalog import Catalog


@dataclass
class TypeField:
    name: str
    index: int | None

    def __eq__(self, other: Self) -> bool:
        return self.name == other.name and self.index == other.index

    def subset_of(self, other: Self):
        return self.name == other.name and (
            other.index is None or self.index == other.index
        )


@dataclass
class LensPath:
    path: list[TypeField]

    @staticmethod
    def from_path(path: str, root_path: str = "catalog") -> "LensPath":
        parts = [TypeField(root_path, None)]
        if path == "":
            return LensPath(parts)
        components = path.split(".")
        for i, c in enumerate(components):
            if matches := parse("{:w}[{:d}]", c):
                name, index = matches
            elif matches := parse("{:w}", c):
                name = matches[0]
                index = None
            else:
                raise RuntimeError(f"Could not parse {c} as column")
            parts.append(TypeField(name, index))
        return LensPath(parts)

    def to_path(self) -> str:
        return ".".join(
            map(
                lambda t: t.name if t.index is None else f"{t.name}[{t.index}]",
                self.path[1:],
            )
        )

    def __eq__(self, other: Self) -> bool:
        return self.path == other.path

    def subset_of(self, other: Self) -> bool:
        if len(self.path) < len(other.path):
            return False
        return all(
            a.subset_of(b) for a, b in zip(self.path[: len(other.path)], other.path)
        )

    def get_db_query(self, level: int = 1) -> str:
        assert level >= 1 and level <= len(self.path)
        parts = []
        for i, field in enumerate(self.path[level:]):
            parts.append(field.name if i == 0 else f"['{field.name}']")
            if field.index is not None:
                parts.append(f"[{field.index+1}]")
        return "".join(parts)

    def to_fspath(self, level: int = 1, last_index: bool = True) -> os.PathLike:
        assert level >= 1 and level <= len(self.path)
        entries = list(
            map(
                lambda p: p.name if p.index is None else f"{p.name}[{p.index}]",
                self.path[:level],
            )
        )
        if not last_index and self.path[level - 1].index is not None:
            entries[-1] = self.path[level - 1].name
        return "/".join(entries)

    def to_volume_path(self) -> os.PathLike:
        entries = map(
            lambda p: p.name if p.index is None else f"{p.name}/{p.index}",
            self.path,
        )
        return "/".join(entries)

    def path_and_query(
        self, level: int = 1, last_index: bool = True
    ) -> tuple[os.PathLike, str]:
        return self.to_fspath(level, last_index), self.get_db_query(level)


class AbstractCatalog(ABC):
    def len(self, path: LensPath) -> int:
        pass

    def write_table(
        self,
        table: pa.Table,
        path: LensPath,
        schema: pa.Schema,
        per_row: bool,
        append: bool,
    ) -> bool:
        pass

    def load_table(self, path: LensPath) -> pa.Table | None:
        pass


class IcebergCatalog(AbstractCatalog):
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    # TODO: make this more efficient
    def len(self, path: LensPath) -> int:
        table, _ = self.load_table(path)
        return table.num_rows

    def write_table(
        self,
        table: pa.Table,
        path: LensPath,
        schema: pa.Schema,
        per_row: bool = False,
        append: bool = False,
    ) -> bool:
        # names = [p.name for p in path.path]
        names = ["catalog"] + [
            p.name if p.index is None else f"{p.name}[{p.index}]" for p in path.path
        ]
        namespace = ".".join(names[:-1])
        self.catalog.create_namespace_if_not_exists(namespace)
        iceberg_table = self.catalog.create_table_if_not_exists(
            identifier=".".join(names),
            schema=schema,
        )
        if append:
            iceberg_table.append(table)
        elif per_row:
            iceberg_table.overwrite(df=table.take([table.num_rows - 1]))
            for row in reversed(range(0, table.num_rows - 1)):
                iceberg_table.append(table.take([row]))
        else:
            iceberg_table.overwrite(df=table)
        return True

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        # names = ["catalog"] + [p.name for p in path.path]
        names = ["catalog"] + [
            p.name if p.index is None else f"{p.name}[{p.index}]" for p in path.path
        ]
        index = None
        print("Getting ident for ", path.path, ":", names)
        for level in reversed(range(1, len(path.path) + 1)):
            print("At level ", level)
            # namespace = ".".join(names[: level - 1])
            identifier = ".".join(names[: level + 1])
            query = path.path[level:]  # if level > 1 else []
            print(identifier, query)
            if self.catalog.table_exists(identifier):
                print(f"{identifier} does exist!")
                break
            index = path.path[level - 1].index
            if index is None:
                print(f"{identifier} does not exist!")
                continue

            # we want to see if x.example as well as x.example[0]
            identifier = ".".join(names[:level] + [path.path[level - 1].name])
            if self.catalog.table_exists(identifier):
                print(f"{identifier} does exist!")
                break
            index = None
            print(f"{identifier} does not exist!")
        iceberg_table = self.catalog.load_table(identifier)
        print(iceberg_table.scan().to_arrow())
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


class LanceCatalog(AbstractCatalog):
    def __init__(self, db_path: os.PathLike):
        self.db_path = db_path

    # TODO: make this more efficient
    def len(self, path: LensPath) -> int:
        table, _ = self.load_table(path)
        return table.num_rows

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

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        offset = None
        limit = None
        for level in reversed(range(1, len(path.path) + 1)):
            field_path, query = path.path_and_query(level, last_index=True)
            fs_path = os.path.join(self.db_path, f"{field_path}.lance")
            if os.path.exists(fs_path):
                break
            offset = path.path[level - 1].index
            if offset is None:
                continue
            field_path, query = path.path_and_query(level, last_index=False)
            fs_path = os.path.join(self.db_path, f"{field_path}.lance")
            if os.path.exists(fs_path):
                limit = 1
                break
            offset = None
        ds = lance.dataset(fs_path)
        print(f"Getting {query} from {fs_path}")
        if query:
            print("Table: ", ds.to_table())
            table = ds.to_table(offset=offset, limit=limit, columns={"value": query})
            # return type.parse_obj(table.to_pylist()[0]["value"])
        else:
            table = ds.to_table(offset=offset, limit=limit)
        return table, query
