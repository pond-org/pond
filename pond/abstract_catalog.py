import os
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

    def get_db_query(self, level: int = 1) -> str:
        assert level >= 1 and level <= len(self.path)
        parts = []
        for i, field in enumerate(self.path[level:]):
            parts.append(field.name if i == 0 else f"['{field.name}']")
            if field.index is not None:
                parts.append(f"[{field.index+1}]")
        return "".join(parts)

    def to_fspath(self, level: int = 1) -> os.PathLike:
        assert level >= 1 and level <= len(self.path)
        entries = map(
            lambda p: p.name if p.index is None else f"{p.name}[{p.index}]",
            self.path[:level],
        )
        return "/".join(entries)

    def path_and_query(self, level: int = 1) -> tuple[os.PathLike, str]:
        return self.to_fspath(level), self.get_db_query(level)


class AbstractCatalog(ABC):
    def write_table(self, table: pa.Table, path: LensPath) -> bool:
        pass

    def load_table(self, path: LensPath) -> pa.Table | None:
        pass


class IcebergCatalog(AbstractCatalog):
    def __init__(self, catalog: Catalog):
        self.catalog = catalog

    def write_table(
        self, table: pa.Table, path: LensPath, schema: pa.Schema, per_row: bool = False
    ) -> bool:
        names = [p.name for p in path.path]
        namespace = ".".join(names[:-1])
        self.catalog.create_namespace_if_not_exists(namespace)
        iceberg_table = self.catalog.create_table_if_not_exists(
            identifier=".".join(names),
            schema=schema,
        )
        if per_row:
            iceberg_table.overwrite(df=table)
        else:
            iceberg_table.overwrite(df=table.take([0]))
            for row in range(1, table.num_rows):
                iceberg_table.append(table.take([row]))
        return True

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        names = ["catalog"] + [p.name for p in path.path]
        print("Getting ident for ", path.path)
        for level in reversed(range(1, len(path.path) + 1)):
            print("At level ", level)
            # namespace = ".".join(names[: level - 1])
            identifier = ".".join(names[: level + 1])
            query = path.path[level:]  # if level > 1 else []
            print(identifier, query)
            if self.catalog.table_exists(identifier):
                print(f"{identifier} does exist!")
                break
            else:
                print(f"{identifier} does not exist!")
        iceberg_table = self.catalog.load_table(identifier)
        print("QUERY: ", query)
        if query:
            print("DOING QUERY!")
            field = ".".join(q.name for q in query)
            table = iceberg_table.scan(selected_fields=(field,)).to_arrow()
            for level, q in enumerate(query[:-1]):
                if level > 0:
                    table = table[0]
                if q.index is not None:
                    print("DEBUG:")
                    print(table)
                    print(table.to_pylist())
                    # table = table[q.name][0][q.index]
                    table = table[q.name][0][q.index]
                    print("AFTER")
                    print(table)
                else:
                    table = table[q.name]
                print(level, ",", q.name, ": ", table)
            if query[-1].name == "dummy":
                table = pa.table({"value": table})
            elif query[-1].index is None:
                print(table)
                table = pa.table({"value": table[0]})
            else:
                print("FINAL: ", table[query[-1].name])
                # print("BEFORE FINAL: ", table.to_pylist())
                # print()
                if query[-1].name == "navigation":
                    table = pa.table(
                        {"value": [table[query[-1].name][query[-1].index]]}
                    )
                else:
                    table = pa.table(
                        {"value": [table[query[-1].name][0][query[-1].index]]}
                    )
        else:
            print("NOT DOING QUERY!")
            table = iceberg_table.scan().to_arrow()
        print("Result: ", table.to_pylist())
        return table, bool(query)


class LanceCatalog(AbstractCatalog):
    def __init__(self, db_path: os.PathLike):
        self.db_path = db_path

    def write_table(
        self, table: pa.Table, path: LensPath, schema: pa.Schema, per_row: bool = False
    ) -> bool:
        fs_path = path.to_fspath(level=len(path.path))
        if per_row:
            ds = lance.write_dataset(
                table.take([0]),
                os.path.join(self.db_path, f"{fs_path}.lance"),
                schema=schema,
                mode="overwrite",
            )
            for row in range(1, table.num_rows):
                ds.insert(table.take([row]))
        else:
            lance.write_dataset(
                table,
                os.path.join(self.db_path, f"{fs_path}.lance"),
                schema=schema,
                mode="overwrite",
            )
        return True

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        for level in reversed(range(1, len(path.path) + 1)):
            field_path, query = path.path_and_query(level)
            fs_path = os.path.join(self.db_path, f"{field_path}.lance")
            if os.path.exists(fs_path):
                break
        ds = lance.dataset(fs_path)
        print(f"Getting {query} from {fs_path}")
        if query:
            print("Table: ", ds.to_table())
            table = ds.to_table(columns={"value": query})
            # return type.parse_obj(table.to_pylist()[0]["value"])
        else:
            table = ds.to_table()
        return table, query
