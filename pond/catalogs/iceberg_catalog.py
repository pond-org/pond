from typing import Optional

from pyiceberg.catalog import load_catalog
import pyarrow as pa

from pond.catalogs.abstract_catalog import AbstractCatalog, LensPath


class IcebergCatalog(AbstractCatalog):
    def __init__(self, name: str, **properties: Optional[str]):
        self.name = name
        self.properties = properties
        self.catalog = load_catalog(name=name, **properties)

    def __getstate__(self):
        return self.name, self.properties

    def __setstate__(self, state):
        (self.name, self.properties) = state
        self.catalog = load_catalog(name=self.name, **self.properties)

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
            # iceberg_table = self.catalog.load_table(".".join(names))
            iceberg_table.append(table)
        elif per_row:
            iceberg_table.overwrite(df=table.take([table.num_rows - 1]))
            for row in reversed(range(0, table.num_rows - 1)):
                iceberg_table.append(table.take([row]))
        else:
            iceberg_table.overwrite(df=table)
        return True

    def exists_at_level(self, path: LensPath) -> bool:
        names = ["catalog"] + [
            p.name if p.index is None else f"{p.name}[{p.index}]" for p in path.path
        ]
        identifier = ".".join(names)
        return self.catalog.table_exists(identifier)

    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        # names = ["catalog"] + [p.name for p in path.path]
        names = ["catalog"] + [
            p.name if p.index is None else f"{p.name}[{p.index}]" for p in path.path
        ]
        index = None
        print("Getting ident for ", path.path, ":", names)
        found = False
        for level in reversed(range(1, len(path.path) + 1)):
            print("At level ", level)
            # namespace = ".".join(names[: level - 1])
            identifier = ".".join(names[: level + 1])
            query = path.path[level:]  # if level > 1 else []
            print(identifier, query)
            if self.catalog.table_exists(identifier):
                print(f"{identifier} does exist!")
                found = True
                break
            index = path.path[level - 1].index
            if index is None:
                print(f"{identifier} does not exist!")
                continue

            # we want to see if x.example as well as x.example[0]
            identifier = ".".join(names[:level] + [path.path[level - 1].name])
            if self.catalog.table_exists(identifier):
                print(f"{identifier} does exist!")
                found = True
                break
            index = None
            print(f"{identifier} does not exist!")
        if not found:
            return None, False
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
