# import pond

import os
from typing import List, Type, get_args, get_origin
from dataclasses import dataclass

from pydantic import BaseModel
from conf.catalog import Catalog, Drive, Navigation, Values
from pydantic_to_pyarrow import get_pyarrow_schema
import pyarrow as pa
import lancedb
import lance
from parse import parse


def process(value1: float) -> int:
    return int(value1)


def get_example_catalog() -> Catalog:
    catalog = Catalog(
        drives=[
            Drive(
                navigation=[Navigation(dummy=True), Navigation(dummy=False)],
                images=[1.0, 2.0, 3.0],
                uncertainty=[0.1, 0.2, 0.3],
            ),
            Drive(
                navigation=[Navigation(dummy=False)],
                images=[4.0, 5.0],
                uncertainty=[0.4, 0.5, 0.6],
            ),
        ],
        values=Values(value1=0.5, value2=2, name="One", names=["Two", "Three"]),
    )
    return catalog


def test_db():
    # with pond.Context(Catalog) as ctx:
    #     ctx.add_transform(process, "values.value1", "values.value2")
    # Catalog.values.value1
    # print(Catalog.values.value1)
    # Catalog.dives[-1].clouds[-1]
    # ctx = Context(Catalog)
    # ctx.register_alias(BinNavigation, pd.DataFrame)
    # ctx.add_transform(process, "values.value1", "values.value2")
    db = lancedb.connect("./.lancedb")
    catalog = Catalog(
        drives=[
            Drive(
                navigation=[Navigation(dummy=True), Navigation(dummy=False)],
                images=[1.0, 2.0, 3.0],
                uncertainty=[0.1, 0.2, 0.3],
            ),
            Drive(
                navigation=[Navigation(dummy=False)],
                images=[4.0, 5.0],
                uncertainty=[0.4, 0.5, 0.6],
            ),
        ],
        values=Values(value1=0.5, value2=2, name="One", names=["Two", "Three"]),
    )
    schema = get_pyarrow_schema(Catalog)
    tbl = db.create_table("catalog", schema=schema, mode="overwrite")
    tbl.add([catalog.dict()])
    # tbl = db.create_table("catalog", data=[catalog], schema=schema)

    # tbl2 = db.open_table("catalog.values.value1")
    # print(tbl2)
    # print(db["catalog"].to_pydantic(Catalog))
    # arr = db["catalog"].to_arrow()
    print("3:")
    print(db["catalog"].search().where("values.value2==3").to_arrow())
    print("2:")
    print(db["catalog"].search().where("values.value2==2").to_arrow())


@dataclass
class TypeField:
    name: str
    index: int | None


@dataclass
class TypePath:
    dataset: str
    path: list[TypeField]

    @staticmethod
    def from_path(path: str) -> "TypePath":
        components = path.split(".")
        dataset = components.pop(0)
        parts = []
        for i, c in enumerate(components):
            if matches := parse("{:w}[{:d}]", c):
                name, index = matches
            elif matches := parse("{:w}", c):
                name = matches[0]
                index = None
            else:
                raise RuntimeError(f"Could not parse {c} as column")
            parts.append(TypeField(name, index))
        return TypePath(dataset, parts)

    def get_db_query(self) -> str:
        parts = []
        for i, field in enumerate(self.path):
            parts.append(field.name if i == 0 else f"['{field.name}']")
            if field.index is not None:
                parts.append(f"[{field.index+1}]")
        return "".join(parts)

    def to_fspath(self) -> os.PathLike:
        entries = [self.dataset]
        entries.extend(
            map(
                lambda p: p.name if p.index is None else f"{p.name}[{p.index}]",
                self.path,
            )
        )
        return "/".join(entries)


def get_tree_type(path: TypePath, root_type: Type[BaseModel]) -> Type[BaseModel]:
    if not path.path:
        return root_type
    field = path.path.pop(0)
    field_type = root_type.model_fields[field.name].annotation
    if field.index is not None:
        print(field_type)
        assert get_origin(field_type) is list
        field_type = get_args(field_type)[0]
    type = get_tree_type(path, field_type) if path else field_type
    path.path.insert(0, field)
    return type


def get_entry_with_type(type_path: TypePath, type: Type[BaseModel]) -> BaseModel:
    db_path = "test_db"
    ds = lance.dataset(f"./{db_path}/{type_path.dataset}.lance")
    if not type_path.path:
        table = ds.to_table()
        return type.parse_obj(table.to_pylist()[0])
    query = type_path.get_db_query()
    print(f"Getting {query} from {type_path.dataset}")
    table = ds.to_table(columns={"value": query})
    return type.parse_obj(table.to_pylist()[0]["value"])


def get_entry(path: str, root_type: Type[BaseModel]) -> BaseModel:
    type_path = TypePath.from_path(path)
    type = get_tree_type(type_path, root_type)
    return get_entry_with_type(type_path, type)


# NOTE: this does not require the root_type but we
# should probably add validation of the type
# This allows setting paths such as
# catalog.drives[0].navigation
# catalog.values
# catalog.values.value1
# NOTE: the type could be
BasicType = str | float | int | bool
EntryType = BaseModel | List[BaseModel] | BasicType | List[BasicType]


def set_entry(path: str, value: EntryType) -> bool:
    db_path = "test_db"
    type_path = TypePath.from_path(path)
    fs_path = type_path.to_fspath()
    print(f"Writing {fs_path}")
    schema = get_pyarrow_schema(type(value))
    table = pa.Table.from_pylist([value.dict()], schema=schema)
    ds = lance.write_dataset(
        table, f"{db_path}/{fs_path}.lance", schema=schema, mode="overwrite"
    )


def test_set_entry():
    catalog = get_example_catalog()
    set_entry("catalog.values", catalog.values)
    set_entry("catalog.drives[0].navigation[0]", catalog.drives[0].navigation[0])
    set_entry("catalog.values.value1", catalog.values.value1)


def test_get_entry_with_type():
    catalog = get_entry_with_type(TypePath.from_path("test"), Catalog)
    values = get_entry_with_type(TypePath.from_path("test.values"), Values)
    drive0 = get_entry_with_type(TypePath.from_path("test.drives[0]"), Drive)
    drive1 = get_entry_with_type(TypePath.from_path("test.drives[1]"), Drive)
    navigation1 = get_entry_with_type(
        TypePath.from_path("test.drives[0].navigation[0]"), Navigation
    )


def test_get_entry():
    catalog = get_entry("test", Catalog)
    values = get_entry("test.values", Catalog)
    drive0 = get_entry("test.drives[0]", Catalog)
    drive1 = get_entry("test.drives[1]", Catalog)
    navigation1 = get_entry("test.drives[0].navigation[0]", Catalog)


def test_get_type():
    catalog = get_tree_type(TypePath.from_path("test"), Catalog)
    print(catalog)
    values = get_tree_type(TypePath.from_path("test.values"), Catalog)
    print(values)
    drive0 = get_tree_type(TypePath.from_path("test.drives[0]"), Catalog)
    print(drive0)
    drive1 = get_tree_type(TypePath.from_path("test.drives[1]"), Catalog)
    print(drive1)
    navigation1 = get_tree_type(
        TypePath.from_path("test.drives[0].navigation[0]"), Catalog
    )
    print(navigation1)


def get_type_of_entry(path: str) -> Type[BaseModel]:
    pass


def write_dataset():
    catalog = get_example_catalog()
    schema = get_pyarrow_schema(Catalog)

    # def producer():
    #     yield pa.RecordBatch.from_pylist([catalog])
    data = pa.Table.from_pylist([catalog.dict()], schema=schema)

    ds = lance.write_dataset(
        data, "test_db/test.lance", schema=schema, mode="overwrite"
    )
    return ds


def test_append():
    catalog = Catalog(drives=[], values=None)
    drives = [
        Drive(
            navigation=[Navigation(dummy=True), Navigation(dummy=False)],
            images=[1.0, 2.0, 3.0],
            uncertainty=[0.1, 0.2, 0.3],
        ),
        Drive(
            navigation=[Navigation(dummy=False)],
            images=[4.0, 5.0],
            uncertainty=[0.4, 0.5, 0.6],
        ),
    ]
    schema = get_pyarrow_schema(Catalog)

    data = pa.Table.from_pylist([catalog.dict()], schema=schema)

    ds = lance.write_dataset(
        data, "test_db/test.lance", schema=schema, mode="overwrite"
    )
    for drive in drives:
        catalog = Catalog(drives=[drive], values=None)
        data = pa.Table.from_pylist([catalog.dict()], schema=schema)
        ds.insert(data)

    print(ds.to_table())
    catalog = get_entry("test", Catalog)
    print(catalog)


def test_dataset():
    # with pond.Context(Catalog) as ctx:
    #     ctx.add_transform(process, "values.value1", "values.value2")
    # Catalog.values.value1
    # print(Catalog.values.value1)
    # Catalog.dives[-1].clouds[-1]
    # ctx = Context(Catalog)
    # ctx.register_alias(BinNavigation, pd.DataFrame)
    # ctx.add_transform(process, "values.value1", "values.value2")
    # table = ds.to_table(
    #     # columns={"images": "drives.0.images"}
    #     # columns={"images": "drives.0.images"}
    #     columns=["drives"],
    #     # offset=0,
    #     # limit=1,
    #     # columns=["values.value2"],
    #     # filter="values.value2 = 3",
    # )
    ds = write_dataset()

    for batch in ds.to_batches(columns=["drives"], batch_size=1):
        print(1)
        print(batch.to_pylist())

    # table = ds.to_table(filter="values.value2 = 3")
    # print(table)

    # table = ds.take(indices=[0], filter="drives[1]", columns=["drives"], limit=0)
    # batches = ds.to_batches(
    #     {
    #         "drives": "drives[1]",
    #     }
    # )
    # for batch in batches:
    #     print(batch.to_table())
    # print(ds.versions())


if __name__ == "__main__":
    # test_db()
    # test_dataset()
    # test_get_entry_with_type()
    # test_get_type()
    # test_get_entry()
    # test_append()
    test_set_entry()
