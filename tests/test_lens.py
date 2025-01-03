# import pond

import os

import pytest

from pydantic import BaseModel
from conf.catalog import Catalog, Drive, Navigation, Values

# from pydantic_to_pyarrow import get_pyarrow_schema
import pyarrow as pa
import lance
import lancedb

from pond import Lens
from pond.lens import TypeField
import pond.lens


@pytest.fixture
def catalog() -> Catalog:
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


def write_dataset(catalog, db_path):
    schema = pond.lens.get_pyarrow_schema(Catalog)

    # def producer():
    #     yield pa.RecordBatch.from_pylist([catalog])
    data = pa.Table.from_pylist([catalog.dict()], schema=schema)

    ds = lance.write_dataset(
        data, os.path.join(db_path, "test.lance"), schema=schema, mode="overwrite"
    )
    return ds


@pytest.mark.skip(reason="no way of currently testing this")
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
    schema = pond.lens.get_pyarrow_schema(Catalog)
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


def test_set_entry(catalog: Catalog, tmp_path_factory):
    path = tmp_path_factory.mktemp("db")
    lens = Lens(Catalog, "values", db_path=path)
    lens.set(catalog.values)
    value = lens.get()
    assert value == catalog.values
    lens = Lens(Catalog, "drives[0].navigation[0]", db_path=path)
    lens.set(catalog.drives[0].navigation[0])
    value = lens.get()
    assert value == catalog.drives[0].navigation[0]
    lens = Lens(Catalog, "drives[0].navigation", db_path=path)
    lens.set(catalog.drives[0].navigation)
    value = lens.get()
    assert value == catalog.drives[0].navigation
    lens = Lens(Catalog, "values.value1", db_path=path)
    lens.set(catalog.values.value1)
    value = lens.get()
    assert value == catalog.values.value1
    lens = Lens(Catalog, "values.names", db_path=path)
    lens.set(catalog.values.names)
    value = lens.get()
    assert value == catalog.values.names


def test_get_entry_with_type(catalog: Catalog, tmp_path_factory):
    path = tmp_path_factory.mktemp("db")
    write_dataset(catalog, path)
    read_catalog = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("", "test"), Catalog, path
    )
    assert read_catalog == catalog
    values = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("values", "test"), Values, path
    )
    assert values == catalog.values
    drive0 = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("drives[0]", "test"), Drive, path
    )
    assert drive0 == catalog.drives[0]
    drive1 = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("drives[1]", "test"), Drive, path
    )
    assert drive1 == catalog.drives[1]
    navigation0 = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("drives[0].navigation[0]", "test"),
        Navigation,
        path,
    )
    assert navigation0 == catalog.drives[0].navigation[0]


def test_get_entry(catalog: Catalog, tmp_path_factory):
    path = tmp_path_factory.mktemp("db")
    write_dataset(catalog, path)
    lens = Lens(Catalog, "", "test", db_path=path)
    read_catalog = lens.get()
    assert read_catalog == catalog
    lens = Lens(Catalog, "values", "test", db_path=path)
    values = lens.get()
    assert values == catalog.values
    lens = Lens(Catalog, "drives[0]", "test", db_path=path)
    drive0 = lens.get()
    assert drive0 == catalog.drives[0]
    lens = Lens(Catalog, "drives[1]", "test", db_path=path)
    drive1 = lens.get()
    assert drive1 == catalog.drives[1]
    lens = Lens(Catalog, "drives[0].navigation[0]", "test", db_path=path)
    navigation0 = lens.get()
    assert navigation0 == catalog.drives[0].navigation[0]


def test_get_type():
    path = pond.lens.LensPath.from_path("")
    catalog = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert catalog == Catalog
    path = pond.lens.LensPath.from_path("values")
    values = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert values == Values
    path = pond.lens.LensPath.from_path("drives[0]")
    drive0 = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert drive0 == Drive
    path = pond.lens.LensPath.from_path("drives[1]")
    drive1 = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert drive1 == Drive
    path = pond.lens.LensPath.from_path("drives[0].navigation[0]")
    navigation0 = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert navigation0 == Navigation


def test_path_and_query():
    path = pond.lens.LensPath.from_path("")
    assert path.path == [TypeField("catalog", None)]
    print(path.path_and_query(1))
    path = pond.lens.LensPath.from_path("values")
    assert path.path == [TypeField("catalog", None), TypeField("values", None)]
    path = pond.lens.LensPath.from_path("drives[0]")
    assert path.path == [TypeField("catalog", None), TypeField("drives", 0)]
    path = pond.lens.LensPath.from_path("drives[0].navigation[0]")
    assert path.path == [
        TypeField("catalog", None),
        TypeField("drives", 0),
        TypeField("navigation", 0),
    ]
    print(path.path_and_query(1))
    print(path.path_and_query(2))


@pytest.mark.skip(reason="no way of currently testing this")
def test_append():
    values = Values(value1=0.5, value2=2, name="One", names=[])
    catalog = Catalog(drives=[], values=values)
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
    schema = pond.lens.get_pyarrow_schema(Catalog)

    data = pa.Table.from_pylist([catalog.dict()], schema=schema)

    ds = lance.write_dataset(
        data, "test_db/test.lance", schema=schema, mode="overwrite"
    )
    for drive in drives:
        catalog = Catalog(drives=[drive], values=values)
        data = pa.Table.from_pylist([catalog.dict()], schema=schema)
        ds.insert(data)

    print(ds.to_table())
    lens = Lens(Catalog, "", "test")
    catalog = lens.get()
    print(catalog)


@pytest.mark.skip(reason="no way of currently testing this")
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
    test_dataset()
    # test_db()
    test_get_entry_with_type()
    test_get_type()
    test_get_entry()
    test_append()
    test_set_entry()
    test_path_and_query()
