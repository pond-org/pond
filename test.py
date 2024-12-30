# import pond

from typing import Type
from pydantic import BaseModel
from conf.catalog import Catalog, Drive, Navigation, Values

# from pydantic_to_pyarrow import get_pyarrow_schema
import pyarrow as pa
import lance
import lancedb

from pond import Lens
import pond.lens


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


def test_set_entry():
    catalog = get_example_catalog()
    lens = Lens("catalog.values", Catalog)
    lens.set(catalog.values)
    lens = Lens("catalog.drives[0].navigation[0]", Catalog)
    lens.set(catalog.drives[0].navigation[0])
    lens = Lens("catalog.drives[0].navigation", Catalog)
    lens.set(catalog.drives[0].navigation[0])
    lens = Lens("catalog.values.value1", Catalog)
    lens.set(catalog.values.value1)
    lens = Lens("catalog.values.names", Catalog)
    lens.set(catalog.values.value1)


def test_get_entry_with_type():
    catalog = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("test"), Catalog
    )
    values = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("test.values"), Values
    )
    drive0 = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("test.drives[0]"), Drive
    )
    drive1 = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("test.drives[1]"), Drive
    )
    navigation1 = pond.lens.get_entry_with_type(
        pond.lens.LensPath.from_path("test.drives[0].navigation[0]"), Navigation
    )


def test_get_entry():
    lens = Lens("test", Catalog)
    catalog = lens.get()
    lens = Lens("test.values", Catalog)
    values = lens.get()
    lens = Lens("test.drives[0]", Catalog)
    drive0 = lens.get()
    lens = Lens("test.drives[1]", Catalog)
    drive1 = lens.get()
    lens = Lens("test.drives[0].navigation[0]", Catalog)
    navigation1 = lens.get()


def test_get_type():
    catalog = pond.lens.get_tree_type(pond.lens.LensPath.from_path("test"), Catalog)
    print(catalog)
    values = pond.lens.get_tree_type(
        pond.lens.LensPath.from_path("test.values"), Catalog
    )
    print(values)
    drive0 = pond.lens.get_tree_type(
        pond.lens.LensPath.from_path("test.drives[0]"), Catalog
    )
    print(drive0)
    drive1 = pond.lens.get_tree_type(
        pond.lens.LensPath.from_path("test.drives[1]"), Catalog
    )
    print(drive1)
    navigation1 = pond.lens.get_tree_type(
        pond.lens.LensPath.from_path("test.drives[0].navigation[0]"), Catalog
    )
    print(navigation1)


def get_type_of_entry(path: str) -> Type[BaseModel]:
    pass


def write_dataset():
    catalog = get_example_catalog()
    schema = pond.lens.get_pyarrow_schema(Catalog)

    # def producer():
    #     yield pa.RecordBatch.from_pylist([catalog])
    data = pa.Table.from_pylist([catalog.dict()], schema=schema)

    ds = lance.write_dataset(
        data, "test_db/test.lance", schema=schema, mode="overwrite"
    )
    return ds


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
    lens = Lens("test", Catalog)
    catalog = lens.get()
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
    test_dataset()
    # test_db()
    test_get_entry_with_type()
    test_get_type()
    test_get_entry()
    test_append()
    test_set_entry()
