# import pond

import pytest

import pyarrow as pa
import lance
import lancedb

from pond import Lens
from pond.lens import TypeField
import pond.lens
from tests.test_utils import (
    catalog,
    filled_iceberg_catalog,
    filled_lance_catalog,
    empty_iceberg_catalog,
    empty_lance_catalog,
)

from conf.catalog import Catalog, Drive, Navigation, Values


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


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_set_get_index(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    lens = Lens(Catalog, "drives", data_catalog)
    print("FIRST LENS PATH: ", lens.lens_path)
    lens.set(catalog.drives)
    lens = Lens(Catalog, "drives[0]", data_catalog)
    print("SECOND LENS PATH: ", lens.lens_path)
    value = lens.get()
    assert value == catalog.drives[0]
    lens = Lens(Catalog, "drives[1]", data_catalog)
    print("SECOND LENS PATH: ", lens.lens_path)
    value = lens.get()
    assert value == catalog.drives[1]


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_set_entry(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    lens = Lens(Catalog, "values", data_catalog)  # , db_path=path)
    lens.set(catalog.values)
    value = lens.get()
    assert value == catalog.values
    lens = Lens(Catalog, "drives[0].navigation[0]", data_catalog)  # , db_path=path)
    lens.set(catalog.drives[0].navigation[0])
    value = lens.get()
    assert value == catalog.drives[0].navigation[0]
    lens = Lens(Catalog, "drives", data_catalog)  # , db_path=path)
    lens.set(catalog.drives)
    value = lens.get()
    assert value == catalog.drives
    lens = Lens(Catalog, "drives[0].navigation", data_catalog)  # , db_path=path)
    lens.set(catalog.drives[0].navigation)
    value = lens.get()
    assert value == catalog.drives[0].navigation
    lens = Lens(Catalog, "values.value1", data_catalog)  # , db_path=path)
    lens.set(catalog.values.value1)
    value = lens.get()
    assert value == catalog.values.value1
    lens = Lens(Catalog, "values.names", data_catalog)  # , db_path=path)
    lens.set(catalog.values.names)
    value = lens.get()
    assert value == catalog.values.names


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_set_part(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    lens = Lens(Catalog, "values", data_catalog)  # , db_path=path)
    lens.set(catalog.values)
    lens = Lens(Catalog, "values.value1", data_catalog)  # , db_path=path)
    value = lens.get()
    assert value == catalog.values.value1
    lens = Lens(Catalog, "drives[0]", data_catalog)  # , db_path=path)
    lens.set(catalog.drives[0])
    lens = Lens(Catalog, "drives[0].navigation", data_catalog)  # , db_path=path)
    value = lens.get()
    assert value == catalog.drives[0].navigation
    lens = Lens(Catalog, "drives[1]", data_catalog)  # , db_path=path)
    lens.set(catalog.drives[1])
    lens = Lens(Catalog, "drives[1].images", data_catalog)  # , db_path=path)
    value = lens.get()
    assert value == catalog.drives[1].images


# def test_get_entry_with_type(catalog: Catalog, tmp_path_factory):
#     path = tmp_path_factory.mktemp("db")
#     write_dataset(catalog, path)
#     read_catalog = pond.lens.get_entry_with_type(
#         pond.lens.LensPath.from_path("", "test"), Catalog, path
#     )
#     assert read_catalog == catalog
#     values = pond.lens.get_entry_with_type(
#         pond.lens.LensPath.from_path("values", "test"), Values, path
#     )
#     assert values == catalog.values
#     drive0 = pond.lens.get_entry_with_type(
#         pond.lens.LensPath.from_path("drives[0]", "test"), Drive, path
#     )
#     assert drive0 == catalog.drives[0]
#     drive1 = pond.lens.get_entry_with_type(
#         pond.lens.LensPath.from_path("drives[1]", "test"), Drive, path
#     )
#     assert drive1 == catalog.drives[1]
#     navigation0 = pond.lens.get_entry_with_type(
#         pond.lens.LensPath.from_path("drives[0].navigation[0]", "test"),
#         Navigation,
#         path,
#     )
#     assert navigation0 == catalog.drives[0].navigation[0]


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("filled_iceberg_catalog",), ("filled_lance_catalog",)]
)
def test_get_entry(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    lens = Lens(Catalog, "", data_catalog, "test")  # , db_path=path)
    read_catalog = lens.get()
    assert read_catalog == catalog
    lens = Lens(Catalog, "values", data_catalog, "test")  # , db_path=path)
    values = lens.get()
    assert values == catalog.values
    lens = Lens(Catalog, "values.value1", data_catalog, "test")  # , db_path=path)
    value1 = lens.get()
    assert value1 == catalog.values.value1
    lens = Lens(
        Catalog, "values.navigation.dummy", data_catalog, "test"
    )  # , db_path=path)
    dummy = lens.get()
    assert dummy == catalog.values.navigation.dummy
    lens = Lens(Catalog, "values.navigation", data_catalog, "test")  # , db_path=path)
    navigation = lens.get()
    assert navigation == catalog.values.navigation
    lens = Lens(Catalog, "drives", data_catalog, "test")  # , db_path=path)
    drives = lens.get()
    assert drives == catalog.drives
    lens = Lens(Catalog, "drives[0]", data_catalog, "test")  # , db_path=path)
    drive0 = lens.get()
    assert drive0 == catalog.drives[0]
    lens = Lens(Catalog, "drives[1]", data_catalog, "test")  # , db_path=path)
    drive1 = lens.get()
    assert drive1 == catalog.drives[1]
    lens = Lens(
        Catalog, "drives[0].navigation[0]", data_catalog, "test"
    )  # , db_path=path)
    navigation0 = lens.get()
    assert navigation0 == catalog.drives[0].navigation[0]
    lens = Lens(
        Catalog, "drives[0].navigation[1]", data_catalog, "test"
    )  # , db_path=path)
    navigation0 = lens.get()
    assert navigation0 == catalog.drives[0].navigation[1]
    lens = Lens(
        Catalog, "drives[0].navigation[1].dummy", data_catalog, "test"
    )  # , db_path=path)
    dummy1 = lens.get()
    assert dummy1 == catalog.drives[0].navigation[1].dummy
    lens = Lens(
        Catalog, "drives[1].navigation[0]", data_catalog, "test"
    )  # , db_path=path)
    navigation1 = lens.get()
    assert navigation1 == catalog.drives[1].navigation[0]


def test_get_type():
    path = pond.lens.LensPath.from_path("")
    catalog, _ = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert catalog == Catalog
    path = pond.lens.LensPath.from_path("values")
    values, _ = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert values == Values
    path = pond.lens.LensPath.from_path("drives[0]")
    drive0, _ = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert drive0 == Drive
    path = pond.lens.LensPath.from_path("drives[1]")
    drive1, _ = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert drive1 == Drive
    path = pond.lens.LensPath.from_path("drives[0].navigation[0]")
    navigation0, _ = pond.lens.get_tree_type(path.path[1:], Catalog)
    assert navigation0 == Navigation


def test_path_and_query():
    path = pond.lens.LensPath.from_path("")
    assert path.path == [TypeField("catalog", None)]
    assert path.path_and_query(1) == ("catalog", "")
    path = pond.lens.LensPath.from_path("values")
    assert path.path == [TypeField("catalog", None), TypeField("values", None)]
    assert path.path_and_query(1) == ("catalog", "values")
    path = pond.lens.LensPath.from_path("drives[0]")
    assert path.path == [TypeField("catalog", None), TypeField("drives", 0)]
    assert path.path_and_query(1) == ("catalog", "drives[1]")
    path = pond.lens.LensPath.from_path("drives[0].navigation[0]")
    assert path.path == [
        TypeField("catalog", None),
        TypeField("drives", 0),
        TypeField("navigation", 0),
    ]
    assert path.path_and_query(1) == ("catalog", "drives[1]['navigation'][1]")
    assert path.path_and_query(2) == ("catalog/drives[0]", "navigation[1]")


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


if __name__ == "__main__":
    # test_db()
    # test_get_entry_with_type()
    test_get_type()
    test_get_entry()
    test_append()
    test_set_entry()
    test_path_and_query()
