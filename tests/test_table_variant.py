import pytest
import pyarrow as pa

from conf.table_catalog import TableCatalog, Car

from pond.field import File
from pond import Lens
from tests.test_utils import (
    empty_iceberg_catalog,
    empty_lance_catalog,
)


@pytest.fixture
def catalog() -> TableCatalog:
    catalog = TableCatalog(
        cars=[
            Car(mileage=200.0, brand="volvo", top_speed=180.0, build_year=1984),
            Car(mileage=300.0, brand="saab", top_speed=230.0, build_year=1991),
            Car(mileage=90.0, brand="seat", top_speed=190.0, build_year=2014),
            Car(mileage=410.0, brand="fiat", top_speed=150.0, build_year=1999),
        ]
    )
    return catalog


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_set_table_entry(request, tmp_path_factory, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    storage_path = tmp_path_factory.mktemp("storage")
    volume_protocol_args = {"dir": {"path": storage_path}}
    root_path = "catalog"

    mileage = pa.array([200.0, 300.0, 90.0, 410.0])
    brand = pa.array(["volvo", "saab", "seat", "fiat"])
    top_speed = pa.array([180.0, 230.0, 190.0, 150.0])
    build_year = [1984, 1991, 2014, 1999]
    # names = ["mileage", "brand", "top_speed", "build_year"]
    schema = pa.schema(
        [
            pa.field("mileage", pa.float64(), nullable=False),
            pa.field("brand", pa.large_string(), nullable=False),
            pa.field("top_speed", pa.float64(), nullable=False),
            pa.field("build_year", pa.int64(), nullable=False),
        ]
    )
    cars = pa.table([mileage, brand, top_speed, build_year], schema=schema)

    lens = Lens(
        TableCatalog, "table:cars", data_catalog, root_path, volume_protocol_args
    )
    lens.set(cars)
    value = lens.get()
    assert value.equals(cars)


# @pytest.mark.parametrize(
#     ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
# )
# def test_get_table_entry(request, catalog, tmp_path_factory, data_catalog_fixture):
#     data_catalog = request.getfixturevalue(data_catalog_fixture)
#     storage_path = tmp_path_factory.mktemp("storage")
#     root_path = "catalog"
#     lens = Lens(VariantCatalog, "", data_catalog, root_path, storage_path)
#     lens.set(catalog)

#     lens = Lens(VariantCatalog, "file:value", data_catalog, root_path, storage_path)
#     value = lens.get()
#     assert value == catalog.value.get()
