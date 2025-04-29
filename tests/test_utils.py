import os
import pytest

import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
import lance

from pond.catalogs.lance_catalog import LanceCatalog
from pond.catalogs.iceberg_catalog import IcebergCatalog
import pond.lens

from conf.catalog import Catalog, Drive, Navigation, Values


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
        values=Values(
            value1=0.5,
            value2=2,
            name="One",
            names=["Two", "Three"],
            navigation=Navigation(dummy=True),
        ),
    )
    return catalog


@pytest.fixture
def filled_iceberg_catalog(catalog: Catalog, tmp_path_factory):
    warehouse_path = tmp_path_factory.mktemp("iceberg_catalog")
    data_catalog = IcebergCatalog(
        "default",
        **{
            "uri": f"sqlite:///{warehouse_path}/pyiceberg_catalog.db",
            "warehouse": f"file://{warehouse_path}",
        },
    )
    write_iceberg_dataset(catalog, data_catalog.catalog)
    return data_catalog


@pytest.fixture
def empty_iceberg_catalog(tmp_path_factory):
    warehouse_path = tmp_path_factory.mktemp("iceberg_catalog")
    data_catalog = IcebergCatalog(
        "default",
        **{
            "uri": f"sqlite:///{warehouse_path}/pyiceberg_catalog.db",
            "warehouse": f"file://{warehouse_path}",
        },
    )
    data_catalog.catalog.create_namespace_if_not_exists("catalog")
    return data_catalog


@pytest.fixture
def empty_lance_catalog(tmp_path_factory):
    path = tmp_path_factory.mktemp("db")
    data_catalog = LanceCatalog(path)
    return data_catalog


@pytest.fixture
def filled_lance_catalog(catalog: Catalog, tmp_path_factory):
    path = tmp_path_factory.mktemp("db")
    data_catalog = LanceCatalog(path)
    write_dataset(catalog, path)
    return data_catalog


def write_dataset(catalog, db_path):
    schema = pond.lens.get_pyarrow_schema(Catalog)

    # def producer():
    #     yield pa.RecordBatch.from_pylist([catalog])
    data = pa.Table.from_pylist([catalog.model_dump()], schema=schema)

    ds = lance.write_dataset(
        data, os.path.join(db_path, "test.lance"), schema=schema, mode="overwrite"
    )
    return ds


def write_iceberg_dataset(catalog, iceberg_catalog):
    schema = pond.lens.get_pyarrow_schema(Catalog)

    data = pa.Table.from_pylist([catalog.model_dump()], schema=schema)

    iceberg_catalog.create_namespace_if_not_exists("catalog")
    iceberg_table = iceberg_catalog.create_table_if_not_exists(
        identifier="catalog.test",
        schema=schema,
    )
    iceberg_table.overwrite(df=data)
    # ds = lance.write_dataset(
    #     data, os.path.join(db_path, "test.lance"), schema=schema, mode="overwrite"
    # )
    # return ds
