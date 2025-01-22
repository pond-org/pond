# import pond

import os
from typing import Any
from functools import partial

import pytest

from pond.abstract_catalog import LanceCatalog
from pond import Lens, Transform
from tests.test_utils import (
    catalog,
    filled_iceberg_catalog,
    filled_lance_catalog,
    empty_iceberg_catalog,
    empty_lance_catalog,
)

from conf.catalog import Catalog, Drive, Navigation, Values


def value1_value2(value1: float) -> int:
    return int(value1)


def value1_value2_not_annotated(value1: Any) -> Any:
    return int(value1)


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_transform(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    lens = Lens(Catalog, "values.value1", data_catalog)
    lens.set(catalog.values.value1)
    transform = Transform(
        value1_value2,
        Catalog,
        "values.value1",
        "values.value2",
        data_catalog,
    )
    transform()
    lens = Lens(Catalog, "values.value2", data_catalog)
    assert lens.get() == value1_value2(catalog.values.value1)
    transform = Transform(
        value1_value2, Catalog, ["values.value1"], ["values.value2"], data_catalog
    )
    transform()
    lens = Lens(Catalog, "values.value2", data_catalog)
    assert lens.get() == value1_value2(catalog.values.value1)
    transform = Transform(
        value1_value2_not_annotated,
        Catalog,
        "values.value1",
        "values.value2",
        data_catalog,
    )
    transform()
    lens = Lens(Catalog, "values.value2", data_catalog)
    assert lens.get() == value1_value2(catalog.values.value1)


def drive_id(input: Drive) -> Drive:
    return input


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_list_items(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    lens = Lens(Catalog, "drives[0]", data_catalog)
    lens.set(catalog.drives[0])
    transform = Transform(drive_id, Catalog, "drives[0]", "drives[1]", data_catalog)
    transform()
    lens = Lens(Catalog, "drives[1]", data_catalog)
    assert catalog.drives[0] == lens.get()


def nav_list_id(input: list[Navigation]) -> list[Navigation]:
    return input


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_list(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    lens = Lens(Catalog, "drives[0]", data_catalog)
    lens.set(catalog.drives[0])
    transform = Transform(
        nav_list_id,
        Catalog,
        "drives[0].navigation",
        "drives[1].navigation",
        data_catalog,
    )
    transform()
    lens = Lens(Catalog, "drives[1].navigation", data_catalog)
    assert catalog.drives[0].navigation == lens.get()
