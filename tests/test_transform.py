# import pond

import os
from typing import Any

import pytest

from conf.catalog import Catalog, Drive, Navigation, Values

from pond import Lens, Transform


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


def value1_value2(value1: float) -> int:
    return int(value1)


def value1_value2_not_annotated(value1: Any) -> Any:
    return int(value1)


def test_transform(catalog: Catalog, tmp_path_factory):
    path = tmp_path_factory.mktemp("db")
    lens = Lens(Catalog, "values.value1", db_path=path)
    lens.set(catalog.values.value1)
    transform = Transform(
        value1_value2,
        Catalog,
        "values.value1",
        "values.value2",
        db_path=path,
    )
    transform()
    lens = Lens(Catalog, "values.value2", db_path=path)
    assert lens.get() == value1_value2(catalog.values.value1)
    transform = Transform(
        value1_value2, Catalog, ["values.value1"], ["values.value2"], db_path=path
    )
    transform()
    lens = Lens(Catalog, "values.value2", db_path=path)
    assert lens.get() == value1_value2(catalog.values.value1)
    transform = Transform(
        value1_value2_not_annotated,
        Catalog,
        "values.value1",
        "values.value2",
        db_path=path,
    )
    transform()
    lens = Lens(Catalog, "values.value2", db_path=path)
    assert lens.get() == value1_value2(catalog.values.value1)
