# Copyright 2025 Nils Bore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# import pond

from typing import Any

import pytest

from conf.catalog import Catalog, Drive, Navigation
from pond import Lens, State, Transform


def value1_value2(value1: float) -> int:
    return int(value1)


def value1_value2_not_annotated(value1: Any) -> Any:
    return int(value1)


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_transform(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog)
    lens = Lens(Catalog, "values.value1", data_catalog)
    lens.set(catalog.values.value1)
    transform = Transform(
        Catalog,
        "values.value1",
        "values.value2",
        value1_value2,
    )
    for unit in transform.get_execute_units(state):
        unit.execute_on(state)
    lens = Lens(Catalog, "values.value2", data_catalog)
    assert lens.get() == value1_value2(catalog.values.value1)
    transform = Transform(Catalog, ["values.value1"], ["values.value2"], value1_value2)
    for unit in transform.get_execute_units(state):
        unit.execute_on(state)
    lens = Lens(Catalog, "values.value2", data_catalog)
    assert lens.get() == value1_value2(catalog.values.value1)
    transform = Transform(
        Catalog,
        "values.value1",
        "values.value2",
        value1_value2_not_annotated,
    )
    for unit in transform.get_execute_units(state):
        unit.execute_on(state)
    lens = Lens(Catalog, "values.value2", data_catalog)
    assert lens.get() == value1_value2(catalog.values.value1)


def drive_id(input: Drive) -> Drive:
    return input


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_list_items(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog)
    lens = Lens(Catalog, "drives[0]", data_catalog)
    lens.set(catalog.drives[0])
    transform = Transform(Catalog, "drives[0]", "drives[1]", drive_id)
    for unit in transform.get_execute_units(state):
        unit.execute_on(state)
    lens = Lens(Catalog, "drives[1]", data_catalog)
    assert catalog.drives[0] == lens.get()


def nav_list_id(input: list[Navigation]) -> list[Navigation]:
    return input


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_list(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog)
    lens = Lens(Catalog, "drives[0]", data_catalog)
    lens.set(catalog.drives[0])
    transform = Transform(
        Catalog,
        "drives[0].navigation",
        "drives[1].navigation",
        nav_list_id,
    )
    for unit in transform.get_execute_units(state):
        unit.execute_on(state)
    lens = Lens(Catalog, "drives[1].navigation", data_catalog)
    assert catalog.drives[0].navigation == lens.get()
