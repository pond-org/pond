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

import pytest

from conf.catalog import Catalog
from pond import State


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_set_get_index(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog)
    state["drives"] = catalog.drives
    assert state["drives[0]"] == catalog.drives[0]
    assert state["drives[1]"] == catalog.drives[1]


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_set_entry(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog)  # , db_path=path)
    state["values"] = catalog.values
    assert state["values"] == catalog.values
    state["drives[0].navigation[0]"] = catalog.drives[0].navigation[0]
    assert state["drives[0].navigation[0]"] == catalog.drives[0].navigation[0]
    state["drives"] = catalog.drives
    assert state["drives"] == catalog.drives
    state["drives[0].navigation"] = catalog.drives[0].navigation
    assert state["drives[0].navigation"] == catalog.drives[0].navigation
    state["values.value1"] = catalog.values.value1
    assert state["values.value1"] == catalog.values.value1
    state["values.names"] = catalog.values.names
    assert state["values.names"] == catalog.values.names


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_set_part(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog)  # , db_path=path)
    state["values"] = catalog.values
    assert state["values.value1"] == catalog.values.value1
    state["drives[0]"] = catalog.drives[0]
    assert state["drives[0].navigation"] == catalog.drives[0].navigation
    state["drives[1]"] = catalog.drives[1]
    assert state["drives[1].images"] == catalog.drives[1].images


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("filled_iceberg_catalog",), ("filled_lance_catalog",)]
)
def test_get_entry(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog, "test")  # , db_path=path)
    assert state[""] == catalog
    assert state["values"] == catalog.values
    assert state["values.value1"] == catalog.values.value1
    assert state["values.navigation.dummy"] == catalog.values.navigation.dummy
    assert state["values.navigation"] == catalog.values.navigation
    assert state["drives"] == catalog.drives
    assert state["drives[0]"] == catalog.drives[0]
    assert state["drives[1]"] == catalog.drives[1]
    assert state["drives[0].navigation[0]"] == catalog.drives[0].navigation[0]
    assert state["drives[0].navigation[1]"] == catalog.drives[0].navigation[1]
    assert (
        state["drives[0].navigation[1].dummy"] == catalog.drives[0].navigation[1].dummy
    )
    assert state["drives[1].navigation[0]"] == catalog.drives[1].navigation[0]
