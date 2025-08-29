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
import pytest

from conf.catalog import Catalog, Navigation
from pond import State, node, pipe


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_decorators(request, catalog, data_catalog_fixture):
    @node(Catalog, "drives[0].navigation", "drives[0].images")
    def nav_to_float(value: list[Navigation]) -> list[float]:
        return [float(nav.dummy) for nav in value]

    @node(Catalog, "drives[0].images", "drives[0].uncertainty")
    def float_id(value: list[float]) -> list[float]:
        return value

    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog)
    state["drives[0].navigation"] = catalog.drives[0].navigation

    p = pipe([nav_to_float, float_id], "drives[0].navigation", "drives[0].uncertainty")

    transforms = p.get_transforms()
    for transform in transforms:
        for unit in transform.get_execute_units(state):
            unit.execute_on(state)
    value = state["drives[0].uncertainty"]
    assert value == nav_to_float.fn(catalog.drives[0].navigation)
    assert len(transforms) == 2
