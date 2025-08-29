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
from pydantic import BaseModel

from pond import State, construct, pipe


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_transform_construct(request, data_catalog_fixture):
    catalog = request.getfixturevalue(data_catalog_fixture)

    class Aggregate(BaseModel):
        a: int
        b: list[float]

    class Catalog(BaseModel):
        agg: Aggregate

    state = State(Catalog, catalog)
    state["agg.a"] = 1
    state["agg.b"] = [2.0, 3.0, 4.0]

    p = pipe(
        [construct(Catalog, "agg"), construct(Catalog)], ["agg.a", "agg.b"], ["agg", ""]
    )
    transforms = p.get_transforms()
    for transform in transforms:
        for unit in transform.get_execute_units(state):
            unit.execute_on(state)
