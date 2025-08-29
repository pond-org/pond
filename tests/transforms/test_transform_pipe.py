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

from conf.catalog import Catalog, Drive, Navigation
from pond.state import State
from pond.transforms.transform import Transform
from pond.transforms.transform_pipe import TransformPipe


def nav_to_float(value: list[Navigation]) -> list[float]:
    return [float(nav.dummy) for nav in value]


def float_id(value: list[float]) -> list[float]:
    return value


def test_transform_pipe():
    t1 = Transform(Drive, "navigation", "images", nav_to_float)
    t2 = Transform(Drive, "images", "uncertainty", float_id)
    p = TransformPipe([t1, t2], "navigation", "uncertainty")
    transforms = p.get_transforms()
    assert len(transforms) == 2

    p1 = TransformPipe([t1], "navigation", "images")
    p2 = TransformPipe([t2], "images", "uncertainty")
    p = TransformPipe([p1, p2], "navigation", "uncertainty")
    transforms = p.get_transforms()
    assert len(transforms) == 2

    p = TransformPipe([t1, p2], "navigation", "uncertainty")
    transforms = p.get_transforms()
    assert len(transforms) == 2

    p = TransformPipe([p], "navigation", "uncertainty")
    transforms = p.get_transforms()
    assert len(transforms) == 2


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_execute_transform_pipe(request, catalog, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(Catalog, data_catalog)
    state["drives[0].navigation"] = catalog.drives[0].navigation
    t1 = Transform(Catalog, "drives[0].navigation", "drives[0].images", nav_to_float)
    t2 = Transform(Catalog, "drives[0].images", "drives[0].uncertainty", float_id)
    p = TransformPipe([t1, t2], "drives[0].navigation", "drives[0].uncertainty")
    transforms = p.get_transforms()
    for transform in transforms:
        for unit in transform.get_execute_units(state):
            unit.execute_on(state)
    value = state["drives[0].uncertainty"]
    assert value == nav_to_float(catalog.drives[0].navigation)
    assert len(transforms) == 2
