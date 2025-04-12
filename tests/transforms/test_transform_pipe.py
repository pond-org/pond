import pytest

from pond.lens import Lens
from pond.state import State
from pond.transform import Transform
from pond.transform_pipe import TransformPipe
from conf.catalog import Catalog, Drive, Navigation

from tests.test_utils import (
    catalog,
    filled_iceberg_catalog,
    filled_lance_catalog,
    empty_iceberg_catalog,
    empty_lance_catalog,
)


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
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
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
