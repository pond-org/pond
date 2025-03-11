import pytest

from pond import State, pipe, node
from conf.catalog import Catalog, Navigation

from tests.test_utils import (
    catalog,
    empty_iceberg_catalog,
    empty_lance_catalog,
)


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
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
