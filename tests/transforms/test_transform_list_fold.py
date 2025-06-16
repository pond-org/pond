import pytest
from pydantic import BaseModel

from pond import State
from pond.transforms.transform_list_fold import TransformListFold


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_transform_list_fold(request, data_catalog_fixture):
    catalog = request.getfixturevalue(data_catalog_fixture)

    class Catalog(BaseModel):
        # inputs
        v1: int
        a: list[int]

        # outputs
        b: int

    def fn(v1: int, a: list[int]) -> int:
        return sum(a) + v1

    state = State(Catalog, catalog)
    state["v1"] = 1
    state["a"] = [2, 3, 4]

    node = TransformListFold(Catalog, ["v1", "a[:]"], "b", fn)

    for unit in node.get_execute_units(state):
        unit.execute_on(state)

    assert state["b"] == 10


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_transform_list_fold_partitioned(request, data_catalog_fixture):
    catalog = request.getfixturevalue(data_catalog_fixture)

    class Catalog(BaseModel):
        # inputs
        v1: int
        a: list[int]

        # outputs
        b: int

    def fn(v1: int, a: list[int]) -> int:
        return sum(a) + v1

    state = State(Catalog, catalog)
    state["v1"] = 1
    state["a[0]"] = 2
    state["a[1]"] = 3
    state["a[2]"] = 4

    node = TransformListFold(Catalog, ["v1", "a[:]"], "b", fn)

    for unit in node.get_execute_units(state):
        unit.execute_on(state)

    assert state["b"] == 10


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_transform_list_fold_partitioned_struct(request, data_catalog_fixture):
    catalog = request.getfixturevalue(data_catalog_fixture)

    class Item(BaseModel):
        value: int

    class Catalog(BaseModel):
        # inputs
        v1: int
        a: list[Item]

        # outputs
        b: int

    def fn(v1: int, a: list[int]) -> int:
        return sum(a) + v1

    state = State(Catalog, catalog)
    state["v1"] = 1
    state["a[0].value"] = 2
    state["a[1].value"] = 3
    state["a[2].value"] = 4

    node = TransformListFold(Catalog, ["v1", "a[:].value"], "b", fn)

    for unit in node.get_execute_units(state):
        unit.execute_on(state)

    assert state["b"] == 10
