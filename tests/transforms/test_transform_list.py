import pytest
from pydantic import BaseModel

from pond import State, node


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_transform_list(request, data_catalog_fixture):
    catalog = request.getfixturevalue(data_catalog_fixture)

    class Catalog(BaseModel):
        # inputs
        v1: int
        a: list[float]
        b: list[int]

        # outputs
        c: list[str]

    @node(Catalog, ["v1", "a[:]", "b[:]"], ["c[:]"])
    def fn(v1: int, a: float, b: int) -> str:
        return f"{v1}+{a}+{b}"

    state = State(Catalog, catalog)
    state["v1"] = 1
    state["a"] = [2.0, 3.0, 4.0]
    state["b"] = [5, 6, 7]

    for unit in fn.get_execute_units(state):
        unit.execute_on(state)

    assert set(state["c"]) == set(["1+2.0+5", "1+3.0+6", "1+4.0+7"])
