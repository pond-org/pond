import pytest

from pydantic import BaseModel

from pond import State, pipe, construct

from tests.test_utils import (
    empty_iceberg_catalog,
    empty_lance_catalog,
    empty_delta_catalog,
)


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
