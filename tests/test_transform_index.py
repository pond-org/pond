import pytest

from pond import State
from pond.abstract_catalog import LensPath
from pond.transform_index import TransformIndex

from tests.test_utils import (
    empty_iceberg_catalog,
    empty_lance_catalog,
)
from tests.test_file_utils import FileCatalog, catalog, filled_storage


def test_get_file_paths():
    t = TransformIndex(FileCatalog)
    outputs = t.get_outputs()
    assert len(outputs) == 4
    assert LensPath.from_path("image") in outputs
    assert LensPath.from_path("images") in outputs
    assert LensPath.from_path("drives") in outputs
    assert LensPath.from_path("values") in outputs
    assert LensPath.from_path("dummy") not in outputs
    t = TransformIndex(FileCatalog, ["image", "values"])
    outputs = t.get_outputs()
    assert len(outputs) == 2
    assert LensPath.from_path("image") in outputs
    assert LensPath.from_path("images") not in outputs
    assert LensPath.from_path("drives") not in outputs
    assert LensPath.from_path("values") in outputs
    assert LensPath.from_path("dummy") not in outputs


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_transform_index(request, catalog, filled_storage, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    state = State(FileCatalog, data_catalog, storage_path=filled_storage)
    t = TransformIndex(FileCatalog)
    t.execute_on(state)
    value = state["image"]
    assert value.path == "catalog/image"
    src = catalog.image.get()
    target = value.get()
    assert (
        target.mode == src.mode
    ), f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
    assert (
        target.size == src.size
    ), f"got size {repr(target.size)}, expected {repr(src.size)}"
    assert target.tobytes() == src.tobytes()

    value = state["images"]
    for i, image in enumerate(catalog.images):
        assert value[i].path == f"catalog/images/test_{i}"
        src = image.get()
        target = value[i].get()
        assert (
            target.mode == src.mode
        ), f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
        assert (
            target.size == src.size
        ), f"got size {repr(target.size)}, expected {repr(src.size)}"
        assert target.tobytes() == src.tobytes()

    value = state["values"]
    assert value.path == "catalog/values"
    assert value.get() == catalog.values.get()

    value = state["drives[0].navigation"]
    assert value.path == "catalog/drives/test_1/navigation"
    assert value.get() == catalog.drives[1].navigation.get()

    value = state["drives[1].images"]
    assert value.path == "catalog/drives/test_0/images"
    assert value.get() == catalog.drives[0].images.get()
