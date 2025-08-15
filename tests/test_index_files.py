import pytest

from pond import Lens, State
from tests.test_file_utils import FileCatalog, catalog  # noqa: F401


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_index_files(
    request,
    catalog,  # noqa: F811
    filled_storage,
    data_catalog_fixture,
):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    # storage_path = tmp_path_factory.mktemp("storage")
    volume_protocol_args = {"dir": {"path": filled_storage}}

    root_path = "catalog"
    lens = Lens(FileCatalog, "", data_catalog, root_path, volume_protocol_args)
    lens.index_files()
    # lens.set(catalog)

    lens = Lens(FileCatalog, "image", data_catalog, root_path, volume_protocol_args)
    value = lens.get()
    assert value.path == "catalog/image"
    src = catalog.image.get()
    target = value.get()
    assert target.mode == src.mode, (
        f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
    )
    assert target.size == src.size, (
        f"got size {repr(target.size)}, expected {repr(src.size)}"
    )
    assert target.tobytes() == src.tobytes()

    lens = Lens(FileCatalog, "images", data_catalog, root_path, volume_protocol_args)
    value = lens.get()
    for i, image in enumerate(catalog.images):
        # TODO: make this work as well
        # lens = Lens(FileCatalog, f"images[{i}]", data_catalog, root_path, volume_protocol_args)
        # value = lens.get()
        assert value[i].path == f"catalog/images/test_{i}"
        src = image.get()
        target = value[i].get()
        assert target.mode == src.mode, (
            f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
        )
        assert target.size == src.size, (
            f"got size {repr(target.size)}, expected {repr(src.size)}"
        )
        assert target.tobytes() == src.tobytes()

    lens = Lens(FileCatalog, "values", data_catalog, root_path, volume_protocol_args)
    value = lens.get()
    assert value.path == "catalog/values"
    assert value.get() == catalog.values.get()

    lens = Lens(
        FileCatalog,
        "drives[0].navigation",
        data_catalog,
        root_path,
        volume_protocol_args,
    )
    value = lens.get()
    assert value.path == "catalog/drives/test_0/navigation"
    assert value.get() == catalog.drives[0].navigation.get()

    lens = Lens(
        FileCatalog, "drives[1].images", data_catalog, root_path, volume_protocol_args
    )
    value = lens.get()
    assert value.path == "catalog/drives/test_1/images"
    assert value.get() == catalog.drives[1].images.get()

    # lens = Lens(FileCatalog, "drives", data_catalog, root_path, storage_path)
    # value = lens.get()
    # assert value == catalog.drives


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_state_index_files(
    request,
    catalog,  # noqa: F811
    filled_storage,
    data_catalog_fixture,
):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    volume_protocol_args = {"dir": {"path": filled_storage}}

    root_path = "catalog"
    state = State(FileCatalog, data_catalog, root_path, volume_protocol_args, "dir")
    state.index_files()

    value = state["image"]
    assert value.path == "catalog/image"
    src = catalog.image.get()
    target = value.get()
    assert target.mode == src.mode, (
        f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
    )
    assert target.size == src.size, (
        f"got size {repr(target.size)}, expected {repr(src.size)}"
    )
    assert target.tobytes() == src.tobytes()

    value = state["images"]
    for i, image in enumerate(catalog.images):
        assert value[i].path == f"catalog/images/test_{i}"
        src = image.get()
        target = value[i].get()
        assert target.mode == src.mode, (
            f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
        )
        assert target.size == src.size, (
            f"got size {repr(target.size)}, expected {repr(src.size)}"
        )
        assert target.tobytes() == src.tobytes()

    value = state["values"]
    assert value.path == "catalog/values"
    assert value.get() == catalog.values.get()

    value = state["drives[0].navigation"]
    assert value.path == "catalog/drives/test_0/navigation"
    assert value.get() == catalog.drives[0].navigation.get()

    value = state["drives[1].images"]
    assert value.path == "catalog/drives/test_1/images"
    assert value.get() == catalog.drives[1].images.get()
