import pytest
from PIL import Image

from conf.variant_catalog import VariantCatalog

from pond.field import File
from pond import Lens
from tests.test_utils import (
    empty_iceberg_catalog,
    empty_lance_catalog,
)


@pytest.fixture
def catalog() -> VariantCatalog:
    catalog = VariantCatalog(
        image=File.set(Image.new("RGBA", (128, 128), "red")),
        images=[
            File.set(Image.new("RGBA", (128, 128), "red")),
            File.set(Image.new("RGBA", (128, 128), "green")),
            File.set(Image.new("RGBA", (128, 128), "blue")),
        ],
        value=File.set(12),
        values=[File.set(1), File.set(2), File.set(3)],
    )
    return catalog


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_set_entry(request, catalog, tmp_path_factory, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    storage_path = tmp_path_factory.mktemp("storage")
    root_path = "catalog"
    lens = Lens(VariantCatalog, "image", data_catalog, root_path, storage_path)
    lens.set(catalog.image)
    value = lens.get()
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
    lens = Lens(VariantCatalog, "images", data_catalog, root_path, storage_path)
    lens.set(catalog.images)
    value_list = lens.get()
    for i, value in enumerate(value_list):
        assert value.path == f"catalog/images/{i}"
        src = catalog.images[i].get()
        target = value.get()
        assert (
            target.mode == src.mode
        ), f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
        assert (
            target.size == src.size
        ), f"got size {repr(target.size)}, expected {repr(src.size)}"
        assert target.tobytes() == src.tobytes()
    lens = Lens(VariantCatalog, "value", data_catalog, root_path, storage_path)
    lens.set(catalog.value)
    value = lens.get()
    assert value.path == "catalog/value"
    assert value.get() == catalog.value.get()
    lens = Lens(VariantCatalog, "values", data_catalog, root_path, storage_path)
    lens.set(catalog.values)
    value_list = lens.get()
    for i, value in enumerate(value_list):
        assert value.path == f"catalog/values/{i}"
        assert value.get() == catalog.values[i].get()


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_set_file_entry(request, catalog, tmp_path_factory, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    storage_path = tmp_path_factory.mktemp("storage")
    root_path = "catalog"
    lens = Lens(VariantCatalog, "file:image", data_catalog, root_path, storage_path)
    lens.set(catalog.image.get())
    lens = Lens(VariantCatalog, "image", data_catalog, root_path, storage_path)
    value = lens.get()
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

    lens = Lens(VariantCatalog, "file:images", data_catalog, root_path, storage_path)
    lens.set([im.get() for im in catalog.images])
    lens = Lens(VariantCatalog, "images", data_catalog, root_path, storage_path)
    value_list = lens.get()
    for i, value in enumerate(value_list):
        assert value.path == f"catalog/images/{i}"
        src = catalog.images[i].get()
        target = value.get()
        assert (
            target.mode == src.mode
        ), f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
        assert (
            target.size == src.size
        ), f"got size {repr(target.size)}, expected {repr(src.size)}"
        assert target.tobytes() == src.tobytes()

    lens = Lens(VariantCatalog, "file:value", data_catalog, root_path, storage_path)
    lens.set(catalog.value.get())
    lens = Lens(VariantCatalog, "value", data_catalog, root_path, storage_path)
    value = lens.get()
    assert value.path == "catalog/value"
    assert value.get() == catalog.value.get()

    lens = Lens(VariantCatalog, "file:values", data_catalog, root_path, storage_path)
    lens.set([v.get() for v in catalog.values])
    lens = Lens(VariantCatalog, "values", data_catalog, root_path, storage_path)
    value_list = lens.get()
    for i, value in enumerate(value_list):
        assert value.path == f"catalog/values/{i}"
        assert value.get() == catalog.values[i].get()


@pytest.mark.parametrize(
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_get_file_entry(request, catalog, tmp_path_factory, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    storage_path = tmp_path_factory.mktemp("storage")
    root_path = "catalog"
    lens = Lens(VariantCatalog, "", data_catalog, root_path, storage_path)
    lens.set(catalog)

    lens = Lens(VariantCatalog, "file:image", data_catalog, root_path, storage_path)
    src = catalog.image.get()
    target = lens.get()
    assert (
        target.mode == src.mode
    ), f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
    assert (
        target.size == src.size
    ), f"got size {repr(target.size)}, expected {repr(src.size)}"
    assert target.tobytes() == src.tobytes()

    lens = Lens(VariantCatalog, "file:images", data_catalog, root_path, storage_path)
    value_list = lens.get()
    for i, target in enumerate(value_list):
        src = catalog.images[i].get()
        assert (
            target.mode == src.mode
        ), f"got mode {repr(target.mode)}, expected {repr(src.mode)}"
        assert (
            target.size == src.size
        ), f"got size {repr(target.size)}, expected {repr(src.size)}"
        assert target.tobytes() == src.tobytes()

    lens = Lens(VariantCatalog, "file:value", data_catalog, root_path, storage_path)
    value = lens.get()
    assert value == catalog.value.get()

    lens = Lens(VariantCatalog, "file:values", data_catalog, root_path, storage_path)
    value_list = lens.get()
    for i, value in enumerate(value_list):
        assert value == catalog.values[i].get()
