import pytest
from PIL import Image

from conf.catalog import Navigation, Values
from conf.file_catalog import FileCatalog, FileDrive
from pond import Lens
from pond.field import File


@pytest.fixture
def catalog() -> FileCatalog:
    catalog = FileCatalog(
        image=File.set(Image.new("RGBA", (128, 128), "red")),
        drives=[
            FileDrive(
                navigation=File.set([Navigation(dummy=True), Navigation(dummy=False)]),
                images=File.set([1.0, 2.0, 3.0]),
                uncertainty=[0.1, 0.2, 0.3],
            ),
            FileDrive(
                navigation=File.set([Navigation(dummy=False)]),
                images=File.set([4.0, 5.0]),
                uncertainty=[0.4, 0.5, 0.6],
            ),
        ],
        values=File.set(
            Values(
                value1=0.5,
                value2=2,
                name="One",
                names=["Two", "Three"],
                navigation=Navigation(dummy=True),
            )
        ),
    )
    return catalog


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",), ("empty_delta_catalog",)],
)
def test_set_file_entry(request, catalog, tmp_path_factory, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    storage_path = tmp_path_factory.mktemp("storage1")
    volume_protocol_args = {"dir": {"path": storage_path}}
    root_path = "catalog"
    lens = Lens(FileCatalog, "image", data_catalog, root_path, volume_protocol_args)
    lens.set(catalog.image)
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
    storage_path = tmp_path_factory.mktemp("storage2")
    lens = Lens(FileCatalog, "values", data_catalog, root_path, volume_protocol_args)
    lens.set(catalog.values)
    value = lens.get()
    assert value.path == "catalog/values"
    assert value.get() == catalog.values.get()
    storage_path = tmp_path_factory.mktemp("storage4")
    lens = Lens(
        FileCatalog,
        "drives[0].navigation",
        data_catalog,
        root_path,
        volume_protocol_args,
    )
    lens.set(catalog.drives[0].navigation)
    value = lens.get()
    assert value.path == "catalog/drives/0/navigation"
    assert value.get() == catalog.drives[0].navigation.get()
    storage_path = tmp_path_factory.mktemp("storage3")
    lens = Lens(FileCatalog, "drives", data_catalog, root_path, volume_protocol_args)
    lens.set(catalog.drives)
    value = lens.get()
    assert value == catalog.drives


@pytest.mark.parametrize(
    ("data_catalog_fixture",),
    [("empty_iceberg_catalog",), ("empty_lance_catalog",)],
)
def test_get_file_part(request, catalog, tmp_path_factory, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    storage_path = tmp_path_factory.mktemp("storage")
    volume_protocol_args = {"dir": {"path": storage_path}}
    root_path = "catalog"
    lens = Lens(FileCatalog, "", data_catalog, root_path, volume_protocol_args)
    lens.set(catalog)

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
    assert value.path == "catalog/drives/0/navigation"
    assert value.get() == catalog.drives[0].navigation.get()

    lens = Lens(FileCatalog, "drives", data_catalog, root_path, volume_protocol_args)
    value = lens.get()
    assert value == catalog.drives
