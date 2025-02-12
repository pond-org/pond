import os
import pickle
import pytest
from pydantic import BaseModel
from PIL import Image

from conf.catalog import Navigation, Values
from conf.file_catalog import (
    FileDrive,
    read_image,
    write_image,
    read_pickle,
    write_pickle,
)

from pond.field import File, Field
from pond import Lens
from tests.test_utils import (
    empty_iceberg_catalog,
    empty_lance_catalog,
)


class FileCatalog(BaseModel):
    image: File[Image.Image] = Field(reader=read_image, writer=write_image, ext="png")
    images: list[File[Image.Image]] = Field(
        reader=read_image, writer=write_image, ext="png"
    )
    drives: list[FileDrive]
    values: File[Values] = Field(reader=read_pickle, writer=write_pickle, ext="pickle")


@pytest.fixture
def catalog() -> FileCatalog:
    catalog = FileCatalog(
        image=File.set(Image.new("RGBA", (128, 128), "red")),
        images=[
            File.set(Image.new("RGBA", (128, 128), "red")),
            File.set(Image.new("RGBA", (128, 128), "green")),
            File.set(Image.new("RGBA", (128, 128), "blue")),
        ],
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
    ("data_catalog_fixture",), [("empty_iceberg_catalog",), ("empty_lance_catalog",)]
)
def test_index_files(request, catalog, tmp_path_factory, data_catalog_fixture):
    data_catalog = request.getfixturevalue(data_catalog_fixture)
    storage_path = tmp_path_factory.mktemp("storage")

    for i, image in enumerate(catalog.images):
        os.makedirs(os.path.join(storage_path, "catalog", "images"), exist_ok=True)
        image.get().save(
            os.path.join(storage_path, "catalog", "images", f"test_{i}.png")
        )
    os.makedirs(os.path.join(storage_path, "catalog"), exist_ok=True)
    catalog.image.get().save(os.path.join(storage_path, "catalog", "image.png"))

    for i, drive in enumerate(catalog.drives):
        navs = drive.navigation.get()
        os.makedirs(
            os.path.join(storage_path, "catalog", "drives", f"{i}"), exist_ok=True
        )
        with open(
            os.path.join(
                storage_path, "catalog", "drives", f"{i}", "navigation.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(
                navs,
                f,
            )

    with open(
        os.path.join(storage_path, "catalog", "values.pickle"),
        "wb",
    ) as f:
        pickle.dump(catalog.values.get(), f)

    root_path = "catalog"
    lens = Lens(FileCatalog, "", data_catalog, root_path, storage_path)
    lens.index_files()
    # lens.set(catalog)

    lens = Lens(FileCatalog, "image", data_catalog, root_path, storage_path)
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

    # lens = Lens(FileCatalog, "values", data_catalog, root_path, storage_path)
    # value = lens.get()
    # assert value.path == "catalog/values"
    # assert value.get() == catalog.values.get()

    # lens = Lens(
    #     FileCatalog, "drives[0].navigation", data_catalog, root_path, storage_path
    # )
    # value = lens.get()
    # assert value.path == "catalog/drives/0/navigation"
    # assert value.get() == catalog.drives[0].navigation.get()

    # lens = Lens(FileCatalog, "drives", data_catalog, root_path, storage_path)
    # value = lens.get()
    # assert value == catalog.drives
