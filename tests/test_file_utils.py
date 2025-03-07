import os
import pickle
import pytest

from pydantic import BaseModel
from PIL import Image

from pond.field import File, Field

from conf.catalog import Navigation, Values
from conf.file_catalog import (
    FileDrive,
    read_image,
    write_image,
    read_pickle,
    write_pickle,
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


@pytest.fixture
def filled_storage(tmp_path_factory, catalog):
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
        images = drive.images.get()
        os.makedirs(
            os.path.join(storage_path, "catalog", "drives", f"test_{i}"), exist_ok=True
        )
        with open(
            os.path.join(
                storage_path, "catalog", "drives", f"test_{i}", "navigation.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(
                navs,
                f,
            )
        with open(
            os.path.join(
                storage_path, "catalog", "drives", f"test_{i}", "images.pickle"
            ),
            "wb",
        ) as f:
            pickle.dump(
                images,
                f,
            )

    with open(
        os.path.join(storage_path, "catalog", "values.pickle"),
        "wb",
    ) as f:
        pickle.dump(catalog.values.get(), f)

    return storage_path
