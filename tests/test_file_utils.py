# Copyright 2025 Nils Bore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
from PIL import Image
from pydantic import BaseModel

from conf.catalog import Navigation, Values
from conf.file_catalog import FileDrive
from pond.field import Field, File
from pond.io.readers import read_image, read_pickle
from pond.io.writers import write_image, write_pickle


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
