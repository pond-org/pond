from typing import Callable, Type

from pydantic import BaseModel

from pond.abstract_transform import AbstractTransform
from pond.transform import Transform
from pond.transform_pipe import TransformPipe
from pond.transform_index import TransformIndex


class node:
    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
    ):
        self.Catalog = Catalog
        self.input = input
        self.output = output

    def __call__(
        self,
        fn: Callable,
    ) -> Transform:
        return Transform(self.Catalog, self.input, self.output, fn)


def pipe(
    transforms: list[AbstractTransform],
    input: list[str] | str,
    output: list[str] | str,
    root_path: str = "catalog",
) -> TransformPipe:
    return TransformPipe(transforms, input, output, root_path)


def index_files(
    Catalog: Type[BaseModel],
    path: list[str] | str = "",
) -> TransformIndex:
    return TransformIndex(Catalog, path)
