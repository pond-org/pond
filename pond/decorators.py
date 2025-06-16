from typing import Callable, Type

from pydantic import BaseModel

from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractTransform,
)
from pond.transforms.transform import Transform
from pond.transforms.transform_construct import TransformConstruct
from pond.transforms.transform_index import TransformIndex
from pond.transforms.transform_list import TransformList
from pond.transforms.transform_list_fold import TransformListFold
from pond.transforms.transform_pipe import TransformPipe


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
    ) -> AbstractExecuteTransform:
        inputs = self.input if isinstance(self.input, list) else [self.input]
        outputs = self.output if isinstance(self.output, list) else [self.output]
        list_input = False
        list_output = False
        for input in inputs:
            if "[:]" in input:
                list_input = True
                break
        for output in outputs:
            if "[:]" in output:
                list_output = True
                break
        if list_input and list_output:
            return TransformList(self.Catalog, self.input, self.output, fn)
        elif list_input:
            return TransformListFold(self.Catalog, self.input, self.output, fn)
        elif list_output:
            raise RuntimeError("Outputs can not use [:] indices without any in input")
        else:
            return Transform(self.Catalog, self.input, self.output, fn)


def pipe(
    transforms: list[AbstractTransform],
    input: list[str] | str = [],
    output: list[str] | str = [],
    root_path: str = "catalog",
) -> TransformPipe:
    return TransformPipe(transforms, input, output, root_path)


def construct(
    Catalog: Type[BaseModel],
    path: str = "",
) -> TransformConstruct:
    return TransformConstruct(Catalog, path)


def index_files(
    Catalog: Type[BaseModel],
    path: list[str] | str = "",
) -> TransformIndex:
    return TransformIndex(Catalog, path)
