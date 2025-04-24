from typing import Callable, Type

from pydantic import BaseModel

from pond.transforms.abstract_transform import (
    AbstractTransform,
    AbstractExecuteTransform,
)
from pond.transforms.transform import Transform
from pond.transforms.transform_pipe import TransformPipe
from pond.transforms.transform_index import TransformIndex
from pond.transforms.transform_list import TransformList

import functools


# def node(
#     Catalog: Type[BaseModel],
#     input: list[str] | str,
#     output: list[str] | str,
# ):
#     # self.Catalog = Catalog
#     # self.input = input
#     # self.output = output

#     def inner(
#         fn: Callable,
#     ) -> AbstractExecuteTransform:
#         @functools.wraps(fn)
#         def wrapper(*args, **kwargs):
#             inputs = input if isinstance(input, list) else [input]
#             for linput in inputs:
#                 if "[:]" in linput:
#                     return TransformList(Catalog, input, output, wrapper)
#             return Transform(Catalog, input, output, wrapper)
#             # return fn(*args, **kwargs)

#         return wrapper

#     return inner


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
        # functools.update_wrapper(self, fn)  ## TA-DA! ##
        for input in inputs:
            if "[:]" in input:
                return TransformList(self.Catalog, self.input, self.output, fn)
        return Transform(self.Catalog, self.input, self.output, fn)
        # return fn(*args, **kwargs)


def pipe(
    transforms: list[AbstractTransform],
    input: list[str] | str = [],
    output: list[str] | str = [],
    root_path: str = "catalog",
) -> TransformPipe:
    return TransformPipe(transforms, input, output, root_path)


def index_files(
    Catalog: Type[BaseModel],
    path: list[str] | str = "",
) -> TransformIndex:
    return TransformIndex(Catalog, path)
