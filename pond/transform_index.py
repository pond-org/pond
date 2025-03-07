import os
import warnings
from collections import OrderedDict
from typing import Callable, Type, get_type_hints, get_origin, get_args, Self, Tuple

from pydantic import BaseModel
from pydantic._internal import _generics

from pond.state import State
from pond.lens import LensInfo, LensPath, TypeField
from pond.field import File
from pond.abstract_transform import AbstractExecuteTransform


def get_file_paths(path: list[TypeField], model_type: Type) -> list[LensPath]:
    if _generics.get_origin(model_type) == File:
        return [LensPath(path)]
    elif (
        get_origin(model_type) is list
        and _generics.get_origin(get_args(model_type)[0]) is File
    ):
        return [LensPath(path)]
    elif get_origin(model_type) is list:
        print(f"Checking list: {path}")
        item_type = get_args(model_type)[0]
        file_paths = get_file_paths(path, item_type)
        # if any items in a list might be a file, we need
        # to index the whole list
        if file_paths:
            return [LensPath(path)]
    elif issubclass(model_type, BaseModel):
        print(f"Checking model: {path}")
        file_paths = []
        for field in model_type.model_fields:
            field_type = model_type.model_fields[field].annotation
            field_path = path + [TypeField(field, None)]
            file_paths.extend(get_file_paths(field_path, field_type))
        return file_paths
    return []


class TransformIndex(AbstractExecuteTransform):
    # TODO: make inputs/outputs work with dicts also
    def __init__(
        self,
        Catalog: Type[BaseModel],
        path: list[str] | str = "",
    ):
        paths = path if isinstance(path, list) else [path]
        self.outputs = []
        for p in paths:
            lens_info = LensInfo(Catalog, p)
            file_paths = get_file_paths(lens_info.lens_path.path, lens_info.type)
            self.outputs.extend(file_paths)

    def get_inputs(self) -> list[LensPath]:
        return []

    def get_outputs(self) -> list[LensPath]:
        return self.outputs

    def get_transforms(self) -> list[Self]:
        return [self]

    def execute_on(self, state: State) -> None:
        state.index_files([o.to_path() for o in self.outputs])
