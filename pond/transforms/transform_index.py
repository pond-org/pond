from typing import Any, Callable, Type, get_args, get_origin

from pydantic import BaseModel
from pydantic._internal import _generics

from pond.field import File
from pond.lens import LensInfo, LensPath, TypeField
from pond.state import State
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractExecuteUnit,
)


def get_file_paths(path: list[TypeField], model_type: Type) -> list[LensPath]:
    if _generics.get_origin(model_type) == File:
        return [LensPath(path)]
    elif (
        get_origin(model_type) is list
        and _generics.get_origin(get_args(model_type)[0]) is File
    ):
        return [LensPath(path)]
    elif get_origin(model_type) is list:
        item_type = get_args(model_type)[0]
        file_paths = get_file_paths(path, item_type)
        # if any items in a list might be a file, we need
        # to index the whole list
        if file_paths:
            return [LensPath(path)]
    elif issubclass(model_type, BaseModel):
        file_paths = []
        for field in model_type.model_fields:
            field_type = model_type.model_fields[field].annotation
            assert field_type is not None
            field_path = path + [TypeField(field, None)]
            file_paths.extend(get_file_paths(field_path, field_type))
        return file_paths
    return []


class ExecuteIndex(AbstractExecuteUnit):
    def __init__(self, inputs: list[LensPath], outputs: list[LensPath]):
        super().__init__(inputs, outputs)

    def load_inputs(self, state: State) -> list[Any]:
        return []

    def save_outputs(self, state: State, rtns: list[Any]) -> list[Any]:
        return []

    def commit(self, state: State, values: list[Any]) -> bool:
        # NOTE: this could be done in different
        # execute units for parallelism
        state.index_files([o.to_path() for o in self.outputs])
        return True

    def run(self, args: list[Any]) -> list[Any]:
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
            lens_info = LensInfo.from_path(Catalog, p)
            file_paths = get_file_paths(lens_info.lens_path.path, lens_info.type)
            self.outputs.extend(file_paths)

    def get_name(self) -> str:
        return "index_" + "+".join(o.to_path() for o in self.outputs)

    def get_docs(self) -> str:
        return "Index files in folders " + ", ".join(
            o.to_volume_path() for o in self.outputs
        )

    def get_fn(self) -> Callable:
        return ExecuteIndex.save_outputs

    def get_inputs(self) -> list[LensPath]:
        return []

    def get_outputs(self) -> list[LensPath]:
        return self.outputs

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        return [self]

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        return [ExecuteIndex(inputs=[], outputs=self.outputs)]
