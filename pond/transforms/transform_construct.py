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
from typing import Any, Callable, Type, get_origin

from pydantic import BaseModel

from pond.field import File
from pond.lens import LensInfo, LensPath, TypeField
from pond.state import State
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractExecuteUnit,
)


class ExecuteConstruct(AbstractExecuteUnit):
    def __init__(self, inputs: list[LensPath], outputs: list[LensPath]):
        super().__init__(inputs, outputs)

    def load_inputs(self, state: State) -> list[Any]:
        return []

    def save_outputs(self, state: State, rtns: list[Any]) -> list[Any]:
        return []

    def commit(self, state: State, values: list[Any]) -> bool:
        return True

    def run(self, args: list[Any]) -> list[Any]:
        return []


class TransformConstruct(AbstractExecuteTransform):
    def __init__(
        self,
        Catalog: Type[BaseModel],
        path: str = "",
    ):
        self.lens_info = LensInfo.from_path(Catalog, path)
        lens_path = self.lens_info.lens_path
        self.outputs = [lens_path]
        if issubclass(self.lens_info.type, File):
            self.inputs = list(self.outputs)
        elif issubclass(self.lens_info.type, BaseModel):
            self.inputs = [
                LensPath(lens_path.path + [TypeField(field, None)])
                for field in self.lens_info.type.model_fields
            ]
        elif get_origin(self.lens_info.type) is list:
            # TODO: maybe something else makes more sense here
            self.inputs = list(self.outputs)
        else:
            self.inputs = list(self.outputs)

    def get_name(self) -> str:
        return "construct_" + "+".join(o.to_path() for o in self.outputs)

    def get_docs(self) -> str:
        return f"Construct object {self.lens_info.lens_path} from parts"

    def get_fn(self) -> Callable:
        return TransformConstruct.__init__

    def get_inputs(self) -> list[LensPath]:
        return self.inputs

    def get_outputs(self) -> list[LensPath]:
        return self.outputs

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        return [self]

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        # NOTE: do we even need to return something here??
        # TODO: checl uses of execute units to see if we can remove empty
        return [ExecuteConstruct(inputs=[], outputs=self.outputs)]
