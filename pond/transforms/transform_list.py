from typing import Callable, Type

from pydantic import BaseModel

from pond.lens import LensPath, TypeField
from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit, ExecuteTransform
from pond.transforms.transform import Transform


# NOTE: this is actually a superset of the functionality
# in transform, so we could use the same unit tests for that part
class TransformList(Transform):
    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        fn: Callable,
    ):
        super().__init__(Catalog, input, output, fn)
        self.input_inds = []
        self.output_inds = []
        wildcard = False
        for input_lens in self.input_lenses.values():
            try:
                index = next(
                    index
                    for index, v in enumerate(input_lens.lens_path.path)
                    if v.index == -1
                )
                wildcard = True
            except StopIteration:
                index = -1
            self.input_inds.append(index)
        if not wildcard:
            raise ValueError("Transform list did not get any inputs with wildcard!")
        wildcard = False
        for output_lens in self.output_lenses.values():
            try:
                index = next(
                    index
                    for index, v in enumerate(output_lens.lens_path.path)
                    if v.index == -1
                )
                wildcard = True
            except StopIteration:
                index = -1
            self.output_inds.append(index)
        if not wildcard:
            raise ValueError("Transform list did not get any outputs with wildcard!")

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        input_lengths = {}
        for (name, input_lens), path_index in zip(
            self.input_lenses.items(), self.input_inds
        ):
            if path_index == -1:
                continue
            i = input_lens.lens_path.clone()
            parent_path = LensPath(
                i.path[:path_index] + [TypeField(i.path[path_index].name, None)]
            )
            lens = state.lens(parent_path.to_path())
            if lens.exists():
                input_lengths[name] = lens.len()
                continue
            list_index = 0
            while True:
                i.path[path_index].index = list_index
                lens = state.lens(i.to_path())
                if not lens.exists():
                    break
                list_index += 1
            input_lengths[name] = list_index

        unique_inputs = set(input_lengths.values())
        assert len(unique_inputs) == 1, (
            f"Input lengths are not the same: {input_lengths}"
        )
        length = unique_inputs.pop()

        for o, path_index in zip(self.output_lenses.values(), self.output_inds):
            # This means we are writing all output values to the same table
            if path_index != -1 and path_index == len(o.lens_path.path) - 1:
                path = o.lens_path.path
                parent_path = LensPath(
                    path[:path_index] + [TypeField(path[path_index].name, None)]
                )
                state[parent_path.to_path()] = []

        execute_units = []
        for index in range(0, length):
            inputs = []
            for il, path_index in zip(self.input_lenses.values(), self.input_inds):
                if path_index != -1:
                    il.set_index(path_index, index)
                    inputs.append(il.lens_path.clone())
                    il.set_index(path_index, -1)
                else:
                    inputs.append(il.lens_path)
            # NOTE: setting output indices is actually not strictly necessary
            outputs = []
            append_outputs = []
            for o, path_index in zip(self.output_lenses.values(), self.output_inds):
                if path_index != -1:
                    o.set_index(path_index, index)
                    outputs.append(o.lens_path.clone())
                    if path_index == len(o.lens_path.path) - 1:
                        append_outputs.append(o.lens_path.clone())
                    o.set_index(path_index, -1)
                else:
                    outputs.append(o.lens_path)
            execute_units.append(
                ExecuteTransform(
                    inputs=inputs,
                    outputs=outputs,
                    fn=self.fn,
                    append_outputs=append_outputs,
                )
            )
        # NOTE: mypy should really accept this?
        return execute_units  # type: ignore

    def needs_commit_lock(self) -> bool:
        # TODO: this should depend on if output is writing to the
        # same table
        return True
