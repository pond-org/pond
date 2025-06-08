import warnings
from collections import OrderedDict
from typing import Callable, Type, get_type_hints, get_args, Self, Tuple

from pydantic import BaseModel
from beartype.door import is_subhint
from beartype.roar import BeartypeDoorNonpepException

from pond.state import State
from pond.lens import LensInfo, LensPath, TypeField, get_cleaned_path
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractExecuteUnit,
    ExecuteTransform,
)


# NOTE: this is actually a superset of the functionality
# in transform, so we could use the same unit tests for that part
class TransformList(AbstractExecuteTransform):
    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        fn: Callable,
    ):
        self.fn = fn
        self.inputs = input if isinstance(input, list) else [input]
        self.outputs = output if isinstance(output, list) else [output]

        types = get_type_hints(self.fn)
        try:
            output_types = types.pop("return")
        except KeyError as e:
            raise RuntimeError(f"Transform does not have return type!") from e
        try:
            if is_subhint(output_types, Tuple):
                output_types = list(get_args(output_types))
        except BeartypeDoorNonpepException:
            pass

        if not isinstance(output_types, list):
            output_types = [output_types]

        # Here we need to go through and match all of the inputs
        # and outputs to make sure the list iterations make sense
        # and that we only have one group to iterate over
        self.input_lenses = OrderedDict()
        for (input_name, input_type), i in zip(types.items(), self.inputs):
            if "[:]" in i:
                assert (
                    i.count("[:]") == 1
                ), "List transforms can only iterate over one expansion"
                replace_element = i.find("[:]") + 1
                lens_path = get_cleaned_path(i[: replace_element - 1], "dummy")
                index = len(lens_path.path) - 1
                input_lens = LensInfo.from_path(
                    Catalog, i[:replace_element] + "0" + i[replace_element + 1 :]
                )
                input_lens.set_index(index, -1)
            else:
                input_lens = LensInfo.from_path(Catalog, i)
                index = -1
            try:
                print("Type: ", input_lens.get_type())
                type_checks = is_subhint(input_lens.get_type(), input_type)
                assert (
                    type_checks
                ), f"Input {input_name} of type {input_type} does not agree with catalog entry {i} with type {input_lens.get_type()}"
                print(f"{input_lens.get_type()} checks with {input_type}!")
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))
            self.input_lenses[i] = (input_lens, index)

        self.output_lenses = OrderedDict()
        found = False
        for output_type, o in zip(output_types, self.outputs):
            if "[:]" in o:
                # assert o.endswith(
                #     "[:]"
                # ), "pond does not yet allow iterating over children for outputs"
                # replace_element = len(o) - 2
                assert (
                    o.count("[:]") == 1
                ), "List transforms can only iterate over one expansion"
                replace_element = o.find("[:]") + 1
                lens_path = get_cleaned_path(o[: replace_element - 1], "dummy")
                index = len(lens_path.path) - 1
                output_lens = LensInfo.from_path(
                    Catalog, o[:replace_element] + "0" + o[replace_element + 1 :]
                )
                output_lens.set_index(index, -1)
                found = True
            else:
                output_lens = LensInfo.from_path(Catalog, o)
                index = -1
            try:
                type_checks = is_subhint(output_lens.get_type(), output_type)
                assert (
                    type_checks
                ), f"Output of type {output_type} does not agree with catalog entry {o} with type {output_lens.get_type()}"
                print(f"{output_lens.get_type()} checks with {output_type}!")
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))
            self.output_lenses[o] = (output_lens, index)
        assert found, "Node output must have at least one list output with list inputs"

    def get_name(self) -> str:
        return self.fn.__name__

    def get_docs(self) -> str:
        return self.fn.__doc__

    def get_fn(self) -> Callable:
        return self.fn

    def get_inputs(self) -> list[LensPath]:
        return [i[0].lens_path for i in self.input_lenses.values()]

    def get_outputs(self) -> list[LensPath]:
        return [o[0].lens_path for o in self.output_lenses.values()]

    def get_transforms(self) -> list[Self]:
        return [self]

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        input_lengths = {}
        for name, (input_lens, index) in self.input_lenses.items():
            if index == -1:
                continue
            i = input_lens.lens_path.clone()
            parent_path = LensPath(
                i.path[:index] + [TypeField(i.path[index].name, None)]
            )
            lens = state.lens(parent_path.to_path())
            if lens.exists():
                input_lengths[name] = lens.len()
                continue
            list_index = 0
            while True:
                i.path[index].index = list_index
                lens = state.lens(i.to_path())
                if not lens.exists():
                    break
                list_index += 1
            input_lengths[name] = list_index

        print(input_lengths)
        unique_inputs = set(input_lengths.values())
        assert (
            len(unique_inputs) == 1
        ), f"Input lengths are not the same: {input_lengths}"
        length = unique_inputs.pop()

        for o in self.output_lenses.values():
            # This means we are writing all output values to the same table
            if o[1] != -1 and o[1] == len(o[0].lens_path.path) - 1:
                path = o[0].lens_path.path
                parent_path = LensPath(
                    path[:index] + [TypeField(path[index].name, None)]
                )
                state[parent_path.to_path()] = []

        execute_units = []
        for index in range(0, length):
            inputs = []
            for i in self.input_lenses.values():
                if i[1] != -1:
                    print(i)
                    i[0].set_index(i[1], index)
                    inputs.append(i[0].lens_path.clone())
                    i[0].set_index(i[1], -1)
                else:
                    inputs.append(i[0].lens_path)
            # NOTE: setting output indices is actually not strictly necessary
            outputs = []
            append_outputs = []
            for o in self.output_lenses.values():
                if o[1] != -1:
                    o[0].set_index(o[1], index)
                    outputs.append(o[0].lens_path.clone())
                    if o[1] == len(o[0].lens_path.path) - 1:
                        append_outputs.append(o[0].lens_path.clone())
                    o[0].set_index(o[1], -1)
                else:
                    outputs.append(o[0].lens_path)
            execute_units.append(
                ExecuteTransform(
                    inputs=inputs,
                    outputs=outputs,
                    fn=self.fn,
                    append_outputs=append_outputs,
                )
            )
        return execute_units

    def needs_commit_lock(self) -> bool:
        # TODO: this should depend on if output is writing to the
        # same table
        return True
