import warnings
from collections import OrderedDict
from typing import Callable, Type, get_type_hints, get_args, Self, Tuple

from pydantic import BaseModel
from beartype.door import is_subhint
from beartype.roar import BeartypeDoorNonpepException

from pond.state import State
from pond.lens import LensInfo, LensPath
from pond.abstract_transform import (
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
                lens = LensInfo.from_path(Catalog, i[: replace_element - 1])
                index = len(lens.lens_path.path) - 1
                input_lens = LensInfo.from_path(
                    Catalog, i[:replace_element] + "0" + i[replace_element + 1 :]
                )
            else:
                lens = LensInfo.from_path(Catalog, i)
                index = -1
                input_lens = None
            try:
                lens_type = (
                    lens.get_type() if input_lens is None else input_lens.get_type()
                )
                type_checks = is_subhint(lens_type, input_type)
                assert (
                    type_checks
                ), f"Input {input_name} of type {input_type} does not agree with catalog entry {i} with type {lens_type}"
                print(f"{lens_type} checks with {input_type}!")
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))
            self.input_lenses[i] = (lens, index, input_lens)

        self.output_lenses = OrderedDict()
        found = False
        for output_type, o in zip(output_types, self.outputs):
            if "[:]" in o:
                assert o.endswith(
                    "[:]"
                ), "pond does not yet allow iterating over children for outputs"
                replace_element = len(o) - 2
                lens = LensInfo.from_path(Catalog, o[: replace_element - 1])
                index = len(lens.lens_path.path) - 1
                output_lens = LensInfo.from_path(
                    Catalog, o[:replace_element] + "0" + o[replace_element + 1 :]
                )
                found = True
            else:
                lens = LensInfo.from_path(Catalog, o)
                index = -1
                output_lens = None
            try:
                lens_type = (
                    lens.get_type() if output_lens is None else output_lens.get_type()
                )
                type_checks = is_subhint(lens_type, output_type)
                assert (
                    type_checks
                ), f"Input of type {output_type} does not agree with catalog entry {o} with type {lens_type}"
                print(f"{lens_type} checks with {output_type}!")
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))
            self.output_lenses[o] = (lens, index, output_lens)
        assert found, "Node output must have at least one list output with list inputs"

    def get_name(self) -> str:
        return self.fn.__name__

    def get_inputs(self) -> list[LensPath]:
        return [i[0].lens_path for i in self.input_lenses.values()]

    def get_outputs(self) -> list[LensPath]:
        return [o[0].lens_path for o in self.output_lenses.values()]

    def get_transforms(self) -> list[Self]:
        return [self]

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        input_lengths = {
            name: state.lens(input_lens.lens_path.to_path()).len()
            for name, (input_lens, _, lens) in self.input_lenses.items()
            if lens is not None
        }
        print(input_lengths)
        unique_inputs = set(input_lengths.values())
        assert (
            len(unique_inputs) == 1
        ), f"Input lengths are not the same: {input_lengths}"
        length = unique_inputs.pop()

        for o in self.output_lenses.values():
            if o[2] is not None:
                state[o[0].lens_path.to_path()] = []

        execute_units = []
        for index in range(0, length):
            inputs = []
            for i in self.input_lenses.values():
                if i[2] is not None:
                    print(i)
                    i[2].set_index(i[1], index)
                    inputs.append(i[2].lens_path.clone())
                else:
                    inputs.append(i[0].lens_path)
            # NOTE: setting output indices is actually not strictly necessary
            outputs = []
            append_outputs = []
            for o in self.output_lenses.values():
                if o[2] is not None:
                    o[2].set_index(o[1], index)
                    append_outputs.append(o[2].lens_path.clone())
                    outputs.append(o[2].lens_path.clone())
                else:
                    outputs.append(o[0])
            execute_units.append(
                ExecuteTransform(
                    inputs=inputs,
                    outputs=outputs,
                    fn=self.fn,
                    append_outputs=append_outputs,
                )
            )
        return execute_units
