import os
import warnings
from collections import OrderedDict
from typing import Callable, Type, get_type_hints, get_args, Self, Tuple

from pydantic import BaseModel
from beartype.door import is_subhint
from beartype.roar import BeartypeDoorNonpepException

from pond.state import State
from pond.lens import LensInfo, LensPath
from pond.abstract_transform import (
    # AbstractTransform,
    AbstractExecuteTransform,
    AbstractExecuteUnit,
    ExecuteTransform,
)

# from fbs_generated import Catalog as GenCatalog


class Transform(AbstractExecuteTransform):
    # TODO: make inputs/outputs work with dicts also
    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        fn: Callable,
    ):
        self.fn = fn
        inputs = input if isinstance(input, list) else [input]
        outputs = output if isinstance(output, list) else [output]
        self.input_lenses = OrderedDict(
            (i, LensInfo.from_path(Catalog, i)) for i in inputs
        )
        self.output_lenses = OrderedDict(
            (o, LensInfo.from_path(Catalog, o)) for o in outputs
        )
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

        for (input_name, input_type), (input_field_name, input_lens) in zip(
            types.items(), self.input_lenses.items(), strict=True
        ):
            try:
                print("Type: ", input_lens.get_type())
                type_checks = is_subhint(input_lens.get_type(), input_type)
                assert (
                    type_checks
                ), f"Input {input_name} of type {input_type} does not agree with catalog entry {input_field_name} with type {input_lens.get_type()}"
                print(f"{input_lens.get_type()} checks with {input_type}!")
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))

        for output_type, (output_field_name, output_lens) in zip(
            output_types, self.output_lenses.items(), strict=True
        ):
            try:
                type_checks = is_subhint(output_lens.get_type(), output_type)
                assert (
                    type_checks
                ), f"Output of type {output_type} does not agree with catalog entry {output_field_name} with type {output_lens.get_type()}"
                print(f"{output_lens.get_type()} checks with {output_type}!")
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))

    def get_name(self) -> str:
        return self.fn.__name__

    def get_inputs(self) -> list[LensPath]:
        return [i.lens_path for i in self.input_lenses.values()]

    def get_outputs(self) -> list[LensPath]:
        return [o.lens_path for o in self.output_lenses.values()]

    def get_transforms(self) -> list[Self]:
        return [self]

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        return [
            ExecuteTransform(
                inputs=[i.lens_path for i in self.input_lenses.values()],
                outputs=[o.lens_path for o in self.output_lenses.values()],
                fn=self.fn,
            )
        ]
