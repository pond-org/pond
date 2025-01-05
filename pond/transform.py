import os
import warnings
from collections import OrderedDict
from typing import Callable, Type, get_type_hints, get_args, Any, Tuple

from beartype.door import is_subhint
from beartype.roar import BeartypeDoorNonpepException

from pond.lens import Lens

# from fbs_generated import Catalog as GenCatalog


class Transform:
    # TODO: make inputs/outputs work with dicts also
    def __init__(
        self,
        fn: Callable,
        input: list[str] | str,
        output: list[str] | str,
        db_path: os.PathLike,
    ):
        self.fn = fn
        self.input_lenses = OrderedDict(
            (i, Lens(self.Catalog, i, db_path=db_path))
            for i in (input if isinstance(input, list) else [input])
        )
        self.output_lenses = OrderedDict(
            (o, Lens(self.Catalog, o, db_path=db_path))
            for o in (output if isinstance(output, list) else [output])
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

        # input_types = list(types.values())

        for (input_name, input_type), input_lens in zip(
            types.items(), self.input_lenses
        ):
            try:
                type_checks = is_subhint(input_lens.get_type(), input_type)
                assert (
                    type_checks
                ), f"Input {input_name} of type {input_type} does not agree with catalog entry {input_lens.path} with type {input_lens.get_type()}"
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))

        for output_type, output_lens in zip(output_types, self.output_lenses):
            try:
                type_checks = is_subhint(output_lens.get_type(), output_type)
                assert (
                    type_checks
                ), f"Output of type {output_type} does not agree with catalog entry {output_lens.path} with type {output_lens.get_type()}"
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))

    def __call__(self) -> None:
        args = [i.get() for i in self.input_lenses.values()]
        rtns = self.fn(*args)
        for rtn, o in zip(rtns, self.output_lenses.values()):
            o.set(rtn)
