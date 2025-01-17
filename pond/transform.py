import os
import warnings
from collections import OrderedDict
from typing import Callable, Type, get_type_hints, get_args, Any, Tuple

from pydantic import BaseModel
from beartype.door import is_subhint
from beartype.roar import BeartypeDoorNonpepException

from pond.abstract_catalog import AbstractCatalog
from pond.lens import Lens

# from fbs_generated import Catalog as GenCatalog


class Transform:
    # TODO: make inputs/outputs work with dicts also
    def __init__(
        self,
        fn: Callable,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        # db_path: os.PathLike,
        catalog: AbstractCatalog,
    ):
        self.fn = fn
        self.input_lenses = OrderedDict(
            (i, Lens(Catalog, i, catalog))  # , db_path=db_path))
            for i in (input if isinstance(input, list) else [input])
        )
        self.output_lenses = OrderedDict(
            (o, Lens(Catalog, o, catalog))  # , db_path=db_path))
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

        for (input_name, input_type), (input_field_name, input_lens) in zip(
            types.items(), self.input_lenses.items(), strict=True
        ):
            try:
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

    def __call__(self) -> None:
        args = [i.get() for i in self.input_lenses.values()]
        rtns = self.fn(*args)
        if isinstance(rtns, tuple) and len(self.output_lenses) > 1:
            rtns_list = list(rtns)
        else:
            rtns_list = [rtns]
        for rtn, o in zip(rtns_list, self.output_lenses.values()):
            o.set(rtn)
