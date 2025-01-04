import os
from collections import OrderedDict
from typing import Callable, Type, get_type_hints, Any

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

    def __call__(self) -> None:
        args = [i.get() for i in self.input_lenses.values()]
        rtns = self.fn(*args)
        for rtn, o in zip(rtns, self.output_lenses.values()):
            o.set(rtn)
