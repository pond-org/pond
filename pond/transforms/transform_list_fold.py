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
from pond.transforms.transform import Transform


# NOTE: this is actually a superset of the functionality
# in transform, so we could use the same unit tests for that part
class TransformListFold(Transform):
    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        fn: Callable,
    ):
        super().__init__(Catalog, input, output, fn, is_list_fold=True)
        self.input_inds = []
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
            raise ValueError(
                f"Transform list fold did not get any inputs with wildcard!"
            )

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        # NOTE: setting output indices is actually not strictly necessary
        return [
            ExecuteTransform(
                inputs=[i.lens_path.clone() for i in self.input_lenses.values()],
                outputs=[o.lens_path for o in self.output_lenses.values()],
                fn=self.fn,
                # input_list_len=length,  # TODO: implement
            )
        ]

    def needs_commit_lock(self) -> bool:
        # TODO: this should depend on if output is writing to the
        # same table
        return True
