from typing import Callable, Type

from pydantic import BaseModel

from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit, ExecuteTransform
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
        wildcard = False
        for input_lens in self.input_lenses.values():
            try:
                next(
                    index
                    for index, v in enumerate(input_lens.lens_path.path)
                    if v.index == -1
                )
                wildcard = True
                break
            except StopIteration:
                pass
        if not wildcard:
            raise ValueError(
                "Transform list fold did not get any inputs with wildcard!"
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
