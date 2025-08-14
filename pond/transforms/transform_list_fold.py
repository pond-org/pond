from typing import Callable, Type

from pydantic import BaseModel

from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit, ExecuteTransform
from pond.transforms.transform import Transform


# NOTE: this is actually a superset of the functionality
# in transform, so we could use the same unit tests for that part
class TransformListFold(Transform):
    """Transform for aggregating arrays to scalar outputs (many-to-one mapping).

    Extends Transform to handle array inputs with scalar outputs, aggregating
    all array elements into a single result. This transform is automatically
    selected by the @node decorator when input paths contain \"[:]\" but
    output paths do not.

    Example:
        @node(Catalog, \"clouds[:].bounds\", \"global_bounds\")
        def merge_bounds(bounds_list: list[Bounds]) -> Bounds:
            return combine_all_bounds(bounds_list)
        # Creates a TransformListFold instance

    Processing Pattern:
        - Input: clouds[0].bounds, clouds[1].bounds, ... → [bounds0, bounds1, ...]
        - Function called once with entire list
        - Output: Single value written to global_bounds

    Note:
        The function receives a list of all array elements as input.
        Requires at least one wildcard in input paths, no wildcards in output paths.
    """

    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        fn: Callable,
    ):
        """Initialize a TransformListFold with wildcard validation.

        Args:
            Catalog: Pydantic model class defining the data schema.
            input: Input path(s) containing at least one \"[:]\" wildcard.
            output: Output path(s) without wildcards (scalar outputs).
            fn: Function that aggregates array elements to scalar result.

        Raises:
            ValueError: If no wildcards found in input paths.

        Note:
            The is_list_fold=True flag affects type validation in the parent class.
            Function signature should accept list types matching the array element types.
        """
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
        """Create a single execute unit that aggregates all array elements.

        Args:
            state: Pipeline state (not used for list fold transforms).

        Returns:
            List containing a single ExecuteTransform unit that processes
            all array elements and produces scalar output.

        Note:
            Unlike TransformList, this creates only one unit that handles
            the entire array aggregation operation.
        """
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
