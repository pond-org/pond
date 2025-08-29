# Copyright 2025 Nils Bore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from collections import OrderedDict
from typing import Callable, Tuple, Type, get_args, get_type_hints

from beartype.door import is_subhint
from beartype.roar import BeartypeDoorNonpepException
from pydantic import BaseModel

from pond.lens import LensInfo, LensPath
from pond.state import State
from pond.transforms.abstract_transform import (  # AbstractTransform,
    AbstractExecuteTransform,
    AbstractExecuteUnit,
    ExecuteTransform,
)

# from fbs_generated import Catalog as GenCatalog


class Transform(AbstractExecuteTransform):
    """Basic transform for scalar input/output processing.

    Handles simple one-to-one data transformations where inputs and outputs
    are scalar values (not arrays). Performs type validation between the
    function signature and catalog schema paths.

    This transform is automatically selected by the @node decorator when
    neither input nor output paths contain array wildcards \"[:]\".

    Example:
        @node(Catalog, \"params.resolution\", \"grid.cell_size\")
        def scale_resolution(res: float) -> float:
            return res * 2.0
        # Creates a Transform instance

    Attributes:
        fn: The wrapped function to execute.
        input_lenses: Ordered mapping of input path strings to LensInfo objects.
        output_lenses: Ordered mapping of output path strings to LensInfo objects.

    Note:
        Type validation ensures function parameter types match catalog schema types.
        The is_list_fold parameter is used internally for type checking context.
    """

    # TODO: make inputs/outputs work with dicts also
    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        fn: Callable,
        is_list_fold: bool = False,
    ):
        """Initialize a Transform with type validation.

        Args:
            Catalog: Pydantic model class defining the data schema.
            input: Input path(s) as string or list of strings. Should not
                contain array wildcards for basic Transform.
            output: Output path(s) as string or list of strings. Should not
                contain array wildcards for basic Transform.
            fn: Function to wrap. Must have type annotations that match
                the data types at the specified input/output paths.
            is_list_fold: Internal flag for type checking context when
                used in list fold operations.

        Raises:
            RuntimeError: If function lacks return type annotation.
            AssertionError: If function parameter types don't match catalog schema.

        Note:
            Performs comprehensive type validation using beartype to ensure
            function signature compatibility with catalog schema types.
        """
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
            raise RuntimeError("Transform does not have return type!") from e
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
                wildcard_index = next(
                    index
                    for index, v in enumerate(input_lens.lens_path.path)
                    if v.index == -1
                )
            except StopIteration:
                wildcard_index = -1
            try:
                input_lens_type = input_lens.get_type()
                if is_list_fold and wildcard_index != -1:
                    input_lens_type = list[input_lens_type]  # type: ignore
                type_checks = is_subhint(input_lens_type, input_type)
                assert type_checks, (
                    f"Input {input_name} of type {input_type} does not agree with catalog entry {input_field_name} with type {input_lens.get_type()}"
                )
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))

        for output_type, (output_field_name, output_lens) in zip(
            output_types, self.output_lenses.items(), strict=True
        ):
            try:
                type_checks = is_subhint(output_lens.get_type(), output_type)
                assert type_checks, (
                    f"Output of type {output_type} does not agree with catalog entry {output_field_name} with type {output_lens.get_type()}"
                )
            except BeartypeDoorNonpepException as m:
                warnings.warn(str(m))

    def get_name(self) -> str:
        """Get the name of the wrapped function.

        Returns:
            The __name__ attribute of the wrapped function.
        """
        return self.fn.__name__

    def get_docs(self) -> str:
        """Get the documentation string of the wrapped function.

        Returns:
            The __doc__ attribute of the wrapped function, or empty string if None.
        """
        return self.fn.__doc__ if self.fn.__doc__ is not None else ""

    def get_fn(self) -> Callable:
        """Get the wrapped function.

        Returns:
            The callable function that implements the transform logic.
        """
        return self.fn

    def get_inputs(self) -> list[LensPath]:
        """Get the input paths for this transform.

        Returns:
            List of LensPath objects extracted from the input lenses.
        """
        return [i.lens_path for i in self.input_lenses.values()]

    def get_outputs(self) -> list[LensPath]:
        """Get the output paths for this transform.

        Returns:
            List of LensPath objects extracted from the output lenses.
        """
        return [o.lens_path for o in self.output_lenses.values()]

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        """Get the list of transforms contained in this transform.

        Returns:
            List containing only this transform (since Transform is atomic).
        """
        return [self]

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        """Create the executable units for this transform.

        Args:
            state: Pipeline state (not used for basic transforms).

        Returns:
            List containing a single ExecuteTransform unit wrapping the function.

        Note:
            Basic transforms always create exactly one execute unit since they
            handle scalar input/output without array expansion.
        """
        return [
            ExecuteTransform(
                inputs=[i.lens_path for i in self.input_lenses.values()],
                outputs=[o.lens_path for o in self.output_lenses.values()],
                fn=self.fn,
            )
        ]
