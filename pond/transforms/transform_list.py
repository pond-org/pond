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
from typing import Callable, Type

from pydantic import BaseModel

from pond.lens import LensPath, TypeField
from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit, ExecuteTransform
from pond.transforms.transform import Transform


# NOTE: this is actually a superset of the functionality
# in transform, so we could use the same unit tests for that part
class TransformList(Transform):
    """Transform for processing arrays with array output (one-to-one mapping).

    Extends Transform to handle array inputs with array outputs, processing
    each element independently. This transform is automatically selected by
    the @node decorator when both input and output paths contain \"[:]\".

    Example:
        @node(Catalog, \"clouds[:].raw_points\", \"clouds[:].filtered_points\")
        def filter_points(points: list[Point]) -> list[Point]:
            return [p for p in points if p.z > 0]
        # Creates a TransformList instance

    Processing Pattern:
        - Input: clouds[0].raw_points, clouds[1].raw_points, ...
        - Function called once per array element
        - Output: clouds[0].filtered_points, clouds[1].filtered_points, ...

    Attributes:
        input_inds: List of wildcard indices in input paths (should contain -1).
        output_inds: List of wildcard indices in output paths (should contain -1).

    Note:
        Requires at least one wildcard in both input and output paths.
        Creates multiple execute units at runtime based on array length.
    """

    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        fn: Callable,
    ):
        """Initialize a TransformList with wildcard validation.

        Args:
            Catalog: Pydantic model class defining the data schema.
            input: Input path(s) containing at least one \"[:]\" wildcard.
            output: Output path(s) containing at least one \"[:]\" wildcard.
            fn: Function to apply to each array element.

        Raises:
            ValueError: If no wildcards found in inputs or outputs.

        Note:
            The function signature must match the element types, not the array types.
            For example, if input is list[Point], function should accept Point.
        """
        super().__init__(Catalog, input, output, fn)
        self.input_inds = []
        self.output_inds = []
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
            raise ValueError("Transform list did not get any inputs with wildcard!")
        wildcard = False
        for output_lens in self.output_lenses.values():
            try:
                index = next(
                    index
                    for index, v in enumerate(output_lens.lens_path.path)
                    if v.index == -1
                )
                wildcard = True
            except StopIteration:
                index = -1
            self.output_inds.append(index)
        if not wildcard:
            raise ValueError("Transform list did not get any outputs with wildcard!")

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        """Create execute units for each array element.

        Determines the length of input arrays and creates one execute unit
        per array element. All input arrays must have the same length.

        Args:
            state: Pipeline state used to determine array lengths.

        Returns:
            List of ExecuteTransform units, one per array element.

        Raises:
            AssertionError: If input arrays have different lengths.

        Note:
            Array lengths are determined by checking existing data in the catalog.
            If the parent array doesn't exist, iterates through indices to find length.
        """
        input_lengths = {}
        for (name, input_lens), path_index in zip(
            self.input_lenses.items(), self.input_inds
        ):
            if path_index == -1:
                continue
            i = input_lens.lens_path.clone()
            parent_path = LensPath(
                i.path[:path_index] + [TypeField(i.path[path_index].name, None)]
            )
            lens = state.lens(parent_path.to_path())
            if lens.exists():
                input_lengths[name] = lens.len()
                continue
            list_index = 0
            while True:
                i.path[path_index].index = list_index
                lens = state.lens(i.to_path())
                if not lens.exists():
                    break
                list_index += 1
            input_lengths[name] = list_index

        unique_inputs = set(input_lengths.values())
        assert len(unique_inputs) == 1, (
            f"Input lengths are not the same: {input_lengths}"
        )
        length = unique_inputs.pop()

        for o, path_index in zip(self.output_lenses.values(), self.output_inds):
            # This means we are writing all output values to the same table
            if path_index != -1 and path_index == len(o.lens_path.path) - 1:
                path = o.lens_path.path
                parent_path = LensPath(
                    path[:path_index] + [TypeField(path[path_index].name, None)]
                )
                state[parent_path.to_path()] = []

        execute_units = []
        for index in range(0, length):
            inputs = []
            for il, path_index in zip(self.input_lenses.values(), self.input_inds):
                if path_index != -1:
                    il.set_index(path_index, index)
                    inputs.append(il.lens_path.clone())
                    il.set_index(path_index, -1)
                else:
                    inputs.append(il.lens_path)
            # NOTE: setting output indices is actually not strictly necessary
            outputs = []
            append_outputs = []
            for o, path_index in zip(self.output_lenses.values(), self.output_inds):
                if path_index != -1:
                    o.set_index(path_index, index)
                    outputs.append(o.lens_path.clone())
                    if path_index == len(o.lens_path.path) - 1:
                        append_outputs.append(o.lens_path.clone())
                    o.set_index(path_index, -1)
                else:
                    outputs.append(o.lens_path)
            execute_units.append(
                ExecuteTransform(
                    inputs=inputs,
                    outputs=outputs,
                    fn=self.fn,
                    append_outputs=append_outputs,
                )
            )
        # NOTE: mypy should really accept this?
        return execute_units  # type: ignore

    def needs_commit_lock(self) -> bool:
        # TODO: this should depend on if output is writing to the
        # same table
        return True
