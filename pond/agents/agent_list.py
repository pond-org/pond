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
"""Agent for array-to-array processing using pydantic-ai.

Provides AgentList class for array element-wise LLM-powered transformations.
Similar to TransformList but uses pydantic-ai agents instead of functions.
"""
from typing import Type

from pydantic import BaseModel

from pond.agents.abstract_agent import ExecuteAgent
from pond.agents.agent import Agent
from pond.lens import LensPath, TypeField
from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit


class AgentList(Agent):
    """Agent transform for processing arrays with array output (one-to-one mapping).

    Extends Agent to handle array inputs with array outputs, processing each
    element independently using an LLM. This agent is automatically selected
    by the @agent_node decorator when both input and output paths contain "[:] ".

    Example:
        @agent_node(
            Catalog,
            "reviews[:].text",
            "reviews[:].sentiment",
            model="openai:gpt-4o",
            instructions="Analyze the sentiment of the review text."
        )
        # Creates an AgentList instance

    Processing Pattern:
        - Input: reviews[0].text, reviews[1].text, ...
        - Agent called once per array element
        - Output: reviews[0].sentiment, reviews[1].sentiment, ...

    Attributes:
        input_inds: List of wildcard indices in input paths (contains -1 for wildcards).
        output_inds: List of wildcard indices in output paths (contains -1 for wildcards).

    Note:
        Requires at least one wildcard in both input and output paths.
        Creates multiple execute units at runtime based on array length.
        Each element is processed independently by the LLM.
    """

    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
        model: str,
        instructions: str,
        prompt: str | None = None,
        input_descriptions: list[str] | None = None,
        output_descriptions: list[str] | None = None,
    ):
        """Initialize an AgentList with wildcard validation.

        Args:
            Catalog: Pydantic model class defining the data schema.
            input: Input path(s) containing at least one "[:] " wildcard.
            output: Output path(s) containing at least one "[:] " wildcard.
            model: Model identifier (e.g., "openai:gpt-4o").
            instructions: System prompt/instructions for the agent.
            prompt: Optional user prompt template.
            input_descriptions: Optional descriptions for input fields.
            output_descriptions: Optional descriptions for output fields.

        Raises:
            ValueError: If no wildcards found in inputs or outputs.

        Note:
            The structured input/output types match the element types,
            not the array types. The agent processes one element at a time.
        """
        super().__init__(
            Catalog,
            input,
            output,
            model,
            instructions,
            prompt,
            input_descriptions,
            output_descriptions,
        )

        # Find wildcard indices in inputs
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
            raise ValueError("AgentList did not get any inputs with wildcard!")

        # Find wildcard indices in outputs
        self.output_inds = []
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
            raise ValueError("AgentList did not get any outputs with wildcard!")

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        """Create execute units for each array element.

        Determines the length of input arrays and creates one execute unit
        per array element. All input arrays must have the same length.

        Args:
            state: Pipeline state used to determine array lengths.

        Returns:
            List of ExecuteAgent units, one per array element.

        Raises:
            AssertionError: If input arrays have different lengths.

        Note:
            Array lengths are determined by checking existing data in the catalog.
            Each element is processed independently by a separate agent execution.
        """
        # Determine array lengths from inputs
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

        # Verify all arrays have the same length
        unique_inputs = set(input_lengths.values())
        assert len(unique_inputs) == 1, (
            f"Input lengths are not the same: {input_lengths}"
        )
        length = unique_inputs.pop()

        # Initialize output arrays if writing to tables
        for o, path_index in zip(self.output_lenses.values(), self.output_inds):
            # This means we are writing all output values to the same table
            if path_index != -1 and path_index == len(o.lens_path.path) - 1:
                path = o.lens_path.path
                parent_path = LensPath(
                    path[:path_index] + [TypeField(path[path_index].name, None)]
                )
                state[parent_path.to_path()] = []

        # Create execute units for each array element
        execute_units = []
        for index in range(0, length):
            # Build input paths with specific indices
            inputs = []
            for il, path_index in zip(self.input_lenses.values(), self.input_inds):
                if path_index != -1:
                    il.set_index(path_index, index)
                    inputs.append(il.lens_path.clone())
                    il.set_index(path_index, -1)
                else:
                    inputs.append(il.lens_path)

            # Build output paths with specific indices
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
                ExecuteAgent(
                    inputs=inputs,
                    outputs=outputs,
                    model=self.model,
                    instructions=self.instructions,
                    input_type=self.input_type,
                    output_type=self.output_type,
                    input_names=self.input_names,
                    output_names=self.output_names,
                    prompt=self.prompt,
                    append_outputs=append_outputs,
                )
            )

        return execute_units  # type: ignore

    def needs_commit_lock(self) -> bool:
        """Check if this agent requires exclusive commit access.

        Returns:
            True, as array agents write to shared output tables.

        Note:
            AgentList always needs a commit lock to prevent concurrent
            modification conflicts when multiple units append to the same table.
        """
        return True
