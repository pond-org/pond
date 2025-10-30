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
"""Agent for array-to-scalar aggregation using pydantic-ai.

Provides AgentListFold class for aggregating array inputs to scalar outputs
using LLMs. Similar to TransformListFold but uses pydantic-ai agents.
"""
from collections import OrderedDict
from typing import Type

from pydantic import BaseModel

from pond.agents.abstract_agent import AbstractAgent, ExecuteAgent
from pond.agents.type_builder import build_input_type, build_output_type
from pond.lens import LensInfo
from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit


class AgentListFold(AbstractAgent):
    """Agent transform for aggregating arrays to scalar outputs (many-to-one mapping).

    Uses pydantic-ai to aggregate array inputs into scalar outputs using an LLM.
    This agent is automatically selected by the @agent_node decorator when input
    paths contain "[:] " but output paths do not.

    Example:
        @agent_node(
            Catalog,
            "reviews[:].text",
            "overall_summary",
            model="openai:gpt-4o",
            instructions="Summarize all reviews into a single cohesive summary."
        )
        # Creates an AgentListFold instance

    Processing Pattern:
        - Input: reviews[0].text, reviews[1].text, ... → [text0, text1, ...]
        - Agent called once with all elements as a list
        - Output: Single value written to overall_summary

    Note:
        The agent receives list[T] types for array inputs and produces
        scalar outputs. Useful for aggregation, summarization, and analysis tasks.
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
        """Initialize an AgentListFold with wildcard validation.

        Args:
            Catalog: Pydantic model class defining the data schema.
            input: Input path(s) containing at least one "[:] " wildcard.
            output: Output path(s) without wildcards (scalar outputs).
            model: Model identifier (e.g., "openai:gpt-4o").
            instructions: System prompt/instructions for the agent.
            prompt: Optional user prompt template.
            input_descriptions: Optional descriptions for input fields.
            output_descriptions: Optional descriptions for output fields.

        Raises:
            ValueError: If no wildcards found in input paths.

        Note:
            The structured input types will be list[T] for array inputs,
            where T is the element type from the catalog schema.
        """
        self.model = model
        self.instructions = instructions
        self.prompt = prompt

        # Parse inputs and outputs
        inputs = input if isinstance(input, list) else [input]
        outputs = output if isinstance(output, list) else [output]

        # Build lens info for type resolution
        self.input_lenses = OrderedDict(
            (i, LensInfo.from_path(Catalog, i)) for i in inputs
        )
        self.output_lenses = OrderedDict(
            (o, LensInfo.from_path(Catalog, o)) for o in outputs
        )

        # Validate that at least one input has a wildcard
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
                "AgentListFold did not get any inputs with wildcard!"
            )

        # Extract types and names
        # For list fold, input types should be list[T] where T is the element type
        input_types = []
        for input_lens in self.input_lenses.values():
            base_type = input_lens.get_type()
            # Check if this input has a wildcard - if so, wrap in list
            has_wildcard = any(v.index == -1 for v in input_lens.lens_path.path)
            if has_wildcard:
                input_types.append(list[base_type])  # type: ignore
            else:
                input_types.append(base_type)

        input_names = [path.split(".")[-1].replace("[:]", "_list") for path in inputs]
        output_types = [lens.get_type() for lens in self.output_lenses.values()]
        output_names = [path.split(".")[-1] for path in outputs]

        # Build structured input/output types
        self.input_type = build_input_type(input_types, input_names, input_descriptions)
        self.output_type = build_output_type(
            output_types, output_names, output_descriptions
        )

        # Store names for later use
        self.input_names = input_names
        self.output_names = output_names

    def get_name(self) -> str:
        """Get the name of this agent.

        Returns:
            String name combining model and agent type.
        """
        return f"AgentListFold({self.model})"

    def get_docs(self) -> str:
        """Get the documentation string for this agent.

        Returns:
            The system instructions as documentation.
        """
        return self.instructions

    def get_fn(self) -> str:
        """Get the model identifier for this agent.

        Returns:
            The model identifier string.

        Note:
            Returns model identifier instead of agent instance for serialization.
        """
        return self.model

    def get_inputs(self) -> list["LensPath"]:  # type: ignore # noqa: F821
        """Get the input paths for this agent.

        Returns:
            List of LensPath objects specifying input data locations.
        """
        from pond.lens import LensPath

        return [lens.lens_path for lens in self.input_lenses.values()]

    def get_outputs(self) -> list["LensPath"]:  # type: ignore # noqa: F821
        """Get the output paths for this agent.

        Returns:
            List of LensPath objects specifying output data locations.
        """
        from pond.lens import LensPath

        return [lens.lens_path for lens in self.output_lenses.values()]

    def get_transforms(self) -> list[AbstractAgent]:
        """Get the executable agent transforms.

        Returns:
            List containing this agent (for compatibility with pipeline).
        """
        return [self]

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        """Create a single execute unit that aggregates all array elements.

        Args:
            state: Pipeline state (not used for list fold agents).

        Returns:
            List containing a single ExecuteAgent unit that processes
            all array elements and produces scalar output.

        Note:
            Unlike AgentList, this creates only one unit that handles
            the entire array aggregation operation.
        """
        return [
            ExecuteAgent(
                inputs=[i.lens_path.clone() for i in self.input_lenses.values()],
                outputs=[o.lens_path for o in self.output_lenses.values()],
                model=self.model,
                instructions=self.instructions,
                input_type=self.input_type,
                output_type=self.output_type,
                input_names=self.input_names,
                output_names=self.output_names,
                prompt=self.prompt,
            )
        ]

    def needs_commit_lock(self) -> bool:
        """Check if this agent requires exclusive commit access.

        Returns:
            True, as aggregation agents may write to shared tables.
        """
        return True
