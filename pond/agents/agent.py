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
"""Agent for scalar input/output processing using pydantic-ai.

Provides Agent class for scalar-to-scalar LLM-powered transformations.
Similar to Transform but uses pydantic-ai agents instead of functions.
"""
from collections import OrderedDict
from typing import Type

from pydantic import BaseModel

from pond.agents.abstract_agent import AbstractAgent, ExecuteAgent
from pond.agents.type_builder import build_input_type, build_output_type
from pond.lens import LensInfo, LensPath
from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit


class Agent(AbstractAgent):
    """Agent transform for scalar input/output processing.

    Uses pydantic-ai to perform LLM-powered transformations where inputs
    and outputs are scalar values (not arrays). Automatically builds
    structured Pydantic types from catalog schemas.

    This agent is selected by the @agent_node decorator when neither
    input nor output paths contain array wildcards "[:] ".

    Example:
        @agent_node(
            Catalog,
            "customer_query",
            "response",
            model="anthropic:claude-sonnet-4-0",
            instructions="Provide helpful customer support responses."
        )
        # Creates an Agent instance

    Attributes:
        model: Model identifier string (e.g., "openai:gpt-4o").
        instructions: System prompt for the agent.
        prompt: Optional user prompt template.
        input_lenses: Ordered mapping of input paths to LensInfo objects.
        output_lenses: Ordered mapping of output paths to LensInfo objects.
        agent: The pydantic-ai Agent instance.
        input_type: Dynamically built Pydantic model for inputs.
        output_type: Dynamically built Pydantic model for outputs.

    Note:
        The agent automatically validates input/output types against
        catalog schemas and creates appropriate structured types for the LLM.
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
        """Initialize an Agent with type building and validation.

        Args:
            Catalog: Pydantic model class defining the data schema.
            input: Input path(s) as string or list of strings. Should not
                contain array wildcards for basic Agent.
            output: Output path(s) as string or list of strings. Should not
                contain array wildcards for basic Agent.
            model: Model identifier (e.g., "openai:gpt-4o",
                "anthropic:claude-sonnet-4-0", "google-gla:gemini-1.5-flash").
            instructions: System prompt/instructions for the agent.
            prompt: Optional user prompt template. If not provided, the
                structured input will be converted to a string prompt.
            input_descriptions: Optional descriptions for each input field,
                used to help the LLM understand the inputs.
            output_descriptions: Optional descriptions for each output field,
                used to guide the LLM's structured output generation.

        Note:
            Dynamically builds Pydantic types based on catalog schema types
            at the specified input/output paths.
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

        # Extract types and names
        input_types = [lens.get_type() for lens in self.input_lenses.values()]
        input_names = [path.split(".")[-1] for path in inputs]  # Use last component as field name
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
            String name combining model and inputs/outputs.
        """
        return f"Agent({self.model})"

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

    def get_inputs(self) -> list[LensPath]:
        """Get the input paths for this agent.

        Returns:
            List of LensPath objects specifying input data locations.
        """
        return [lens.lens_path for lens in self.input_lenses.values()]

    def get_outputs(self) -> list[LensPath]:
        """Get the output paths for this agent.

        Returns:
            List of LensPath objects specifying output data locations.
        """
        return [lens.lens_path for lens in self.output_lenses.values()]

    def get_transforms(self) -> list[AbstractAgent]:
        """Get the executable agent transforms.

        Returns:
            List containing this agent (for compatibility with pipeline).
        """
        return [self]

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        """Get the executable units for this agent.

        Args:
            state: Pipeline state (not used for basic agents).

        Returns:
            List containing a single ExecuteAgent unit.

        Note:
            Basic agents create exactly one execute unit.
        """
        return [
            ExecuteAgent(
                inputs=self.get_inputs(),
                outputs=self.get_outputs(),
                model=self.model,
                instructions=self.instructions,
                input_type=self.input_type,
                output_type=self.output_type,
                input_names=self.input_names,
                output_names=self.output_names,
                prompt=self.prompt,
            )
        ]
