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
"""Abstract base classes for pydantic-ai agent transforms.

This module provides the foundation for agent-based transforms that use
pydantic-ai for LLM-powered data transformations. Agents parallel the
transform system but use LLMs instead of user-defined functions.
"""
from abc import abstractmethod
from typing import Any, Type

import dill  # type: ignore
from pydantic import BaseModel
from pydantic_ai import Agent

from pond.lens import LensPath
from pond.state import State
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractExecuteUnit,
)


class ExecuteAgent(AbstractExecuteUnit):
    """Executable unit that runs a pydantic-ai agent.

    Similar to ExecuteTransform but uses a pydantic-ai Agent instead of
    a user function. Handles structured input/output via dynamically
    created Pydantic models.

    Attributes:
        agent: The pydantic-ai Agent instance.
        input_type: Pydantic model for structured agent input.
        output_type: Pydantic model for structured agent output.
        input_names: List of field names for the input model.
        output_names: List of field names for the output model.
        prompt: Optional user prompt to pass to the agent.
        append_outputs: List of output paths that should append rather than overwrite.

    Note:
        Agents run synchronously using agent.run_sync(). For async execution,
        pipeline runners should handle async/await orchestration.
    """

    def __init__(
        self,
        inputs: list[LensPath],
        outputs: list[LensPath],
        agent: Agent,
        input_type: Type[BaseModel],
        output_type: Type[BaseModel],
        input_names: list[str],
        output_names: list[str],
        prompt: str | None = None,
        append_outputs: list[LensPath] = [],
    ):
        """Initialize an ExecuteAgent.

        Args:
            inputs: List of input paths for data loading.
            outputs: List of output paths for data storage.
            agent: The pydantic-ai Agent instance to execute.
            input_type: Pydantic model class for structured input.
            output_type: Pydantic model class for structured output.
            input_names: Field names in the input model.
            output_names: Field names in the output model.
            prompt: Optional user prompt to pass when running the agent.
            append_outputs: Output paths that should append to existing data.

        Note:
            The agent must be configured with the output_type as its result type.
        """
        super().__init__(inputs, outputs)
        self.agent = agent
        self.input_type = input_type
        self.output_type = output_type
        self.input_names = input_names
        self.output_names = output_names
        self.prompt = prompt
        self.append_outputs = append_outputs

    def __getstate__(self):
        """Prepare instance state for pickling using dill.

        Returns:
            Serialized state containing all necessary attributes.

        Note:
            Uses dill to handle serialization of the agent and types.
        """
        return dill.dumps(
            (
                self.inputs,
                self.outputs,
                self.agent,
                self.input_type,
                self.output_type,
                self.input_names,
                self.output_names,
                self.prompt,
                self.append_outputs,
            )
        )

    def __setstate__(self, state):
        """Restore instance state after unpickling.

        Args:
            state: Serialized state from __getstate__.
        """
        (
            self.inputs,
            self.outputs,
            self.agent,
            self.input_type,
            self.output_type,
            self.input_names,
            self.output_names,
            self.prompt,
            self.append_outputs,
        ) = dill.loads(state)

    def load_inputs(self, state: State) -> list[Any]:
        """Load input data from the catalog, handling array wildcards.

        For inputs with wildcard indices (index == -1), attempts to load
        the entire array first. If not available, iterates through indices
        to build the array dynamically.

        Args:
            state: Pipeline state with catalog access.

        Returns:
            List of loaded input values, with arrays expanded as needed.

        Note:
            Uses the same wildcard expansion logic as ExecuteTransform.
        """
        args = []
        for i in self.inputs:
            try:
                index = next(ind for ind, v in enumerate(i.path) if v.index == -1)
                parent = LensPath(i.path[: index + 1])
                parent.path[-1].index = None
                value = state[parent.to_path()]
                if value is not None:
                    args.append(value)
                    continue
                input_list = []
                for list_index in range(0, 100000):
                    i.path[index].index = list_index
                    value = state[i.to_path()]
                    if value is None:
                        break
                    input_list.append(value)
                args.append(input_list)
            except StopIteration:
                args.append(state[i.to_path()])
                continue
        return args

    def save_outputs(self, state: State, rtns: list[Any]) -> list[Any]:
        """Convert output values to catalog-compatible Arrow tables.

        Args:
            state: Pipeline state with catalog access.
            rtns: List of computed output values from the agent.

        Returns:
            List of Arrow tables ready for catalog storage.

        Note:
            Uses the lens system to convert Python objects to appropriate
            Arrow table representations.
        """
        values = []
        for rtn, o in zip(rtns, self.outputs):
            values.append(state.lens(o.to_path()).create_table(rtn))
        return values

    def commit(self, state: State, values: list[Any]) -> bool:
        """Commit Arrow tables to the catalog.

        Args:
            state: Pipeline state with catalog access.
            values: List of Arrow tables from save_outputs.

        Returns:
            True if all commits were successful.

        Note:
            Respects append_outputs list to determine append vs. overwrite.
        """
        for val, o in zip(values, self.outputs):
            append = o in self.append_outputs
            state.lens(o.to_path()).write_table(val, append)
        return True

    def run(self, args: list[Any]) -> list[Any]:
        """Execute the pydantic-ai agent with loaded arguments.

        Creates a structured input instance from the arguments and runs
        the agent. Extracts individual output field values from the
        structured agent response.

        Args:
            args: List of arguments loaded by load_inputs.

        Returns:
            List of output values extracted from the agent response.

        Note:
            Uses run_sync for synchronous execution. The agent validates
            its output against the output_type schema automatically.
        """
        # Create input instance
        input_kwargs = dict(zip(self.input_names, args, strict=True))
        input_instance = self.input_type(**input_kwargs)

        # Run agent
        if self.prompt:
            result = self.agent.run_sync(self.prompt, deps=input_instance)
        else:
            # If no prompt, use input as the prompt (for simple cases)
            prompt_str = str(input_instance)
            result = self.agent.run_sync(prompt_str, deps=input_instance)

        # Extract output values
        output_values = [
            getattr(result.output, field_name) for field_name in self.output_names
        ]

        return output_values


class AbstractAgent(AbstractExecuteTransform):
    """Abstract base class for agent-based transforms.

    Extends AbstractExecuteTransform to support pydantic-ai agents.
    Concrete implementations include Agent, AgentList, and AgentListFold.

    Attributes:
        model: Model identifier (e.g., "openai:gpt-4o", "anthropic:claude-sonnet-4-0").
        instructions: System prompt/instructions for the agent.
        prompt: Optional user prompt template.

    Note:
        Agents use dynamically built Pydantic types based on catalog schemas
        to enable structured input/output for LLM interactions.
    """

    @abstractmethod
    def get_model(self) -> str:
        """Get the model identifier for this agent.

        Returns:
            Model string in format "provider:model-name".
        """
        pass

    @abstractmethod
    def get_instructions(self) -> str:
        """Get the system instructions for this agent.

        Returns:
            System prompt that guides the agent's behavior.
        """
        pass

    @abstractmethod
    def get_prompt(self) -> str | None:
        """Get the user prompt template for this agent.

        Returns:
            Optional user prompt string, or None for default behavior.
        """
        pass
