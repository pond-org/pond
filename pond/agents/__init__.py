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
"""Pydantic-AI agent transforms module.

This module provides LLM-powered data transformations using pydantic-ai.
Agents are an alternative to function-based transforms, allowing you to
use Large Language Models to process data in your pipelines.

Key Components:
    - Agent: Scalar input/output agent transforms
    - AgentList: Array-to-array element-wise agent transforms
    - AgentListFold: Array-to-scalar aggregation agent transforms
    - type_builder: Utilities for dynamically creating Pydantic types

Usage:
    from pond.decorators import agent_node

    @agent_node(
        Catalog,
        "customer_query",
        "support_response",
        model="anthropic:claude-sonnet-4-0",
        instructions="Provide helpful customer support responses."
    )

    # The decorator automatically creates an Agent instance
"""
from pond.agents.abstract_agent import AbstractAgent, ExecuteAgent
from pond.agents.agent import Agent
from pond.agents.agent_list import AgentList
from pond.agents.agent_list_fold import AgentListFold
from pond.agents.type_builder import (
    build_input_type,
    build_output_type,
    build_pydantic_type_from_catalog_type,
    extract_all_field_values,
    extract_field_value,
)

__all__ = [
    "AbstractAgent",
    "Agent",
    "AgentList",
    "AgentListFold",
    "ExecuteAgent",
    "build_input_type",
    "build_output_type",
    "build_pydantic_type_from_catalog_type",
    "extract_field_value",
    "extract_all_field_values",
]
