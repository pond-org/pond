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
"""Dynamic Pydantic type builder for agent inputs and outputs.

This module provides utilities to dynamically create Pydantic BaseModel classes
based on catalog schema types. These dynamically created types are used as
structured input/output types for pydantic-ai agents.
"""
from typing import Any, Type, get_args, get_origin

from pydantic import BaseModel, Field, create_model


def build_pydantic_type_from_catalog_type(
    field_name: str,
    catalog_type: Type,
    description: str | None = None,
) -> Type[BaseModel]:
    """Build a Pydantic model from a catalog type for use with pydantic-ai.

    Creates a single-field Pydantic BaseModel that wraps the catalog type.
    This allows using any catalog type as structured input/output for agents.

    Args:
        field_name: The name for the field in the generated model.
        catalog_type: The Python type from the catalog schema (e.g., int, str,
            list[str], or a Pydantic BaseModel subclass).
        description: Optional description for the field, used by the LLM
            to understand the field's purpose.

    Returns:
        A dynamically created Pydantic BaseModel class with a single field
        of the specified type.

    Example:
        >>> InputType = build_pydantic_type_from_catalog_type(
        ...     "temperature", float, "Temperature in celsius"
        ... )
        >>> # Creates a model like:
        >>> # class InputType(BaseModel):
        >>> #     temperature: float = Field(description="Temperature in celsius")
    """
    # If the catalog type is already a BaseModel, use it directly
    if isinstance(catalog_type, type) and issubclass(catalog_type, BaseModel):
        return catalog_type

    # Build field definition with optional description
    field_kwargs: dict[str, Any] = {}
    if description:
        field_kwargs["description"] = description

    # Create a dynamic model with a single field
    model = create_model(
        f"AgentType_{field_name}",
        __base__=BaseModel,
        **{field_name: (catalog_type, Field(**field_kwargs))},
    )

    return model


def build_input_type(
    input_types: list[Type],
    input_names: list[str],
    descriptions: list[str] | None = None,
) -> Type[BaseModel]:
    """Build a Pydantic model for multiple agent inputs.

    Creates a Pydantic BaseModel with fields corresponding to each input
    parameter. The agent will receive this structured input type.

    Args:
        input_types: List of Python types for each input field.
        input_names: List of field names corresponding to each input.
        descriptions: Optional list of descriptions for each field.

    Returns:
        A dynamically created Pydantic BaseModel class with fields for
        all inputs.

    Example:
        >>> InputType = build_input_type(
        ...     [float, str],
        ...     ["temperature", "location"],
        ...     ["Temperature in celsius", "Location name"]
        ... )
        >>> # Creates a model like:
        >>> # class InputType(BaseModel):
        >>> #     temperature: float = Field(description="Temperature in celsius")
        >>> #     location: str = Field(description="Location name")
    """
    if descriptions is None:
        descriptions = [None] * len(input_types)  # type: ignore

    # Build field definitions
    fields: dict[str, Any] = {}
    for name, type_, desc in zip(input_names, input_types, descriptions, strict=True):
        field_kwargs: dict[str, Any] = {}
        if desc:
            field_kwargs["description"] = desc

        fields[name] = (type_, Field(**field_kwargs))

    # Create dynamic model
    model = create_model("AgentInput", __base__=BaseModel, **fields)

    return model


def build_output_type(
    output_types: list[Type],
    output_names: list[str],
    descriptions: list[str] | None = None,
) -> Type[BaseModel]:
    """Build a Pydantic model for multiple agent outputs.

    Creates a Pydantic BaseModel with fields corresponding to each output.
    The agent will return this structured output type.

    Args:
        output_types: List of Python types for each output field.
        output_names: List of field names corresponding to each output.
        descriptions: Optional list of descriptions for each field.

    Returns:
        A dynamically created Pydantic BaseModel class with fields for
        all outputs.

    Example:
        >>> OutputType = build_output_type(
        ...     [str, int],
        ...     ["summary", "confidence"],
        ...     ["Summary text", "Confidence score 0-100"]
        ... )
        >>> # Creates a model like:
        >>> # class OutputType(BaseModel):
        >>> #     summary: str = Field(description="Summary text")
        >>> #     confidence: int = Field(description="Confidence score 0-100")
    """
    if descriptions is None:
        descriptions = [None] * len(output_types)  # type: ignore

    # Build field definitions
    fields: dict[str, Any] = {}
    for name, type_, desc in zip(output_names, output_types, descriptions, strict=True):
        field_kwargs: dict[str, Any] = {}
        if desc:
            field_kwargs["description"] = desc

        fields[name] = (type_, Field(**field_kwargs))

    # Create dynamic model
    model = create_model("AgentOutput", __base__=BaseModel, **fields)

    return model


def extract_field_value(model_instance: BaseModel, field_name: str) -> Any:
    """Extract a field value from a Pydantic model instance.

    Helper function to get a specific field value from a model returned
    by an agent.

    Args:
        model_instance: A Pydantic model instance.
        field_name: Name of the field to extract.

    Returns:
        The value of the specified field.

    Example:
        >>> result = agent.run_sync("query")
        >>> temperature = extract_field_value(result.output, "temperature")
    """
    return getattr(model_instance, field_name)


def extract_all_field_values(model_instance: BaseModel) -> list[Any]:
    """Extract all field values from a Pydantic model instance as a list.

    Returns field values in the order they were defined in the model.

    Args:
        model_instance: A Pydantic model instance.

    Returns:
        List of all field values in definition order.

    Example:
        >>> result = agent.run_sync("query")
        >>> outputs = extract_all_field_values(result.output)
        >>> summary, confidence = outputs
    """
    return [getattr(model_instance, field_name) for field_name in model_instance.model_fields]
