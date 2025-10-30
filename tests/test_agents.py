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
"""Unit tests for pydantic-ai agent transforms using TestModel."""
import pytest
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from pond.catalogs.lance_catalog import LanceCatalog
from pond.decorators import agent
from pond.state import State


# Test catalog schema
class Review(BaseModel):
    """A single review."""

    text: str
    sentiment: str = ""


class Analysis(BaseModel):
    """Analysis results."""

    overall_sentiment: str = ""
    recommendation: str = ""


class AgentTestCatalog(BaseModel):
    """Test catalog for agent tests."""

    reviews: list[Review] = []
    analysis: Analysis = Analysis()
    query: str = ""
    response: str = ""


def test_agent_scalar_to_scalar(tmp_path):
    """Test Agent (scalar -> scalar) with TestModel."""
    test_model = TestModel()

    my_agent = agent(
        AgentTestCatalog,
        "query",
        "response",
        model="test",
        instructions="Respond to the query with helpful information.",
    )

    # Setup state and initialize data
    data_catalog = LanceCatalog(tmp_path)
    state = State(AgentTestCatalog, data_catalog)

    # Write input data
    state.lens("query").write_table(
        state.lens("query").create_table("What is the weather?")
    )

    # Run agent
    units = my_agent.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify output
    result = state["response"]
    assert result is not None
    assert result != ""


def test_agent_list_array_to_array(tmp_path):
    """Test AgentList (array -> array) with TestModel."""
    test_model = TestModel()

    analyze_sentiment = agent(
        AgentTestCatalog,
        "reviews[:].text",
        "reviews[:].sentiment",
        model="test",
        instructions="Analyze the sentiment: Positive, Negative, or Neutral.",
    )

    # Setup state
    data_catalog = LanceCatalog(tmp_path)
    state = State(AgentTestCatalog, data_catalog)

    # Write input reviews
    reviews_data = [
        Review(text="Great product!"),
        Review(text="Terrible experience."),
        Review(text="It's okay."),
    ]
    state.lens("reviews").write_table(
        state.lens("reviews").create_table(reviews_data)
    )

    # Run agent
    units = analyze_sentiment.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify all reviews have sentiments
    for i in range(len(reviews_data)):
        sentiment = state[f"reviews[{i}].sentiment"]
        assert sentiment is not None
        assert sentiment != ""


def test_agent_list_fold_array_to_scalar(tmp_path):
    """Test AgentListFold (array -> scalar) with TestModel."""
    test_model = TestModel()

    aggregate_sentiment = agent(
        AgentTestCatalog,
        "reviews[:].text",
        "analysis.overall_sentiment",
        model="test",
        instructions="Aggregate all reviews into overall sentiment.",
    )

    # Setup state
    data_catalog = LanceCatalog(tmp_path)
    state = State(AgentTestCatalog, data_catalog)

    # Write input reviews
    reviews_data = [
        Review(text="Great!"),
        Review(text="Love it!"),
        Review(text="Amazing!"),
    ]
    state.lens("reviews").write_table(
        state.lens("reviews").create_table(reviews_data)
    )

    # Initialize analysis object
    state.lens("analysis").write_table(
        state.lens("analysis").create_table(Analysis())
    )

    # Run agent
    units = aggregate_sentiment.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify overall sentiment was set
    result = state["analysis.overall_sentiment"]
    assert result is not None
    assert result != ""


def test_agent_multiple_inputs(tmp_path):
    """Test agent with multiple inputs."""
    test_model = TestModel()

    multi_input_agent = agent(
        AgentTestCatalog,
        ["analysis.overall_sentiment", "query"],
        "response",
        model="test",
        instructions="Generate response based on sentiment and query.",
    )

    # Setup state
    data_catalog = LanceCatalog(tmp_path)
    state = State(AgentTestCatalog, data_catalog)

    # Write input data
    state.lens("analysis").write_table(
        state.lens("analysis").create_table(Analysis(overall_sentiment="Positive"))
    )
    state.lens("query").write_table(
        state.lens("query").create_table("Should I buy this?")
    )

    # Run agent
    units = multi_input_agent.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify response
    result = state["response"]
    assert result is not None
    assert result != ""


def test_agent_with_descriptions(tmp_path):
    """Test agent with input/output descriptions."""
    test_model = TestModel()

    described_agent = agent(
        AgentTestCatalog,
        "query",
        "response",
        model="test",
        instructions="Respond helpfully.",
        input_descriptions=["User's question"],
        output_descriptions=["Helpful response to the question"],
    )

    # Setup state
    data_catalog = LanceCatalog(tmp_path)
    state = State(AgentTestCatalog, data_catalog)

    # Write input data
    state.lens("query").write_table(
        state.lens("query").create_table("Test query")
    )

    # Run agent
    units = described_agent.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify execution completed
    result = state["response"]
    assert result is not None


def test_agent_list_format_inputs_outputs(tmp_path):
    """Test Agent with list format for inputs and outputs."""
    test_model = TestModel()

    my_agent = agent(
        AgentTestCatalog,
        ["query"],  # List format
        ["response"],  # List format
        model="test",
        instructions="Respond to the query.",
    )

    # Setup state
    data_catalog = LanceCatalog(tmp_path)
    state = State(AgentTestCatalog, data_catalog)

    # Write input data
    state.lens("query").write_table(state.lens("query").create_table("Test query"))

    # Run agent
    units = my_agent.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify output
    result = state["response"]
    assert result is not None
    assert result != ""


def test_agent_list_mixed_inputs(tmp_path):
    """Test AgentList with mix of scalar and multiple array inputs."""
    test_model = TestModel()

    # Create a catalog with additional fields
    class Product(BaseModel):
        name: str
        price: float = 0.0
        description: str = ""

    class MixedCatalog(BaseModel):
        category: str = ""
        products: list[Product] = []

    mixed_agent = agent(
        MixedCatalog,
        ["category", "products[:].name", "products[:].price"],
        "products[:].description",
        model="test",
        instructions="Generate product description based on category, name and price.",
    )

    # Setup state
    data_catalog = LanceCatalog(tmp_path)
    state = State(MixedCatalog, data_catalog)

    # Write input data
    state.lens("category").write_table(
        state.lens("category").create_table("Electronics")
    )
    products_data = [
        Product(name="Laptop", price=999.99),
        Product(name="Mouse", price=29.99),
    ]
    state.lens("products").write_table(
        state.lens("products").create_table(products_data)
    )

    # Run agent
    units = mixed_agent.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify all products have descriptions
    for i in range(len(products_data)):
        description = state[f"products[{i}].description"]
        assert description is not None
        assert description != ""


def test_agent_list_fold_partitioned(tmp_path):
    """Test AgentListFold with partitioned array writes."""
    test_model = TestModel()

    aggregate_agent = agent(
        AgentTestCatalog,
        "reviews[:].text",
        "analysis.overall_sentiment",
        model="test",
        instructions="Aggregate all reviews into overall sentiment.",
    )

    # Setup state
    data_catalog = LanceCatalog(tmp_path)
    state = State(AgentTestCatalog, data_catalog)

    # Write reviews individually (partitioned)
    state.lens("reviews[0]").write_table(
        state.lens("reviews[0]").create_table(Review(text="Excellent!"))
    )
    state.lens("reviews[1]").write_table(
        state.lens("reviews[1]").create_table(Review(text="Very good!"))
    )
    state.lens("reviews[2]").write_table(
        state.lens("reviews[2]").create_table(Review(text="Fantastic!"))
    )

    # Initialize analysis object
    state.lens("analysis").write_table(
        state.lens("analysis").create_table(Analysis())
    )

    # Run agent
    units = aggregate_agent.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify overall sentiment was set
    result = state["analysis.overall_sentiment"]
    assert result is not None
    assert result != ""


def test_agent_multiple_outputs(tmp_path):
    """Test agent with multiple outputs."""
    test_model = TestModel()

    multi_output_agent = agent(
        AgentTestCatalog,
        "reviews[:].text",
        ["analysis.overall_sentiment", "analysis.recommendation"],
        model="test",
        instructions="Analyze reviews and provide sentiment and recommendation.",
    )

    # Setup state
    data_catalog = LanceCatalog(tmp_path)
    state = State(AgentTestCatalog, data_catalog)

    # Write input reviews
    reviews_data = [
        Review(text="Great!"),
        Review(text="Love it!"),
    ]
    state.lens("reviews").write_table(
        state.lens("reviews").create_table(reviews_data)
    )

    # Initialize analysis object
    state.lens("analysis").write_table(
        state.lens("analysis").create_table(Analysis())
    )

    # Run agent
    units = multi_output_agent.get_execute_units(state)
    for unit in units:
        unit.override_model(test_model)
        unit.execute_on(state)

    # Verify both outputs were set
    sentiment = state["analysis.overall_sentiment"]
    recommendation = state["analysis.recommendation"]
    assert sentiment is not None
    assert sentiment != ""
    assert recommendation is not None
    assert recommendation != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
