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

from pond.decorators import agent, construct, pipe
from pond.runners.sequential_runner import SequentialRunner
from pond.state import State


# Test catalog schema
class Review(BaseModel):
    """A single review."""

    text: str
    sentiment: str | None = None
    category: str | None = None


class Analysis(BaseModel):
    """Analysis results."""

    overall_sentiment: str | None = None
    recommendation: str | None = None


class TestCatalog(BaseModel):
    """Test catalog for agent tests."""

    reviews: list[Review] | None = None
    analysis: Analysis | None = None
    query: str | None = None
    response: str | None = None


def test_agent_scalar_to_scalar():
    """Test Agent (scalar -> scalar) with TestModel."""
    # Create agent with test model
    test_model = TestModel()

    my_agent = agent(
        TestCatalog,
        "query",
        "response",
        model="test",
        instructions="Respond to the query with helpful information.",
    )

    # Override the model with TestModel
    # We need to access the agent instance created inside get_execute_units
    catalog = TestCatalog(query="What is the weather?")
    state = State(catalog)

    # Create pipeline
    pipeline = pipe([construct(TestCatalog), my_agent], output="response")

    # Get execute units and override their agents
    for transform in pipeline.get_transforms():
        units = transform.get_execute_units(state)
        for unit in units:
            # Override the lazy agent with test model
            unit._agent = test_model

    # Run the pipeline
    runner = SequentialRunner()
    runner.run(state, pipeline)

    # Verify output exists
    assert state.catalog.response is not None
    # TestModel returns a success message
    assert "success" in state.catalog.response.lower() or state.catalog.response != ""


def test_agent_list_array_to_array():
    """Test AgentList (array -> array) with TestModel."""
    test_model = TestModel(custom_result_text="Positive")

    # Create agent that analyzes sentiment for each review
    analyze_sentiment = agent(
        TestCatalog,
        "reviews[:].text",
        "reviews[:].sentiment",
        model="test",
        instructions="Analyze the sentiment: Positive, Negative, or Neutral.",
    )

    # Setup test data
    catalog = TestCatalog(
        reviews=[
            Review(text="Great product!"),
            Review(text="Terrible experience."),
            Review(text="It's okay."),
        ]
    )
    state = State(catalog)

    # Create pipeline
    pipeline = pipe([construct(TestCatalog), analyze_sentiment])

    # Override agents in execute units
    for transform in pipeline.get_transforms():
        units = transform.get_execute_units(state)
        for unit in units:
            unit._agent = test_model

    # Run pipeline
    runner = SequentialRunner()
    runner.run(state, pipeline)

    # Verify all reviews have sentiments
    assert state.catalog.reviews is not None
    for review in state.catalog.reviews:
        assert review.sentiment is not None
        assert review.sentiment != ""


def test_agent_list_fold_array_to_scalar():
    """Test AgentListFold (array -> scalar) with TestModel."""
    test_model = TestModel(custom_result_text="Overall Positive")

    # Create agent that aggregates reviews to overall sentiment
    aggregate_sentiment = agent(
        TestCatalog,
        "reviews[:].text",
        "analysis.overall_sentiment",
        model="test",
        instructions="Aggregate all reviews into overall sentiment.",
    )

    # Setup test data
    catalog = TestCatalog(
        reviews=[
            Review(text="Great!"),
            Review(text="Love it!"),
            Review(text="Amazing!"),
        ],
        analysis=Analysis(),
    )
    state = State(catalog)

    # Create pipeline
    pipeline = pipe([construct(TestCatalog), aggregate_sentiment])

    # Override agents in execute units
    for transform in pipeline.get_transforms():
        units = transform.get_execute_units(state)
        for unit in units:
            unit._agent = test_model

    # Run pipeline
    runner = SequentialRunner()
    runner.run(state, pipeline)

    # Verify overall sentiment was set
    assert state.catalog.analysis is not None
    assert state.catalog.analysis.overall_sentiment is not None
    assert state.catalog.analysis.overall_sentiment != ""


def test_agent_multiple_inputs():
    """Test agent with multiple inputs."""
    test_model = TestModel(custom_result_text="Recommendation based on inputs")

    # Agent with multiple inputs
    multi_input_agent = agent(
        TestCatalog,
        ["analysis.overall_sentiment", "query"],
        "response",
        model="test",
        instructions="Generate response based on sentiment and query.",
    )

    # Setup test data
    catalog = TestCatalog(
        analysis=Analysis(overall_sentiment="Positive"),
        query="Should I buy this?",
    )
    state = State(catalog)

    # Create pipeline
    pipeline = pipe([construct(TestCatalog), multi_input_agent])

    # Override agents
    for transform in pipeline.get_transforms():
        units = transform.get_execute_units(state)
        for unit in units:
            unit._agent = test_model

    # Run pipeline
    runner = SequentialRunner()
    runner.run(state, pipeline)

    # Verify response was generated
    assert state.catalog.response is not None
    assert state.catalog.response != ""


def test_agent_with_descriptions():
    """Test agent with input/output descriptions."""
    test_model = TestModel()

    # Agent with field descriptions
    described_agent = agent(
        TestCatalog,
        "query",
        "response",
        model="test",
        instructions="Respond helpfully.",
        input_descriptions=["User's question"],
        output_descriptions=["Helpful response to the question"],
    )

    # Setup and run
    catalog = TestCatalog(query="Test query")
    state = State(catalog)

    pipeline = pipe([construct(TestCatalog), described_agent])

    for transform in pipeline.get_transforms():
        units = transform.get_execute_units(state)
        for unit in units:
            unit._agent = test_model

    runner = SequentialRunner()
    runner.run(state, pipeline)

    # Verify execution completed
    assert state.catalog.response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
