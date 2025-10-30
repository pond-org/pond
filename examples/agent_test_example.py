#!/usr/bin/env python3
"""Example demonstrating pydantic-ai agents with TestModel.

This example shows how to test agents locally without real LLM API calls
using pydantic-ai's TestModel. TestModel is useful for:
- Unit testing without API costs
- Fast development iterations
- Avoiding rate limits
- Deterministic testing
"""
from pydantic import BaseModel
from pydantic_ai.models.test import TestModel

from pond.decorators import agent, construct, pipe
from pond.runners.sequential_runner import SequentialRunner
from pond.state import State


# Define the data catalog schema
class ProductReview(BaseModel):
    """A single product review."""

    text: str
    sentiment: str | None = None
    helpful_score: int | None = None


class ProductAnalysis(BaseModel):
    """Product analysis data."""

    reviews: list[ProductReview] | None = None
    overall_sentiment: str | None = None
    recommendation: str | None = None
    summary: str | None = None


# Example 1: Scalar agent with TestModel
classify_sentiment = agent(
    ProductAnalysis,
    "overall_sentiment",
    "recommendation",
    model="test",  # Will be overridden with TestModel
    instructions="Based on overall sentiment, recommend whether to buy the product.",
)

# Example 2: Array agent with TestModel
analyze_reviews = agent(
    ProductAnalysis,
    "reviews[:].text",
    "reviews[:].sentiment",
    model="test",
    instructions="Analyze each review's sentiment: Positive, Negative, or Neutral.",
    output_descriptions=["Sentiment classification"],
)

# Example 3: Array fold agent with TestModel
summarize_all = agent(
    ProductAnalysis,
    "reviews[:].text",
    "overall_sentiment",
    model="test",
    instructions="Determine overall sentiment from all reviews.",
)

# Example 4: Generate summary
generate_summary = agent(
    ProductAnalysis,
    "reviews[:].text",
    "summary",
    model="test",
    instructions="Create a brief summary of all reviews.",
)


def main():
    """Run the example with TestModel."""
    print("=" * 70)
    print("Pydantic-AI Agents with TestModel Example")
    print("=" * 70)
    print()
    print("TestModel allows testing agents without real LLM API calls.")
    print("It generates synthetic responses that match the expected schema.")
    print()

    # Create test data
    catalog = ProductAnalysis(
        reviews=[
            ProductReview(text="Excellent product! Highly recommend."),
            ProductReview(text="Good quality, met expectations."),
            ProductReview(text="Disappointed with the quality."),
            ProductReview(text="Amazing! Best purchase ever."),
        ]
    )

    print(f"Input: {len(catalog.reviews)} product reviews")
    for i, review in enumerate(catalog.reviews, 1):
        print(f"  {i}. {review.text}")
    print()

    # Create pipeline
    pipeline = pipe(
        [
            construct(ProductAnalysis),
            analyze_reviews,      # Analyze each review
            summarize_all,        # Aggregate to overall sentiment
            generate_summary,     # Generate text summary
            classify_sentiment,   # Make recommendation
        ],
        output="recommendation",
    )

    # Create state
    state = State(catalog)

    # Create TestModel with custom output
    # You can customize the response to match your domain
    test_model = TestModel(
        custom_result_text="Test response - this simulates LLM output"
    )

    # Override all agents in the pipeline with TestModel
    print("Overriding agents with TestModel...")
    for transform in pipeline.get_transforms():
        units = transform.get_execute_units(state)
        for unit in units:
            # Replace the lazy agent with TestModel
            if hasattr(unit, '_agent'):
                unit._agent = test_model
    print()

    # Run pipeline
    print("Running pipeline...")
    runner = SequentialRunner()
    runner.run(state, pipeline)
    print()

    # Display results
    print("=" * 70)
    print("Results:")
    print("=" * 70)
    print()

    print("Individual Review Sentiments:")
    if state.catalog.reviews:
        for i, review in enumerate(state.catalog.reviews, 1):
            print(f"  {i}. {review.text[:50]}...")
            print(f"     Sentiment: {review.sentiment}")
        print()

    print(f"Overall Sentiment: {state.catalog.overall_sentiment}")
    print()
    print(f"Summary: {state.catalog.summary}")
    print()
    print(f"Recommendation: {state.catalog.recommendation}")
    print()

    print("=" * 70)
    print("Note: TestModel generates synthetic responses.")
    print("For real LLM responses, use models like 'openai:gpt-4o'")
    print("and set appropriate API keys.")
    print("=" * 70)


def example_custom_test_model():
    """Example showing how to customize TestModel responses."""
    print("\n\nCustom TestModel Example:")
    print("-" * 70)

    # You can customize TestModel to return specific values
    custom_model = TestModel(
        custom_result_text="Positive"  # Force specific output
    )

    # Simple agent
    simple_agent = agent(
        ProductAnalysis,
        "overall_sentiment",
        "recommendation",
        model="test",
        instructions="Make a recommendation.",
    )

    catalog = ProductAnalysis(overall_sentiment="Positive")
    state = State(catalog)

    pipeline = pipe([construct(ProductAnalysis), simple_agent])

    # Override with custom model
    for transform in pipeline.get_transforms():
        units = transform.get_execute_units(state)
        for unit in units:
            if hasattr(unit, '_agent'):
                unit._agent = custom_model

    runner = SequentialRunner()
    runner.run(state, pipeline)

    print(f"Input: {catalog.overall_sentiment}")
    print(f"Output: {state.catalog.recommendation}")
    print()


if __name__ == "__main__":
    # Note: TestModel doesn't require API keys
    # No need for: export OPENAI_API_KEY="your-key"
    main()
    example_custom_test_model()
