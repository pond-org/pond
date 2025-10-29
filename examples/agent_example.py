#!/usr/bin/env python3
"""Example demonstrating pydantic-ai agent transforms in pond.

This example shows how to use the agent() function to create
LLM-powered data transformations in pipelines.
"""
from pydantic import BaseModel

from pond.decorators import agent, construct, pipe
from pond.runners.sequential_runner import SequentialRunner
from pond.state import State


# Define the data catalog schema
class Review(BaseModel):
    """A single customer review."""

    text: str
    sentiment: str | None = None
    rating: int | None = None


class ProductData(BaseModel):
    """Product review data catalog."""

    reviews: list[Review] | None = None
    overall_summary: str | None = None
    recommendation: str | None = None


# Example 1: Agent (scalar -> scalar)
# Generates a recommendation based on the overall summary
generate_recommendation = agent(
    ProductData,
    "overall_summary",
    "recommendation",
    model="openai:gpt-4o-mini",  # Using mini for faster/cheaper execution
    instructions="Based on the summary, provide a brief product recommendation (one sentence).",
)

# Example 2: AgentList (array -> array)
# Analyzes sentiment for each review
analyze_sentiments = agent(
    ProductData,
    "reviews[:].text",
    "reviews[:].sentiment",
    model="openai:gpt-4o-mini",
    instructions="Analyze the sentiment of the review text. Return one of: Positive, Negative, Neutral.",
    output_descriptions=["Sentiment classification (Positive/Negative/Neutral)"],
)

# Example 3: AgentListFold (array -> scalar)
# Aggregates all reviews into a single summary
summarize_reviews = agent(
    ProductData,
    "reviews[:].text",
    "overall_summary",
    model="openai:gpt-4o-mini",
    instructions="Summarize all the reviews into a cohesive 2-3 sentence summary.",
    input_descriptions=["List of all customer review texts"],
    output_descriptions=["Overall summary of all reviews"],
)

# Example 4: Another agent (array -> array)
# Converts sentiments to numeric ratings
sentiment_to_rating = agent(
    ProductData,
    "reviews[:].sentiment",
    "reviews[:].rating",
    model="openai:gpt-4o-mini",
    instructions="Convert sentiment to a rating: Positive=5, Neutral=3, Negative=1.",
)


def main():
    """Run the example pipeline."""
    # Create initial catalog with sample reviews
    catalog = ProductData(
        reviews=[
            Review(text="This product is amazing! Best purchase ever."),
            Review(text="Decent quality, met my expectations."),
            Review(text="Very disappointing. Would not recommend."),
            Review(text="Great value for money. Happy with it."),
        ]
    )

    # Build the pipeline
    pipeline = pipe(
        [
            construct(ProductData),
            analyze_sentiments,  # Analyze sentiment for each review
            sentiment_to_rating,  # Convert sentiments to ratings
            summarize_reviews,  # Aggregate all reviews to summary
            generate_recommendation,  # Generate final recommendation
        ],
        output="recommendation",
    )

    # Create state and run pipeline
    state = State(catalog)
    runner = SequentialRunner()

    print("Starting agent pipeline...")
    print(f"Input: {len(catalog.reviews)} reviews")
    print()

    # Run the pipeline
    runner.run(state, pipeline)

    # Display results
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print()
    print("Individual Review Analysis:")
    for i, review in enumerate(state.catalog.reviews):
        print(f"  Review {i+1}:")
        print(f"    Text: {review.text}")
        print(f"    Sentiment: {review.sentiment}")
        print(f"    Rating: {review.rating}/5")
        print()

    print(f"Overall Summary: {state.catalog.overall_summary}")
    print()
    print(f"Recommendation: {state.catalog.recommendation}")
    print()


if __name__ == "__main__":
    # Note: You'll need to set your OpenAI API key:
    # export OPENAI_API_KEY="your-key-here"
    main()
