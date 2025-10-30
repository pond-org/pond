# Pond Agents Module

LLM-powered data transformations using pydantic-ai as an alternative to function-based transforms.

## Overview

The agents module extends pond's transform system with Large Language Model (LLM) capabilities through pydantic-ai. Agents provide the same input/output mapping patterns as transforms but use LLMs instead of user-defined functions.

## Quick Start

Use the `agent()` function to create LLM-powered transforms:

```python
from pond.decorators import agent

# Automatically selects the right agent type based on input/output patterns
analyze_sentiment = agent(
    Catalog,
    "review.text",
    "review.sentiment",
    model="openai:gpt-4o",
    instructions="Analyze sentiment: return Positive, Negative, or Neutral."
)
```

## Agent Types

The `agent()` function automatically selects the appropriate type based on your input/output paths:

### 1. Agent (Scalar → Scalar)

**When**: Neither input nor output paths contain `[:]`

**Example**:
```python
generate_description = agent(
    Catalog,
    "product.features",
    "product.description",
    model="anthropic:claude-sonnet-4-0",
    instructions="Generate a compelling product description from the features.",
)
```

### 2. AgentList (Array → Array)

**When**: Both input and output paths contain `[:]`

**Example**:
```python
analyze_sentiments = agent(
    Catalog,
    "reviews[:].text",
    "reviews[:].sentiment",
    model="openai:gpt-4o",
    instructions="Analyze sentiment: return Positive, Negative, or Neutral.",
)
```

### 3. AgentListFold (Array → Scalar)

**When**: Input paths contain `[:]` but output paths don't

**Example**:
```python
summarize_reviews = agent(
    Catalog,
    "reviews[:].text",
    "overall_summary",
    model="anthropic:claude-sonnet-4-0",
    instructions="Summarize all reviews into a cohesive 2-3 sentence summary.",
)
```

## Supported Models

Pydantic-ai supports multiple model providers:

- **OpenAI**: `"openai:gpt-4o"`, `"openai:gpt-4o-mini"`
- **Anthropic**: `"anthropic:claude-sonnet-4-0"`, `"anthropic:claude-opus-4-0"`
- **Google**: `"google-gla:gemini-1.5-flash"`, `"google-gla:gemini-1.5-pro"`
- **Local**: `"ollama:llama2"`, `"ollama:mistral"`
- And many more (Groq, DeepSeek, Cohere, etc.)

## Configuration

### Model Credentials

Set environment variables for your chosen provider:

```bash
# OpenAI
export OPENAI_API_KEY="your-key"

# Anthropic
export ANTHROPIC_API_KEY="your-key"

# Google
export GEMINI_API_KEY="your-key"
```

### Agent Parameters

```python
agent(
    Catalog,              # Your catalog schema
    input,                # str | list[str] - Input path(s)
    output,               # str | list[str] - Output path(s)
    model,                # str - Model identifier
    instructions,         # str - System prompt
    prompt=None,          # Optional user prompt template
    input_descriptions=None,   # Optional field descriptions
    output_descriptions=None,  # Optional field descriptions
)
```

## Pipeline Integration

Agents work seamlessly with pond pipelines:

```python
from pond.decorators import agent, pipe, construct
from pond.runners.sequential_runner import SequentialRunner

# Create agents
analyze = agent(Catalog, "reviews[:].text", "reviews[:].sentiment", ...)
summarize = agent(Catalog, "reviews[:].text", "overall_summary", ...)
recommend = agent(Catalog, "overall_summary", "recommendation", ...)

# Build pipeline
pipeline = pipe([
    construct(Catalog),
    analyze,
    summarize,
    recommend,
], output="recommendation")

# Run
runner = SequentialRunner()
runner.run(state, pipeline)
```

## Multiple Inputs/Outputs

Agents support multiple inputs and outputs:

```python
# Multiple inputs
multi_input_agent = agent(
    Catalog,
    ["product.name", "product.price", "product.features"],
    "product.marketing_copy",
    model="openai:gpt-4o",
    instructions="Generate marketing copy from product details.",
)

# Multiple outputs
multi_output_agent = agent(
    Catalog,
    "customer_feedback",
    ["analysis.sentiment", "analysis.priority"],
    model="anthropic:claude-sonnet-4-0",
    instructions="Analyze feedback and determine sentiment and priority.",
)
```

## Best Practices

### 1. Clear Instructions

Write specific, actionable instructions:

```python
# Good
instructions="Analyze sentiment. Return exactly one of: Positive, Negative, Neutral."

# Less effective
instructions="Look at the text and tell me what you think."
```

### 2. Field Descriptions

Use descriptions to guide the LLM:

```python
agent(
    Catalog,
    "reviews[:].text",
    "reviews[:].sentiment",
    model="openai:gpt-4o",
    instructions="Classify sentiment",
    input_descriptions=["Customer review text to analyze"],
    output_descriptions=["Sentiment: Positive, Negative, or Neutral"],
)
```

### 3. Choose Appropriate Models

- **Fast tasks**: Use smaller/faster models (gpt-4o-mini, gemini-1.5-flash)
- **Complex reasoning**: Use larger models (gpt-4o, claude-sonnet-4-0)
- **Cost-sensitive**: Consider model pricing and token usage

### 4. Error Handling

Pydantic-ai automatically retries if LLM output doesn't match the schema:

```python
# If LLM returns invalid type, it will be prompted to try again
# Output schema ensures type safety
```

## Architecture

The agents module mirrors the transforms architecture:

```
agent() function
    ↓ (auto-selects based on paths)
    ├── Agent (scalar → scalar)
    ├── AgentList (array → array)
    └── AgentListFold (array → scalar)

ExecuteAgent (execution unit)
├── load_inputs() - Load from catalog
├── run() - Execute pydantic-ai agent (created lazily)
├── save_outputs() - Convert to Arrow tables
└── commit() - Write to catalog
```

### Lazy Agent Instantiation

The pydantic-ai Agent is created lazily in `ExecuteAgent` to support:
- **Process parallelization**: Proper serialization for multiprocessing
- **Resource efficiency**: Only create agents when needed
- **Thread safety**: Each process gets its own agent instance

## Examples

See `examples/agent_example.py` for a complete working example demonstrating all three agent types.

## Comparison: Agents vs Transforms

| Feature | Transforms | Agents |
|---------|-----------|--------|
| Implementation | User-defined Python functions | LLM-powered via pydantic-ai |
| Use cases | Deterministic data processing | Text analysis, generation, reasoning |
| Type safety | Static type checking | Dynamic with validation |
| Performance | Fast, local execution | Slower, API calls |
| Cost | Free | API usage costs |
| Flexibility | Code-based logic | Natural language instructions |

## Limitations

1. **Latency**: LLM API calls add latency compared to local functions
2. **Cost**: Each agent execution incurs API costs
3. **Reliability**: LLMs may produce inconsistent outputs
4. **Offline**: Requires internet connectivity (except local models)

## Advanced Usage

### Dynamic Type Building

Agents automatically build Pydantic types from catalog schemas:

```python
# Input: catalog.customer_query (str)
# Agent builds: class AgentInput(BaseModel): customer_query: str

# Output: catalog.support_response (str)
# Agent builds: class AgentOutput(BaseModel): support_response: str
```

The LLM receives structured inputs and produces validated structured outputs based on catalog types.

### Type Builder Utilities

For advanced use cases, use the type builder directly:

```python
from pond.agents.type_builder import (
    build_input_type,
    build_output_type,
    extract_field_value,
)

# Build custom types
InputType = build_input_type(
    [str, int],
    ["name", "age"],
    ["User name", "User age"]
)

# Extract values from agent output
result = agent_run(...)
value = extract_field_value(result.output, "field_name")
```

## Future Enhancements

Potential improvements:

- Caching/memoization for repeated inputs
- Batch processing optimization
- Streaming support for long outputs
- Tool registration for agents
- Custom dependency injection

## Contributing

To extend the agents module:

1. Add new agent types in `pond/agents/`
2. Follow the existing architecture patterns
3. Ensure type safety with Pydantic
4. Add tests in `tests/agents/`
5. Update this README with examples
