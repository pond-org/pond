# Pond Agents Module

LLM-powered data transformations using pydantic-ai as an alternative to function-based transforms.

## Overview

The agents module extends pond's transform system with Large Language Model (LLM) capabilities through pydantic-ai. Agents provide the same input/output mapping patterns as transforms but use LLMs instead of user-defined functions.

## Agent Types

### 1. Agent (Scalar → Scalar)

Processes single values to produce single outputs.

```python
from pond.agents.agent import Agent

# Generate a product description from features
generate_description = Agent(
    Catalog,
    "product.features",
    "product.description",
    model="anthropic:claude-sonnet-4-0",
    instructions="Generate a compelling product description from the features.",
)
```

### 2. AgentList (Array → Array)

Processes array elements independently, one-to-one mapping.

```python
from pond.agents.agent_list import AgentList

# Analyze sentiment for each review
analyze_sentiments = AgentList(
    Catalog,
    "reviews[:].text",
    "reviews[:].sentiment",
    model="openai:gpt-4o",
    instructions="Analyze sentiment: return Positive, Negative, or Neutral.",
)
```

### 3. AgentListFold (Array → Scalar)

Aggregates array elements into a single output.

```python
from pond.agents.agent_list_fold import AgentListFold

# Summarize all reviews
summarize_reviews = AgentListFold(
    Catalog,
    "reviews[:].text",
    "overall_summary",
    model="anthropic:claude-sonnet-4-0",
    instructions="Summarize all reviews into a cohesive 2-3 sentence summary.",
)
```

## Decorator Syntax (Optional)

You can also use the `@agent_node` decorator:

```python
from pond.decorators import agent_node

@agent_node(
    Catalog,
    "customer_query",
    "support_response",
    model="openai:gpt-4o",
    instructions="Provide helpful customer support responses."
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

- **Catalog** (`Type[BaseModel]`): Your catalog schema
- **input** (`str | list[str]`): Input path(s) in the catalog
- **output** (`str | list[str]`): Output path(s) in the catalog
- **model** (`str`): Model identifier (e.g., "openai:gpt-4o")
- **instructions** (`str`): System prompt guiding agent behavior
- **prompt** (`str | None`): Optional user prompt template
- **input_descriptions** (`list[str] | None`): Descriptions for input fields
- **output_descriptions** (`list[str] | None`): Descriptions for output fields

## Dynamic Type Building

Agents automatically build Pydantic types from catalog schemas:

```python
# Input: catalog.customer_query (str)
# Agent builds: class AgentInput(BaseModel): customer_query: str

# Output: catalog.support_response (str)
# Agent builds: class AgentOutput(BaseModel): support_response: str
```

The LLM receives structured inputs and produces validated structured outputs based on catalog types.

## Pipeline Integration

Agents work seamlessly with pond pipelines:

```python
from pond.decorators import pipe, construct
from pond.runners.sequential_runner import SequentialRunner

pipeline = pipe([
    construct(Catalog),
    analyze_sentiments,    # AgentList
    summarize_reviews,     # AgentListFold
    generate_report,       # Agent
], output="report")

runner = SequentialRunner()
runner.run(state, pipeline)
```

## Multiple Inputs/Outputs

Agents support multiple inputs and outputs:

```python
# Multiple inputs
multi_input_agent = Agent(
    Catalog,
    ["product.name", "product.price", "product.features"],
    "product.marketing_copy",
    model="openai:gpt-4o",
    instructions="Generate marketing copy from product details.",
)

# Multiple outputs
multi_output_agent = Agent(
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
Agent(
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
AbstractAgent (base class)
├── Agent (scalar → scalar)
├── AgentList (array → array)
└── AgentListFold (array → scalar)

ExecuteAgent (execution unit)
├── load_inputs() - Load from catalog
├── run() - Execute pydantic-ai agent
├── save_outputs() - Convert to Arrow tables
└── commit() - Write to catalog
```

## Type Builder Utilities

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
result = agent.run_sync("query")
value = extract_field_value(result.output, "field_name")
```

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
