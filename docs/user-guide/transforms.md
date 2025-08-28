# Building Transforms

Transforms are the core computational units in PyPond. They define how data flows through your pipeline, processing information from input paths to output paths with automatic type validation.

## Transform Basics

### The @node Decorator

The `@node` decorator is the primary way to create transforms:

```python
from pond import node
from pydantic import BaseModel

# Define your catalog structure
class Input(BaseModel):
    path: str
    
class Output(BaseModel):
    path: str
    
class Catalog(BaseModel):
    input: Input
    output: Output

@node(Catalog, "input.path", "output.path")
def my_transform(input_value: str) -> str:
    # Your processing logic here
    processed_value = input_value.upper()  # Example processing
    return processed_value
```

### Transform Selection

PyPond automatically selects the appropriate transform type based on your input/output patterns:

#### Scalar Transforms
For simple one-to-one processing:

```python
from pond import node
from pydantic import BaseModel

class Params(BaseModel):
    threshold: float
    
class Results(BaseModel):
    filtered_count: int
    
class Catalog(BaseModel):
    params: Params
    results: Results

@node(Catalog, "params.threshold", "results.filtered_count")
def count_above_threshold(threshold: float) -> int:
    # Process scalar input, return scalar output
    data = [1.0, 2.5, 3.8, 1.2, 4.9, 0.8]  # Example data
    return len([x for x in data if x > threshold])
```

#### Array Transforms (TransformList)
When both input and output use array wildcards `[:]`:

```python
from pond import node
from pydantic import BaseModel

class DataItem(BaseModel):
    values: list[float]
    normalized: list[float]
    
class Catalog(BaseModel):
    raw_data: list[DataItem]
    processed_data: list[DataItem]

@node(Catalog, "raw_data[:].values", "processed_data[:].normalized")
def normalize_values(values: list[float]) -> list[float]:
    # Process each array element
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    return [(x - mean) / std for x in values]
```

#### Array Fold Transforms (TransformListFold)
When input uses wildcards `[:]` but output is scalar:

```python
from pond import node
from pydantic import BaseModel

class Measurement(BaseModel):
    temperature: float
    
class Summary(BaseModel):
    avg_temperature: float
    
class Catalog(BaseModel):
    measurements: list[Measurement]
    summary: Summary

@node(Catalog, "measurements[:].temperature", "summary.avg_temperature")
def compute_average_temperature(temps: list[float]) -> float:
    # Reduce array to scalar
    return sum(temps) / len(temps)
```

## Multiple Inputs and Outputs

### Multiple Inputs
Combine data from different paths:

```python
from pond import node
from pydantic import BaseModel

class Params(BaseModel):
    scale: float
    
class Data(BaseModel):
    raw_values: list[float]
    scaled_values: list[float]
    
class Catalog(BaseModel):
    params: Params
    data: Data

@node(Catalog, ["params.scale", "data.raw_values"], "data.scaled_values")
def scale_data(scale: float, values: list[float]) -> list[float]:
    return [v * scale for v in values]
```

### Multiple Outputs
Return tuple for multiple outputs:

```python
from pond import node
from pydantic import BaseModel

class Data(BaseModel):
    values: list[float]
    
class Stats(BaseModel):
    mean: float
    std: float
    
class Catalog(BaseModel):
    data: Data
    stats: Stats

@node(Catalog, "data.values", ["stats.mean", "stats.std"])
def compute_statistics(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    return mean, std
```

### Mixed Array and Scalar
Combine array and scalar inputs:

```python
from pond import node
from pydantic import BaseModel

class Params(BaseModel):
    threshold: float
    
class Measurement(BaseModel):
    value: float
    
class Results(BaseModel):
    above_threshold: int
    
class Catalog(BaseModel):
    params: Params
    measurements: list[Measurement]
    results: Results

@node(Catalog, ["params.threshold", "measurements[:].value"], "results.above_threshold")
def count_above_threshold(threshold: float, values: list[float]) -> int:
    return len([v for v in values if v > threshold])
```

## Working with Files

### File Inputs
Use the `file:` variant to access file contents directly:

```python
from pond import node, Field, File
from pond.io.readers import read_npz
from pond.io.writers import write_npz
from pydantic import BaseModel
import numpy as np

class ProcessedData(BaseModel):
    values: list[float]
    
class Dataset(BaseModel):
    raw_data: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npz")
    processed_data: ProcessedData
    
class Catalog(BaseModel):
    datasets: list[Dataset]

@node(Catalog, "file:datasets[:].raw_data", "datasets[:].processed_data")
def process_arrays(data: np.ndarray) -> ProcessedData:
    # data is a numpy array loaded from file
    # Apply some processing (e.g., normalization)
    processed = (data - data.mean()) / data.std()
    return ProcessedData(values=processed.tolist())
```

### File Outputs
Store results as files:

```python
from pond import node, Field, File
from pond.io.writers import write_plotly_png
from pydantic import BaseModel
import plotly.graph_objects as go
import plotly.express as px

class Dataset(BaseModel):
    measurements: list[float]
    analysis_plot: File[go.Figure] = Field(writer=write_plotly_png, ext="png")
    
class Catalog(BaseModel):
    datasets: list[Dataset]

@node(Catalog, "datasets[:].measurements", "file:datasets[:].analysis_plot")
def create_plots(measurements: list[float]) -> go.Figure:
    # Return plotly figure - it will be saved as PNG automatically
    fig = px.line(y=measurements, title="Analysis Plot")
    return fig
```

## Working with Tables

### Table Inputs
Use the `table:` variant for efficient computation:

```python
from pond import node
from pydantic import BaseModel
import pyarrow as pa
import pyarrow.compute as pc
from datetime import datetime

class EventData(BaseModel):
    value: float
    timestamp: datetime
    
class EventStats(BaseModel):
    total: int
    avg_value: float
    latest_time: str
    
class Summary(BaseModel):
    event_stats: EventStats
    
class Event(BaseModel):
    data: EventData
    
class Catalog(BaseModel):
    events: list[Event]
    summary: Summary

@node(Catalog, "table:events", "summary.event_stats")
def analyze_events(events_table: pa.Table) -> EventStats:
    # Efficient computation on PyArrow table
    total_events = len(events_table)
    avg_value = pc.mean(events_table["data.value"]).as_py()
    max_timestamp = pc.max(events_table["data.timestamp"]).as_py()
    
    return EventStats(
        total=total_events,
        avg_value=avg_value,
        latest_time=max_timestamp.isoformat()
    )
```

## Advanced Patterns

### Conditional Processing

```python
from pond import node
from pydantic import BaseModel

class Config(BaseModel):
    enabled: bool
    
class Data(BaseModel):
    raw: list[float]
    processed: list[float]
    
class Catalog(BaseModel):
    config: Config
    data: Data

@node(Catalog, ["config.enabled", "data.raw"], "data.processed")
def conditional_process(enabled: bool, raw_data: list[float]) -> list[float]:
    if enabled:
        return [x * 2 for x in raw_data]
    else:
        return raw_data  # Pass through unchanged
```

### Error Handling

```python
from pond import node
from pydantic import BaseModel
from typing import Any

class SafeResult(BaseModel):
    success: bool
    value: float = 0.0
    error: str = ""
    
class Input(BaseModel):
    risky_data: list[float]
    
class Output(BaseModel):
    safe_result: SafeResult
    
class Catalog(BaseModel):
    input: Input
    output: Output

@node(Catalog, "input.risky_data", "output.safe_result")
def safe_process(data: list[float]) -> SafeResult:
    try:
        # Example risky computation
        if len(data) == 0:
            raise ValueError("Empty data")
        result = sum(data) / len(data)  # Could fail with division by zero
        return SafeResult(success=True, value=result)
    except Exception as e:
        return SafeResult(success=False, error=str(e))
```

### Nested Data Processing

```python
from pond import node
from pydantic import BaseModel

class ExperimentSummary(BaseModel):
    total_trials: int
    total_results: int
    avg_result: float
    best_trial: int
    
class Trial(BaseModel):
    results: list[float]
    
class Experiment(BaseModel):
    trials: list[Trial]
    summary: ExperimentSummary
    
class Catalog(BaseModel):
    experiments: list[Experiment]

@node(Catalog, "experiments[:].trials", "experiments[:].summary")
def summarize_experiment(trials: list[Trial]) -> ExperimentSummary:
    # Extract results from trials
    trial_results = [trial.results for trial in trials]
    all_results = [result for trial_result in trial_results for result in trial_result]
    
    return ExperimentSummary(
        total_trials=len(trial_results),
        total_results=len(all_results),
        avg_result=sum(all_results) / len(all_results),
        best_trial=max(range(len(trial_results)), 
                      key=lambda i: max(trial_results[i]))
    )
```

## Type Validation

PyPond performs strict type checking between your function signatures and catalog schema:

### Good: Types Match
```python
from pond import node
from pydantic import BaseModel

class Data(BaseModel):
    items: list[str]
    
class Results(BaseModel):
    count: int
    percentage: float
    
class Catalog(BaseModel):
    data: Data
    results: Results

@node(Catalog, "data.items", "results.count")
def count_items(items: list[str]) -> int:  # ✓ Returns int, matches schema
    return len(items)
```

### Error: Type Mismatch
```python
from pond import node
from pydantic import BaseModel

class Data(BaseModel):
    items: list[str]
    
class Results(BaseModel):
    count: int
    
class Catalog(BaseModel):
    data: Data
    results: Results

# This would cause a type validation error (commented out to avoid test failure):
# @node(Catalog, "data.items", "results.count")
# def count_items(items: list[str]) -> str:  # ✗ Returns str, expected int
#     return str(len(items))  
print("Example: Type mismatch - function returns str but catalog expects int")
```

## Performance Tips

### Batch Processing
```python
from pond import node
from pydantic import BaseModel
import numpy as np

class LargeDataItem(BaseModel):
    data: list[float]  # Simplified for example
    processed: list[float]
    
class Catalog(BaseModel):
    large_dataset: list[LargeDataItem]

# Good: Process in batches
@node(Catalog, "large_dataset[:].data", "large_dataset[:].processed")
def process_batch(data: list[float]) -> list[float]:
    # Each array is processed independently, enabling parallelization
    return [x * 2.0 for x in data]  # Example processing
```

### Minimize Data Movement
```python
from pond import node
from pydantic import BaseModel
import pyarrow as pa
import pyarrow.compute as pc

class Measurement(BaseModel):
    values: float
    
class StatsSummary(BaseModel):
    mean: float
    count: int
    
class Statistics(BaseModel):
    summary: StatsSummary
    
class Catalog(BaseModel):
    measurements: list[Measurement]
    statistics: Statistics

# Good: Use table variant for computation-heavy operations
@node(Catalog, "table:measurements", "statistics.summary")
def compute_stats(table: pa.Table) -> StatsSummary:
    # Computation happens on columnar data - very efficient
    return StatsSummary(
        mean=pc.mean(table["values"]).as_py(),
        count=len(table)
    )
```

### Cache Expensive Operations
```python
from pond import node
from pydantic import BaseModel

class RawData(BaseModel):
    values: list[float]
    
class Features(BaseModel):
    feature_vector: list[float]
    
class Raw(BaseModel):
    data: RawData
    
class Intermediate(BaseModel):
    features: Features
    
class Results(BaseModel):
    prediction: float
    
class Catalog(BaseModel):
    raw: Raw
    intermediate: Intermediate
    results: Results

# Use intermediate results to avoid recomputation
@node(Catalog, "raw.data", "intermediate.features")
def extract_features(data: RawData) -> Features:
    # Expensive feature extraction (simplified example)
    feature_vector = [sum(data.values), len(data.values), max(data.values)]
    return Features(feature_vector=feature_vector)

@node(Catalog, "intermediate.features", "results.prediction")
def make_prediction(features: Features) -> float:
    # Fast prediction using cached features
    return sum(features.feature_vector) / len(features.feature_vector)
```

## Testing Transforms

### Unit Testing
```python
def test_my_transform():
    # Test the function directly
    result = my_transform.fn(test_input)
    assert result == expected_output

def test_transform_types():
    # Test type compatibility
    from pond.transforms.transform import Transform
    transform = Transform(Catalog, "input.path", "output.path", my_transform.fn)
    # Validates types automatically
```

### Integration Testing
```python
def test_pipeline_integration():
    catalog = IcebergCatalog(name="test")
    state = State(Catalog, catalog)
    
    # Set up test data
    state["input.path"] = test_data
    
    # Run transform
    runner = SequentialRunner()
    pipeline = pipe([my_transform])
    runner.run(state, pipeline, [])
    
    # Check results
    result = state["output.path"]
    assert result == expected_result
```

## Best Practices

### Function Design
- **Pure functions**: Avoid side effects, use only inputs and return outputs
- **Type hints**: Always provide complete type annotations  
- **Documentation**: Add docstrings explaining the transform's purpose
- **Error handling**: Handle edge cases gracefully

### Path Design
- **Descriptive names**: Use clear, hierarchical path names
- **Consistent patterns**: Follow naming conventions across your project
- **Future-proof**: Design paths that can evolve with your schema

### Performance
- **Batch operations**: Prefer array operations over scalar loops
- **Appropriate variants**: Use `table:` for computation, `file:` for direct access
- **Lazy evaluation**: Let PyPond handle data loading efficiently

This comprehensive approach to building transforms ensures your data pipelines are both powerful and maintainable.