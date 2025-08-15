# Building Transforms

Transforms are the core computational units in PyPond. They define how data flows through your pipeline, processing information from input paths to output paths with automatic type validation.

## Transform Basics

### The @node Decorator

The `@node` decorator is the primary way to create transforms:

```python
from pond import node

@node(Catalog, "input.path", "output.path")
def my_transform(input_value: InputType) -> OutputType:
    # Your processing logic here
    return processed_value
```

### Transform Selection

PyPond automatically selects the appropriate transform type based on your input/output patterns:

#### Scalar Transforms
For simple one-to-one processing:

```python
@node(Catalog, "params.threshold", "results.filtered_count")
def count_above_threshold(threshold: float) -> int:
    # Process scalar input, return scalar output
    data = get_data_somehow()
    return len([x for x in data if x > threshold])
```

#### Array Transforms (TransformList)
When both input and output use array wildcards `[:]`:

```python
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
@node(Catalog, "measurements[:].temperature", "summary.avg_temperature")
def compute_average_temperature(temps: list[float]) -> float:
    # Reduce array to scalar
    return sum(temps) / len(temps)
```

## Multiple Inputs and Outputs

### Multiple Inputs
Combine data from different paths:

```python
@node(Catalog, ["params.scale", "data.raw_values"], "data.scaled_values")
def scale_data(scale: float, values: list[float]) -> list[float]:
    return [v * scale for v in values]
```

### Multiple Outputs
Return tuple for multiple outputs:

```python
@node(Catalog, "data.values", ["stats.mean", "stats.std"])
def compute_statistics(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    return mean, std
```

### Mixed Array and Scalar
Combine array and scalar inputs:

```python
@node(Catalog, ["params.threshold", "measurements[:].value"], "results.above_threshold")
def count_above_threshold(threshold: float, values: list[float]) -> int:
    return len([v for v in values if v > threshold])
```

## Working with Files

### File Inputs
Use the `file:` variant to access file contents directly:

```python
@node(Catalog, "file:datasets[:].raw_data", "datasets[:].processed_data")
def process_arrays(data_list: list[np.ndarray]) -> list[ProcessedData]:
    # data_list is a list of numpy arrays loaded from files
    results = []
    for data in data_list:
        result = apply_processing(data)
        results.append(ProcessedData(values=result.tolist()))
    return results
```

### File Outputs
Store results as files:

```python
@node(Catalog, "datasets[:].measurements", "file:datasets[:].analysis_plot")
def create_plots(measurements: list[float]) -> go.Figure:
    # Return plotly figure - it will be saved as PNG automatically
    fig = px.line(y=measurements)
    return fig
```

## Working with Tables

### Table Inputs
Use the `table:` variant for efficient computation:

```python
@node(Catalog, "table:events[:].data", "summary.event_stats")
def analyze_events(events_table: pa.Table) -> dict[str, float]:
    import pyarrow.compute as pc
    
    # Efficient computation on PyArrow table
    total_events = len(events_table)
    avg_value = pc.mean(events_table["value"]).as_py()
    max_timestamp = pc.max(events_table["timestamp"]).as_py()
    
    return {
        "total": total_events,
        "avg_value": avg_value,
        "latest_time": max_timestamp.isoformat()
    }
```

## Advanced Patterns

### Conditional Processing

```python
@node(Catalog, ["config.enabled", "data.raw"], "data.processed")
def conditional_process(enabled: bool, raw_data: list[float]) -> list[float]:
    if enabled:
        return [x * 2 for x in raw_data]
    else:
        return raw_data  # Pass through unchanged
```

### Error Handling

```python
@node(Catalog, "input.risky_data", "output.safe_result")
def safe_process(data: list[float]) -> dict[str, Any]:
    try:
        result = risky_computation(data)
        return {"success": True, "value": result}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### Nested Data Processing

```python
@node(Catalog, "experiments[:].trials[:].results", "experiments[:].summary")
def summarize_experiment(trial_results: list[list[float]]) -> ExperimentSummary:
    # Flatten nested results
    all_results = [result for trial in trial_results for result in trial]
    
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
class Results(BaseModel):
    count: int
    percentage: float

@node(Catalog, "data.items", "results.count")
def count_items(items: list[str]) -> int:  # ✓ Returns int, matches schema
    return len(items)
```

### Error: Type Mismatch
```python
@node(Catalog, "data.items", "results.count")
def count_items(items: list[str]) -> str:  # ✗ Returns str, expected int
    return str(len(items))
```

## Performance Tips

### Batch Processing
```python
# Good: Process in batches
@node(Catalog, "large_dataset[:].data", "large_dataset[:].processed")
def process_batch(data: np.ndarray) -> np.ndarray:
    # Each array is processed independently, enabling parallelization
    return efficient_processing(data)
```

### Minimize Data Movement
```python
# Good: Use table variant for computation-heavy operations
@node(Catalog, "table:measurements[:].values", "statistics.summary")
def compute_stats(table: pa.Table) -> dict:
    # Computation happens on columnar data - very efficient
    import pyarrow.compute as pc
    return {
        "mean": pc.mean(table["value"]).as_py(),
        "count": len(table)
    }
```

### Cache Expensive Operations
```python
# Use intermediate results to avoid recomputation
@node(Catalog, "raw.data", "intermediate.features")
def extract_features(data: RawData) -> Features:
    # Expensive feature extraction
    return expensive_computation(data)

@node(Catalog, "intermediate.features", "results.prediction")
def make_prediction(features: Features) -> float:
    # Fast prediction using cached features
    return model.predict(features)
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