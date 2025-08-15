# Pipeline Execution

This guide covers how to execute PyPond pipelines using different runners and execution strategies.

## Execution Overview

PyPond provides two main execution strategies:

- **Sequential Runner**: Executes transforms one at a time in order
- **Parallel Runner**: Executes transforms in parallel with automatic dependency resolution

## Sequential Execution

Sequential execution is ideal for development, debugging, and simple pipelines.

### Basic Usage

```python
from pond.runners.sequential_runner import SequentialRunner

runner = SequentialRunner()
runner.run(state, pipeline, hooks=[])
```

### Characteristics

- **Deterministic**: Transforms execute in the exact order they appear in the pipeline
- **Single-threaded**: One transform at a time
- **Easy debugging**: Errors are easy to trace and fix
- **Consistent state**: Pipeline state is always consistent

### When to Use Sequential

- **Development**: When building and testing new pipelines
- **Debugging**: When you need to step through execution
- **Simple pipelines**: When transforms are fast and dependencies are linear
- **Resource constraints**: When memory or CPU resources are limited

## Parallel Execution

Parallel execution maximizes throughput by running independent transforms concurrently.

### Basic Usage

```python
from pond.runners.parallel_runner import ParallelRunner

runner = ParallelRunner()
runner.run(state, pipeline, hooks=[])
```

### Dependency Resolution

The parallel runner automatically analyzes transform dependencies:

```python
# These transforms can run in parallel
@node(Catalog, "input.data_a", "output.result_a")
def process_a(data: DataA) -> ResultA:
    return transform_a(data)

@node(Catalog, "input.data_b", "output.result_b") 
def process_b(data: DataB) -> ResultB:
    return transform_b(data)

# This transform must wait for both above to complete
@node(Catalog, ["output.result_a", "output.result_b"], "final.combined")
def combine_results(a: ResultA, b: ResultB) -> Combined:
    return combine(a, b)
```

### Process Pool Configuration

The parallel runner uses multiprocessing with configurable worker count:

```python
# Currently hardcoded to 10 workers, but can be configured
# Future versions will allow customization
runner = ParallelRunner()
```

### When to Use Parallel

- **Production workloads**: When performance matters
- **Independent transforms**: When transforms can run concurrently
- **Large datasets**: When processing time is significant
- **Multi-core systems**: When you want to utilize available CPU cores

## Pipeline Construction

### Simple Pipelines

```python
from pond import pipe

def simple_pipeline():
    return pipe([
        transform_a,
        transform_b,
        transform_c,
    ])
```

### Pipelines with Dependencies

```python
def complex_pipeline():
    return pipe([
        # Preprocessing
        load_data,
        clean_data,
        validate_data,
        
        # Parallel processing branches
        compute_features,
        compute_statistics,
        
        # Final aggregation
        combine_results,
        generate_report,
    ], input="raw_data", output="final_report")
```

### Nested Pipelines

```python
def preprocessing():
    return pipe([
        load_raw_files,
        parse_format,
        validate_schema,
    ], output="clean_data")

def analysis():
    return pipe([
        compute_metrics,
        run_analysis,
        generate_insights,
    ], input="clean_data", output="results")

def main_pipeline():
    return pipe([
        preprocessing(),
        analysis(),
    ])
```

## Error Handling

### Sequential Runner Errors

In sequential execution, errors stop the pipeline immediately:

```python
try:
    runner.run(state, pipeline, hooks=[])
except RuntimeError as e:
    print(f"Pipeline failed at: {e}")
    # Examine state to see what completed
    # Fix the issue and restart from failed transform
```

### Parallel Runner Errors

In parallel execution, one failure stops all execution:

```python
try:
    runner.run(state, pipeline, hooks=[])
except RuntimeError as e:
    print(f"Pipeline failed: {e}")
    # All running transforms are terminated
    # May need to clean up partial results
```

## Hooks Integration

Both runners support the full hooks system for monitoring and extensibility.

### Progress Monitoring

```python
from pond.hooks.ui_hook import UIHook

# Hamilton UI integration
ui_hook = UIHook(port=8080, username="analyst", project="pipeline")

runner.run(state, pipeline, hooks=[ui_hook])
```

### Custom Monitoring

```python
class TimingHook(AbstractHook):
    def __init__(self):
        self.start_time = None
        self.transform_times = {}
    
    def pre_node_execute(self, transform):
        self.start_time = time.time()
    
    def post_node_execute(self, transform, success, error):
        if self.start_time:
            duration = time.time() - self.start_time
            self.transform_times[transform.get_name()] = duration
            print(f"{transform.get_name()}: {duration:.2f}s")

timing_hook = TimingHook()
runner.run(state, pipeline, hooks=[timing_hook])
```

## Performance Optimization

### Parallel Runner Tips

```python
# Good: Independent transforms that can parallelize
@node(Catalog, "datasets[:].raw", "datasets[:].processed")
def process_dataset(raw: RawData) -> ProcessedData:
    return expensive_processing(raw)

# Less optimal: Sequential dependencies
@node(Catalog, "step1.result", "step2.input")
def step1(data): return process_step1(data)

@node(Catalog, "step2.input", "step3.input") 
def step2(data): return process_step2(data)
```

### Resource Management

```python
# For memory-intensive operations
def memory_efficient_pipeline():
    return pipe([
        # Process in smaller batches
        batch_processor,
        # Clean up intermediate results
        cleanup_temp_data,
        # Continue with final processing
        final_aggregation,
    ])
```

### I/O Optimization

```python
# Minimize catalog reads/writes
@node(Catalog, ["input.a", "input.b", "input.c"], "output.combined")
def combine_inputs(a, b, c):
    # Better: Single transform with multiple inputs
    return combine_all(a, b, c)

# Avoid: Multiple separate reads
# @node(Catalog, "input.a", "temp.a_processed")
# @node(Catalog, "input.b", "temp.b_processed") 
# @node(Catalog, "input.c", "temp.c_processed")
# @node(Catalog, ["temp.a_processed", "temp.b_processed", "temp.c_processed"], "output.combined")
```

## Execution Patterns

### Development Workflow

```python
# 1. Start with sequential for development
dev_runner = SequentialRunner()

# 2. Add comprehensive logging
dev_hooks = [LoggingHook(), TimingHook()]

# 3. Run pipeline
dev_runner.run(state, pipeline, dev_hooks)

# 4. Switch to parallel for production
prod_runner = ParallelRunner()
prod_hooks = [UIHook(), MetricsHook()]
prod_runner.run(state, pipeline, prod_hooks)
```

### Incremental Processing

```python
def incremental_pipeline(state, new_data_only=False):
    if new_data_only:
        # Process only new data
        pipeline = pipe([process_new_data, merge_with_existing])
    else:
        # Full reprocessing
        pipeline = pipe([process_all_data, generate_full_results])
    
    runner = ParallelRunner()
    runner.run(state, pipeline, hooks=[])
```

### Conditional Execution

```python
def adaptive_pipeline(state):
    # Check what needs processing
    if not state.lens("preprocessed_data").exists():
        preprocessing_needed = True
    else:
        preprocessing_needed = False
    
    transforms = []
    if preprocessing_needed:
        transforms.extend([load_data, preprocess_data])
    
    transforms.extend([main_analysis, generate_output])
    
    pipeline = pipe(transforms)
    runner = ParallelRunner()
    runner.run(state, pipeline, hooks=[])
```

## Best Practices

### Runner Selection

- **Sequential for**:
  - Development and debugging
  - Simple, fast pipelines  
  - When dependencies are strictly linear
  - Resource-constrained environments

- **Parallel for**:
  - Production workloads
  - Complex pipelines with independent branches
  - Large datasets requiring significant processing
  - Multi-core systems

### Pipeline Design

- **Minimize dependencies**: Design transforms to be as independent as possible
- **Batch processing**: Group related operations into single transforms
- **Resource awareness**: Consider memory and CPU requirements
- **Error handling**: Design for graceful failure recovery

### Monitoring

- **Always use hooks**: Monitor pipeline execution in production
- **Log performance**: Track execution times and resource usage
- **Health checks**: Validate pipeline state and results
- **Alerting**: Set up notifications for failures

This comprehensive approach to pipeline execution ensures your data processing workflows are both efficient and reliable.