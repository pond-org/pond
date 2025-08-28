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
import tempfile
from pond import State, node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.sequential_runner import SequentialRunner
from pydantic import BaseModel

# Example pipeline setup
class Input(BaseModel):
    data: list[float]
    
class Output(BaseModel):
    result: float
    
class Pipeline(BaseModel):
    input: Input
    output: Output

@node(Pipeline, "input.data", "output.result")
def compute_average(data: list[float]) -> float:
    return sum(data) / len(data)

# Setup catalog and state
catalog_temp = tempfile.mkdtemp(prefix="seq_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="seq_warehouse_")
catalog = IcebergCatalog(
    "sequential",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(Pipeline, catalog)

# Set input data and run pipeline
state["input.data"] = [1.0, 2.0, 3.0, 4.0, 5.0]
pipeline = pipe([compute_average], input="input", output="output.result")
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
import tempfile
from pond import State, node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.parallel_runner import ParallelRunner
from pydantic import BaseModel

# Example pipeline setup
class Input(BaseModel):
    data_a: list[float]
    data_b: list[float]
    
class Output(BaseModel):
    result_a: float
    result_b: float
    
class Pipeline(BaseModel):
    input: Input
    output: Output

@node(Pipeline, "input.data_a", "output.result_a")
def compute_average_a(data: list[float]) -> float:
    return sum(data) / len(data)

@node(Pipeline, "input.data_b", "output.result_b")
def compute_average_b(data: list[float]) -> float:
    return sum(data) / len(data)

# Setup catalog and state
catalog_temp = tempfile.mkdtemp(prefix="par_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="par_warehouse_")
catalog = IcebergCatalog(
    "parallel",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(Pipeline, catalog)

# Set input data and run pipeline
state["input.data_a"] = [1.0, 2.0, 3.0]
state["input.data_b"] = [4.0, 5.0, 6.0]
pipeline = pipe([compute_average_a, compute_average_b], input="input", output=["output.result_a", "output.result_b"])
runner = ParallelRunner(max_workers=0)  # Use max_workers=0 for docs testing
runner.run(state, pipeline, hooks=[])
```

### Dependency Resolution

The parallel runner automatically analyzes transform dependencies:

```python
from pond import node
from pydantic import BaseModel

# Define data models
class DataA(BaseModel):
    values: list[float]
    
class DataB(BaseModel):
    values: list[float]
    
class ResultA(BaseModel):
    result: float
    
class ResultB(BaseModel):
    result: float
    
class Combined(BaseModel):
    total: float
    
class Input(BaseModel):
    data_a: DataA
    data_b: DataB
    
class Output(BaseModel):
    result_a: ResultA
    result_b: ResultB
    
class Final(BaseModel):
    combined: Combined
    
class Catalog(BaseModel):
    input: Input
    output: Output
    final: Final

# These transforms can run in parallel
@node(Catalog, "input.data_a", "output.result_a")
def process_a(data: DataA) -> ResultA:
    return ResultA(result=sum(data.values) / len(data.values))

@node(Catalog, "input.data_b", "output.result_b") 
def process_b(data: DataB) -> ResultB:
    return ResultB(result=sum(data.values) / len(data.values))

# This transform must wait for both above to complete
@node(Catalog, ["output.result_a", "output.result_b"], "final.combined")
def combine_results(a: ResultA, b: ResultB) -> Combined:
    return Combined(total=a.result + b.result)
```

### Process Pool Configuration

The parallel runner uses multiprocessing with configurable worker count:

```python
from pond.runners.parallel_runner import ParallelRunner

# Configurable number of worker processes
runner = ParallelRunner(max_workers=10)  # Default is 10 workers
print(f"Parallel runner created with 10 workers for production use")
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
import tempfile
from pond import State, node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.sequential_runner import SequentialRunner
from pydantic import BaseModel

# Example with error handling
class Input(BaseModel):
    data: list[float]
    
class Output(BaseModel):
    result: float

class Pipeline(BaseModel):
    input: Input
    output: Output

@node(Pipeline, "input.data", "output.result")
def failing_transform(data: list[float]) -> float:
    if len(data) == 0:
        raise ValueError("Empty data provided")
    return sum(data) / len(data)

# Setup
catalog_temp = tempfile.mkdtemp(prefix="error_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="error_warehouse_")
catalog = IcebergCatalog(
    "error_test",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(Pipeline, catalog)
pipeline = pipe([failing_transform], input="input", output="output.result")
runner = SequentialRunner()

# Set input data and run with error handling
state["input.data"] = [1.0, 2.0, 3.0, 4.0, 5.0]
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
import tempfile
from pond import State, node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.parallel_runner import ParallelRunner
from pydantic import BaseModel

# Example with parallel error handling
class Input(BaseModel):
    data: list[float]
    
class Output(BaseModel):
    result: float

class Pipeline(BaseModel):
    input: Input
    output: Output

@node(Pipeline, "input.data", "output.result")
def failing_transform(data: list[float]) -> float:
    if len(data) == 0:
        raise ValueError("Empty data provided")
    return sum(data) / len(data)

# Setup
catalog_temp = tempfile.mkdtemp(prefix="par_error_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="par_error_warehouse_")
catalog = IcebergCatalog(
    "par_error_test",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(Pipeline, catalog)
pipeline = pipe([failing_transform], input="input", output="output.result")
runner = ParallelRunner(max_workers=0)  # Use max_workers=0 for docs testing

# Set input data and run with parallel error handling
state["input.data"] = [1.0, 2.0, 3.0, 4.0, 5.0]
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
import tempfile
from pond import State, node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.sequential_runner import SequentialRunner
from pond.hooks.abstract_hook import AbstractHook
from pydantic import BaseModel

# Simple progress monitoring
class ProgressHook(AbstractHook):
    def __init__(self):
        self.completed_count = 0
        
    def pre_node_execute(self, transform):
        print(f"Starting: {transform.get_name()}")
        
    def post_node_execute(self, transform, success, error):
        if success:
            self.completed_count += 1
            print(f"Completed: {transform.get_name()} (Total: {self.completed_count})")
        else:
            print(f"Failed: {transform.get_name()} - {error}")

# Example pipeline setup
class Input(BaseModel):
    data: list[float]
    
class Output(BaseModel):
    result: float
    
class Pipeline(BaseModel):
    input: Input
    output: Output

@node(Pipeline, "input.data", "output.result")
def compute_average(data: list[float]) -> float:
    return sum(data) / len(data)

# Setup catalog and state
catalog_temp = tempfile.mkdtemp(prefix="progress_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="progress_warehouse_")
catalog = IcebergCatalog(
    "progress_test",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(Pipeline, catalog)

# Set input data and run with progress monitoring
state["input.data"] = [1.0, 2.0, 3.0, 4.0, 5.0]
pipeline = pipe([compute_average], input="input", output="output.result")
runner = SequentialRunner()
progress_hook = ProgressHook()
runner.run(state, pipeline, hooks=[progress_hook])
```

### Custom Monitoring

```python
import time
import tempfile
from pond import State, node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.sequential_runner import SequentialRunner
from pond.hooks.abstract_hook import AbstractHook
from pydantic import BaseModel

# Custom timing hook
class TimingHook(AbstractHook):
    def __init__(self):
        self.start_time = None
        self.transform_times = {}
    
    def pre_node_execute(self, transform):
        self.start_time = time.time()
        print(f"Starting: {transform.get_name()}")
    
    def post_node_execute(self, transform, success, error):
        if self.start_time:
            duration = time.time() - self.start_time
            self.transform_times[transform.get_name()] = duration
            print(f"{transform.get_name()}: {duration:.2f}s")

# Example pipeline setup
class Input(BaseModel):
    data: list[float]
    
class Output(BaseModel):
    result: float
    
class Pipeline(BaseModel):
    input: Input
    output: Output

@node(Pipeline, "input.data", "output.result")
def compute_average(data: list[float]) -> float:
    # Add small delay for timing demonstration
    time.sleep(0.1)
    return sum(data) / len(data)

# Setup catalog and state
catalog_temp = tempfile.mkdtemp(prefix="timing_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="timing_warehouse_")
catalog = IcebergCatalog(
    "timing_test",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(Pipeline, catalog)

# Set sample data and run with timing
state["input.data"] = [1.0, 2.0, 3.0, 4.0, 5.0]
pipeline = pipe([compute_average], input="input", output="output.result")
runner = SequentialRunner()
timing_hook = TimingHook()
runner.run(state, pipeline, hooks=[timing_hook])
```

## Performance Optimization

### Parallel Runner Tips

```python
from pond import node
from pydantic import BaseModel

# Define data models
class RawData(BaseModel):
    values: list[float]
    
class ProcessedData(BaseModel):
    normalized: list[float]
    
class Dataset(BaseModel):
    raw: RawData
    processed: ProcessedData
    
class Step1Result(BaseModel):
    intermediate: list[float]
    
class Step2Input(BaseModel):
    intermediate: list[float]
    
class Step3Input(BaseModel):
    final: list[float]
    
class Catalog(BaseModel):
    datasets: list[Dataset]
    step1: Step1Result
    step2: Step2Input
    step3: Step3Input

# Good: Independent transforms that can parallelize
@node(Catalog, "datasets[:].raw", "datasets[:].processed")
def process_dataset(raw: RawData) -> ProcessedData:
    # Expensive processing - each dataset processed independently
    normalized = [(x - sum(raw.values)/len(raw.values)) for x in raw.values]
    return ProcessedData(normalized=normalized)

# Less optimal: Sequential dependencies
@node(Catalog, "step1.intermediate", "step2.intermediate")
def step1(data: list[float]) -> list[float]:
    return [x * 2 for x in data]

@node(Catalog, "step2.intermediate", "step3.final") 
def step2(data: list[float]) -> list[float]:
    return [x + 1 for x in data]
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
from pond import node
from pydantic import BaseModel

# Define data models
class InputA(BaseModel):
    values: list[float]
    
class InputB(BaseModel):
    values: list[float]
    
class InputC(BaseModel):
    values: list[float]
    
class Input(BaseModel):
    a: InputA
    b: InputB
    c: InputC
    
class Combined(BaseModel):
    result: float
    
class Output(BaseModel):
    combined: Combined
    
class Catalog(BaseModel):
    input: Input
    output: Output

# Minimize catalog reads/writes
@node(Catalog, ["input.a", "input.b", "input.c"], "output.combined")
def combine_inputs(a: InputA, b: InputB, c: InputC) -> Combined:
    # Better: Single transform with multiple inputs
    all_values = a.values + b.values + c.values
    result = sum(all_values) / len(all_values)
    return Combined(result=result)

# This approach minimizes I/O compared to multiple separate reads
print("I/O optimization example: single transform with multiple inputs")
```

## Execution Patterns

### Development Workflow

```python
import tempfile
from pond import State, node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.sequential_runner import SequentialRunner
from pond.runners.parallel_runner import ParallelRunner
from pydantic import BaseModel

# Example development workflow
class Input(BaseModel):
    data: list[float]
    
class Output(BaseModel):
    result: float
    
class Pipeline(BaseModel):
    input: Input
    output: Output

@node(Pipeline, "input.data", "output.result")
def compute_average(data: list[float]) -> float:
    return sum(data) / len(data)

# Setup
catalog_temp = tempfile.mkdtemp(prefix="workflow_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="workflow_warehouse_")
catalog = IcebergCatalog(
    "workflow",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(Pipeline, catalog)
state["input.data"] = [1.0, 2.0, 3.0, 4.0, 5.0]
pipeline = pipe([compute_average], input="input", output="output.result")

# 1. Start with sequential for development
dev_runner = SequentialRunner()
print("Development phase: Sequential execution")
dev_runner.run(state, pipeline, hooks=[])

# 2. Switch to parallel for production
prod_runner = ParallelRunner(max_workers=0)  # Use max_workers=0 for docs testing
print("Production phase: Parallel execution")
prod_runner.run(state, pipeline, hooks=[])
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
    
    runner = ParallelRunner(max_workers=0)  # Use max_workers=0 for docs testing
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
    runner = ParallelRunner(max_workers=0)  # Use max_workers=0 for docs testing
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