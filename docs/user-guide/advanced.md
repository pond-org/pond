# Advanced Usage

This guide covers advanced PyPond patterns and techniques for building sophisticated data pipelines.

## Complex Pipeline Patterns

### Conditional Pipeline Construction

Build pipelines that adapt to different conditions:

```python
def adaptive_pipeline(config: dict):
    transforms = []
    
    # Always include preprocessing
    transforms.extend([
        load_raw_data,
        validate_schema,
        clean_data
    ])
    
    # Conditional feature extraction
    if config.get("enable_advanced_features"):
        transforms.extend([
            extract_advanced_features,
            validate_features
        ])
    else:
        transforms.append(extract_basic_features)
    
    # Optional ML training
    if config.get("train_model"):
        transforms.extend([
            prepare_training_data,
            train_model,
            evaluate_model
        ])
    
    # Always generate results
    transforms.extend([
        generate_predictions,
        create_report
    ])
    
    return pipe(transforms, input="raw_data", output="final_report")
```

### Multi-Stage Pipelines

Break complex workflows into logical stages:

```python
def data_ingestion_stage():
    return pipe([
        discover_source_files,
        validate_file_formats,
        standardize_schemas,
        create_metadata
    ], output="ingested_data")

def preprocessing_stage():
    return pipe([
        quality_check,
        data_cleaning,
        feature_engineering,
        validation
    ], input="ingested_data", output="preprocessed_data")

def analysis_stage():
    return pipe([
        exploratory_analysis,
        statistical_modeling,
        result_validation
    ], input="preprocessed_data", output="analysis_results")

def reporting_stage():
    return pipe([
        generate_visualizations,
        create_summary_tables,
        compile_final_report
    ], input="analysis_results", output="final_deliverable")

def full_pipeline():
    return pipe([
        data_ingestion_stage(),
        preprocessing_stage(),
        analysis_stage(),
        reporting_stage()
    ])
```

### Pipeline Composition with Branching

Create pipelines with parallel processing branches:

```python
def parallel_analysis_pipeline():
    # Shared preprocessing
    preprocessing = pipe([
        load_data,
        clean_data,
        validate_data
    ], output="clean_data")
    
    # Parallel analysis branches
    statistical_branch = pipe([
        compute_descriptive_stats,
        run_hypothesis_tests,
        generate_stat_report
    ], input="clean_data", output="statistical_results")
    
    ml_branch = pipe([
        prepare_ml_features,
        train_models,
        evaluate_performance
    ], input="clean_data", output="ml_results")
    
    visualization_branch = pipe([
        create_exploratory_plots,
        generate_dashboards,
        export_visualizations
    ], input="clean_data", output="visual_results")
    
    # Final aggregation
    final_stage = pipe([
        combine_analysis_results,
        create_executive_summary
    ], input=["statistical_results", "ml_results", "visual_results"], 
       output="comprehensive_report")
    
    return pipe([
        preprocessing,
        statistical_branch,
        ml_branch, 
        visualization_branch,
        final_stage
    ])
```

## Advanced Data Access Patterns

### Dynamic Path Construction

Build paths programmatically for processing collections of data:

```python
def process_experiment_results(experiment_id: str, trial_count: int):
    transforms = []
    
    for trial_idx in range(trial_count):
        # Create transform for each trial
        @node(Catalog, f"experiments[{experiment_id}].trials[{trial_idx}].raw_data", 
                      f"experiments[{experiment_id}].trials[{trial_idx}].processed_data")
        def process_trial(raw_data: RawData) -> ProcessedData:
            return process_single_trial(raw_data)
        
        transforms.append(process_trial)
    
    # Aggregate all trial results
    @node(Catalog, f"experiments[{experiment_id}].trials[:].processed_data",
                   f"experiments[{experiment_id}].summary")
    def aggregate_trials(trial_results: list[ProcessedData]) -> ExperimentSummary:
        return aggregate_experiment_results(trial_results)
    
    transforms.append(aggregate_trials)
    return pipe(transforms)
```

### Cross-Catalog Operations

Work with multiple catalogs simultaneously:

```python
def cross_catalog_analysis():
    # Historical data catalog
    historical_catalog = IcebergCatalog(name="historical", db_path="./warehouse/historical")
    historical_state = State(HistoricalSchema, historical_catalog)
    
    # Real-time data catalog  
    realtime_catalog = LanceCatalog(db_path="./realtime_data")
    realtime_state = State(RealtimeSchema, realtime_catalog)
    
    # Analysis catalog
    analysis_catalog = DeltaCatalog(db_path="./analysis_results") 
    analysis_state = State(AnalysisSchema, analysis_catalog)
    
    # Cross-catalog transform
    @node(AnalysisSchema, ["historical_baseline", "realtime_metrics"], "comparison_results")
    def compare_datasets(baseline: HistoricalData, metrics: RealtimeData) -> ComparisonResults:
        return perform_comparison(baseline, metrics)
    
    # Load data from different catalogs
    historical_data = historical_state["baseline_metrics"]
    realtime_data = realtime_state["current_metrics"]
    
    # Store in analysis catalog
    analysis_state["historical_baseline"] = historical_data
    analysis_state["realtime_metrics"] = realtime_data
    
    # Run comparison
    pipeline = pipe([compare_datasets])
    runner = SequentialRunner()
    runner.run(analysis_state, pipeline, [])
    
    return analysis_state["comparison_results"]
```

### Advanced Lens Operations

Use lenses for sophisticated data access and manipulation patterns:

```python
import tempfile
import numpy as np
from pydantic import BaseModel
from pond import State
from pond.catalogs.iceberg_catalog import IcebergCatalog

# Setup catalog for example
temp_dir = tempfile.mkdtemp(prefix="pypond_advanced_")
catalog = IcebergCatalog(
    "default",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# Define example schema
class DatasetMetadata(BaseModel):
    status: str
    priority: str

class Dataset(BaseModel):
    metadata: DatasetMetadata
    raw_data: str
    processed_data: str = ""

class Catalog(BaseModel):
    datasets: list[Dataset]
    high_priority_results: list[str] = []

state = State(Catalog, catalog)

def process_dataset(raw_data: str) -> str:
    return f"processed_{raw_data}"

def batch_process(datasets: list[Dataset]) -> list[str]:
    return [f"batch_processed_{d.raw_data}" for d in datasets]

def advanced_lens_operations(state: State):
    # Conditional data processing
    datasets_lens = state.lens("datasets[:]")
    
    for i in range(datasets_lens.len()):
        dataset_lens = state.lens(f"datasets[{i}]")
        
        if dataset_lens.lens("metadata.status").get() == "pending":
            # Process only pending datasets
            raw_data = dataset_lens.lens("raw_data").get()
            processed = process_dataset(raw_data)
            dataset_lens.lens("processed_data").set(processed)
            dataset_lens.lens("metadata.status").set("completed")
    
    # Batch operations with filtering
    pending_datasets = []
    for i in range(datasets_lens.len()):
        dataset_lens = state.lens(f"datasets[{i}]")
        if dataset_lens.lens("metadata.priority").get() == "high":
            pending_datasets.append(dataset_lens.get())
    
    # Process high-priority datasets in batch
    if pending_datasets:
        batch_results = batch_process(pending_datasets)
        for i, result in enumerate(batch_results):
            # Store results back
            state.lens(f"high_priority_results[{i}]").set(result)
```

## Performance Optimization

### Memory-Efficient Processing

Process large datasets efficiently with chunked operations:

```python
import tempfile
import pyarrow as pa
import pyarrow.compute as pc
from pydantic import BaseModel
from pond import node
from pond.catalogs.iceberg_catalog import IcebergCatalog

# Setup catalog for example
temp_dir = tempfile.mkdtemp(prefix="pypond_streaming_")
catalog = IcebergCatalog(
    "default",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# Define example schema
class Summary(BaseModel):
    statistics: dict

class DataPoint(BaseModel):
    value: float
    category: str

class Catalog(BaseModel):
    summary: Summary
    large_dataset: list[DataPoint] = []

@node(Catalog, "table:large_dataset", "summary.statistics")
def streaming_analysis(table: pa.Table) -> dict:
    # Efficiently process large Arrow tables in memory-managed chunks
    chunk_size = 10000
    total_rows = len(table)
    
    running_sum = 0
    running_count = 0
    min_val = float('inf')
    max_val = float('-inf')
    
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = table.slice(start_idx, end_idx - start_idx)
        
        # Compute statistics on chunk
        chunk_sum = pc.sum(chunk['value']).as_py()
        chunk_min = pc.min(chunk['value']).as_py()
        chunk_max = pc.max(chunk['value']).as_py()
        
        # Update running statistics
        running_sum += chunk_sum
        running_count += len(chunk)
        min_val = min(min_val, chunk_min)
        max_val = max(max_val, chunk_max)
    
    return {
        "mean": running_sum / running_count,
        "min": min_val,
        "max": max_val,
        "count": running_count
    }
```

### Parallel File Processing

Optimize file operations for parallel execution:

```python
import tempfile
import time
import numpy as np
from pydantic import BaseModel
from pond import node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog

# Setup catalog for example
temp_dir = tempfile.mkdtemp(prefix="pypond_parallel_")
catalog = IcebergCatalog(
    "default",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# Define example models
from pond import Field, File
from pond.io.readers import read_npz
from pond.io.writers import write_npz

class ProcessingResult(BaseModel):
    features: list[float]
    statistics: dict
    insights: list[str]
    metadata: dict

class LargeFile(BaseModel):
    data: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npz")

class ProcessedFile(BaseModel):
    results: ProcessingResult

class Catalog(BaseModel):
    processed_files: list[ProcessedFile] = []
    large_files: list[LargeFile] = []

def extract_features(data: np.ndarray) -> list[float]:
    return [float(np.mean(data)), float(np.std(data))]

def compute_statistics(data: np.ndarray) -> dict:
    return {"size": len(data), "min": float(np.min(data)), "max": float(np.max(data))}

def analyze_patterns(data: np.ndarray) -> list[str]:
    return ["pattern1", "pattern2"]

@node(Catalog, "file:large_files[:].data", "processed_files[:].results")
def parallel_file_processor(data: np.ndarray) -> ProcessingResult:
    # This transform processes each file independently
    # Perfect for parallel execution
    
    # CPU-intensive processing
    features = extract_features(data)
    statistics = compute_statistics(data)
    insights = analyze_patterns(data)
    
    return ProcessingResult(
        features=features,
        statistics=statistics,
        insights=insights,
        metadata={"processing_time": time.time()}
    )

def index_large_files():
    pass

def aggregate_results():
    pass

def generate_summary_report():
    pass

def efficient_file_pipeline():
    return pipe([
        index_large_files,          # Discover files
        parallel_file_processor,    # Process in parallel
        aggregate_results,          # Combine results
        generate_summary_report     # Create final output
    ])
```

### Caching and Memoization

Implement intelligent caching for expensive operations:

```python
import tempfile
from pydantic import BaseModel
from pond import node
from pond.catalogs.iceberg_catalog import IcebergCatalog

# Setup catalog for example
temp_dir = tempfile.mkdtemp(prefix="pypond_cache_")
catalog = IcebergCatalog(
    "default",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# Define example models
class InputData(BaseModel):
    value: float

class OutputData(BaseModel):
    result: float

class Input(BaseModel):
    data: InputData

class Output(BaseModel):
    processed: OutputData

class Catalog(BaseModel):
    input: Input
    output: Output

def expensive_computation(data: InputData) -> OutputData:
    return OutputData(result=data.value * 2)

class CachedTransform:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.cache = {}
    
    def __call__(self, input_data: InputData) -> OutputData:
        # Create cache key from input
        cache_key = hash(input_data.model_dump_json())
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Expensive computation
        result = expensive_computation(input_data)
        
        # Cache result
        self.cache[cache_key] = result
        return result

# Use in pipeline
cached_processor = CachedTransform("./cache")

@node(Catalog, "input.data", "output.processed")
def cached_transform(data: InputData) -> OutputData:
    return cached_processor(data)
```

## Error Handling and Resilience

### Graceful Error Handling

Handle errors without stopping the entire pipeline:

```python
import tempfile
import logging
from typing import Optional
from pydantic import BaseModel
from pond import node
from pond.catalogs.iceberg_catalog import IcebergCatalog

# Setup catalog for example
temp_dir = tempfile.mkdtemp(prefix="pypond_resilient_")
catalog = IcebergCatalog(
    "default",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# Setup logger
logger = logging.getLogger(__name__)

# Define example models
class RiskyData(BaseModel):
    id: str
    value: float

class SafeResult(BaseModel):
    success: bool
    data: Optional[str]
    error: Optional[str]

class ProcessingSummary(BaseModel):
    total_processed: int
    successful_count: int
    failed_count: int
    success_rate: float
    errors: list[str]

class RiskyInput(BaseModel):
    data: RiskyData

class SafeOutput(BaseModel):
    results: SafeResult

class Catalog(BaseModel):
    risky_inputs: list[RiskyInput] = []
    safe_outputs: list[SafeOutput] = []
    final_summary: ProcessingSummary

def risky_processing(data: RiskyData) -> str:
    if data.value < 0:
        raise ValueError("Negative values not allowed")
    return f"processed_{data.id}"

@node(Catalog, "risky_inputs[:].data", "safe_outputs[:].results")
def resilient_processor(data: RiskyData) -> SafeResult:
    try:
        # Potentially failing operation
        result = risky_processing(data)
        return SafeResult(
            success=True,
            data=result,
            error=None
        )
    except Exception as e:
        # Log error and return safe result
        logger.error(f"Processing failed for {data.id}: {e}")
        return SafeResult(
            success=False,
            data=None,
            error=str(e)
        )

@node(Catalog, "safe_outputs[:].results", "final_summary")
def summarize_results(results: list[SafeResult]) -> ProcessingSummary:
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    return ProcessingSummary(
        total_processed=len(results),
        successful_count=len(successful),
        failed_count=len(failed),
        success_rate=len(successful) / len(results),
        errors=[r.error for r in failed if r.error]
    )
```

### Retry Logic

Implement retry mechanisms for unreliable operations:

```python
import tempfile
import time
from pydantic import BaseModel
from pond import node
from pond.catalogs.iceberg_catalog import IcebergCatalog

# Setup catalog for example
temp_dir = tempfile.mkdtemp(prefix="pypond_retry_")
catalog = IcebergCatalog(
    "default",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# Define example models
class SourceData(BaseModel):
    value: str

class ProcessedData(BaseModel):
    result: str

class UnreliableSource(BaseModel):
    data: SourceData

class Processed(BaseModel):
    result: ProcessedData

class Catalog(BaseModel):
    unreliable_source: UnreliableSource
    processed: Processed

def process_unreliable_source(data: SourceData) -> ProcessedData:
    # Simulate unreliable processing
    return ProcessedData(result=f"processed_{data.value}")

def with_retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        raise last_exception
            return wrapper
    return decorator

@node(Catalog, "unreliable_source.data", "processed.result")
def unreliable_transform(data: SourceData) -> ProcessedData:
    # Apply retry logic manually to avoid decorator conflicts
    max_attempts = 3
    delay = 1.0
    last_exception = None
    
    for attempt in range(max_attempts):
        try:
            return process_unreliable_source(data)
        except Exception as e:
            last_exception = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (2 ** attempt))  # Exponential backoff
                continue
            else:
                raise last_exception
```

## Advanced Hooks

### Custom Performance Monitoring

Create sophisticated monitoring hooks:

```python
import time
import json
from pond.hooks.abstract_hook import AbstractHook

class PerformanceMonitoringHook(AbstractHook):
    def __init__(self):
        self.metrics = {
            "transform_times": {},
            "memory_usage": {},
            "cpu_usage": {},
            "data_sizes": {}
        }
        self.start_times = {}
    
    def pre_node_execute(self, transform):
        try:
            import psutil
        except ImportError:
            psutil = None
        
        name = transform.get_name()
        self.start_times[name] = time.time()
        
        # Record system metrics if psutil available
        if psutil:
            process = psutil.Process()
            self.metrics["memory_usage"][name] = process.memory_info().rss
            self.metrics["cpu_usage"][name] = psutil.cpu_percent()
    
    def post_node_execute(self, transform, success, error):
        name = transform.get_name()
        
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics["transform_times"][name] = duration
    
    def post_pipe_execute(self, pipe, success, error):
        # Generate performance report
        report = self.generate_performance_report()
        
        # Save to file or send to monitoring system
        with open("performance_report.json", "w") as f:
            json.dump(report, f, indent=2)
    
    def generate_performance_report(self):
        return {
            "total_transforms": len(self.metrics["transform_times"]),
            "total_execution_time": sum(self.metrics["transform_times"].values()),
            "slowest_transforms": sorted(
                self.metrics["transform_times"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "average_execution_time": sum(self.metrics["transform_times"].values()) / len(self.metrics["transform_times"]) if self.metrics["transform_times"] else 0
        }
```

### Data Lineage Tracking

Track data flow and transformations:

```python
import time
import json
from pond.hooks.abstract_hook import AbstractHook

class LineageTrackingHook(AbstractHook):
    def __init__(self):
        self.lineage_graph = {
            "nodes": [],
            "edges": []
        }
    
    def pre_node_execute(self, transform):
        # Record transform information
        node = {
            "id": transform.get_name(),
            "type": "transform",
            "inputs": [path.to_path() for path in transform.get_inputs()],
            "outputs": [path.to_path() for path in transform.get_outputs()],
            "timestamp": time.time()
        }
        self.lineage_graph["nodes"].append(node)
        
        # Record data flow edges
        for input_path in transform.get_inputs():
            for output_path in transform.get_outputs():
                edge = {
                    "from": input_path.to_path(),
                    "to": output_path.to_path(),
                    "transform": transform.get_name(),
                    "timestamp": time.time()
                }
                self.lineage_graph["edges"].append(edge)
    
    def post_pipe_execute(self, pipe, success, error):
        # Export lineage information
        with open("data_lineage.json", "w") as f:
            json.dump(self.lineage_graph, f, indent=2)
```

## Best Practices Summary

### Pipeline Design
- **Modular composition**: Build pipelines from reusable components
- **Clear dependencies**: Make data dependencies explicit
- **Error boundaries**: Handle errors gracefully without stopping everything
- **Resource awareness**: Consider memory and CPU requirements

### Performance
- **Parallel-friendly design**: Create transforms that can run independently  
- **Batch operations**: Process multiple items together when possible
- **Lazy evaluation**: Load data only when needed
- **Appropriate variants**: Use table/file variants for optimal performance

### Maintainability
- **Documentation**: Document complex pipeline logic
- **Testing**: Test transforms and pipelines thoroughly
- **Monitoring**: Track performance and data quality
- **Version control**: Track schema and pipeline changes

### Scalability
- **Catalog selection**: Choose appropriate storage backends
- **Resource planning**: Plan for data growth
- **Optimization**: Profile and optimize bottlenecks
- **Distribution**: Design for potential distributed execution

These advanced patterns enable you to build sophisticated, production-ready data pipelines with PyPond.