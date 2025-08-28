# PyPond Documentation

<div align="center">
  <img src="assets/pypond-icon.svg" alt="PyPond" width="256" height="256">
</div>

PyPond is a Python library for building data pipelines using hierarchical pydantic structures and arrow format storage. It provides a flexible and powerful framework for processing structured data with type safety and efficient storage backends.

## Key Features

- **Type-Safe Pipelines**: Built on pydantic models with automatic type validation
- **Multiple Storage Backends**: Support for Apache Iceberg, Lance, and Delta Lake
- **Flexible Data Access**: Path-based lens system for navigating complex data structures
- **Parallel Execution**: Built-in support for parallel pipeline execution
- **File Integration**: Seamless handling of unstructured files alongside structured data
- **Extensible Hooks**: Plugin system for monitoring, visualization, and custom behaviors

## Quick Start

```python
from pond import State, node, pipe, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.sequential_runner import SequentialRunner
from pydantic import BaseModel
import numpy as np

# Define your data schema
class Parameters(BaseModel):
    threshold: float
    scale: float

class Results(BaseModel):
    processed_count: int
    avg_value: float
    output_data: File[np.ndarray] = Field(ext="npy")

class Catalog(BaseModel):
    params: Parameters
    results: Results

# Create transforms using the @node decorator
@node(Catalog, "params.threshold", "results.processed_count")
def count_above_threshold(threshold: float) -> int:
    # Your processing logic here
    return 42

@node(Catalog, ["results.processed_count", "params.scale"], "results.avg_value") 
def compute_average(count: int, scale: float) -> float:
    return count * scale

# Create a pipeline
def my_pipeline():
    return pipe([
        count_above_threshold,
        compute_average,
    ], input="params")

# Execute the pipeline
import tempfile
temp_dir = tempfile.mkdtemp(prefix="pypond_quickstart_")
catalog = IcebergCatalog(
    "quickstart_example",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")
state = State(Catalog, catalog)

# Set input parameters using state accessors
state["params.threshold"] = 0.5
state["params.scale"] = 2.0

# Run the pipeline
runner = SequentialRunner()
pipeline = my_pipeline()
runner.run(state, pipeline, hooks=[])

# Access results using state accessors
count = state["results.processed_count"]
avg = state["results.avg_value"]
```

## Architecture Overview

PyPond is built around several key components that work together to create a powerful data pipeline system:

### Catalog System
The catalog system provides the foundation for data storage and retrieval using multiple backend formats including Apache Iceberg, Lance, and Delta Lake. It handles schema evolution, versioning, and efficient querying of structured data.

### Transform System  
Transforms are the computational units of pypond. They are automatically typed functions that process data between different locations in the catalog. The system supports three main types of transforms based on input/output patterns:

- **Transform**: For scalar input/output operations
- **TransformList**: For array-to-array transformations  
- **TransformListFold**: For array-to-scalar reductions

### State and Data Access
The `State` object provides the main interface for interacting with your data catalog. You can:

- Set data using dictionary-style access: `state["path.to.field"] = value`
- Get data using dictionary-style access: `value = state["path.to.field"]` 
- Access the underlying lens system for advanced operations: `state.lens("path")`

### Execution System
The execution system orchestrates pipeline runs using different strategies. It includes sequential execution for development/debugging and parallel execution for production workloads with automatic dependency resolution.

### I/O System
The I/O system handles reading and writing various file formats including images, point clouds, numpy arrays, and other scientific data formats. It integrates with the broader ecosystem through fsspec for filesystem abstraction.

## Navigation

- **[User Guide](user-guide/index.md)**: Learn how to use PyPond effectively
- **[API Reference](api-reference/index.md)**: Detailed documentation of all classes and functions  
- **[Examples](examples/index.md)**: Practical examples and tutorials
- **[Development](development/index.md)**: Information for contributors and developers

## Community

- **GitHub**: [nilsbore/pypond](https://github.com/nilsbore/pypond)
- **Issues**: [Report bugs or request features](https://github.com/nilsbore/pypond/issues)