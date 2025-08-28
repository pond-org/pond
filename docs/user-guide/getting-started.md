# Getting Started

This guide will get you up and running with PyPond quickly.

## Installation

PyPond requires Python 3.12 or later. Install using pip:

```bash
pip install pypond
```

## Dependencies

PyPond has several key dependencies that enable its functionality:
- **pydantic**: For type-safe data models
- **pyarrow**: For columnar data processing
- **fsspec**: For filesystem abstraction
- **beartype**: For runtime type checking

Catalog backends:
- **pyiceberg**: Apache Iceberg support
- **lancedb**: Lance database support  
- **deltalake**: Delta Lake support

## Your First Pipeline

Let's create a simple data processing pipeline:

### 1. Define Your Data Model

```python
from pydantic import BaseModel
from pond import Field, File
import numpy as np

class Parameters(BaseModel):
    threshold: float
    multiplier: float

class Results(BaseModel):
    filtered_count: int
    processed_data: list[float]

class Catalog(BaseModel):
    params: Parameters
    results: Results
```

### 2. Create Transforms

```python
from pond import node, pipe
import numpy as np
from pydantic import BaseModel
from pond import Field, File

# Redefine models for this block
class Parameters(BaseModel):
    threshold: float
    multiplier: float

class Results(BaseModel):
    filtered_count: int
    processed_data: list[float]

class Catalog(BaseModel):
    params: Parameters
    results: Results

@node(Catalog, "params.threshold", "results.filtered_count")
def count_above_threshold(threshold: float) -> int:
    # Your processing logic
    data = np.random.randn(1000)
    return int(np.sum(data > threshold))

@node(Catalog, ["params.multiplier", "results.filtered_count"], "results.processed_data")
def create_processed_data(multiplier: float, count: int) -> list[float]:
    return [multiplier] * count
```

### 3. Build a Pipeline

```python
def my_pipeline():
    return pipe([
        count_above_threshold,
        create_processed_data,
    ], input="params")
```

### 4. Execute the Pipeline

```python
import tempfile
import numpy as np
from pydantic import BaseModel
from pond import Field, File, State, node, pipe
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.runners.sequential_runner import SequentialRunner

# Redefine models for this block
class Parameters(BaseModel):
    threshold: float
    multiplier: float

class Results(BaseModel):
    filtered_count: int
    processed_data: list[float]

class Catalog(BaseModel):
    params: Parameters
    results: Results

# Redefine transforms for this block
@node(Catalog, "params.threshold", "results.filtered_count")
def count_above_threshold(threshold: float) -> int:
    # Your processing logic
    data = np.random.randn(1000)
    return int(np.sum(data > threshold))

@node(Catalog, ["params.multiplier", "results.filtered_count"], "results.processed_data")
def create_processed_data(multiplier: float, count: int) -> list[float]:
    return [multiplier] * count

def my_pipeline():
    return pipe([
        count_above_threshold,
        create_processed_data,
    ], input="params")

# Set up state with catalog (using proper Iceberg configuration)
temp_dir = tempfile.mkdtemp(prefix="pypond_example_")
test_catalog = IcebergCatalog(
    "default",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
test_catalog.catalog.create_namespace_if_not_exists("catalog")
state = State(Catalog, test_catalog)

# Set input parameters
state["params.threshold"] = 0.5
state["params.multiplier"] = 2.0

# Run pipeline
runner = SequentialRunner()
pipeline = my_pipeline()
runner.run(state, pipeline, hooks=[])

# Get results
count = state["results.filtered_count"]
processed = state["results.processed_data"]  # This loads the list
print(f"Found {count} values above threshold")
print(f"Processed data length: {len(processed)}")

# Verify results are reasonable
assert isinstance(count, int)
assert count > 0
assert len(processed) == count
assert all(x == 2.0 for x in processed)  # All values should be multiplier
```

## Key Concepts

### State Management
The `State` object is your main interface to the data catalog:
- Use `state["path"] = value` to store data
- Use `value = state["path"]` to retrieve data
- The state automatically handles type conversion and validation

### Transform Selection
PyPond automatically selects the right transform type based on your input/output patterns:
- Scalar → Scalar: Uses `Transform`
- Array → Array: Uses `TransformList` 
- Array → Scalar: Uses `TransformListFold`

### Catalog Backends
Choose the storage backend that fits your needs:
- **Iceberg**: Best for large-scale analytics and schema evolution
- **Lance**: Optimized for vector data and fast queries
- **Delta Lake**: ACID transactions and time travel

## Next Steps

Now that you have a basic pipeline running:

1. **[Core Concepts](core-concepts.md)**: Understand PyPond's architecture
2. **[Working with Catalogs](catalogs.md)**: Learn about different storage backends
3. **[Building Transforms](transforms.md)**: Create more complex data processing functions
4. **[File Handling](files.md)**: Work with unstructured files alongside structured data