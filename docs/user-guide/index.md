# User Guide

Welcome to the PyPond user guide! This guide will walk you through all the key concepts and features of PyPond, from basic usage to advanced patterns.

## What is PyPond?

PyPond is a Python library for building type-safe data pipelines using hierarchical pydantic structures and arrow format storage. It combines the flexibility of Python with the performance and reliability of modern data storage formats.

## Getting Started

New to PyPond? Start here:

- **[Getting Started](getting-started.md)**: Installation and your first pipeline
- **[Core Concepts](core-concepts.md)**: Understanding PyPond's architecture
- **[Working with Catalogs](catalogs.md)**: Data storage and retrieval

## Building Pipelines

Learn how to create and execute data pipelines:

- **[Building Transforms](transforms.md)**: Creating data processing functions
- **[Pipeline Execution](execution.md)**: Running pipelines sequentially or in parallel
- **[File Handling](files.md)**: Working with unstructured files

## Advanced Topics

For experienced users looking to go deeper:

- **[Advanced Usage](advanced.md)**: Complex patterns and best practices

## Quick Reference

Common patterns and examples:

### Basic Transform
```python
from pond import node
from pydantic import BaseModel

class Input(BaseModel):
    field: float
    
class Output(BaseModel):
    result: float
    
class Catalog(BaseModel):
    input: Input
    output: Output

@node(Catalog, "input.field", "output.result")
def my_transform(value: float) -> float:
    return value * 2.0
```

### Array Processing
```python
from pond import node
from pydantic import BaseModel

class Item(BaseModel):
    value: float
    
class ProcessedItem(BaseModel):
    value: float
    
class Input(BaseModel):
    items: list[Item]
    
class Output(BaseModel):
    processed: list[ProcessedItem]
    
class Catalog(BaseModel):
    input: Input
    output: Output

@node(Catalog, "input.items[:]", "output.processed[:]")
def process_items(item: Item) -> ProcessedItem:
    return ProcessedItem(value=item.value * 2)
```

### File Handling
```python
from pydantic import BaseModel
from pond import Field, File
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

class MyData(BaseModel):
    data_file: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
```

### State Access
```python
import tempfile
from pond import State
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pydantic import BaseModel

class Params(BaseModel):
    threshold: float
    
class Results(BaseModel):
    output: float
    
class MySchema(BaseModel):
    params: Params
    results: Results

# Setup (in real usage, you'd configure this once)
catalog_temp = tempfile.mkdtemp(prefix="example_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="example_warehouse_")
catalog = IcebergCatalog(
    "example",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(MySchema, catalog)

# Set data
state["params.threshold"] = 0.5

# Set result for demonstration
state["results.output"] = 42.0

# Get data  
result = state["results.output"]
print(f"Retrieved result: {result}")

# Advanced lens access
lens = state.lens("params.threshold")
print(f"Threshold exists: {lens.exists()}")
```