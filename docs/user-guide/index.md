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
@node(Catalog, "input.field", "output.result")
def my_transform(value: float) -> float:
    return value * 2.0
```

### Array Processing
```python
@node(Catalog, "input.items[:]", "output.processed[:]")
def process_items(item: Item) -> ProcessedItem:
    return ProcessedItem(value=item.value * 2)
```

### File Handling
```python
class MyData(BaseModel):
    data_file: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
```

### State Access
```python
# Set data
state["params.threshold"] = 0.5

# Get data  
result = state["results.output"]

# Advanced lens access
lens = state.lens("complex.path[:].field")
```