# Pond

<div align="center">
  <img src="docs/assets/pypond-icon.svg" alt="PyPond" width="256" height="256">
</div>

<div align="center">

[![Test and lint](https://github.com/nilsbore/pypond/actions/workflows/tests.yml/badge.svg)](https://github.com/nilsbore/pypond/actions/workflows/tests.yml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

</div>

Pond is a Python library for building data pipelines using hierarchical pydantic structures and arrow format storage. It provides a flexible and powerful framework for processing structured data with type safety and efficient storage backends.

## ✨ Key Features

- **Type-Safe Pipelines**: Built on pydantic models with automatic type validation
- **Multiple Storage Backends**: Support for Apache Iceberg, Lance, and Delta Lake  
- **Flexible Data Access**: Path-based lens system for navigating complex data structures
- **Parallel Execution**: Built-in support for parallel pipeline execution
- **File Integration**: Seamless handling of unstructured files alongside structured data
- **Extensible Hooks**: Plugin system for monitoring, visualization, and custom behaviors

## 🚀 Quick Start

### Installation

Clone the repository and install dependencies using uv:

```bash
git clone https://github.com/nilsbore/pypond.git
cd pypond
uv sync
```

### Your First Pipeline

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

class Catalog(BaseModel):
    params: Parameters
    results: Results

# Create transforms using the @node decorator
@node(Catalog, "params.threshold", "results.processed_count")
def count_above_threshold(threshold: float) -> int:
    # Your processing logic here
    data = np.random.randn(1000)
    return int(np.sum(data > threshold))

@node(Catalog, ["results.processed_count", "params.scale"], "results.avg_value") 
def compute_average(count: int, scale: float) -> float:
    return count * scale

# Create and execute pipeline
catalog = IcebergCatalog(name="default")
state = State(Catalog, catalog)
state["params.threshold"] = 0.5
state["params.scale"] = 2.0

# Run the pipeline
pipeline = pipe([count_above_threshold, compute_average], input="params")
runner = SequentialRunner()
runner.run(state, pipeline)

# Access results
count = state["results.processed_count"]
avg = state["results.avg_value"]
print(f"Processed {count} items with average: {avg}")
```

## 🏗️ Architecture

PyPond is built around several key components that work together:

### Catalog System
The catalog system provides the foundation for data storage and retrieval using multiple backend formats including Apache Iceberg, Lance, and Delta Lake. It handles schema evolution, versioning, and efficient querying of structured data.

### Transform System  
Transforms are the computational units of PyPond. They are automatically typed functions that process data between different locations in the catalog:

- **Transform**: For scalar input/output operations
- **TransformList**: For array-to-array transformations  
- **TransformListFold**: For array-to-scalar reductions

### State and Data Access
The `State` object provides the main interface for interacting with your data catalog:

- Set data: `state["path.to.field"] = value`
- Get data: `value = state["path.to.field"]` 
- Access lens system: `state.lens("path")`

### Execution System
The execution system orchestrates pipeline runs using different strategies:
- **Sequential execution**: For development and debugging
- **Parallel execution**: For production workloads with automatic dependency resolution

## 🛠️ Development

### Run Example Workflow

PyPond uses Hamilton UI for visualizing workflow runs:

```bash
uv run hamilton ui
```

Once the project is set up, run the example with UI integration:

```bash
uv run python example.py --ui
```

## 📚 Documentation

- **[Main docs page](https://pond-org.github.io/pond/)**: Entry point to documentation
- **[User Guide](https://pond-org.github.io/pond/user-guide/)**: Learn how to use PyPond effectively
- **[API Reference](https://pond-org.github.io/pond/api-reference/)**: Detailed documentation of all classes and functions

## 🤝 Contributing

We are not currently accepting external contributions as PyPond is in early development. We're focusing on establishing the core architecture and foundational features before opening to community contributions.

However, you can help by:
- **Starring the repository** to show interest
- **Reporting bugs** through detailed issue reports
- **Sharing feedback** on the project's direction
- **Spreading the word** about the project

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details and future plans for community involvement.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

PyPond developers ❤️ the concepts behind [Kedro](https://github.com/kedro-org/kedro), particularly the data catalog approach. PyPond modernizes these ideas with arrow-native storage, pydantic schemas, and flexible execution models.

