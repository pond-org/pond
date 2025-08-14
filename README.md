# Pond
[![Test and lint](https://github.com/nilsbore/pypond/actions/workflows/tests.yml/badge.svg)](https://github.com/nilsbore/pypond/actions/workflows/tests.yml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

* Pond is a python library for defining **workflow graphs**, similar to [kedro](https://github.com/kedro-org/kedro).
* Pond developers :heart: the concepts behind kedro and in particular the **data catalog**.
* The highlight of Pond is a modern data catalog based on **arrow**.
* There are escape hatches for **unstructured file storage** using fsspec.
* Pond **reduces software complexity** in the presence of large computation graphs.
* Basic support for **parallel execution**.

## Architecture Overview

Pond is built around hierarchical pydantic data structures that define your data products, with pipeline nodes that map functions between different parts of the structure. Here are the key components:

### Data Catalog System
- **Multiple storage backends**: Supports Iceberg, Lance, and Delta Lake for structured arrow data
- **Path-based access**: Navigate nested data using expressions like `clouds[0].bounds` or `params.resolution`
- **Type validation**: Automatic schema validation between pipeline stages and catalog entries

### Transform System  
- **Node decorators**: Use `@node` to define pipeline functions with clear input/output mappings
- **List processing**: Built-in support for processing arrays of data with `[:]` syntax
- **Pipeline composition**: Chain transforms together using the `pipe()` function
- **Type safety**: Runtime validation ensures transform inputs match catalog schema types

### Flexible Data Access
- **Lens system**: Unified interface for accessing both structured arrow data and unstructured files
- **Multiple variants**: Access the same data as pydantic objects (`default:`), arrow tables (`table:`), or raw files (`file:`)
- **File integration**: Seamless handling of unstructured data via fsspec with custom readers/writers

### Execution Models
- **Sequential execution**: Process transforms one by one for debugging and development  
- **Parallel execution**: Automatically execute independent transforms concurrently
- **Progress tracking**: Integration with Hamilton UI for workflow visualization and monitoring

The library excels at workflows involving mixed structured/unstructured data, like the included LiDAR processing example that transforms point cloud files into elevation heatmaps through a series of typed, validated pipeline stages.

## Run example workflow
Pond uses hamilton for visualizing workflow runs. See details [here](https://hamilton.dagworks.io/en/latest/hamilton-ui/ui/#get-started) for details on how to set up a project in the GUI.
```
uv run hamilton ui
```
Once the project is set up, run using the following command.
```sh
uv run python example.py
```
