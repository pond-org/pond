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

## Run example workflow
Pond uses hamilton for visualizing workflow runs. See details [here](https://hamilton.dagworks.io/en/latest/hamilton-ui/ui/#get-started) for details on how to set up a project in the GUI.
```
uv run hamilton ui
```
Once the project is set up, run using the following command.
```sh
uv run python example.py
```
