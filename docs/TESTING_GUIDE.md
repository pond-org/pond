# Documentation Testing Guide

This guide explains how to write testable code examples in PyPond documentation.

## Overview

PyPond uses **pytest-codeblocks** to automatically test all Python code blocks in the documentation. This ensures that:

- All examples remain up-to-date and functional
- Documentation doesn't drift from the actual API
- Users can trust that examples will work as shown

## How It Works

When you run `uv run python -m pytest docs`, pytest-codeblocks:

1. Extracts all ````python` code blocks from markdown files
2. Executes each block as a separate test
3. Reports any errors or failures

## Using Shared Setup Functions

**CRITICAL**: Always import and use the setup functions from `docs/conftest.py` directly. Do not copy their implementation.

### Import the Conftest Module

At the start of code blocks that need catalog setup, import the conftest module:

```python
# pytest-codeblocks:skip
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from conftest import create_test_catalog

# Now use the shared function
catalog = create_test_catalog()
```

### Available Functions

From `docs/conftest.py`:
- `create_test_catalog()`: Returns properly configured IcebergCatalog
- `sample_data` fixture: Provides deterministic test data
- Other catalog fixtures: `docs_iceberg_catalog`, `docs_lance_catalog`, `docs_delta_catalog`

## Writing Testable Examples

### Simple Examples

For basic examples without catalogs:

```python
from pydantic import BaseModel
from pond import Field

class Parameters(BaseModel):
    threshold: float
    multiplier: float

params = Parameters(threshold=0.5, multiplier=2.0)
print(f"Threshold: {params.threshold}")
```

### Examples with Catalogs

**DO THIS** - Use the shared function:

```python
# pytest-codeblocks:skip
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from conftest import create_test_catalog
from pond import State

# Use the shared catalog setup
catalog = create_test_catalog()
state = State(MyModel, catalog)
```

**DON'T DO THIS** - Don't duplicate the setup code:

```python
# pytest-codeblocks:skip
# ❌ Don't copy the implementation from conftest.py
temp_dir = tempfile.mkdtemp(prefix="pypond_example_")
catalog = IcebergCatalog(
    "default",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")
```

### Complete Pipeline Examples

```python
# pytest-codeblocks:skip
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from conftest import create_test_catalog
import numpy as np
from pydantic import BaseModel
from pond import State, node, pipe
from pond.runners.sequential_runner import SequentialRunner

# Define models
class Pipeline(BaseModel):
    params: dict
    results: dict

# Define transforms
@node(Pipeline, "params.threshold", "results.count")
def count_above_threshold(threshold: float) -> int:
    np.random.seed(42)  # Consistent with sample_data
    data = np.random.randn(100)
    return int(np.sum(data > threshold))

# Use shared catalog setup
catalog = create_test_catalog()
state = State(Pipeline, catalog)
state["params.threshold"] = 0.5

# Execute pipeline
runner = SequentialRunner()
pipeline = pipe([count_above_threshold], input="params")
runner.run(state, pipeline, hooks=[])

result = state["results.count"]
assert isinstance(result, int)
```

## Best Practices

### 1. Always Import Conftest

```python
# pytest-codeblocks:skip
# At the top of any code block needing shared setup
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from conftest import create_test_catalog, sample_data
```

### 2. Use Deterministic Data

```python
# Use the same seed as sample_data fixture
import numpy as np
np.random.seed(42)
```

### 3. Self-Contained But Shared

Each code block should import what it needs, but use shared functions:

```python
# pytest-codeblocks:skip
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from conftest import create_test_catalog
from pond import State
from pydantic import BaseModel

class MyModel(BaseModel):
    data: dict

catalog = create_test_catalog()  # Use shared function
state = State(MyModel, catalog)
```

## Maintaining the Shared Functions

### When You Need New Setup Patterns

1. **Add to `docs/conftest.py`** - Don't put setup code in documentation
2. **Import and use** the new function in your docs
3. **Update this guide** with the new available functions

### Example: Adding New Catalog Type

If you need a new catalog setup pattern:

1. Add to `docs/conftest.py`:
```python
def create_test_lance_catalog():
    """Helper to create Lance catalog for docs."""
    # implementation here
    return catalog
```

2. Use in documentation:
```python
# pytest-codeblocks:skip
from conftest import create_test_lance_catalog
catalog = create_test_lance_catalog()
```

## Running Tests

```bash
# Test all documentation
uv run python -m pytest docs --codeblocks -v

# Test specific file
uv run python -m pytest docs/user-guide/getting-started.md -v
```

## Current Implementation Notes

### Self-Contained Code Blocks (Current Status)

Currently, most documentation examples are self-contained with their own setup code. This was done to fix immediate testing issues, but the conftest approach above is preferred for future documentation.

**Common Patterns in Current Docs:**

```python
# Current pattern - will be refactored to use conftest functions later
import tempfile
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond import State

temp_dir = tempfile.mkdtemp(prefix="pypond_example_")
catalog = IcebergCatalog(
    "example_name",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# IMPORTANT: File operations need volume protocol args
volume_protocol_args = {"dir": {"path": temp_dir}}
state = State(Catalog, catalog, volume_protocol_args=volume_protocol_args)
```

### Key Lessons from Recent Fixes

1. **Volume Protocol Args**: File operations require `volume_protocol_args = {"dir": {"path": warehouse_dir}}` in State initialization
2. **Unique Identifiers**: Use unique catalog names and temp directory prefixes to avoid conflicts
3. **PyPond Limitations**: Only one wildcard supported in paths (e.g., `datasets[:].files[:]` won't work)
4. **Complete Setup**: IcebergCatalog needs both `uri` and `warehouse` parameters

### Future Refactoring Plan

The current self-contained examples should be refactored to use conftest functions that handle:
- Proper catalog configuration with URI and warehouse
- Volume protocol args setup
- Unique naming to avoid test conflicts
- Cleanup after tests

## Troubleshooting

### Import Errors

If you get import errors for conftest:
- Make sure the sys.path.append lines are correct
- Check that you're importing from the right location

### Volume Protocol Errors

If you get `KeyError: 'dir'` when working with files:
- Ensure `volume_protocol_args = {"dir": {"path": temp_dir}}` is passed to State
- This is required for any file operations (`state["file:path"]` assignments)

### Catalog Errors

Don't try to fix catalog setup in your documentation - fix it in `docs/conftest.py` and use the shared function.

### Wildcard Path Errors

If you get "Only one wildcard currently supported":
- PyPond currently supports only one `[:]` wildcard per path
- Use loops instead: `for i in range(n): state[f"items[{i}].files[:]"]`

## Contributing

When writing new documentation:

1. **Import from conftest** - don't duplicate setup code
2. **Add new functions to conftest** if you need new patterns  
3. **Keep examples focused** on the concept being demonstrated
4. **Test locally** before submitting

The key principle: documentation should demonstrate the API, while `conftest.py` handles all the setup complexity.