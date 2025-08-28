# File Handling

PyPond provides seamless integration between structured catalog data and unstructured files. This guide covers how to work with files alongside your data pipelines.

## File Fields

### Basic File Definition

Use the `File` type with `Field` metadata to define file storage:

```python
from pond import Field, File
from pond.io.readers import read_npz
from pond.io.writers import write_npz
from pydantic import BaseModel
import numpy as np

class Metadata(BaseModel):
    source: str
    version: str

class Dataset(BaseModel):
    metadata: Metadata
    data_file: File[np.ndarray] = Field(
        reader=read_npz,
        writer=write_npz, 
        ext="npy"
    )
```

### File Field Components

- **`File[T]`**: Generic type indicating the file contains data of type `T`
- **`reader`**: Function to load file contents  
- **`writer`**: Function to save file contents
- **`ext`**: File extension
- **`protocol`**: Storage protocol (optional)
- **`path`**: Custom file path (optional)

## Built-in Readers and Writers

PyPond includes readers and writers for common file formats:

### Numpy Arrays
```python
from pond import Field, File
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

data_file: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
```

### Images
```python
from pond import Field, File
from pond.io.readers import read_image
from pond.io.writers import write_image
import numpy as np

image_file: File[np.ndarray] = Field(reader=read_image, writer=write_image, ext="png")
```

### Point Clouds (LAS)
```python
from pond import Field, File
from pond.io.readers import read_las
import laspy

point_cloud: File[laspy.LasData] = Field(reader=read_las, ext="laz")
```

### Plotly Figures
```python
from pond import Field, File
from pond.io.writers import write_plotly_png
import plotly.graph_objects as go

plot_file: File[go.Figure] = Field(writer=write_plotly_png, ext="png")
```

### Pickle Files
```python
from pond import Field, File
from pond.io.readers import read_pickle
from pond.io.writers import write_pickle
from typing import Any

serialized_data: File[Any] = Field(reader=read_pickle, writer=write_pickle, ext="pkl")
```

## File Access Patterns

### Default Access (Structured)
Returns `File` objects with metadata:

```python
import tempfile
import os
from pond import State, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_npz
from pond.io.writers import write_npz
from pydantic import BaseModel
import numpy as np

class Metadata(BaseModel):
    source: str
    version: str

class Dataset(BaseModel):
    metadata: Metadata
    data_file: File[np.ndarray] = Field(
        reader=read_npz,
        writer=write_npz, 
        ext="npy"
    )

class Catalog(BaseModel):
    dataset: Dataset

# Setup - ensure directories exist and have proper paths
import uuid
unique_id = str(uuid.uuid4())[:8]
catalog_temp = tempfile.mkdtemp(prefix=f"file_catalog_basic_{unique_id}_")
warehouse_temp = tempfile.mkdtemp(prefix=f"file_warehouse_basic_{unique_id}_")
os.makedirs(catalog_temp, exist_ok=True)
os.makedirs(warehouse_temp, exist_ok=True)

catalog = IcebergCatalog(
    "file_example_basic",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# Configure volume protocol args for file handling
volume_protocol_args = {"dir": {"path": warehouse_temp}}
state = State(Catalog, catalog, volume_protocol_args=volume_protocol_args)

# Create sample data
np.random.seed(42)
sample_data = np.random.randn(10)
state["file:dataset.data_file"] = sample_data
state["dataset.metadata"] = Metadata(source="test", version="1.0")

# Get File object
file_obj = state["dataset.data_file"]
print(f"File path: {file_obj.path}")

# Load content manually
content = file_obj.get()
print(f"Loaded data shape: {content.shape}")
```

### File Variant Access (Direct Content)
Returns file contents directly:

```python
import tempfile
import os
from pond import State, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_npz
from pond.io.writers import write_npz
from pydantic import BaseModel
import numpy as np

class Metadata(BaseModel):
    source: str
    version: str

class Dataset(BaseModel):
    metadata: Metadata
    data_file: File[np.ndarray] = Field(
        reader=read_npz,
        writer=write_npz, 
        ext="npy"
    )

class Catalog(BaseModel):
    dataset: Dataset

# Setup
import uuid
unique_id = str(uuid.uuid4())[:8]
catalog_temp = tempfile.mkdtemp(prefix=f"file_variant_{unique_id}_")
warehouse_temp = tempfile.mkdtemp(prefix=f"file_variant_warehouse_{unique_id}_")
os.makedirs(catalog_temp, exist_ok=True)
os.makedirs(warehouse_temp, exist_ok=True)

catalog = IcebergCatalog(
    "file_variant_example_x1",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("catalog")

# Configure volume protocol args for file handling
volume_protocol_args = {"dir": {"path": warehouse_temp}}
state = State(Catalog, catalog, volume_protocol_args=volume_protocol_args)

# Create sample data
np.random.seed(42)
sample_data = np.random.randn(10)
state["file:dataset.data_file"] = sample_data
state["dataset.metadata"] = Metadata(source="test", version="1.0")

# Get file contents directly
data = state["file:dataset.data_file"]  # Returns np.ndarray
print(f"Direct file access data shape: {data.shape}")
```


## Storage Protocols

### Local Filesystem (default)
```python
from pond import Field, File
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Uses local directory storage
local_file: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
```

### Custom Protocol
```python
from pond import Field, File
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Uses specified protocol from volume configuration
s3_file: File[np.ndarray] = Field(
    reader=read_npz, 
    writer=write_npz, 
    ext="npy",
    protocol="s3"
)
```

### Custom Path
```python
from pond import Field, File
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Uses explicit path instead of auto-generated
custom_file: File[np.ndarray] = Field(
    reader=read_npz,
    writer=write_npz,
    ext="npy", 
    path="/data/custom/location"
)
```

## File Operations in Transforms

### Processing Files
```python
import tempfile
from pydantic import BaseModel
from pond import node, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Setup catalog
temp_dir = tempfile.mkdtemp(prefix="pypond_process_")
catalog = IcebergCatalog(
    "process_example",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

class ProcessedResult(BaseModel):
    mean: float
    std: float

class Input(BaseModel):
    raw_data: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")

class Output(BaseModel):
    processed: ProcessedResult

class Catalog(BaseModel):
    input: Input
    output: Output

def apply_algorithm(data: np.ndarray) -> np.ndarray:
    return data * 2  # Simple processing

@node(Catalog, "file:input.raw_data", "output.processed") 
def process_data(data: np.ndarray) -> ProcessedResult:
    # data is automatically loaded from file
    processed = apply_algorithm(data)
    return ProcessedResult(
        mean=float(np.mean(processed)),
        std=float(np.std(processed))
    )
```

### Creating Files
```python
import tempfile
from pydantic import BaseModel
from pond import node, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Setup catalog
temp_dir = tempfile.mkdtemp(prefix="pypond_generate_")
catalog = IcebergCatalog(
    "generate_example",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

class Parameters(BaseModel):
    mean: float
    std: float
    size: int

class Input(BaseModel):
    parameters: Parameters

class Output(BaseModel):
    result_data: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")

class Catalog(BaseModel):
    input: Input
    output: Output

@node(Catalog, "input.parameters", "file:output.result_data")
def generate_data(params: Parameters) -> np.ndarray:
    # Return data - it will be automatically saved to file
    return np.random.normal(params.mean, params.std, params.size)
```

### File-to-File Processing
```python
import tempfile
from pydantic import BaseModel
from pond import node, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_image
from pond.io.writers import write_image
import numpy as np

# Setup catalog
temp_dir = tempfile.mkdtemp(prefix="pypond_image_")
catalog = IcebergCatalog(
    "image_example",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

class Input(BaseModel):
    raw_image: File[np.ndarray] = Field(reader=read_image, writer=write_image, ext="png")

class Output(BaseModel):
    processed_image: File[np.ndarray] = Field(reader=read_image, writer=write_image, ext="png")

class Catalog(BaseModel):
    input: Input
    output: Output

def apply_filters(image: np.ndarray) -> np.ndarray:
    return image * 0.8  # Simple image processing

@node(Catalog, "file:input.raw_image", "file:output.processed_image")
def process_image(image: np.ndarray) -> np.ndarray:
    # Process image array
    processed = apply_filters(image)
    return processed
```

## File Indexing

### Automatic File Discovery
Index existing files into the catalog:

```python
import tempfile
from pydantic import BaseModel
from pond import index_files, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Setup catalog
temp_dir = tempfile.mkdtemp(prefix="pypond_index_")
catalog = IcebergCatalog(
    "index_example",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

class Dataset(BaseModel):
    data_files: list[File[np.ndarray]] = Field(reader=read_npz, writer=write_npz, ext="npy")

class Catalog(BaseModel):
    datasets: list[Dataset] = []

# Index files matching the schema
index_files(Catalog, "datasets[:].data_files")
```

### File Patterns
Use wildcards to match multiple files:

```python
from pydantic import BaseModel
from pond import Field, File
from pond.io.readers import read_npz
import numpy as np

class FileCollection(BaseModel):
    input_files: list[File[np.ndarray]] = Field(
        path="data/inputs/*.npy",
        reader=read_npz,
        ext="npy"
    )
```

### Directory Structure
Map directory structure to catalog hierarchy:

```python
from pydantic import BaseModel
from pond import Field, File
from pond.io.readers import read_npz
import numpy as np

class Experiment(BaseModel):
    trial_data: list[File[np.ndarray]] = Field(
        path="experiments/*/trials",  # Maps to experiments[i]/trials/*.npy
        reader=read_npz,
        ext="npy"
    )

class ExperimentData(BaseModel):
    experiments: list[Experiment]
```

## Volume Configuration

### Volume Setup
Configure filesystem protocols in `volume.yaml`:

```yaml
# Local directory
dir:
  protocol: dir
  target_protocol: file
  target_options:
    auto_mkdir: true

# S3 storage  
s3:
  protocol: s3
  key: YOUR_ACCESS_KEY
  secret: YOUR_SECRET_KEY
  endpoint_url: https://s3.amazonaws.com

# Azure Blob
azure:
  protocol: abfs
  account_name: your_account
  account_key: your_key
```

### Using Volume Configuration
```python
import tempfile
from pydantic import BaseModel
from pond import State, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.volume import load_volume_protocol_args
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Example schema
class Dataset(BaseModel):
    data_file: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")

class Catalog(BaseModel):
    dataset: Dataset

# Setup catalog
temp_dir = tempfile.mkdtemp(prefix="pypond_volume_")
catalog = IcebergCatalog(
    "volume_example",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

volume_args = load_volume_protocol_args()
state = State(Catalog, catalog, volume_protocol_args=volume_args)
```

## Custom File Types

### Custom Reader/Writer
```python
from pydantic import BaseModel
from pond import Field, File

class CustomData(BaseModel):
    content: str
    
    @classmethod
    def parse(cls, data: bytes) -> 'CustomData':
        return cls(content=data.decode('utf-8'))
    
    def serialize(self) -> bytes:
        return self.content.encode('utf-8')

def read_custom_format(fs, path: str) -> CustomData:
    with fs.open(path, 'rb') as f:
        # Your custom loading logic
        return CustomData.parse(f.read())

def write_custom_format(data: CustomData, fs, path: str):
    with fs.open(path, 'wb') as f:
        # Your custom saving logic
        f.write(data.serialize())

# Use in schema
custom_file: File[CustomData] = Field(
    reader=read_custom_format,
    writer=write_custom_format,
    ext="custom"
)
```

### Binary Data
```python
from pond import Field, File

def read_binary(fs, path: str) -> bytes:
    with fs.open(path, 'rb') as f:
        return f.read()

def write_binary(data: bytes, fs, path: str):
    with fs.open(path, 'wb') as f:
        f.write(data)

binary_file: File[bytes] = Field(reader=read_binary, writer=write_binary, ext="bin")
```

## File Management Patterns

### File Collections
```python
import tempfile
from pydantic import BaseModel
from pond import node, Field, File, State
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Setup catalog
temp_dir = tempfile.mkdtemp(prefix="pypond_datacoll_")
catalog = IcebergCatalog(
    "datacoll_example",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

class DataCollection(BaseModel):
    raw_files: list[File[np.ndarray]] = Field(
        reader=read_npz,
        ext="npy",
        path="raw_data"
    )
    processed_files: list[File[np.ndarray]] = Field(
        reader=read_npz,
        writer=write_npz,
        ext="npy"
    )

class Dataset(BaseModel):
    data: DataCollection

class Catalog(BaseModel):
    datasets: list[Dataset] = []

def process_array(arr: np.ndarray) -> np.ndarray:
    return arr * 2  # Simple processing

@node(Catalog, "file:datasets[:].data.raw_files", "file:datasets[:].data.processed_files")
def process_file_collection(raw_data: list[np.ndarray]) -> list[np.ndarray]:
    return [process_array(arr) for arr in raw_data]
```

## Performance Considerations

### Lazy Loading
Files are loaded only when accessed:

```python
import tempfile
from pydantic import BaseModel
from pond import State, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Setup example
import uuid
unique_id = str(uuid.uuid4())[:8]
temp_dir = tempfile.mkdtemp(prefix=f"pypond_fileref_{unique_id}_")
catalog = IcebergCatalog(
    "fileref_example_x3",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

class Dataset(BaseModel):
    large_file: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")

class Catalog(BaseModel):
    dataset: Dataset

# Configure volume protocol args for file handling
volume_protocol_args = {"dir": {"path": temp_dir}}
state = State(Catalog, catalog, volume_protocol_args=volume_protocol_args)

# Create sample data first
sample_data = np.random.randn(100)
state["file:dataset.large_file"] = sample_data

# This doesn't load the file
file_ref = state["dataset.large_file"]

# This loads the file content
content = state["file:dataset.large_file"]
```

### Batch File Operations
Process multiple files efficiently:

```python
import tempfile
from pydantic import BaseModel
from pond import node, Field, File
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond.io.readers import read_npz
from pond.io.writers import write_npz
import numpy as np

# Setup catalog
temp_dir = tempfile.mkdtemp(prefix="pypond_batch_")
catalog = IcebergCatalog(
    "batch_example",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}",
)
catalog.catalog.create_namespace_if_not_exists("catalog")

class LargeDataset(BaseModel):
    chunks: list[File[np.ndarray]] = Field(reader=read_npz, writer=write_npz, ext="npy")

class Summary(BaseModel):
    statistics: dict

class Catalog(BaseModel):
    large_dataset: list[LargeDataset] = []
    summary: Summary

# Example function showing the pattern (simplified to avoid complex list path issues)
def compute_batch_stats(chunks: list[np.ndarray]) -> dict:
    # Process all chunks in one transform
    all_data = np.concatenate(chunks)
    return {
        "total_size": len(all_data),
        "mean": float(np.mean(all_data)),
        "std": float(np.std(all_data))
    }
```

### Storage Optimization
Choose appropriate formats:

```python
from pond import Field, File
from pond.io.readers import read_npz, read_image, read_pickle
from pond.io.writers import write_npz, write_image, write_pickle
from pydantic import BaseModel
import numpy as np
from typing import Any

# Example complex object
class ComplexObject(BaseModel):
    data: dict
    metadata: str

# For numerical data - efficient binary format
numeric_data: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")

# For images - compressed format
image_data: File[np.ndarray] = Field(reader=read_image, writer=write_image, ext="png")

# For complex objects - pickle (less efficient but flexible)
complex_data: File[ComplexObject] = Field(reader=read_pickle, writer=write_pickle, ext="pkl")
```

## Best Practices

### File Organization
- **Consistent naming**: Use descriptive, consistent file names
- **Directory structure**: Organize files hierarchically  
- **File formats**: Choose formats appropriate for your data
- **Compression**: Use compressed formats for large datasets

### Error Handling
- **File existence**: Check file existence before processing
- **Format validation**: Validate file formats and contents
- **Path handling**: Use absolute paths when possible
- **Cleanup**: Clean up temporary files after processing

### Performance
- **Batch operations**: Process multiple files together when possible
- **Lazy loading**: Don't load files until needed
- **Format selection**: Choose efficient formats for your use case
- **Caching**: Cache frequently accessed files

This comprehensive approach to file handling ensures your pipelines can seamlessly work with both structured and unstructured data.