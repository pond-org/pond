# File Handling

PyPond provides seamless integration between structured catalog data and unstructured files. This guide covers how to work with files alongside your data pipelines.

## File Fields

### Basic File Definition

Use the `File` type with `Field` metadata to define file storage:

```python
from pond import Field, File
from pydantic import BaseModel
import numpy as np

class Dataset(BaseModel):
    metadata: dict[str, str]
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
from pond.io.readers import read_npz
from pond.io.writers import write_npz

data_file: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
```

### Images
```python
from pond.io.readers import read_image
from pond.io.writers import write_image

image_file: File[np.ndarray] = Field(reader=read_image, writer=write_image, ext="png")
```

### Point Clouds (LAS)
```python
from pond.io.readers import read_las
import laspy

point_cloud: File[laspy.LasData] = Field(reader=read_las, ext="laz")
```

### Plotly Figures
```python
from pond.io.writers import write_plotly_png
import plotly.graph_objects as go

plot_file: File[go.Figure] = Field(writer=write_plotly_png, ext="png")
```

### Pickle Files
```python
from pond.io.readers import read_pickle
from pond.io.writers import write_pickle

serialized_data: File[Any] = Field(reader=read_pickle, writer=write_pickle, ext="pkl")
```

## File Access Patterns

### Default Access (Structured)
Returns `File` objects with metadata:

```python
# Get File object
file_obj = state["dataset.data_file"]
print(f"File path: {file_obj.path}")

# Load content manually
content = file_obj.get()
```

### File Variant Access (Direct Content)
Returns file contents directly:

```python
# Get file contents directly
data = state["file:dataset.data_file"]  # Returns np.ndarray
```

### Array File Access
For arrays of files:

```python
class MultiDataset(BaseModel):
    data_files: list[File[np.ndarray]] = Field(reader=read_npz, writer=write_npz, ext="npy")

# Access all file contents
all_data = state["file:datasets[:].data_files"]  # Returns list[np.ndarray]
```

## Storage Protocols

### Local Filesystem (default)
```python
# Uses local directory storage
local_file: File[np.ndarray] = Field(reader=read_npz, writer=write_npz, ext="npy")
```

### Custom Protocol
```python
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
@node(Catalog, "input.parameters", "file:output.result_data")
def generate_data(params: Parameters) -> np.ndarray:
    # Return data - it will be automatically saved to file
    return np.random.normal(params.mean, params.std, params.size)
```

### File-to-File Processing
```python
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
from pond import index_files

# Index files matching the schema
index_files(Catalog, "datasets[:].data_files")
```

### File Patterns
Use wildcards to match multiple files:

```python
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
class ExperimentData(BaseModel):
    experiments: list[Experiment]

class Experiment(BaseModel):
    trial_data: list[File[np.ndarray]] = Field(
        path="experiments/*/trials",  # Maps to experiments[i]/trials/*.npy
        reader=read_npz,
        ext="npy"
    )
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
from pond.volume import load_volume_protocol_args

volume_args = load_volume_protocol_args()
state = State(Catalog, catalog, volume_protocol_args=volume_args)
```

## Custom File Types

### Custom Reader/Writer
```python
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

@node(Catalog, "file:datasets[:].raw_files", "file:datasets[:].processed_files")
def process_file_collection(raw_data: list[np.ndarray]) -> list[np.ndarray]:
    return [process_array(arr) for arr in raw_data]
```

## Performance Considerations

### Lazy Loading
Files are loaded only when accessed:

```python
# This doesn't load the file
file_ref = state["dataset.large_file"]

# This loads the file content
content = state["file:dataset.large_file"]
```

### Batch File Operations
Process multiple files efficiently:

```python
@node(Catalog, "file:large_dataset[:].chunks", "summary.statistics")
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