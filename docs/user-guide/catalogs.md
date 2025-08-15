# Working with Catalogs

Catalogs are the storage layer of PyPond, providing persistent, efficient storage for your structured data. This guide covers the different catalog options and how to work with them effectively.

## Overview

PyPond supports three main catalog backends, each optimized for different use cases:

| Catalog | Best For | Key Features |
|---------|----------|--------------|
| **Iceberg** | Analytics, Large Datasets | ACID transactions, schema evolution, partitioning |
| **Lance** | Vector Data, ML Workloads | Fast queries, vector search, version control |
| **Delta Lake** | Streaming, Data Lakes | ACID transactions, time travel, streaming support |

## Iceberg Catalog

Apache Iceberg is ideal for large-scale analytics workloads with evolving schemas.

### Setup

```python
from pond.catalogs.iceberg_catalog import IcebergCatalog

# Local filesystem
catalog = IcebergCatalog(name="analytics")

# With custom warehouse location
catalog = IcebergCatalog(name="analytics", db_path="./warehouse")

# With S3 backend (requires proper credentials)
catalog = IcebergCatalog(
    name="analytics",
    db_path="s3://my-bucket/warehouse"
)
```

### Features

#### Schema Evolution
Iceberg handles schema changes gracefully:

```python
# Version 1 of your model
class DataV1(BaseModel):
    id: int
    name: str

# Later, evolve to version 2  
class DataV2(BaseModel):
    id: int
    name: str
    description: str  # New field
    deprecated_field: str = None  # Optional for backward compatibility
```

#### Time Travel
Access historical versions of your data:

```python
# This is handled automatically by Iceberg
# Each write creates a new snapshot
state["results.data"] = new_data  # Creates snapshot N
# Previous versions remain accessible
```

#### Partitioning
Optimize queries with automatic partitioning:

```python
class TimeSeriesData(BaseModel):
    timestamp: datetime
    sensor_id: str
    value: float

# Iceberg can partition by timestamp for efficient time-range queries
```

## Lance Catalog

Lance is optimized for vector data and machine learning workloads.

### Setup

```python
from pond.catalogs.lance_catalog import LanceCatalog

# Local Lance database
catalog = LanceCatalog(db_path="./lance_data")

# The database will be created automatically
```

### Features

#### Vector Optimization
Lance excels at handling vector and array data:

```python
class EmbeddingData(BaseModel):
    text: str
    embedding: list[float]  # High-dimensional vectors
    metadata: dict[str, str]

# Lance stores and queries vectors efficiently
@node(Catalog, "documents[:].text", "documents[:].embedding")
def compute_embeddings(text: str) -> list[float]:
    # Your embedding model here
    return [0.1, 0.2, 0.3, ...]  # 768-dimensional vector
```

#### Fast Queries
Optimized for analytical queries:

```python
# Lance provides efficient filtering and aggregation
lens = state.lens("table:documents[:].embedding")
table = lens.get()  # Returns PyArrow table for fast operations
```

#### Version Control
Built-in versioning for reproducibility:

```python
# Each write is versioned automatically
# Perfect for ML experiment tracking
```

## Delta Lake Catalog

Delta Lake provides ACID guarantees and streaming support.

### Setup

```python
from pond.catalogs.delta_catalog import DeltaCatalog

# Local Delta Lake
catalog = DeltaCatalog(db_path="./delta_data")

# With S3 backend
catalog = DeltaCatalog(db_path="s3://bucket/delta-tables/")
```

### Features

#### ACID Transactions
Guaranteed consistency for concurrent operations:

```python
# Multiple processes can safely write to the same catalog
# Delta Lake ensures consistency
```

#### Streaming Support
Integrate with streaming data sources:

```python
class StreamingData(BaseModel):
    timestamp: datetime
    event_type: str
    payload: dict[str, Any]

# Delta Lake handles incremental updates efficiently
```

#### Time Travel
Access historical versions:

```python
# Delta Lake maintains transaction log
# Enables querying data at specific points in time
```

## Catalog Operations

### Basic Operations

```python
# Create state with catalog
state = State(MyModel, catalog)

# Store data
state["path.to.data"] = value

# Retrieve data  
value = state["path.to.data"]

# Check existence
lens = state.lens("path.to.data")
if lens.exists():
    count = lens.len()
    data = lens.get()
```

### Bulk Operations

```python
# Efficient bulk loading
data_list = [item1, item2, item3, ...]
state["items[:]"] = data_list

# Append to existing data
new_items = [item4, item5]
lens = state.lens("items[:]")
lens.set(new_items, append=True)
```

### Table Access

Use the table variant for efficient computation:

```python
# Get PyArrow table for analysis
table = state.lens("table:items[:]").get()

# Perform computations
import pyarrow.compute as pc
filtered = pc.filter(table, pc.greater(table["value"], 0.5))
aggregated = pc.sum(filtered["amount"])
```

## Performance Considerations

### Iceberg
- **Best**: Large datasets (>1GB), complex schemas, analytical queries
- **Partition**: By frequently-queried columns (time, category)
- **Schema**: Design for evolution - add optional fields

### Lance  
- **Best**: Vector data, ML features, frequent updates
- **Optimize**: For your query patterns - Lance adapts automatically
- **Vector data**: Store embeddings and high-dimensional data directly

### Delta Lake
- **Best**: Streaming data, frequent updates, ACID requirements
- **Compact**: Regularly compact small files for better performance
- **Partition**: By time or frequently-filtered columns

## File System Integration

All catalogs integrate with PyPond's file system through volume protocols:

```python
from pond.volume import load_volume_protocol_args

# Load volume configuration
volume_args = load_volume_protocol_args()

# Create state with volume support
state = State(
    MyModel, 
    catalog, 
    volume_protocol_args=volume_args
)
```

### Volume Configuration

Create a `volume.yaml` file:

```yaml
# Local filesystem
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
```

### Mixed Storage

Combine structured catalog storage with unstructured files:

```python
class MixedData(BaseModel):
    # Structured data in catalog
    metadata: dict[str, str]
    
    # Files in volume storage
    raw_file: File[bytes] = Field(protocol="s3", ext="bin")
    processed_file: File[np.ndarray] = Field(protocol="dir", ext="npy")
```

## Migration Between Catalogs

You can migrate data between catalog backends:

```python
# Source catalog
source_catalog = LanceCatalog(db_path="./source")
source_state = State(MyModel, source_catalog)

# Destination catalog  
dest_catalog = IcebergCatalog(name="migrated")
dest_state = State(MyModel, dest_catalog)

# Migrate data
data = source_state["all.data"]
dest_state["all.data"] = data
```

## Best Practices

### Schema Design
```python
# Good: Optional fields for evolution
class EvolvableModel(BaseModel):
    id: int
    name: str
    new_field: str = None  # Can be added later
    
# Avoid: Required fields that can't be added later
```

### Path Naming
```python
# Good: Hierarchical, descriptive paths
state["experiments.run_001.results.accuracy"]

# Avoid: Flat, cryptic paths  
state["exp1_acc"]
```

### Batch Operations
```python
# Good: Batch writes for efficiency
items = [process(item) for item in input_data]
state["processed_items[:]"] = items

# Avoid: Item-by-item writes
for i, item in enumerate(input_data):
    state[f"processed_items[{i}]"] = process(item)
```

### Error Handling
```python
# Check existence before operations
lens = state.lens("optional.data")
if lens.exists():
    data = lens.get()
    # Process data
else:
    # Handle missing data case
    data = initialize_default_data()
```

This comprehensive approach to catalog management ensures your data pipelines are both performant and maintainable.