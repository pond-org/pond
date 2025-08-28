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
import tempfile
from pond.catalogs.iceberg_catalog import IcebergCatalog

# Local filesystem with proper configuration
temp_dir = tempfile.mkdtemp(prefix="analytics_")
catalog = IcebergCatalog(
    "analytics",
    type="sql",
    uri=f"sqlite:///{temp_dir}/catalog.db",
    warehouse=f"file://{temp_dir}"
)

# S3 backend would require:
# catalog = IcebergCatalog(
#     "analytics", 
#     type="glue",  # or "hive"
#     uri="s3://my-bucket/warehouse"
# )
```

### Features

#### Schema Evolution
Iceberg handles schema changes gracefully:

```python
from pydantic import BaseModel
from typing import Optional

# Version 1 of your model
class DataV1(BaseModel):
    id: int
    name: str

# Later, evolve to version 2  
class DataV2(BaseModel):
    id: int
    name: str
    description: str  # New field
    deprecated_field: Optional[str] = None  # Optional for backward compatibility
```

#### Time Travel
Access historical versions of your data:

```python
# This is handled automatically by Iceberg
# Each write creates a new snapshot
example_data = {"values": [1, 2, 3], "timestamp": "2024-01-01"}
# state["results.data"] = example_data  # Creates snapshot N
# Previous versions remain accessible
print("Iceberg automatically handles versioning with each write operation")
```

#### Partitioning
Optimize queries with automatic partitioning:

```python
from pydantic import BaseModel
from datetime import datetime

class TimeSeriesData(BaseModel):
    timestamp: datetime
    sensor_id: str
    value: float

# Iceberg can partition by timestamp for efficient time-range queries
print("Iceberg supports partitioning for efficient time-range queries")
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
from pydantic import BaseModel
from pond import node

class Metadata(BaseModel):
    source: str
    category: str

class EmbeddingData(BaseModel):
    text: str
    embedding: list[float]  # High-dimensional vectors
    metadata: Metadata

class Document(BaseModel):
    text: str
    embedding: list[float]
    metadata: Metadata
    
class Catalog(BaseModel):
    documents: list[Document]

# Lance stores and queries vectors efficiently
@node(Catalog, "documents[:].text", "documents[:].embedding")
def compute_embeddings(text: str) -> list[float]:
    # Your embedding model here (simplified example)
    return [0.1, 0.2, 0.3] * 256  # 768-dimensional vector
```

#### Fast Queries
Optimized for analytical queries:

```python
# Lance provides efficient filtering and aggregation (requires state setup)
print("Lance provides efficient vector operations:")
print("- Fast similarity search")
print("- Efficient filtering and aggregation")
print("- PyArrow integration for analytics")
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
from pydantic import BaseModel
from datetime import datetime
from typing import Any

class Payload(BaseModel):
    user_id: str
    action: str
    value: float

class StreamingData(BaseModel):
    timestamp: datetime
    event_type: str
    payload: Payload

# Delta Lake handles incremental updates efficiently
print("Delta Lake provides ACID guarantees for streaming data")
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
import tempfile
from pond import State
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pydantic import BaseModel

class Data(BaseModel):
    values: list[float]
    
class MyModel(BaseModel):
    path: Data

# Create state with catalog
catalog_temp = tempfile.mkdtemp(prefix="example_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="example_warehouse_")
catalog = IcebergCatalog(
    "example",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(MyModel, catalog)

# Store data
example_data = Data(values=[1.0, 2.0, 3.0])
state["path"] = example_data

# Retrieve data  
retrieved_value = state["path"]
print(f"Retrieved: {retrieved_value.values}")

# Check existence
lens = state.lens("path")
if lens.exists():
    data = lens.get()
    print(f"Data exists: {data.values}")
```

### Bulk Operations

```python
import tempfile
from pond import State
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pydantic import BaseModel

class Item(BaseModel):
    value: float
    name: str
    
class BulkModel(BaseModel):
    items: list[Item]

# Setup catalog
catalog_temp = tempfile.mkdtemp(prefix="bulk_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="bulk_warehouse_")
catalog = IcebergCatalog(
    "bulk",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(BulkModel, catalog)

# Efficient bulk loading
item1 = Item(value=1.0, name="first")
item2 = Item(value=2.0, name="second")
item3 = Item(value=3.0, name="third")
data_list = [item1, item2, item3]
state["items"] = data_list

# Append to existing data
item4 = Item(value=4.0, name="fourth")
item5 = Item(value=5.0, name="fifth")
new_items = [item4, item5]
lens = state.lens("items")
lens.set(new_items, append=True)

print(f"Total items after append: {lens.len()}")
```

### Table Access

Use the table variant for efficient computation:

```python
import tempfile
import pyarrow.compute as pc
from pond import State
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pydantic import BaseModel

class Item(BaseModel):
    value: float
    name: str
    
class BulkModel(BaseModel):
    items: list[Item]

# Setup catalog
catalog_temp = tempfile.mkdtemp(prefix="table_catalog_")
warehouse_temp = tempfile.mkdtemp(prefix="table_warehouse_")
catalog = IcebergCatalog(
    "table",
    type="sql",
    uri=f"sqlite:///{catalog_temp}/catalog.db",
    warehouse=f"file://{warehouse_temp}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(BulkModel, catalog)

# Add some sample data
item1 = Item(value=1.0, name="first")
item2 = Item(value=2.0, name="second")
item3 = Item(value=3.0, name="third")
state["items"] = [item1, item2, item3]

# Get PyArrow table for analysis
table = state.lens("table:items").get()

# Perform computations
filtered = pc.filter(table, pc.greater(table["value"], 1.5))
result_count = len(filtered)
print(f"Items with value > 1.5: {result_count}")
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
import tempfile
from pond.volume import load_volume_protocol_args
from pond import State
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pydantic import BaseModel

class MyModel(BaseModel):
    data: list[float]

# Load volume configuration
volume_args = load_volume_protocol_args()

# Create state with volume support
temp_dir = tempfile.mkdtemp(prefix="volume_")
catalog_db = tempfile.mkdtemp(prefix="volume_catalog_db_")
catalog = IcebergCatalog(
    "volume",
    type="sql",
    uri=f"sqlite:///{catalog_db}/catalog.db",
    warehouse=f"file://{temp_dir}"
)
catalog.catalog.create_namespace_if_not_exists("default")
state = State(
    MyModel, 
    catalog, 
    volume_protocol_args=volume_args
)

print("State created with volume storage support")
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
from pydantic import BaseModel
from pond import Field, File
import numpy as np

class Metadata(BaseModel):
    source: str
    version: str

class MixedData(BaseModel):
    # Structured data in catalog
    metadata: Metadata
    
    # Files in volume storage
    raw_file: File[bytes] = Field(protocol="s3", ext="bin")
    processed_file: File[np.ndarray] = Field(protocol="dir", ext="npy")

print("Mixed storage: structured data in catalog, files in volume storage")
```

## Migration Between Catalogs

You can migrate data between catalog backends:

```python
from pond.catalogs.lance_catalog import LanceCatalog
from pond.catalogs.iceberg_catalog import IcebergCatalog
from pond import State
from pydantic import BaseModel
import tempfile

class MigrationData(BaseModel):
    values: list[float]
    
class MyModel(BaseModel):
    all: MigrationData

# Source catalog
source_temp = tempfile.mkdtemp(prefix="source_")
source_catalog = LanceCatalog(db_path=source_temp)
source_state = State(MyModel, source_catalog)

# Destination catalog
dest_catalog_temp = tempfile.mkdtemp(prefix="dest_catalog_")
dest_warehouse_temp = tempfile.mkdtemp(prefix="dest_warehouse_")
dest_catalog = IcebergCatalog(
    "migrated",
    type="sql", 
    uri=f"sqlite:///{dest_catalog_temp}/catalog.db",
    warehouse=f"file://{dest_warehouse_temp}"
)
dest_catalog.catalog.create_namespace_if_not_exists("default")
dest_state = State(MyModel, dest_catalog)

# Migrate data (example)
sample_data = MigrationData(values=[1.0, 2.0, 3.0])
source_state["all"] = sample_data
data = source_state["all"]
dest_state["all"] = data

print("Data migration completed")
```

## Best Practices

### Schema Design

### Path Naming

### Batch Operations

### Error Handling

This comprehensive approach to catalog management ensures your data pipelines are both performant and maintainable.