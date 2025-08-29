# Copyright 2025 Nils Bore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared test fixtures for pytest-codeblocks documentation testing."""

import tempfile
import shutil
from pathlib import Path
import pytest
import numpy as np

# Only import what's needed for fixtures - avoid namespace conflicts
from pond.catalogs.iceberg_catalog import IcebergCatalog


@pytest.fixture(scope="session")
def temp_warehouse():
    """Create a temporary warehouse directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="pypond_docs_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def docs_test_iceberg_catalog(tmp_path_factory):
    """Create an Iceberg catalog for documentation examples."""
    warehouse_path = tmp_path_factory.mktemp("docs_iceberg_catalog")
    test_catalog = IcebergCatalog(
        "default",
        type="sql",
        uri=f"sqlite:///{warehouse_path}/catalog.db",
        warehouse=f"file://{warehouse_path}",
    )
    test_catalog.catalog.create_namespace_if_not_exists("catalog")
    return test_catalog


@pytest.fixture(scope="function")
def sample_data():
    """Generate sample data for documentation examples."""
    np.random.seed(42)  # Deterministic for testing
    return {
        "raw_data": np.random.randn(1000),
        "threshold": 0.5,
        "multiplier": 2.0,
        "metadata": {"source": "test_sensor", "version": "1.0"}
    }


# Infrastructure helper functions for documentation examples

@pytest.fixture
def create_docs_test_catalog(tmp_path_factory):
    """Creates properly configured IcebergCatalog for docs examples."""
    def _create():
        warehouse_path = tmp_path_factory.mktemp("pypond_docs")
        test_catalog = IcebergCatalog(
            "default",
            type="sql",
            uri=f"sqlite:///{warehouse_path}/catalog.db",
            warehouse=f"file://{warehouse_path}",
        )
        test_catalog.catalog.create_namespace_if_not_exists("catalog")
        return test_catalog
    return _create


@pytest.fixture  
def create_docs_state(tmp_path_factory):
    """Creates State object with catalog for given model."""
    def _create(model_class):
        from pond import State  # Import only when needed
        warehouse_path = tmp_path_factory.mktemp("pypond_docs")
        test_catalog = IcebergCatalog(
            "default",
            type="sql",
            uri=f"sqlite:///{warehouse_path}/catalog.db",
            warehouse=f"file://{warehouse_path}",
        )
        test_catalog.catalog.create_namespace_if_not_exists("catalog")
        return State(model_class, test_catalog)
    return _create


@pytest.fixture
def setup_docs_example(tmp_path_factory):
    """Returns configured catalog and state for given model."""
    def _setup(model_class):
        from pond import State  # Import only when needed
        warehouse_path = tmp_path_factory.mktemp("pypond_docs")
        test_catalog = IcebergCatalog(
            "default",
            type="sql",
            uri=f"sqlite:///{warehouse_path}/catalog.db",
            warehouse=f"file://{warehouse_path}",
        )
        test_catalog.catalog.create_namespace_if_not_exists("catalog")
        state = State(model_class, test_catalog)
        return test_catalog, state
    return _setup


def get_deterministic_data(size=1000, seed=42):
    """Returns consistent numpy test data."""
    np.random.seed(seed)
    return np.random.randn(size)


@pytest.fixture
def create_sample_storage(tmp_path_factory):
    """Creates sample storage with test files for File field examples."""
    def _create_storage():
        import os
        from pond.io.writers import write_npz
        
        storage_path = tmp_path_factory.mktemp("sample_storage")
        
        # Create some sample numpy arrays and save them
        np.random.seed(42)
        for i in range(3):
            data = np.random.randn(100)
            file_path = storage_path / f"sample_{i}.npz" 
            # Fix function signature - write_npz takes (array, fs, path)
            import fsspec
            fs = fsspec.filesystem('file')
            write_npz(data, fs, str(file_path))
            
        return str(storage_path)
    return _create_storage


@pytest.fixture
def setup_file_example_with_storage(tmp_path_factory):
    """Sets up a complete example with catalog and sample files for File field demos."""
    def _setup():
        import os
        from pond.io.writers import write_npz
        
        # Create temporary directories  
        base_path = tmp_path_factory.mktemp("pypond_docs")
        storage_dir = base_path / "storage"
        storage_dir.mkdir(exist_ok=True)
        
        # Create sample files
        np.random.seed(42)
        for i in range(3):
            data = np.random.randn(100) 
            file_path = storage_dir / f"dataset_{i}.npz"
            # Fix function signature - write_npz takes (array, fs, path)
            import fsspec
            fs = fsspec.filesystem('file')
            write_npz(data, fs, str(file_path))
        
        # Create catalog
        test_catalog = IcebergCatalog(
            "default", 
            type="sql",
            uri=f"sqlite:///{base_path}/catalog.db",
            warehouse=f"file://{base_path}",
        )
        test_catalog.catalog.create_namespace_if_not_exists("catalog")
        
        return test_catalog, str(storage_dir)
    return _setup


# Configure pytest-codeblocks
pytest_codeblocks_use_black = False