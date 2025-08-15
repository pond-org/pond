# API Reference

This section provides detailed documentation for all PyPond classes, functions, and modules.

## Core Components

- **[Decorators](core/decorators.md)**: Transform decorators (@node, @pipe, etc.)
- **[Field](core/field.md)**: File field definitions and metadata
- **[State](core/state.md)**: Pipeline state management and data access
- **[Lens](core/lens.md)**: Path-based data access system
- **[Volume](core/volume.md)**: Filesystem configuration

## Transform System

- **[Abstract Transform](transforms/abstract.md)**: Base transform interfaces
- **[Transform](transforms/transform.md)**: Scalar transformations
- **[Transform List](transforms/transform-list.md)**: Array transformations
- **[Transform List Fold](transforms/transform-list-fold.md)**: Array reduction transformations
- **[Transform Pipe](transforms/transform-pipe.md)**: Pipeline composition

## Catalog Backends

- **[Abstract Catalog](catalogs/abstract.md)**: Base catalog interface
- **[Iceberg Catalog](catalogs/iceberg.md)**: Apache Iceberg integration
- **[Lance Catalog](catalogs/lance.md)**: Lance database integration  
- **[Delta Catalog](catalogs/delta.md)**: Delta Lake integration

## Execution

- **[Abstract Runner](runners/abstract.md)**: Base runner interface
- **[Sequential Runner](runners/sequential.md)**: Sequential execution
- **[Parallel Runner](runners/parallel.md)**: Parallel execution with dependency resolution

## I/O System

- **[Readers](io/readers.md)**: File reading functions
- **[Writers](io/writers.md)**: File writing functions

## Extensibility

- **[Abstract Hook](hooks/abstract.md)**: Hook system for extensibility
- **[UI Hook](hooks/ui.md)**: Hamilton UI integration hook
- **[Marimo Progress Bar Hook](hooks/marimo-progress.md)**: Marimo notebook progress tracking