from typing import Callable, Type

from fastapi import FastAPI
from pydantic import BaseModel

from pond.api.input_transform import FastAPIInputTransform
from pond.api.output_transform import FastAPIOutputTransform
from pond.transforms.abstract_transform import (
    AbstractExecuteTransform,
    AbstractTransform,
)
from pond.transforms.transform import Transform
from pond.transforms.transform_construct import TransformConstruct
from pond.transforms.transform_index import TransformIndex
from pond.transforms.transform_list import TransformList
from pond.transforms.transform_list_fold import TransformListFold
from pond.transforms.transform_pipe import TransformPipe


class node:
    """Decorator class for creating data transformation nodes.

    A node represents a single transformation step in a data pipeline,
    mapping input data paths to output data paths through a function.
    Automatically selects the appropriate transform type based on input/output patterns.

    Transform Type Selection Rules:
    - Transform: Scalar input/output (e.g., input="params.value", output="result.total")
    - TransformList: Array input AND array output (e.g., input="clouds[:].points", output="clouds[:].bounds")
    - TransformListFold: Array input, scalar output (e.g., input="clouds[:].points", output="global_bounds")

    Examples:
        # Basic transform - processes single values
        @node(Catalog, "params.resolution", "grid.cell_size")
        def scale_resolution(res: float) -> float:
            return res * 2.0

        # List transform - processes each array element independently
        @node(Catalog, "clouds[:].raw_points", "clouds[:].filtered_points")
        def filter_points(points: list[Point]) -> list[Point]:
            return [p for p in points if p.z > 0]

        # List fold transform - aggregates array elements to single value
        @node(Catalog, "clouds[:].bounds", "global_bounds")
        def merge_bounds(bounds_list: list[Bounds]) -> Bounds:
            return combine_all_bounds(bounds_list)

    Attributes:
        Catalog: The pydantic model class defining the data structure schema.
        input: Input path(s) as string or list of strings. Supports array notation
            like "clouds[:].points" for processing all array elements.
        output: Output path(s) as string or list of strings. Must be compatible
            with the transform function's return type.

    Note:
        The decorator analyzes input/output patterns to choose between Transform,
        TransformList, or TransformListFold implementations automatically.
        Array notation "[:]" triggers collection processing behavior.
    """

    def __init__(
        self,
        Catalog: Type[BaseModel],
        input: list[str] | str,
        output: list[str] | str,
    ):
        """Initialize a node decorator.

        Args:
            Catalog: Pydantic model class defining the data schema structure.
                Must be a subclass of BaseModel with proper field annotations.
            input: Input data path(s). Can be a single path string or list of paths.
                Supports array wildcard notation "[:] " for processing collections.
            output: Output data path(s). Can be a single path string or list of paths.
                Must match the return type structure of the decorated function.

        Note:
            Path strings use dot notation (e.g., "data.field[0].subfield").
            Array wildcards "[:] " enable collection processing transforms.
        """
        self.Catalog = Catalog
        self.input = input
        self.output = output

    def __call__(
        self,
        fn: Callable,
    ) -> AbstractExecuteTransform:
        """Apply the node decorator to a function.

        Analyzes the input/output path patterns to automatically select
        the appropriate transform implementation and wraps the function.

        Transform Selection Logic:
        1. Checks for "[:]" wildcard in input paths → list_input = True
        2. Checks for "[:]" wildcard in output paths → list_output = True
        3. Selects transform type:
           - list_input=True + list_output=True → TransformList
           - list_input=True + list_output=False → TransformListFold
           - list_input=False + list_output=False → Transform
           - list_input=False + list_output=True → RuntimeError (unsupported)

        Args:
            fn: The function to be wrapped as a transform node. Must have
                type annotations that match the input/output path types.

        Returns:
            An AbstractExecuteTransform instance of the appropriate type:
            - Transform: For scalar input/output
            - TransformList: For array input and array output
            - TransformListFold: For array input and scalar output

        Raises:
            RuntimeError: If output paths use "[:] " notation without corresponding
                array inputs, which is not supported.

        Note:
            The function must have proper type annotations that align with the
            data types at the specified input/output paths in the catalog schema.
        """
        inputs = self.input if isinstance(self.input, list) else [self.input]
        outputs = self.output if isinstance(self.output, list) else [self.output]
        list_input = False
        list_output = False
        for input in inputs:
            if "[:]" in input:
                list_input = True
                break
        for output in outputs:
            if "[:]" in output:
                list_output = True
                break
        if list_input and list_output:
            return TransformList(self.Catalog, self.input, self.output, fn)
        elif list_input:
            return TransformListFold(self.Catalog, self.input, self.output, fn)
        elif list_output:
            raise RuntimeError("Outputs can not use [:] indices without any in input")
        else:
            return Transform(self.Catalog, self.input, self.output, fn)


def pipe(
    transforms: list[AbstractTransform],
    input: list[str] | str = [],
    output: list[str] | str = [],
    root_path: str = "catalog",
) -> TransformPipe:
    """Create a pipeline of transforms and/or sub-pipelines that execute in sequence.

    Composes multiple transforms into a single executable unit that
    processes data through each transform step in order. Creates a TransformPipe
    that coordinates execution and data flow between transform stages.

    The transforms list can contain:
    - Individual transform nodes (created with @node decorator)
    - Other pipelines (created with pipe() function) for hierarchical composition
    - TransformConstruct, TransformIndex, and other transform implementations

    Example:
        # Pipeline with individual transforms
        basic_pipeline = pipe([
            index_files(Catalog, "input_files"),
            parse_data,
            validate_data
        ], output="processed_data")

        # Pipeline composing other pipelines
        preprocessing = pipe([parse_files, clean_data])
        processing = pipe([compute_features, analyze_data])

        full_pipeline = pipe([
            preprocessing,  # Sub-pipeline
            processing,     # Sub-pipeline
            generate_report # Individual transform
        ], input="raw_data", output="final_report")

    Args:
        transforms: List of AbstractTransform instances to execute sequentially.
            Can include individual transforms, other pipelines, or mixed.
            Each transform's outputs should align with the next transform's inputs.
        input: Optional input path(s) for the overall pipeline. Empty list
            means the pipeline doesn't require external inputs.
        output: Optional output path(s) for the overall pipeline. Empty list
            means the pipeline doesn't produce external outputs.
        root_path: Root path name for resolving relative paths. Defaults to "catalog".

    Returns:
        TransformPipe instance that executes all transforms in sequence.

    Note:
        Default mutable arguments [] are used for backward compatibility.
        The pipeline validates that transform input/output types are compatible.
        Pipelines can be nested to arbitrary depth for modular composition.
    """
    return TransformPipe(transforms, input, output, root_path)


def construct(
    Catalog: Type[BaseModel],
    path: str = "",
) -> TransformConstruct:
    """Create a transform that constructs pydantic model instances.

    Creates a TransformConstruct that builds instances of the specified
    pydantic model class, useful for initializing data structures.

    Example:
        init_catalog = construct(Catalog)
        # Creates an empty Catalog instance to start the pipeline

    Args:
        Catalog: Pydantic model class to construct instances of.
            Must be a subclass of BaseModel.
        path: Optional path where the constructed instance will be stored.
            Empty string uses the root path.

    Returns:
        TransformConstruct instance configured for the specified model type.

    Note:
        The construct transform is typically used at the beginning of pipelines
        to initialize data structures before populating them with data.
    """
    return TransformConstruct(Catalog, path)


def index_files(
    Catalog: Type[BaseModel],
    path: list[str] | str = "",
    root_path: str = "catalog",
) -> TransformIndex:
    """Create a transform that indexes files from the filesystem into the catalog.

    Scans filesystem locations for files matching the schema patterns
    and creates catalog entries for discovered files. Useful for ingesting
    existing file collections into pond pipelines. Creates a TransformIndex
    that discovers files based on field metadata (extensions, paths, protocols).

    Example:
        discover_las_files = index_files(Catalog, "cloud_files")
        # Scans for .laz files and creates File[LasData] entries

    Args:
        Catalog: Pydantic model class defining the file schema structure.
            Must contain File fields with appropriate metadata (reader, ext, path).
        path: Path(s) to index files from. Can be a string path or list of paths.
            Empty string indexes from the root path.
        root_path: Root path name for resolving relative paths. Defaults to "catalog".

    Returns:
        TransformIndex instance that will discover and index files.

    Note:
        File discovery behavior depends on field metadata (extensions, protocols).
        The transform scans for files matching the expected patterns and creates
        File instances that can be processed by subsequent transforms.
    """
    return TransformIndex(Catalog, path, root_path)


def fastapi_input(
    Catalog: Type[BaseModel], input: list[str] | str, app: FastAPI
) -> FastAPIInputTransform:
    """Create a transform that waits for HTTP inputs to populate state paths.

    Registers POST endpoints on the provided FastAPI app for setting state values.
    The transform blocks execution until all specified input paths have been
    populated via HTTP requests. URLs are automatically derived from path strings.

    Args:
        Catalog: Pydantic model class defining the data schema structure.
        input: Input path(s) to expose as HTTP endpoints. Can be a single path
            string or list of paths. Supports array notation like "clouds[:].points".
        app: FastAPI application instance to register endpoints on.

    Returns:
        FastAPIInputTransform that waits for HTTP inputs.

    Note:
        URLs are automatically generated from paths:
        - "params" → POST /input/params
        - "clouds[0].points" → POST /input/clouds/0/points
        - File[DataT] fields support multipart form uploads
        - Regular fields accept JSON payloads
    """
    input_paths = [input] if isinstance(input, str) else input
    return FastAPIInputTransform(Catalog, input_paths, app)


def fastapi_output(
    Catalog: Type[BaseModel], output: list[str] | str, app: FastAPI
) -> FastAPIOutputTransform:
    """Create a transform that exposes state paths as HTTP endpoints.

    Registers GET endpoints on the provided FastAPI app for retrieving state values.
    The transform runs after pipeline computation to serve results via HTTP.
    URLs are automatically derived from path strings.

    Args:
        Catalog: Pydantic model class defining the data schema structure.
        output: Output path(s) to expose as HTTP endpoints. Can be a single path
            string or list of paths. Supports array notation like "clouds[:].bounds".
        app: FastAPI application instance to register endpoints on.

    Returns:
        FastAPIOutputTransform that exposes outputs via HTTP endpoints.

    Note:
        URLs are automatically generated from paths:
        - "heightmap_plot" → GET /output/heightmap_plot
        - "clouds[0].bounds" → GET /output/clouds/0/bounds
        - File[DataT] fields return file downloads
        - Regular fields return JSON responses
    """
    output_paths = [output] if isinstance(output, str) else output
    return FastAPIOutputTransform(Catalog, output_paths, app)
