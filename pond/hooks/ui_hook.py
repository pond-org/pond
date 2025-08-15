import datetime
import hashlib
import inspect
import os
import random
import traceback
from datetime import timezone

# from types import ModuleType
from typing import Optional, Type, Union

from hamilton_sdk.api.clients import (  # type: ignore
    BasicSynchronousHamiltonClient,
    ResourceDoesNotExistException,
    UnauthorizedException,
)
from hamilton_sdk.api.projecttypes import GitInfo  # type: ignore
from hamilton_sdk.driver import (  # type: ignore
    _derive_url,
    _derive_version_control_info,
    _get_fully_qualified_function_path,
    validate_tags,
)
from hamilton_sdk.tracking.data_observation import ObservationType  # type: ignore
from hamilton_sdk.tracking.runs import Status, TrackingState  # type: ignore
from hamilton_sdk.tracking.trackingtypes import TaskRun  # type: ignore
from loguru import logger
from pydantic import BaseModel

from pond.hooks.abstract_hook import AbstractHook
from pond.lens import LensInfo, LensPath, get_cleaned_path
from pond.transforms.abstract_transform import AbstractExecuteTransform
from pond.transforms.transform_index import TransformIndex
from pond.transforms.transform_pipe import TransformPipe

LONG_SCALE = float(0xFFFFFFFFFFFFFFF)
"""Large scale constant used for hash normalization.

Used to convert hash values to floating point numbers in the range [0, 1]
for deterministic sampling operations.
"""


def get_node_name(name: str, task_id: Optional[str]) -> str:
    """Generate a node name for Hamilton tracking.

    Args:
        name: Base name of the transform/node.
        task_id: Optional task identifier for parallel execution.

    Returns:
        Combined node name, either the base name or prefixed with task_id.

    Note:
        Used to create unique node names for Hamilton UI when transforms
        are executed in parallel with different task identifiers.
    """
    return name if task_id is None else f"{task_id}-{name}"


def _convert_classifications(transform: AbstractExecuteTransform) -> list[str]:
    """Convert PyPond transform to Hamilton node classifications.

    Args:
        transform: PyPond transform to classify.

    Returns:
        List of string classifications for Hamilton UI display.
        Currently returns ["data_loader"] for TransformIndex,
        ["transform"] for other transforms.

    Note:
        Classifications help Hamilton UI categorize and display
        different types of pipeline nodes appropriately.
    """
    out = []
    # if (
    #     node_.tags.get("hamilton.data_loader")
    #     and node_.tags.get("hamilton.data_loader.has_metadata") is not False
    # ):
    #     out.append("data_loader")
    # elif node_.tags.get("hamilton.data_saver"):
    #     out.append("data_saver")
    # elif node_.user_defined:
    #     out.append("input")
    # else:
    #     out.append("transform")
    if isinstance(transform, TransformIndex):
        out.append("data_loader")
    else:
        out.append("transform")
    return out


def _convert_node_dependencies(
    root_type: Type[BaseModel],
    transform: AbstractExecuteTransform,
    transform_dict: dict[str, AbstractExecuteTransform],
    deps: set[str],
) -> dict:
    """Convert PyPond dependencies to Hamilton dependency format.

    Args:
        root_type: Root pydantic model type for type resolution.
        transform: Transform whose dependencies are being converted.
        transform_dict: Mapping of transform names to transform objects.
        deps: Set of dependency names (transform names or input paths).

    Returns:
        Dictionary containing Hamilton-compatible dependency information
        with dependency names, type specifications, and schema versions.

    Note:
        Resolves dependency types by checking if they are transforms
        (with output types) or external inputs (with schema types).
    """
    dependencies = []
    dependency_specs = []
    dependency_specs_type = "python_type"
    dependency_specs_schema_version = 1
    for dep in deps:
        dependencies.append(dep)
        if dep in transform_dict:
            # Dep is a transform
            dependency_specs.append(
                {"type_name": str(transform_dict[dep].get_output_type(root_type))}
            )
        else:
            # Dep is an input
            dependency_specs.append(
                {"type_name": str(LensInfo.from_path(root_type, dep).get_type())}
            )

    return {
        "dependencies": dependencies,
        "dependency_specs": dependency_specs,
        "dependency_specs_type": dependency_specs_type,
        "dependency_specs_schema_version": dependency_specs_schema_version,
    }


def _extract_node_templates_from_function_graph(
    # fn_graph: graph.FunctionGraph,
    root_type: Type[BaseModel],
    transforms: list[AbstractExecuteTransform],
    dependencies: dict[str, set[str]],
    inputs: dict[str, Type],
) -> list[dict]:
    """Convert PyPond transforms to Hamilton node templates.

    Args:
        root_type: Root pydantic model type for the pipeline.
        transforms: List of PyPond transforms to convert.
        dependencies: Mapping of transform names to their dependencies.
        inputs: External inputs to the pipeline with their types.

    Returns:
        List of dictionaries containing Hamilton node template information
        including names, types, documentation, and dependencies.

    Note:
        Creates Hamilton-compatible node definitions for both transforms
        and external inputs, enabling visualization and tracking in Hamilton UI.
    """
    transform_dict = {transform.get_name(): transform for transform in transforms}
    node_templates = []
    for transform in transforms:
        name = transform.get_name()
        node_templates.append(
            dict(
                name=name,
                output={"type_name": str(transform.get_output_type(root_type))},
                output_type="python_type",
                output_schema_version=1,  # TODO -- merge this with _convert_node_dependencies
                documentation=transform.get_docs(),
                tags={},  # node_.tags,  # TODO -- ensure serializable
                classifications=_convert_classifications(transform),
                code_artifact_pointers=[
                    _get_fully_qualified_function_path(transform.get_fn())
                ],
                **_convert_node_dependencies(
                    root_type, transform, transform_dict, dependencies[name]
                ),
            )
        )
    for input_name, input_type in inputs.items():
        node_templates.append(
            dict(
                name=input_name,
                output={"type_name": str(input_type)},
                output_type="python_type",
                output_schema_version=1,  # TODO -- merge this with _convert_node_dependencies
                documentation="User provided input",
                tags={},  # node_.tags,  # TODO -- ensure serializable
                classifications=["input"],
                code_artifact_pointers=[],
                dependencies=[],
                dependency_specs=[],
                dependency_specs_type="python_type",
                dependency_specs_schema_version=1,
            )
        )
    return node_templates


def extract_code_artifacts_from_function_graph(
    transforms: list[AbstractExecuteTransform], vcs_info: GitInfo, repo_base_path: str
) -> list[dict]:
    """Extract source code artifacts from PyPond transforms.

    Args:
        transforms: List of transforms to extract code from.
        vcs_info: Git repository information for URL generation.
        repo_base_path: Base path of the repository for relative paths.

    Returns:
        List of dictionaries containing code artifact information including
        function names, file paths, line numbers, and repository URLs.

    Note:
        Extracts source code information for Hamilton UI to display
        code locations and enable navigation to source files.
        Handles decorated functions by unwrapping to find actual source.
    """
    seen = set()
    out = []
    for transform in transforms:
        fn = transform.get_fn()
        fn_name = _get_fully_qualified_function_path(fn)
        if fn_name not in seen:
            seen.add(fn_name)
            # need to handle decorators -- they will return the wrong sourcefile.
            unwrapped_fn = inspect.unwrap(fn)
            if unwrapped_fn != fn:
                # TODO: pull decorator stuff too
                source_file = inspect.getsourcefile(unwrapped_fn)
            else:
                source_file = inspect.getsourcefile(fn)
            if source_file is not None:
                path = os.path.relpath(source_file, repo_base_path)
            else:
                path = ""
            try:
                source_lines = inspect.getsourcelines(fn)
                out.append(
                    dict(
                        name=fn_name,
                        type="p_function",
                        path=path,
                        start=inspect.getsourcelines(fn)[1] - 1,
                        end=inspect.getsourcelines(fn)[1] - 1 + len(source_lines[0]),
                        url=_derive_url(vcs_info, path, source_lines[1]),
                    )
                )
            except OSError:
                # This is an error state where somehow we don't have
                # source code.
                out.append(
                    dict(
                        name=fn_name,
                        type="p_function",
                        path=path,
                        start=0,
                        end=0,
                        url=_derive_url(vcs_info, path, 0),
                    )
                )
    return out


def extract_code_artifacts_from_inputs(
    inputs: dict[str, Type], vcs_info: GitInfo, repo_base_path: str
) -> list[dict]:
    """Extract source code artifacts from pipeline input types.

    Args:
        inputs: Dictionary mapping input names to their Python types.
        vcs_info: Git repository information for URL generation.
        repo_base_path: Base path of the repository for relative paths.

    Returns:
        List of dictionaries containing code artifact information for
        input type definitions including class names, file paths, and URLs.

    Note:
        Extracts type definition information for Hamilton UI to show
        where input types are defined in the codebase.
        Skips built-in types that don't have source files.
    """
    seen = set()
    out = []
    for input_name, input_type in inputs.items():
        class_name = _get_fully_qualified_function_path(input_type)
        if class_name not in seen:
            seen.add(class_name)
            # need to handle decorators -- they will return the wrong sourcefile.
            try:
                source_file = inspect.getsourcefile(input_type)
            except TypeError:
                # Happens e.g. for built-in types
                continue
            if source_file is not None:
                path = os.path.relpath(source_file, repo_base_path)
            else:
                path = ""
            try:
                source_lines = inspect.getsourcelines(input_type)
                out.append(
                    dict(
                        name=class_name,
                        type="p_class",
                        path=path,
                        start=inspect.getsourcelines(input_type)[1] - 1,
                        end=inspect.getsourcelines(input_type)[1]
                        - 1
                        + len(source_lines[0]),
                        url=_derive_url(vcs_info, path, source_lines[1]),
                    )
                )
            except OSError:
                # This is an error state where somehow we don't have
                # source code.
                out.append(
                    dict(
                        name=class_name,
                        type="p_class",
                        path=path,
                        start=0,
                        end=0,
                        url=_derive_url(vcs_info, path, 0),
                    )
                )
    return out


def _slurp_code(
    transforms: list[AbstractExecuteTransform], repo_base: str
) -> list[dict]:
    """Read complete source files containing transform functions.

    Args:
        transforms: List of transforms to extract source modules from.
        repo_base: Base repository path for relative path calculation.

    Returns:
        List of dictionaries containing full source file contents
        with relative file paths for Hamilton UI code viewing.

    Note:
        Reads entire Python modules containing transform functions
        to provide complete source context in Hamilton UI.
    """
    modules = set()
    for transform in transforms:
        module = inspect.getmodule(transform.get_fn())
        if module is not None:
            modules.add(module)
    out = []
    for module in modules:
        if hasattr(module, "__file__") and module.__file__ is not None:
            module_path = os.path.relpath(module.__file__, repo_base)
            with open(module.__file__, "r") as f:
                out.append({"path": module_path, "contents": f.read()})
    return out


def add_dependency(
    transform_name: str,
    path: LensPath,
    transforms: list[AbstractExecuteTransform],
    dependencies: dict[str, set[str]],
):
    """Add a dependency to a transform based on path availability.

    Args:
        transform_name: Name of the transform that needs the dependency.
        path: LensPath representing the required data.
        transforms: List of all transforms in the pipeline.
        dependencies: Dictionary to update with the new dependency.

    Note:
        Searches through all transforms to find which one produces
        the required path, then adds that transform as a dependency.
        Uses subset relationships to handle array wildcards correctly.
    """
    for transform in transforms:
        outputs = transform.get_outputs()
        for o in outputs:
            if path.subset_of(o):
                dependencies[transform_name].add(transform.get_name())
                break


def compute_dependencies(
    transforms: list[AbstractExecuteTransform],
    inputs: dict[str, Type],
) -> dict[str, set[str]]:
    """Compute dependency graph for all transforms in the pipeline.

    Args:
        transforms: List of all transforms in the pipeline.
        inputs: External inputs available to the pipeline.

    Returns:
        Dictionary mapping transform names to sets of their dependencies
        (either other transform names or external input names).

    Note:
        Analyzes each transform's input requirements and determines
        which other transforms or external inputs satisfy those requirements.
        Essential for Hamilton UI to understand execution dependencies.
    """
    dependencies: dict[str, set[str]] = {}
    for transform in transforms:
        name = transform.get_name()
        dependencies[name] = set()
        for i in transform.get_inputs():
            add_dependency(name, i, transforms, dependencies)
            for input_name in inputs.keys():
                p = get_cleaned_path(input_name, "catalog")
                if i.subset_of(p):
                    dependencies[name].add(input_name)
    return dependencies


def process_result(
    transform: AbstractExecuteTransform,
) -> tuple[Optional[ObservationType], Optional[ObservationType], list[ObservationType]]:
    """Process transform execution results for Hamilton observability.

    Args:
        transform: The executed transform to process results for.

    Returns:
        Tuple containing:
        - Statistics observability data (or None)
        - Schema observability data (or None)
        - List of additional observability data

    Note:
        Currently returns a disabled result summary placeholder.
        This function provides a hook for future result processing
        and observability data extraction.
    """
    schema = None
    additional: list[ObservationType] = []
    statistics = {
        "observability_type": "primitive",
        "observability_value": {
            "type": "str",
            "value": "RESULT SUMMARY DISABLED",
        },
        "observability_schema_version": "0.0.1",
    }
    return statistics, schema, additional


class UIHook(AbstractHook):
    def __init__(
        self,
        project_id: int,
        username: str,
        dag_name: str,
        # root_type: Type[BaseModel],
        tags: dict[str, str] = {},
        api_key: Optional[str] = None,
        hamilton_api_url="http://localhost:8241",
        hamilton_ui_url="http://localhost:8241",
        verify: Union[str, bool] = True,
        run_id: str = "dev",
    ):
        """Initialize the Hamilton UI integration hook.

        This hook integrates PyPond pipeline execution with Hamilton UI for
        monitoring, visualization, and tracking of pipeline runs.

        Args:
            project_id: The ID of the Hamilton project to track runs under.
            username: The username for Hamilton API authentication.
            dag_name: The name of the DAG/pipeline for display in Hamilton UI.
            tags: Dictionary of tags to help curate and organize the DAG.
            api_key: The API key for Hamilton authentication. Optional.
            hamilton_api_url: Hamilton API endpoint URL. Defaults to localhost:8241.
            hamilton_ui_url: Hamilton UI endpoint URL. Defaults to localhost:8241.
            verify: SSL verification setting passed to requests library.
            run_id: Identifier for this specific run. Defaults to "dev".

        Note:
            This hook requires a running Hamilton UI instance and proper
            authentication credentials. The project must exist in Hamilton
            before running pipelines with this hook.
        """
        self.project_id = project_id
        self.api_key = api_key
        self.username = username
        self.run_id = run_id
        # self.root_type = root_type
        self.client = BasicSynchronousHamiltonClient(
            api_key, username, hamilton_api_url, verify=verify
        )
        self.initialized = False
        self.project_version = None
        self.base_tags = tags
        validate_tags(self.base_tags)
        self.dag_name = dag_name
        self.hamilton_ui_url = hamilton_ui_url
        logger.debug("Validating authentication against Hamilton BE API...")
        self.client.validate_auth()
        logger.debug(f"Ensuring project {self.project_id} exists...")
        try:
            self.client.project_exists(self.project_id)
        except UnauthorizedException:
            logger.exception(
                f"Authentication failed. Please check your username and try again. "
                f"Username: {self.username}..."
            )
            raise
        except ResourceDoesNotExistException:
            logger.error(
                f"Project {self.project_id} does not exist/is accessible. Please create it first in the UI! "
                f"You can do so at {self.hamilton_ui_url}/dashboard/projects"
            )
            raise
        self.dag_template_id_cache: dict[int, str] = {}
        self.tracking_states: dict[str, TrackingState] = {}
        self.dw_run_ids: dict[str, str] = {}
        self.task_runs: dict[str, TaskRun] = {}
        self.dependencies: dict[str, set[str]] = {}
        super().__init__()
        # set this to a float to sample blocks. 0.1 means 10% of blocks will be sampled.
        # set this to an int to sample blocks by modulo.
        self.special_parallel_sample_strategy = None
        # set this to some constant value if you want to generate the same sample each time.
        # if you're using a float value.
        self.seed: float | None = None

    def stop(self):
        """Stop the Hamilton client connection.

        Initiates shutdown of the Hamilton client if running in a
        remote environment. Used for cleanup when the hook is no longer needed.
        """
        self.client.stop()

    def post_graph_construct(
        self,
        transforms: list[AbstractExecuteTransform],
        inputs: dict[str, Type],
    ):
        """Register the pipeline DAG template with Hamilton.

        Args:
            transforms: List of transforms in the pipeline.
            inputs: External inputs to the pipeline with their types.

        Note:
            Creates a DAG template in Hamilton with all transform information,
            code artifacts, and dependency relationships. This template is
            reused for multiple pipeline runs.
        """
        if self.seed is None:
            self.seed = random.random()
        logger.debug("post_graph_construct")
        self.dependencies = compute_dependencies(transforms, inputs)
        fg_id = id(transforms)
        if fg_id in self.dag_template_id_cache:
            logger.warning("Skipping creation of DAG template as it already exists.")
            return
        module_hash = str(random.getrandbits(128))  # driver._get_modules_hash(modules)
        vcs_info = _derive_version_control_info(module_hash)
        dag_hash = str(random.getrandbits(128))  # driver.hash_dag(graph)
        code_hash = str(
            random.getrandbits(128)
        )  # driver.hash_dag_modules(graph, modules)

        nodes = _extract_node_templates_from_function_graph(
            self.root_type, transforms, self.dependencies, inputs
        )

        code_artifacts = extract_code_artifacts_from_function_graph(
            transforms, vcs_info, vcs_info.local_repo_base_path
        )
        input_artifacts = extract_code_artifacts_from_inputs(
            inputs, vcs_info, vcs_info.local_repo_base_path
        )
        # raise ValueError("BREAK")

        dag_template_id = self.client.register_dag_template_if_not_exists(
            project_id=self.project_id,
            dag_hash=dag_hash,
            code_hash=code_hash,
            name=self.dag_name,
            nodes=nodes,
            code_artifacts=code_artifacts + input_artifacts,
            config={},  # graph.config,
            tags=self.base_tags,
            code=_slurp_code(transforms, vcs_info.local_repo_base_path),
            vcs_info=vcs_info,
        )
        self.dag_template_id_cache[fg_id] = dag_template_id

    def pre_pipe_execute(
        self,
        pipe: TransformPipe,
    ):
        """Start a new DAG run in Hamilton UI.

        Args:
            pipe: The transform pipeline about to be executed.

        Note:
            Creates a new run instance in Hamilton UI and initializes
            tracking state. Logs the Hamilton UI URL where results
            can be viewed in real-time.
        """
        logger.debug("pre_graph_execute %s", self.run_id)
        transforms = pipe.get_transforms()
        inputs = {
            i.to_path(): LensInfo(self.root_type, i).get_type()
            for i in pipe.get_inputs()
        }
        self.post_graph_construct(transforms, inputs)
        final_vars: list[str] = []

        fg_id = id(transforms)
        if fg_id in self.dag_template_id_cache:
            dag_template_id = self.dag_template_id_cache[fg_id]
        else:
            raise ValueError(
                "DAG template ID not found in cache. This should never happen."
            )
        tracking_state = TrackingState(self.run_id)
        self.tracking_states[self.run_id] = tracking_state  # cache
        tracking_state.clock_start()
        dw_run_id = self.client.create_and_start_dag_run(
            dag_template_id=dag_template_id,
            tags=self.base_tags,
            inputs=inputs if inputs is not None else {},
            outputs=final_vars,
        )
        self.dw_run_ids[self.run_id] = dw_run_id
        self.task_runs[self.run_id] = {}
        logger.warning(
            f"\nCapturing execution run. Results can be found at "
            f"{self.hamilton_ui_url}/dashboard/project/{self.project_id}/runs/{dw_run_id}\n"
        )
        # return dw_run_id

    def pre_node_execute(
        self,
        transform: AbstractExecuteTransform,
        # kwargs: dict[str, Any],
        # task_id: Optional[str] = None,
    ):
        """Record the start of transform execution.

        Args:
            transform: The transform about to be executed.

        Note:
            Creates a task run record in Hamilton with start time,
            dependencies, and sampling information. Updates the
            Hamilton UI with execution status.
        """
        task_id = None
        logger.debug("pre_node_execute %s %s", self.run_id, task_id)
        tracking_state = self.tracking_states[self.run_id]
        if tracking_state.status == Status.UNINITIALIZED:  # not thread safe?
            tracking_state.update_status(Status.RUNNING)

        name = transform.get_name()
        in_sample = self.is_in_sample(task_id)
        task_run = TaskRun(node_name=name, is_in_sample=in_sample)
        task_run.status = Status.RUNNING
        task_run.start_time = datetime.datetime.now(timezone.utc)
        tracking_state.update_task(name, task_run)
        self.task_runs[self.run_id][name] = task_run

        task_update = dict(
            node_template_name=name,
            node_name=get_node_name(name, task_id),
            realized_dependencies=list(self.dependencies[name]),
            status=task_run.status,
            start_time=task_run.start_time,
            end_time=None,
        )
        # we need a 1-1 mapping of updates for the sample stuff to work.
        self.client.update_tasks(
            self.dw_run_ids[self.run_id],
            attributes=[None],
            task_updates=[task_update],
            in_samples=[task_run.is_in_sample],
        )

    def get_hash(self, block_value: int):
        """Generate a deterministic hash for sampling decisions.

        Args:
            block_value: Integer value to generate hash for, typically a block ID.

        Returns:
            Integer hash value derived from seed, salt, and block value.
            Used for deterministic sampling in parallel execution contexts.

        Note:
            Uses SHA-1 hashing with a salted seed to ensure consistent
            sampling decisions across pipeline runs. The hash is truncated
            to 15 hex digits for conversion to integer.
        """
        full_salt = "%s.%s%s" % (self.seed, "POND", ".")
        hash_str = "%s%s" % (full_salt, str(block_value))
        hash_str_b = hash_str.encode("ascii")
        return int(hashlib.sha1(hash_str_b).hexdigest()[:15], 16)

    def get_deterministic_random(self, block_value: int):
        """Generate a deterministic random number for sampling.

        Args:
            block_value: Integer value to generate random number for.

        Returns:
            Float value between 0 and 1, deterministically derived
            from the block value using the hash function.

        Note:
            Normalizes the hash value by dividing by LONG_SCALE to
            produce a consistent floating-point value in [0, 1) range.
            Used for probabilistic sampling strategies.
        """
        zero_to_one = self.get_hash(block_value) / LONG_SCALE
        return zero_to_one  # should be between 0 and 1

    def is_in_sample(self, task_id: Optional[str]) -> bool:
        """Determine whether a task should be included in sampling.

        Args:
            task_id: Optional task identifier, typically for parallel execution.
                Expected format for sampled tasks: "expand-{info}.block.{id}".

        Returns:
            True if the task should be sampled (tracked in detail),
            False if it should be skipped for performance reasons.

        Note:
            Sampling strategies:
            - Float strategy: Sample blocks probabilistically based on random value
            - Int strategy: Sample every Nth block using modulo operation
            - All non-parallel tasks are always included in sample

            Used to reduce tracking overhead for large parallel workloads
            while maintaining representative monitoring coverage.
        """
        if (
            self.special_parallel_sample_strategy is not None
            and task_id is not None
            and task_id.startswith("expand-")
            and "block" in task_id
        ):
            in_sample = False
            block_id = int(task_id.split(".")[1])
            if isinstance(self.special_parallel_sample_strategy, float):
                # if it's a float we want to sample blocks
                if (
                    self.get_deterministic_random(block_id)
                    < self.special_parallel_sample_strategy
                ):
                    in_sample = True
            elif isinstance(self.special_parallel_sample_strategy, int):
                # if it's an int we want to take the modulo of the block id so all the
                # nodes for a block will be captured or not.
                if block_id % self.special_parallel_sample_strategy == 0:
                    in_sample = True
            else:
                raise ValueError(
                    f"Unknown special_parallel_sample_strategy: "
                    f"{self.special_parallel_sample_strategy}"
                )
        else:
            in_sample = True
        return in_sample

    def post_node_execute(
        self,
        transform: AbstractExecuteTransform,
        # kwargs: dict[str, Any],
        success: bool,
        error: Optional[Exception],
        # result_type: Optional[Type],
        # task_id: Optional[str] = None,
    ):
        """Record the completion of transform execution in Hamilton UI.

        Args:
            transform: The transform that just finished executing.
            success: Whether the transform executed successfully.
            error: Exception that occurred during execution, if any.

        Note:
            Updates Hamilton UI with:
            - Task completion time and status (SUCCESS/FAILURE)
            - Result summary or error traceback
            - Additional observability data if available
            - Forces sampling for failed tasks to ensure error visibility

            The task update includes all attributes in the correct order
            for optimal Hamilton UI display.
        """
        task_id = None
        logger.debug("post_node_execute %s %s", self.run_id, task_id)
        name = transform.get_name()
        task_run: TaskRun = self.task_runs[self.run_id][name]
        tracking_state = self.tracking_states[self.run_id]
        task_run.end_time = datetime.datetime.now(timezone.utc)

        other_results = []
        if success:
            task_run.status = Status.SUCCESS
            task_run.result_type = transform.get_output_type(self.root_type)
            result_summary, schema, additional_attributes = process_result(transform)
            if result_summary is None:
                result_summary = {
                    "observability_type": "observability_failure",
                    "observability_schema_version": "0.0.3",
                    "observability_value": {
                        "type": str(str),
                        "value": "Failed to process result.",
                    },
                }
            other_results = (
                [schema] if schema is not None else []
            ) + additional_attributes

            task_run.result_summary = result_summary
            task_attr = dict(
                node_name=get_node_name(name, task_id),
                name="result_summary",
                type=task_run.result_summary["observability_type"],
                # 0.0.3 -> 3
                schema_version=int(
                    task_run.result_summary["observability_schema_version"].split(".")[
                        -1
                    ]
                ),
                value=task_run.result_summary["observability_value"],
                attribute_role="result_summary",
            )

        else:
            assert error is not None
            task_run.status = Status.FAILURE
            task_run.is_in_sample = True  # override any sampling
            # if isinstance(error, dq_base.DataValidationError):
            #     task_run.error = runs.serialize_data_quality_error(error)
            # else:
            task_run.error = traceback.format_exception(
                type(error), error, error.__traceback__
            )
            task_attr = dict(
                node_name=get_node_name(name, task_id),
                name="stack_trace",
                type="error",
                schema_version=1,
                value={
                    "stack_trace": task_run.error,
                },
                attribute_role="error",
            )

        # `result_summary` or "error" is first because the order influences UI display order
        attributes = [task_attr]
        for i, other_result in enumerate(other_results):
            other_attr = dict(
                node_name=get_node_name(name, task_id),
                name=other_result.get(
                    "name", f"Attribute {i + 1}"
                ),  # retrieve name if specified
                type=other_result["observability_type"],
                # 0.0.3 -> 3
                schema_version=int(
                    other_result["observability_schema_version"].split(".")[-1]
                ),
                value=other_result["observability_value"],
                attribute_role="result_summary",
            )
            attributes.append(other_attr)
        tracking_state.update_task(name, task_run)
        task_update = dict(
            node_template_name=name,
            node_name=get_node_name(name, task_id),
            realized_dependencies=list(self.dependencies[name]),
            status=task_run.status,
            start_time=task_run.start_time,
            end_time=task_run.end_time,
        )
        self.client.update_tasks(
            self.dw_run_ids[self.run_id],
            attributes=attributes,
            task_updates=[task_update for _ in attributes],
            in_samples=[task_run.is_in_sample for _ in attributes],
        )

    def post_pipe_execute(
        self,
        pipe: TransformPipe,
        success: bool,
        error: Optional[Exception],
    ):
        """Record the completion of pipeline execution in Hamilton UI.

        Args:
            pipe: The transform pipeline that finished executing.
            success: Whether the entire pipeline executed successfully.
            error: Exception that occurred during pipeline execution, if any.

        Note:
            Finalizes the Hamilton UI run record by:
            - Setting final run status (SUCCESS/FAILURE)
            - Updating end times for all tasks
            - Handling aborted or incomplete tasks
            - Logging the Hamilton UI URL for viewing results

            Ensures all task states are properly finalized even if
            the pipeline was interrupted or failed partway through.
        """
        logger.debug("post_graph_execute %s", self.run_id)
        dw_run_id = self.dw_run_ids[self.run_id]
        tracking_state = self.tracking_states[self.run_id]
        tracking_state.clock_end(status=Status.SUCCESS if success else Status.FAILURE)
        finally_block_time = datetime.datetime.utcnow()
        if tracking_state.status != Status.SUCCESS:
            # TODO: figure out how to handle crtl+c stuff
            # -- we are at the mercy of Hamilton here.
            tracking_state.status = Status.FAILURE
            # this assumes the task map only has things that have been processed, not
            # nodes that have yet to be computed.
            for task_name, task_run in tracking_state.task_map.items():
                if task_run.status != Status.SUCCESS:
                    task_run.status = Status.FAILURE
                    task_run.end_time = finally_block_time
                    if task_run.error is None:  # we likely aborted it.
                        # Note if we start to do concurrent execution we'll likely
                        # need to adjust this.
                        task_run.error = ["Run was likely aborted."]
                if task_run.end_time is None and task_run.status == Status.SUCCESS:
                    task_run.end_time = finally_block_time

        self.client.log_dag_run_end(
            dag_run_id=dw_run_id,
            status=tracking_state.status.value,
        )
        logger.warning(
            f"\nCaptured execution run. Results can be found at "
            f"{self.hamilton_ui_url}/dashboard/project/{self.project_id}/runs/{dw_run_id}\n"
        )
