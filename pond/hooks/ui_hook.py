import os

# from types import ModuleType
from typing import Union, Optional, Type
import datetime
from datetime import timezone
import random
import inspect
import traceback

import hashlib
from loguru import logger
from pydantic import BaseModel

from hamilton_sdk.api.clients import (
    BasicSynchronousHamiltonClient,
    UnauthorizedException,
    ResourceDoesNotExistException,
)
from hamilton_sdk.api.projecttypes import GitInfo
from hamilton_sdk.tracking.runs import Status, TrackingState
from hamilton_sdk.tracking.trackingtypes import TaskRun
from hamilton_sdk.tracking.data_observation import ObservationType

from hamilton_sdk.driver import (
    validate_tags,
    _get_fully_qualified_function_path,
    _derive_url,
    _derive_version_control_info,
)

from pond.lens import LensPath, LensInfo, get_cleaned_path
from pond.transforms.abstract_transform import AbstractExecuteTransform
from pond.transforms.transform_pipe import TransformPipe
from pond.transforms.transform_index import TransformIndex
from pond.hooks.abstract_hook import AbstractHook

LONG_SCALE = float(0xFFFFFFFFFFFFFFF)


def get_node_name(name: str, task_id: Optional[str]) -> str:
    return name if task_id is None else f"{task_id}-{name}"


def _convert_classifications(transform: AbstractExecuteTransform) -> list[str]:
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
    """Converts a function graph to a list of nodes that the DAGWorks graph can understand.

    @param fn: Function graph to convert
    @return: A list of node objects
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
    """Converts a function graph to a list of code artifacts that the function graph uses.

    @param fn_graph: Function graph to convert.
    @return: A list of node objects.
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
    """Converts a function graph to a list of code artifacts that the function graph uses.

    @param fn_graph: Function graph to convert.
    @return: A list of node objects.
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
    modules = set()
    for transform in transforms:
        module = inspect.getmodule(transform.get_fn())
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
    for transform in transforms:
        outputs = transform.get_outputs()
        # print(f"Transform {transform.get_name()} outputs: {outputs}")
        for o in outputs:
            if path.subset_of(o):
                dependencies[transform_name].add(transform.get_name())
                break


def compute_dependencies(
    transforms: list[AbstractExecuteTransform],
    inputs: dict[str, Type],
) -> dict[str, set[str]]:
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
    schema = None
    additional = []
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
        tags: dict[str, str] = None,
        api_key: str = None,
        hamilton_api_url="http://localhost:8241",
        hamilton_ui_url="http://localhost:8241",
        verify: Union[str, bool] = True,
        run_id: str = "dev",
    ):
        """This hooks into Hamilton execution to track DAG runs in Hamilton UI.

        :param project_id: the ID of the project
        :param username: the username for the API key.
        :param dag_name: the name of the DAG.
        :param tags: any tags to help curate and organize the DAG
        :param client_factory: a factory to create the client to phone Hamilton with.
        :param api_key: the API key to use. See us if you want to use this.
        :param hamilton_api_url: API endpoint.
        :param hamilton_ui_url: UI Endpoint.
        :param verify: SSL verification to pass-through to requests
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
        self.base_tags = tags if tags is not None else {}
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
        self.dag_template_id_cache = {}
        self.tracking_states = {}
        self.dw_run_ids = {}
        self.task_runs = {}
        self.dependencies: dict[str, set[str]] = {}
        super().__init__()
        # set this to a float to sample blocks. 0.1 means 10% of blocks will be sampled.
        # set this to an int to sample blocks by modulo.
        self.special_parallel_sample_strategy = None
        # set this to some constant value if you want to generate the same sample each time.
        # if you're using a float value.
        self.seed = None

    def stop(self):
        """Initiates stop if run in remote environment"""
        self.client.stop()

    def post_graph_construct(
        self,
        transforms: list[AbstractExecuteTransform],
        inputs: dict[str, Type],
    ):
        """Registers the DAG to get an ID."""
        if self.seed is None:
            self.seed = random.random()
        logger.debug("post_graph_construct")
        self.dependencies = compute_dependencies(transforms, inputs)
        print(self.dependencies)
        fg_id = id(transforms)
        if fg_id in self.dag_template_id_cache:
            logger.warning("Skipping creation of DAG template as it already exists.")
            return
        module_hash = str(random.getrandbits(128))  # driver._get_modules_hash(modules)
        print("Module hash: ", module_hash)
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
        """Creates a DAG run."""
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
        """Captures start of node execution."""
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
        """Creates a deterministic hash."""
        full_salt = "%s.%s%s" % (self.seed, "POND", ".")
        hash_str = "%s%s" % (full_salt, str(block_value))
        hash_str = hash_str.encode("ascii")
        return int(hashlib.sha1(hash_str).hexdigest()[:15], 16)

    def get_deterministic_random(self, block_value: int):
        """Gets a random number between 0 & 1 given the block value."""
        zero_to_one = self.get_hash(block_value) / LONG_SCALE
        return zero_to_one  # should be between 0 and 1

    def is_in_sample(self, task_id: str) -> bool:
        """Determines if what we're tracking is considered in sample.

        This should only be used at the node level right now and is intended
        for parallel blocks that could be quick large.
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
        """Captures end of node execution."""
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
                    "name", f"Attribute {i+1}"
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
        """Captures end of DAG execution."""
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
