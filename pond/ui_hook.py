import os
from types import ModuleType
from typing import Any, Callable, Union, Optional, Type
import datetime
from datetime import timezone
import random
import inspect
import traceback

import hashlib
from loguru import logger
from pydantic import BaseModel

try:
    import git
except ImportError:
    git = None

from hamilton_sdk.api.clients import (
    BasicSynchronousHamiltonClient,
    UnauthorizedException,
    ResourceDoesNotExistException,
)
from hamilton_sdk.tracking.runs import Status, TrackingState
from hamilton_sdk.api.projecttypes import GitInfo
from hamilton_sdk.tracking.trackingtypes import TaskRun
from hamilton_sdk.tracking.data_observation import ObservationType

from pond.lens import LensPath
from pond.abstract_transform import AbstractExecuteTransform


LONG_SCALE = float(0xFFFFFFFFFFFFFFF)


def get_node_name(name: str, task_id: Optional[str]) -> str:
    return name if task_id is None else f"{task_id}-{name}"


def validate_tags(tags: Any):
    """Validates that tags are a dictionary of strings to strings.

    :param tags: Tags to validate
    :raises ValueError: If tags are not a dictionary of strings to strings
    """
    if not isinstance(tags, dict):
        raise ValueError(f"Tags must be a dictionary, but got {tags}")
    for key, value in tags.items():
        if not isinstance(key, str):
            raise ValueError(f"Tag keys must be strings, but got {key}")
        if not isinstance(value, str):
            raise ValueError(f"Tag values must be strings, but got {value}")


# def _generate_unique_temp_module_name() -> str:
#     """Generates a unique module name that is a valid python variable."""
#     return f"temporary_module_{str(uuid.uuid4()).replace('-', '_')}"


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
        dependency_specs.append(
            {"type_name": str(transform_dict[dep].get_output_type(root_type))}
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
    return node_templates


def _get_fully_qualified_function_path(fn: Callable) -> str:
    """Gets the fully qualified path of a function.

    :param fn: Function to get the path of
    :return: Fully qualified path of the function
    """
    module = inspect.getmodule(fn)
    fn_name = fn.__name__
    if module is not None:
        fn_name = f"{module.__name__}.{fn_name}"
    return fn_name


def _derive_url(vcs_info: GitInfo, path: str, line: int) -> str:
    """Derives a URL from a VCS info, a path, and a line number.

    @param vcs_info: VCS info
    @param path: Path
    @param line: Line number
    @return: A URL
    """
    if vcs_info.repository == "Error: No repository to link to.":
        return "Error: No repository to link to."
    if vcs_info.repository.endswith(".git"):
        repo_url = vcs_info.repository[:-4]
    else:
        repo_url = vcs_info.repository
    return f"{repo_url}/blob/{vcs_info.commit_hash}/{path}#L{line}"


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


def _derive_version_control_info(module_hash: str) -> GitInfo:
    """Derive the git info for the current project.
    Currently, this decides whether we're in a git repository.
    This is not going to work for everything, but we'll see what the customers want.
    We might end up having to pass this data in...
    """
    default = GitInfo(
        branch="unknown",
        commit_hash=module_hash,
        committed=False,
        repository="Error: No repository to link to.",
        local_repo_base_path=os.getcwd(),
    )
    if git is None:
        return default
    try:
        repo = git.Repo(".", search_parent_directories=True)
    except git.exc.InvalidGitRepositoryError:
        logger.warning(
            "Warning: We are not currently in a git repository. We recommend using that as a "
            "way to version the "
            "project *if* your hamilton code lives within this repository too. If it does not,"
            " then we'll try to "
            "version code based on the python modules passed to the Driver. "
            "Incase you want to get set up with git quickly you can run:\n "
            "git init && git add . && git commit -m 'Initial commit'\n"
            "Still have questions? Reach out to stefan @ dagworks.io, elijah @ dagworks.io "
            "and we'll try to help you as soon as possible."
        )
        return default
    if "COLAB_RELEASE_TAG" in os.environ:
        logger.warning(
            "We currently do not support logging version information inside a google"
            "colab notebook. This is something we are planning to do. "
            "If you have any questions, please reach out to support@dagworks.io"
            "and we'll try to help you as soon as possible."
        )
        return default

    try:
        commit = repo.head.commit
    except Exception:
        return default
    try:
        repo_url = repo.remote().url
    except Exception:
        # TODO: change this to point to our docs on what to do.
        repo_url = "Error: No repository to link to."
    try:
        branch_name = repo.active_branch.name
    except Exception:
        branch_name = "unknown"  # detached head
        logger.warning(
            "Warning: we are unable to determine the branch name. "
            "This is likely because you are in a detached head state. "
            "If you are in a detached head state, you can check out a "
            "branch by running `git checkout -b <branch_name>`. "
            "If you intend to be (if you are using some sort of CI"
            "system that checks out a detached head) then you can ignore this."
        )
    return GitInfo(
        branch=branch_name,
        commit_hash=commit.hexsha,
        committed=not repo.is_dirty(),
        repository=repo_url,
        local_repo_base_path=repo.working_dir,
    )


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
) -> dict[str, set[str]]:
    dependencies: dict[str, set[str]] = {}
    for transform in transforms:
        name = transform.get_name()
        dependencies[name] = set()
        for i in transform.get_inputs():
            add_dependency(name, i, transforms, dependencies)
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


class UIClient:
    def __init__(
        self,
        project_id: int,
        username: str,
        dag_name: str,
        root_type: Type[BaseModel],
        tags: dict[str, str] = None,
        api_key: str = None,
        hamilton_api_url="http://localhost:8241",
        hamilton_ui_url="http://localhost:8241",
        verify: Union[str, bool] = True,
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
        self.root_type = root_type
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
        # graph: h_graph.FunctionGraph,
        transforms: list[AbstractExecuteTransform],
        # modules: list[ModuleType],
        # config: dict[str, Any],
    ):
        """Registers the DAG to get an ID."""
        if self.seed is None:
            self.seed = random.random()
        logger.debug("post_graph_construct")
        self.dependencies = compute_dependencies(transforms)
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
            self.root_type, transforms, self.dependencies
        )
        print("Nodes: ", nodes)

        code_artifacts = extract_code_artifacts_from_function_graph(
            transforms, vcs_info, vcs_info.local_repo_base_path
        )
        print("CODE: ", code_artifacts)

        dag_template_id = self.client.register_dag_template_if_not_exists(
            project_id=self.project_id,
            dag_hash=dag_hash,
            code_hash=code_hash,
            name=self.dag_name,
            nodes=nodes,
            code_artifacts=code_artifacts,
            config={},  # graph.config,
            tags=self.base_tags,
            code=_slurp_code(transforms, vcs_info.local_repo_base_path),
            vcs_info=vcs_info,
        )
        self.dag_template_id_cache[fg_id] = dag_template_id

    def pre_graph_execute(
        self,
        run_id: str,
        transforms: list[AbstractExecuteTransform],
        final_vars: list[str],
        inputs: dict[str, Any],
    ):
        """Creates a DAG run."""
        logger.debug("pre_graph_execute %s", run_id)
        fg_id = id(transforms)
        if fg_id in self.dag_template_id_cache:
            dag_template_id = self.dag_template_id_cache[fg_id]
        else:
            raise ValueError(
                "DAG template ID not found in cache. This should never happen."
            )
        tracking_state = TrackingState(run_id)
        self.tracking_states[run_id] = tracking_state  # cache
        tracking_state.clock_start()
        dw_run_id = self.client.create_and_start_dag_run(
            dag_template_id=dag_template_id,
            tags=self.base_tags,
            inputs=inputs if inputs is not None else {},
            outputs=final_vars,
        )
        self.dw_run_ids[run_id] = dw_run_id
        self.task_runs[run_id] = {}
        logger.warning(
            f"\nCapturing execution run. Results can be found at "
            f"{self.hamilton_ui_url}/dashboard/project/{self.project_id}/runs/{dw_run_id}\n"
        )
        return dw_run_id

    def pre_node_execute(
        self,
        run_id: str,
        transform: AbstractExecuteTransform,
        # kwargs: dict[str, Any],
        task_id: Optional[str] = None,
    ):
        """Captures start of node execution."""
        logger.debug("pre_node_execute %s %s", run_id, task_id)
        tracking_state = self.tracking_states[run_id]
        if tracking_state.status == Status.UNINITIALIZED:  # not thread safe?
            tracking_state.update_status(Status.RUNNING)

        name = transform.get_name()
        in_sample = self.is_in_sample(task_id)
        task_run = TaskRun(node_name=name, is_in_sample=in_sample)
        task_run.status = Status.RUNNING
        task_run.start_time = datetime.datetime.now(timezone.utc)
        tracking_state.update_task(name, task_run)
        self.task_runs[run_id][name] = task_run

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
            self.dw_run_ids[run_id],
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
        run_id: str,
        transform: AbstractExecuteTransform,
        # kwargs: dict[str, Any],
        success: bool,
        error: Optional[Exception],
        result_type: Optional[Type],
        task_id: Optional[str] = None,
    ):
        """Captures end of node execution."""
        logger.debug("post_node_execute %s %s", run_id, task_id)
        name = transform.get_name()
        task_run: TaskRun = self.task_runs[run_id][name]
        tracking_state = self.tracking_states[run_id]
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
            self.dw_run_ids[run_id],
            attributes=attributes,
            task_updates=[task_update for _ in attributes],
            in_samples=[task_run.is_in_sample for _ in attributes],
        )

    def post_graph_execute(
        self,
        run_id: str,
        transforms: list[AbstractExecuteTransform],
        success: bool,
        error: Optional[Exception],
    ):
        """Captures end of DAG execution."""
        logger.debug("post_graph_execute %s", run_id)
        dw_run_id = self.dw_run_ids[run_id]
        tracking_state = self.tracking_states[run_id]
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
