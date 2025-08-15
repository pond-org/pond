import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing.context import SpawnContext

from loguru import logger

from pond.hooks.abstract_hook import AbstractHook
from pond.runners.abstract_runner import AbstractRunner
from pond.state import State
from pond.transforms.abstract_transform import AbstractExecuteUnit
from pond.transforms.transform_pipe import TransformPipe


def execute_unit(state: State, unit: AbstractExecuteUnit, t: int, lock):
    """Execute a single unit in a separate process.

    This function is called by the process pool to execute individual
    transform units. Uses a lock to ensure thread-safe commits to the catalog.

    Args:
        state: Pipeline state with catalog access.
        unit: The execute unit to run.
        t: Transform index for tracking completion.
        lock: Multiprocessing lock for thread-safe catalog commits.

    Returns:
        Tuple of (unit, transform_index) for completion tracking.

    Note:
        The lock ensures that catalog commits are atomic across processes.
        Only the commit operation is locked, not the entire execution.
    """
    args = unit.load_inputs(state)
    rtns = unit.run(args)
    values = unit.save_outputs(state, rtns)
    with lock:
        unit.commit(state, values)
    return unit, t


class ParallelRunner(AbstractRunner):
    """Parallel pipeline execution runner with dependency resolution.

    Executes pipeline transforms in parallel while respecting data dependencies.
    Uses a process pool to run independent transforms concurrently and employs
    dependency tracking to ensure correct execution order.

    Execution Strategy:
    - Analyzes transform dependencies based on input/output paths
    - Schedules transforms for parallel execution when dependencies are met
    - Uses multiprocessing with spawn context for process isolation
    - Employs locks for thread-safe catalog commits

    Features:
    - Automatic dependency resolution
    - Parallel execution of independent transforms
    - Full hook integration with proper timing
    - Error handling with cleanup

    Note:
        Uses spawn context for better cross-platform compatibility.
        Maximum worker count is currently hardcoded to 10.
    """

    def __init__(self):
        """Initialize a parallel runner."""
        super().__init__()

    def run(self, state: State, pipe: TransformPipe, hooks: list[AbstractHook]):
        """Execute the pipeline in parallel with dependency resolution.

        Args:
            state: Pipeline state containing catalog and configuration.
            pipe: The transform pipeline to execute.
            hooks: List of hooks called at each execution stage.

        Raises:
            RuntimeError: If any transform fails during parallel execution.

        Note:
            Uses ProcessPoolExecutor with spawn context for process isolation.
            Transforms are scheduled as soon as their dependencies are satisfied.
            Catalog commits are synchronized using multiprocessing locks.
        """
        # Initialize hooks and pipeline state
        for hook in hooks:
            hook.initialize(state.root_type)
            hook.pre_pipe_execute(pipe)
        error = None
        success = True

        # Set up dependency tracking
        transforms = dict(enumerate(pipe.get_transforms()))
        dependencies = {
            t: transform.get_inputs() for t, transform in transforms.items()
        }
        todo = set(transforms.keys())
        inputs = pipe.get_inputs()
        produced = list(inputs)
        futures = set()
        execute_units_finished = {}

        with multiprocessing.Manager() as m:
            lock = m.Lock()
            with ProcessPoolExecutor(
                max_workers=10,
                # mp_context=ForkContext(),
                mp_context=SpawnContext(),
            ) as pool:
                while True:
                    ready = {
                        t
                        for t in todo
                        if all(
                            any(i.subset_of(p) for p in produced)
                            for i in dependencies[t]
                        )
                    }
                    logger.info(f"Found {len(ready)} nodes to execute")
                    todo -= ready
                    for t in ready:
                        transform = transforms[t]
                        for hook in hooks:
                            hook.pre_node_execute(transform)
                        execute_units = transform.get_execute_units(state)
                        execute_units_finished[t] = len(execute_units)
                        for unit in execute_units:
                            futures.add(
                                pool.submit(
                                    execute_unit,
                                    state,
                                    unit,
                                    t,
                                    lock,
                                )
                            )

                    if not futures:
                        logger.info("No futures left in pool, stopping...")
                        break

                    done, futures = wait(
                        futures, return_when=FIRST_COMPLETED, timeout=0.1
                    )

                    for future in done:
                        try:
                            # unit, rtns, t = future.result()
                            unit, t = future.result()
                            # unit.save_outputs(state, rtns)
                        except Exception as e:
                            error = e
                            success = False
                            for hook in hooks:
                                hook.post_node_execute(transform, success, error)
                                hook.post_pipe_execute(pipe, success, error)
                            raise RuntimeError(
                                f"Failed at transform {transform.get_name()}"
                            ) from error

                        execute_units_finished[t] = execute_units_finished[t] - 1
                        if execute_units_finished[t] == 0:
                            transform = transforms[t]
                            for hook in hooks:
                                hook.post_node_execute(transform, success, error)
                            for o in transform.get_outputs():
                                produced.append(o)

        for hook in hooks:
            hook.post_pipe_execute(pipe, success, error)
