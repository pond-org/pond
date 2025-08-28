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
    - FastAPI transforms are executed on the main thread to avoid serialization issues
    - Uses multiprocessing with spawn context for process isolation
    - Employs locks for thread-safe catalog commits

    Features:
    - Automatic dependency resolution
    - Parallel execution of independent transforms
    - Full hook integration with proper timing
    - Error handling with cleanup

    Note:
        Uses spawn context for better cross-platform compatibility.
        Worker count is configurable via max_workers parameter.
    """

    def __init__(self, max_workers: int = 10):
        """Initialize a parallel runner.

        Args:
            max_workers: Maximum number of worker processes to use. Defaults to 10.
                        Set to 0 to use sequential execution (useful for debugging
                        or testing environments where multiprocessing conflicts occur).
        """
        super().__init__()
        self.max_workers = max_workers

    def _is_fastapi_transform(self, transform) -> bool:
        """Conservative check for FastAPI transforms that can't be pickled."""
        return "FastAPI" in transform.__class__.__name__

    def _execute_on_main_thread(self, transform, state, hooks, lock):
        """Execute FastAPI transform on main thread with proper error handling."""
        error = None
        success = True

        try:
            for hook in hooks:
                hook.pre_node_execute(transform)

            execute_units = transform.get_execute_units(state)

            for unit in execute_units:
                args = unit.load_inputs(state)
                rtns = unit.run(args)
                values = unit.save_outputs(state, rtns)

                with lock:
                    unit.commit(state, values)

        except Exception as e:
            error = e
            success = False
            for hook in hooks:
                hook.post_node_execute(transform, success, error)
            raise RuntimeError(f"Failed at transform {transform.get_name()}") from error

        # Mark transform as completed
        for hook in hooks:
            hook.post_node_execute(transform, success, error)

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
            When max_workers=0, falls back to sequential execution to avoid
            multiprocessing overhead and potential testing conflicts.
        """
        # Special case: if max_workers=0, use sequential execution
        if self.max_workers == 0:
            from pond.runners.sequential_runner import SequentialRunner

            sequential_runner = SequentialRunner()
            return sequential_runner.run(state, pipe, hooks)

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
                max_workers=self.max_workers,
                # mp_context=ForkContext(),
                mp_context=SpawnContext(),
            ) as pool:
                while True:
                    # Check for cancellation before scheduling new transforms
                    if any(hook.is_cancellation_requested() for hook in hooks):
                        # Cancel all pending futures
                        for future in futures:
                            future.cancel()
                        raise InterruptedError("Pipeline execution was canceled")

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

                    # Split ready transforms into parallel and FastAPI transforms
                    parallel_ready = []
                    fastapi_ready = []

                    for t in ready:
                        transform = transforms[t]
                        is_fastapi = self._is_fastapi_transform(transform)
                        if is_fastapi:
                            fastapi_ready.append(t)
                        else:
                            parallel_ready.append(t)

                    # Submit ALL parallel transforms first
                    for t in parallel_ready:
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

                    # Then execute FastAPI transforms on main thread
                    for t in fastapi_ready:
                        transform = transforms[t]
                        self._execute_on_main_thread(transform, state, hooks, lock)
                        # Immediately add outputs to produced since execution is complete
                        for o in transform.get_outputs():
                            produced.append(o)

                    # Check if we're done - no more todos and no pending futures
                    if not todo and not futures:
                        logger.info("All transforms completed, stopping...")
                        break

                    # If we just executed FastAPI transforms, continue to check for newly ready transforms
                    if fastapi_ready and not futures:
                        continue

                    if not futures and todo:
                        # This shouldn't happen - we have remaining work but no futures
                        # This could indicate a dependency issue
                        logger.warning(
                            f"No futures but {len(todo)} transforms remaining"
                        )
                        break

                    # Only wait for futures if we have any
                    if futures:
                        done, futures = wait(
                            futures, return_when=FIRST_COMPLETED, timeout=0.1
                        )
                    else:
                        # No futures to wait for, continue to check for more ready transforms
                        done = set()
                        continue

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
