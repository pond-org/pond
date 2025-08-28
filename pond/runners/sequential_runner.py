from pond.hooks.abstract_hook import AbstractHook
from pond.runners.abstract_runner import AbstractRunner
from pond.state import State
from pond.transforms.transform_pipe import TransformPipe


class SequentialRunner(AbstractRunner):
    """Sequential pipeline execution runner.

    Executes pipeline transforms one at a time in the order they appear
    in the pipeline. Provides deterministic execution with full hook
    integration at each step.

    Execution Flow:
    1. Initialize and call pre_pipe_execute hooks
    2. For each transform:
       - Call pre_node_execute hooks
       - Execute all units for the transform
       - Call post_node_execute hooks
       - Stop on first error
    3. Call post_pipe_execute hooks

    Note:
        Sequential execution ensures transforms complete fully before
        proceeding to the next transform, making debugging easier.
    """

    def __init__(self):
        """Initialize a sequential runner."""
        super().__init__()

    def run(self, state: State, pipe: TransformPipe, hooks: list[AbstractHook]):
        """Execute the pipeline sequentially with full hook integration.

        Args:
            state: Pipeline state containing catalog and configuration.
            pipe: The transform pipeline to execute.
            hooks: List of hooks called at each execution stage.

        Raises:
            RuntimeError: If any transform fails, wrapped with the failing transform name.

        Note:
            Execution stops at the first transform failure. All hooks are called
            even in error cases to ensure proper cleanup and reporting.
        """
        # Initialize hooks and start pipeline execution
        for hook in hooks:
            hook.initialize(state.root_type)
            hook.pre_pipe_execute(pipe)
        error = None
        success = True
        # Execute each transform sequentially
        for transform in pipe.get_transforms():
            # Check for cancellation before each transform
            if any(hook.is_cancellation_requested() for hook in hooks):
                raise InterruptedError("Pipeline execution was canceled")

            for hook in hooks:
                hook.pre_node_execute(transform)
            try:
                # Execute all units for this transform
                for unit in transform.get_execute_units(state):
                    unit.execute_on(state)
            except Exception as e:
                error = e
                success = False
            for hook in hooks:
                hook.post_node_execute(transform, success, error)
            if error is not None:
                raise RuntimeError(
                    f"Failed at transform {transform.get_name()}"
                ) from error
        # Finalize pipeline execution
        for hook in hooks:
            hook.post_pipe_execute(pipe, success, error)
