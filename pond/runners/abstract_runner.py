from abc import ABC

from pond.hooks.abstract_hook import AbstractHook
from pond.state import State
from pond.transforms.transform_pipe import TransformPipe


class AbstractRunner(ABC):
    """Abstract base class for pipeline execution runners.

    Runners control how pipeline transforms are executed, including execution
    order, parallelization, and hook integration. Different runners can provide
    sequential execution, parallel execution, or other execution strategies.

    The runner is responsible for:
    - Orchestrating transform execution
    - Managing execution state and dependencies
    - Integrating with hooks for extensibility
    - Handling errors and cleanup

    Note:
        Concrete implementations must define the execution strategy while
        maintaining hook integration and state management.
    """

    def run(self, state: State, pipe: TransformPipe, hooks: list[AbstractHook]):
        """Execute the pipeline with the given state and hooks.

        Args:
            state: Pipeline state containing catalog and configuration.
            pipe: The transform pipeline to execute.
            hooks: List of hooks for extensibility. Hooks can provide various
                functionality such as monitoring, visualization, logging,
                debugging, profiling, or custom pipeline behaviors.

        Note:
            This method must be implemented by concrete runner classes.
            The execution strategy (sequential, parallel, etc.) is determined
            by the specific runner implementation.
        """
        pass
