from abc import ABC
from typing import Optional, Type

from pydantic import BaseModel

from pond.transforms.abstract_transform import AbstractExecuteTransform
from pond.transforms.transform_pipe import TransformPipe


class AbstractHook(ABC):
    """Abstract base class for pipeline execution hooks.

    Hooks provide extensibility points during pipeline execution, allowing
    custom functionality such as monitoring, visualization, logging, debugging,
    profiling, or other pipeline behaviors. Hooks are called at specific
    stages of pipeline and transform execution.

    Hook Lifecycle:
    1. initialize() - Called once before pipeline execution
    2. pre_pipe_execute() - Called before pipeline starts
    3. For each transform:
       - pre_node_execute() - Called before transform execution
       - post_node_execute() - Called after transform completion/failure
    4. post_pipe_execute() - Called after pipeline completes/fails

    Attributes:
        root_type: The pydantic model class defining the data schema,
            set during initialization.

    Note:
        All hook methods are optional - implement only the ones needed
        for your specific hook functionality.
    """

    def __init__(self):
        """Initialize a new hook instance."""
        self.root_type = None

    def initialize(self, root_type: Type[BaseModel]):
        """Initialize the hook with the pipeline's root data type.

        Args:
            root_type: The pydantic model class that defines the pipeline's
                data structure schema.

        Note:
            Called once before pipeline execution begins. Use this to
            set up any resources or state needed for the hook.
        """
        self.root_type = root_type

    def pre_pipe_execute(self, pipe: TransformPipe):
        """Called before pipeline execution begins.

        Args:
            pipe: The transform pipeline that will be executed.

        Note:
            Override this method to perform setup operations before
            the pipeline starts executing transforms.
        """
        pass

    def post_pipe_execute(
        self, pipe: TransformPipe, success: bool, error: Optional[Exception]
    ):
        """Called after pipeline execution completes.

        Args:
            pipe: The transform pipeline that was executed.
            success: True if pipeline completed successfully, False if failed.
            error: Exception that caused failure, or None if successful.

        Note:
            Override this method to perform cleanup operations or handle
            pipeline completion/failure. Called even when pipeline fails.
        """
        pass

    def pre_node_execute(self, node: AbstractExecuteTransform):
        """Called before each transform execution.

        Args:
            node: The transform that is about to be executed.

        Note:
            Override this method to perform operations before each
            transform runs, such as logging or progress tracking.
        """
        pass

    def post_node_execute(
        self, node: AbstractExecuteTransform, success: bool, error: Optional[Exception]
    ):
        """Called after each transform execution completes.

        Args:
            node: The transform that was executed.
            success: True if transform completed successfully, False if failed.
            error: Exception that caused transform failure, or None if successful.

        Note:
            Override this method to handle transform completion/failure.
            Called for every transform, even when they fail.
        """
        pass

    def is_cancellation_requested(self) -> bool:
        """Check if pipeline cancellation has been requested.

        Returns:
            False by default. Override in hooks that support cancellation.

        Note:
            This method provides a default implementation that allows runners
            to check cancellation on any hook without needing to verify if
            the method exists. Hooks that support cancellation should override
            this method to return their actual cancellation state.
        """
        return False
