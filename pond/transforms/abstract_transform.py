from abc import ABC, abstractmethod
from typing import Any, Callable, Type

import dill  # type: ignore
from pydantic import BaseModel

from pond.lens import LensInfo, LensPath
from pond.state import State

# AbstractExecuteTransform = NewType("AbstractExecuteTransform", None)


class AbstractTransform(ABC):
    """Abstract base class for all transform operations in pond pipelines.

    Defines the interface for transform operations that can be composed into
    data processing pipelines. Transforms specify input and output paths in
    the data catalog and can be executed on pipeline state.

    The transform hierarchy:
    - AbstractTransform: Base interface for all transforms
    - AbstractExecuteTransform: Transforms that can be directly executed
    - Concrete implementations: Transform, TransformList, TransformListFold, etc.

    Note:
        All transforms must specify their input/output paths and provide
        executable units that can be run by pipeline runners.
    """

    def get_input_types(self, root_type: Type[BaseModel]) -> list[Type]:
        """Get the Python types for all input paths.

        Resolves input path specifications to their corresponding Python types
        based on the root data model schema.

        Args:
            root_type: The pydantic model class defining the data schema.

        Returns:
            List of Python types corresponding to each input path.

        Note:
            Uses LensInfo to resolve path strings to actual types from the schema.
        """
        return [LensInfo(root_type, p).get_type() for p in self.get_inputs()]

    def get_output_type(self, root_type: Type[BaseModel]) -> Type | None:
        """Get the Python type for the transform output.

        Resolves output path specifications to their corresponding Python types.
        For multiple outputs, returns a tuple type.

        Args:
            root_type: The pydantic model class defining the data schema.

        Returns:
            Single type for one output, tuple type for multiple outputs,
            or None for transforms with no outputs.

        Note:
            Multiple outputs are represented as a tuple of their individual types.
        """
        outputs = [LensInfo(root_type, p).get_type() for p in self.get_outputs()]
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple[*outputs]  # type: ignore

    @abstractmethod
    def get_inputs(self) -> list[LensPath]:
        """Get the input paths for this transform.

        Returns:
            List of LensPath objects specifying where to read input data.

        Note:
            Input paths must exist in the catalog when the transform executes.
        """
        pass

    @abstractmethod
    def get_outputs(self) -> list[LensPath]:
        """Get the output paths for this transform.

        Returns:
            List of LensPath objects specifying where to write output data.

        Note:
            Output paths will be created in the catalog when the transform executes.
        """
        pass

    @abstractmethod
    def get_transforms(self) -> list["AbstractExecuteTransform"]:
        """Get the executable transform units for this transform.

        Returns:
            List of AbstractExecuteTransform instances that implement
            the actual computation logic.

        Note:
            For simple transforms, this returns [self]. For composite transforms
            like pipelines, this returns all constituent transforms.
        """
        pass

    def call(self, state: State) -> Any:
        """Execute the transform on the given state.

        Runs all executable units associated with this transform and
        returns the computed results.

        Args:
            state: The pipeline state containing catalog and data access.

        Returns:
            The result of executing the transform. Single values for
            transforms with one output, lists for multiple outputs.

        Note:
            This method coordinates execution across multiple execute units
            but doesn't handle state persistence - that's done by the units.
        """
        units = [
            unit
            for transform in self.get_transforms()
            for unit in transform.get_execute_units(state)
        ]
        rtns = [unit.run(unit.load_inputs(state)) for unit in units]
        # TODO: this is not entirely correct
        # and should probably depend on if the
        # transform is expanded or not
        if len(rtns) == 1:
            return rtns[0]
        else:
            return rtns


class AbstractExecuteUnit(ABC):
    """Abstract base class for executable units within transforms.

    An execute unit represents the smallest executable component in a pipeline.
    It handles loading inputs, running computation, and saving outputs.
    Execute units are created by transforms and executed by pipeline runners.

    Attributes:
        inputs: List of LensPath objects specifying input data locations.
        outputs: List of LensPath objects specifying output data locations.

    Note:
        Execute units handle the low-level details of data loading/saving
        while transforms handle higher-level orchestration.
    """

    def __init__(self, inputs: list[LensPath], outputs: list[LensPath]):
        """Initialize an execute unit with input/output paths.

        Args:
            inputs: List of LensPath objects for input data locations.
            outputs: List of LensPath objects for output data locations.
        """
        self.inputs = inputs
        self.outputs = outputs

    def get_inputs(self) -> list[LensPath]:
        """Get the input paths for this execute unit.

        Returns:
            List of LensPath objects specifying input data locations.
        """
        return self.inputs

    def get_outputs(self) -> list[LensPath]:
        """Get the output paths for this execute unit.

        Returns:
            List of LensPath objects specifying output data locations.
        """
        return self.outputs

    @abstractmethod
    def load_inputs(self, state: State) -> list[Any]:
        """Load input data from the state catalog.

        Args:
            state: Pipeline state with catalog access.

        Returns:
            List of loaded input values corresponding to input paths.

        Note:
            Must handle array wildcards and missing data appropriately.
        """
        pass

    @abstractmethod
    def save_outputs(self, state: State, outputs: list[Any]) -> list[Any]:
        """Prepare output data for storage in the catalog.

        Args:
            state: Pipeline state with catalog access.
            outputs: List of computed output values to prepare for storage.

        Returns:
            List of prepared values ready for catalog storage.

        Note:
            Converts Python objects to catalog-compatible format (Arrow tables).
        """
        pass

    @abstractmethod
    def commit(self, state: State, values: list[Any]) -> bool:
        """Commit prepared values to the catalog.

        Args:
            state: Pipeline state with catalog access.
            values: List of prepared values from save_outputs.

        Returns:
            True if commit was successful.

        Note:
            This is the final step that persists data to storage.
        """
        pass

    @abstractmethod
    def run(self, args: list[Any]) -> list[Any]:
        """Execute the core computation logic.

        Args:
            args: List of input arguments loaded by load_inputs.

        Returns:
            List of computed output values.

        Note:
            This method contains the actual computation logic and should
            be pure (no side effects on state).
        """
        pass

    def execute_on(self, state: State) -> None:
        """Execute this unit on the given state.

        Orchestrates the complete execution cycle: load inputs, run computation,
        save outputs, and commit to storage.

        Args:
            state: Pipeline state with catalog access.

        Note:
            This is the main entry point for executing a unit and handles
            the complete data flow from input loading to output persistence.
        """
        args = self.load_inputs(state)
        rtns = self.run(args)
        values = self.save_outputs(state, rtns)
        self.commit(state, values)


class ExecuteTransform(AbstractExecuteUnit):
    """Concrete implementation of an executable transform unit.

    Wraps a user function and handles data loading, execution, and storage.
    Supports both scalar and array processing patterns with automatic
    wildcard expansion for array inputs.

    Attributes:
        fn: The user function to execute.
        append_outputs: List of output paths that should append rather than overwrite.

    Note:
        This is the primary execute unit used by Transform, TransformList,
        and TransformListFold implementations.
    """

    def __init__(
        self,
        inputs: list[LensPath],
        outputs: list[LensPath],
        fn: Callable,
        append_outputs: list[LensPath] = [],
        # input_list_len: int = -1,
    ):
        """Initialize an ExecuteTransform.

        Args:
            inputs: List of input paths for data loading.
            outputs: List of output paths for data storage.
            fn: The callable to execute with loaded inputs.
            append_outputs: Output paths that should append to existing data
                rather than overwrite. Defaults to empty list.

        Note:
            Default mutable argument [] is used for backward compatibility.
            The function must have type annotations matching the path types.
        """
        super().__init__(inputs, outputs)
        self.fn = fn  # wrapper
        self.append_outputs = append_outputs
        # self.input_list_len = input_list_len

    def __getstate__(self):
        """Prepare instance state for pickling using dill.

        Returns:
            Serialized state containing all necessary attributes.

        Note:
            Uses dill instead of pickle to handle function serialization.
        """
        return dill.dumps((self.inputs, self.outputs, self.fn, self.append_outputs))

    def __setstate__(self, state):
        """Restore instance state after unpickling.

        Args:
            state: Serialized state from __getstate__.

        Note:
            Deserializes using dill to restore function objects.
        """
        self.inputs, self.outputs, self.fn, self.append_outputs = dill.loads(state)

    def load_inputs(self, state: State) -> list[Any]:
        """Load input data from the catalog, handling array wildcards.

        For inputs with wildcard indices (index == -1), attempts to load
        the entire array first. If not available, iterates through indices
        to build the array dynamically.

        Args:
            state: Pipeline state with catalog access.

        Returns:
            List of loaded input values, with arrays expanded as needed.

        Note:
            Uses a maximum iteration limit of 100000 for dynamic array loading.
            Array indices are resolved at runtime based on available data.
        """
        args = []
        for i in self.inputs:
            try:
                index = next(ind for ind, v in enumerate(i.path) if v.index == -1)
                parent = LensPath(i.path[: index + 1])
                parent.path[-1].index = None
                value = state[parent.to_path()]
                if value is not None:
                    args.append(value)
                    continue
                input_list = []
                for list_index in range(0, 100000):
                    i.path[index].index = list_index
                    value = state[i.to_path()]
                    if value is None:
                        break
                    input_list.append(value)
                args.append(input_list)
            except StopIteration:
                args.append(state[i.to_path()])
                continue
            # if self.input_list_len == -1:
            #     raise ValueError("Need to provide list len for execute transform to provide list inputs!")
        return args

    def save_outputs(self, state: State, rtns: list[Any]) -> list[Any]:
        """Convert output values to catalog-compatible Arrow tables.

        Args:
            state: Pipeline state with catalog access.
            rtns: List of computed output values from the function.

        Returns:
            List of Arrow tables ready for catalog storage.

        Note:
            Uses the lens system to convert Python objects to appropriate
            Arrow table representations based on the output path schemas.
        """
        values = []
        for rtn, o in zip(rtns, self.outputs):
            values.append(state.lens(o.to_path()).create_table(rtn))
        return values

    def commit(self, state: State, values: list[Any]) -> bool:
        """Commit Arrow tables to the catalog.

        Args:
            state: Pipeline state with catalog access.
            values: List of Arrow tables from save_outputs.

        Returns:
            True if all commits were successful.

        Note:
            Respects append_outputs list to determine whether to append
            or overwrite existing data at each output path.
        """
        for val, o in zip(values, self.outputs):
            append = o in self.append_outputs
            state.lens(o.to_path()).write_table(val, append)
        return True

    def run(self, args: list[Any]) -> list[Any]:
        """Execute the wrapped function with loaded arguments.

        Args:
            args: List of arguments loaded by load_inputs.

        Returns:
            List of output values, handling both single and tuple returns.

        Note:
            Automatically handles tuple unpacking for multi-output functions
            and normalizes single outputs to list format.
        """
        rtns = self.fn(*args)
        if isinstance(rtns, tuple) and len(self.outputs) > 1:
            rtns_list = list(rtns)
        else:
            rtns_list = [rtns]
        return rtns_list


class AbstractExecuteTransform(AbstractTransform):
    """Abstract base class for directly executable transforms.

    Extends AbstractTransform with metadata and execution capabilities.
    Concrete implementations include Transform, TransformList, TransformListFold,
    and composite transforms like TransformPipe.

    Note:
        These transforms can be directly executed by pipeline runners
        and provide metadata for pipeline introspection and visualization.
    """

    @abstractmethod
    def get_name(self) -> str:
        """Get the display name of this transform.

        Returns:
            Human-readable name for the transform, typically the function name.
        """
        pass

    @abstractmethod
    def get_docs(self) -> str:
        """Get the documentation string for this transform.

        Returns:
            Documentation text, typically from the function docstring.
        """
        pass

    @abstractmethod
    def get_fn(self) -> Callable:
        """Get the underlying callable for this transform.

        Returns:
            The function or callable that implements the transform logic.
        """
        pass

    @abstractmethod
    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        """Get the executable units for this transform.

        Args:
            state: Pipeline state, may be needed for dynamic unit creation.

        Returns:
            List of executable units that implement this transform.

        Note:
            Some transforms may create units dynamically based on state
            (e.g., array processing with runtime-determined lengths).
        """
        pass

    def needs_commit_lock(self) -> bool:
        """Check if this transform requires exclusive commit access.

        Returns:
            True if the transform needs a commit lock for thread safety.

        Note:
            Most transforms don't need commit locks, but some may require
            exclusive access to prevent concurrent modification conflicts.
        """
        return False
