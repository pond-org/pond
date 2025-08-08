from abc import ABC, abstractmethod
from typing import Any, Callable, Type

import dill  # type: ignore
from pydantic import BaseModel

from pond.lens import LensInfo, LensPath
from pond.state import State

# AbstractExecuteTransform = NewType("AbstractExecuteTransform", None)


class AbstractTransform(ABC):
    def get_input_types(self, root_type: Type[BaseModel]) -> list[Type]:
        return [LensInfo(root_type, p).get_type() for p in self.get_inputs()]

    def get_output_type(self, root_type: Type[BaseModel]) -> Type | None:
        outputs = [LensInfo(root_type, p).get_type() for p in self.get_outputs()]
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple[*outputs]  # type: ignore

    @abstractmethod
    def get_inputs(self) -> list[LensPath]:
        pass

    @abstractmethod
    def get_outputs(self) -> list[LensPath]:
        pass

    @abstractmethod
    def get_transforms(self) -> list["AbstractExecuteTransform"]:
        pass

    def call(self, state: State) -> Any:
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
    def __init__(self, inputs: list[LensPath], outputs: list[LensPath]):
        self.inputs = inputs
        self.outputs = outputs

    def get_inputs(self) -> list[LensPath]:
        return self.inputs

    def get_outputs(self) -> list[LensPath]:
        return self.outputs

    @abstractmethod
    def load_inputs(self, state: State) -> list[Any]:
        pass

    @abstractmethod
    def save_outputs(self, state: State, outputs: list[Any]) -> list[Any]:
        pass

    @abstractmethod
    def commit(self, state: State, values: list[Any]) -> bool:
        pass

    @abstractmethod
    def run(self, args: list[Any]) -> list[Any]:
        pass

    def execute_on(self, state: State) -> None:
        args = self.load_inputs(state)
        rtns = self.run(args)
        values = self.save_outputs(state, rtns)
        self.commit(state, values)


class ExecuteTransform(AbstractExecuteUnit):
    def __init__(
        self,
        inputs: list[LensPath],
        outputs: list[LensPath],
        fn: Callable,
        append_outputs: list[LensPath] = [],
        # input_list_len: int = -1,
    ):
        super().__init__(inputs, outputs)
        self.fn = fn  # wrapper
        self.append_outputs = append_outputs
        # self.input_list_len = input_list_len

    def __getstate__(self):
        return dill.dumps((self.inputs, self.outputs, self.fn, self.append_outputs))

    def __setstate__(self, state):
        self.inputs, self.outputs, self.fn, self.append_outputs = dill.loads(state)

    def load_inputs(self, state: State) -> list[Any]:
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
        values = []
        for rtn, o in zip(rtns, self.outputs):
            values.append(state.lens(o.to_path()).create_table(rtn))
        return values

    def commit(self, state: State, values: list[Any]) -> bool:
        for val, o in zip(values, self.outputs):
            append = o in self.append_outputs
            state.lens(o.to_path()).write_table(val, append)
        return True

    def run(self, args: list[Any]) -> list[Any]:
        rtns = self.fn(*args)
        if isinstance(rtns, tuple) and len(self.outputs) > 1:
            rtns_list = list(rtns)
        else:
            rtns_list = [rtns]
        return rtns_list


class AbstractExecuteTransform(AbstractTransform):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_docs(self) -> str:
        pass

    @abstractmethod
    def get_fn(self) -> Callable:
        pass

    @abstractmethod
    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        pass

    def needs_commit_lock(self) -> bool:
        return False
