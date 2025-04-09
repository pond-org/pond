from abc import ABC
from typing import NewType, Callable, Type

from pydantic import BaseModel

from pond.lens import LensPath, LensInfo
from pond.state import State

AbstractExecuteTransform = NewType("AbstractExecuteTransform", None)


class AbstractTransform(ABC):
    
    def get_input_types(self, root_type: Type[BaseModel]) -> dict[str, Type]:
        return [LensInfo(root_type, p).get_type() for p in self.get_inputs()]

    def get_output_type(self, root_type: Type[BaseModel]) -> Type:
        outputs = [LensInfo(root_type, p).get_type() for p in self.get_outputs()]
        if len(outputs) == 0:
            return None
        elif len(outputs) == 1:
            return outputs[0]
        else:
            return tuple[*outputs]

    def get_inputs(self) -> list[LensPath]:
        pass

    def get_outputs(self) -> list[LensPath]:
        pass

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        pass


class AbstractExecuteUnit(ABC):
    def __init__(self, inputs: list[LensPath], outputs: list[LensPath]):
        self.inputs = inputs
        self.outputs = outputs

    def get_inputs(self) -> list[LensPath]:
        return self.inputs

    def get_outputs(self) -> list[LensPath]:
        return self.outputs

    def execute_on(self, state: State) -> None:
        pass


class ExecuteTransform(AbstractExecuteUnit):
    def __init__(
        self,
        inputs: list[LensPath],
        outputs: list[LensPath],
        fn: Callable,
        append_outputs: list[LensPath] = [],
    ):
        super().__init__(inputs, outputs)
        self.fn = fn
        self.append_outputs = append_outputs

    def execute_on(self, state: State) -> None:
        args = [state[i.to_path()] for i in self.inputs]
        rtns = self.fn(*args)
        if isinstance(rtns, tuple) and len(self.outputs) > 1:
            rtns_list = list(rtns)
        else:
            rtns_list = [rtns]
        for rtn, o in zip(rtns_list, self.outputs):
            # state[o.to_path()] = rtn
            append = o in self.append_outputs
            if append:
                print(f"WILL APPEND TO {o.to_path()}")
            state.lens(o.to_path()).set(rtn, append)


class AbstractExecuteTransform(AbstractTransform):
    def get_name(self) -> str:
        pass

    def get_docs(self) -> str:
        pass

    def get_fn(self) -> Callable:
        pass

    def get_execute_units(self, state: State) -> list[AbstractExecuteUnit]:
        pass
