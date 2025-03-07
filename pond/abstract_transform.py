from abc import ABC
from typing import NewType

from pond.lens import LensPath
from pond.state import State

AbstractExecuteTransform = NewType("AbstractExecuteTransform", None)


class AbstractTransform(ABC):
    def get_inputs(self) -> list[LensPath]:
        pass

    def get_outputs(self) -> list[LensPath]:
        pass

    def get_transforms(self) -> list[AbstractExecuteTransform]:
        pass


class AbstractExecuteTransform(AbstractTransform):
    def execute_on(self, state: State) -> None:
        pass
