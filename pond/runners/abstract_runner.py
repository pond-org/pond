from abc import ABC

from pond.state import State
from pond.transforms.transform_pipe import TransformPipe
from pond.hooks.abstract_hook import AbstractHook


class AbstractRunner(ABC):
    def run(self, state: State, pipe: TransformPipe, hooks: list[AbstractHook]):
        pass
