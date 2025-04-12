from abc import ABC
from typing import Type, Optional

from pydantic import BaseModel

from pond.abstract_transform import AbstractExecuteTransform
from pond.transform_pipe import TransformPipe


class BaseHook(ABC):
    def __init__(self):
        self.root_type = None

    def initialize(self, root_type: Type[BaseModel]):
        self.root_type = root_type

    def pre_pipe_execute(self, pipe: TransformPipe):
        pass

    def post_pipe_execute(
        self, pipe: TransformPipe, success: bool, error: Optional[Exception]
    ):
        pass

    def pre_node_execute(self, node: AbstractExecuteTransform):
        pass

    def post_node_execute(
        self, node: AbstractExecuteTransform, success: bool, error: Optional[Exception]
    ):
        pass
