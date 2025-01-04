import os
from typing import Callable, Type, get_type_hints, Any

from pydantic import BaseModel

from pond.transforms import Transform


class Context:
    def __init__(self, Catalog: Type[BaseModel], db_path: os.PathLike):
        """
        Here we should first pick apart Catalog into a tree
        that has a name and a type attached to each node.
        Actually, we should probably represent it as a dict
        with the fully qualified name as the key
        """
        assert issubclass(
            Catalog, BaseModel
        ), "Catalog argument needs to be a pydantic.BaseModel!"
        self.db_path = db_path
        self.transforms: list[Transform] = []

    def add_transform(
        self, fn: Callable, input: list[str] | str, output: list[str] | str
    ):
        self.transforms.append(Transform(fn, input, output, self.db_path))
