import os
from typing import Callable, Type, get_type_hints, Any
import typing
from dataclasses import dataclass
import dataclasses

# from fbs_generated import Catalog as GenCatalog


@dataclass
class TypeNode:
    type: Type
    fbs_storage: bool


class Context:
    def __init__(self, root: os.PathLike, Catalog: Type):
        """
        Here we should first pick apart Catalog into a tree
        that has a name and a type attached to each node.
        Actually, we should probably represent it as a dict
        with the fully qualified name as the key
        """
        self.root = root
        self.typetree: dict[str, TypeNode] = {}
        assert dataclasses.is_dataclass(Catalog)
        for name, type in get_type_hints(Catalog).items():
            self.add_typetree("", name, type)
            print(name, type)
        print(self.typetree)

    def add_typetree(self, namespace: str, name: str, type: Type) -> bool:
        fbs_storage = True
        path = f"{namespace}.{name}" if namespace else name
        print(typing.get_origin(type))
        if typing.get_origin(type) == list:
            item_type = typing.get_args(type)[0]
            print(item_type)
            fbs_storage = self.add_typetree(namespace, f"[{name}]", item_type)
        elif dataclasses.is_dataclass(type):
            for child_name, child_type in get_type_hints(type).items():
                child_fbs_storage = self.add_typetree(path, child_name, child_type)
                fbs_storage = fbs_storage and child_fbs_storage
        elif type in {str, float, int, bool}:
            pass
        else:
            print(name, type)
            raise RuntimeError("pond only supports lists and dataclasses")
        print("Adding ", path, type)
        self.typetree[path] = TypeNode(type, fbs_storage)
        return fbs_storage

    def add_transform(
        self, fn: Callable, input: list[str] | str, output: list[str] | str
    ):
        pass

    # def load(self, dataset: str) -> Any:
    #     """
    #     Here we need to inspect the dataset by getting the type
    #     from the typetree
    #     """

    #     # TODO: for lists we will need to add element names here
    #     # something like [dives:Mungo_high].[clouds:test.laz] or something
    #     # in that case, we get the type from the first element and the
    #     # path name from the second
    #     typenode = self.typetree[dataset]
    #     if not typenode.fbs_storage:
    #         raise RuntimeError("pond does not yet support custom dataloaders")
    #     type = typenode.type
    #     path = os.path.join(self.root, *dataset.split(sep="."))

    #     buf = open(path, "rb").read()
    #     buf = bytearray(buf)

    #     if typing.get_origin(type) == list:
    #         item_type = typing.get_args(type)[0]
    #         print(item_type)
    #         fbs_storage = self.add_typetree(namespace, f"[{name}]", item_type)
    #     elif dataclasses.is_dataclass(type):
    #         for child_name, child_type in get_type_hints(type).items():
    #             child_fbs_storage = self.add_typetree(path, child_name, child_type)
    #             fbs_storage = fbs_storage and child_fbs_storage
    #     elif type in {str, float, int, bool}:
    #         pass
    #     else:
    #         print(name, type)
    #         raise RuntimeError("pond only supports lists and dataclasses")
