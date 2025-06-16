from typing import Any, Type

from pydantic import BaseModel

from pond.catalogs.abstract_catalog import AbstractCatalog
from pond.lens import Lens


class State:
    def __init__(
        self,
        root_type: Type[BaseModel],
        catalog: AbstractCatalog,
        root_path: str = "catalog",
        # storage_path: str = ".",
        volume_protocol_args: dict[str, Any] = {},
        default_volume_protocol: str = "dir",
    ):
        self.root_type = root_type
        self.catalog = catalog
        self.root_path = root_path
        self.volume_protocol_args = volume_protocol_args
        self.default_volume_protocol = default_volume_protocol
        # self.storage_path = storage_path

    def lens(self, path: str) -> Lens:
        lens = Lens(
            self.root_type,
            path,
            self.catalog,
            self.root_path,
            self.volume_protocol_args,
            self.default_volume_protocol,
        )
        return lens

    def index_files(self, paths: list[str] = [""]):
        for path in paths:
            lens = self.lens(path)
            lens.index_files()

    def __getitem__(self, path: str) -> Any:
        return self.lens(path).get()

    def __setitem__(self, path: str, value: Any):
        self.lens(path).set(value)
