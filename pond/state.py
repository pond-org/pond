from typing import Type, Any

from pydantic import BaseModel
from pond.lens import Lens
from pond.abstract_catalog import AbstractCatalog


class State:
    def __init__(
        self,
        root_type: Type[BaseModel],
        catalog: AbstractCatalog,
        root_path: str = "catalog",
        storage_path: str = ".",
    ):
        self.root_type = root_type
        self.catalog = catalog
        self.root_path = root_path
        self.storage_path = storage_path

    def lens(self, path: str) -> Lens:
        lens = Lens(
            self.root_type, path, self.catalog, self.root_path, self.storage_path
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
