from typing import Generic, TypeVar, Callable

import pydantic
from pydantic import BaseModel, ConfigDict


DataT = TypeVar("DataT")


class File(BaseModel, Generic[DataT]):
    # This is needed in order to store the "loader"
    model_config = ConfigDict(extra="allow")
    path: str

    @staticmethod
    def save(path: str, loader: Callable[[], DataT]) -> "File[DataT]":
        return File(path=path, loader=loader)

    @staticmethod
    def set(object: DataT) -> "File[DataT]":
        return File(path="", object=object)

    def get(self) -> DataT:
        return self.object

    def load(self) -> DataT:
        return self.loader()


def Field(
    *args, reader=None, writer=None, ext="pickle", json_schema_extra={}, **kwargs
):
    json_schema_extra = {
        "reader": reader,
        "writer": writer,
        "ext": ext,
        **json_schema_extra,
    }
    return pydantic.Field(*args, json_schema_extra=json_schema_extra, **kwargs)
