from typing import Callable, Generic, TypeVar

import pydantic
from pydantic import BaseModel, ConfigDict

DataT = TypeVar("DataT")


class File(BaseModel, Generic[DataT]):
    # This is needed in order to store the "loader"
    model_config = ConfigDict(extra="allow")
    path: str

    @staticmethod
    def save(path: str, loader: Callable[[], DataT]) -> "File[DataT]":
        return File(path=path, loader=loader)  # type: ignore

    @staticmethod
    def set(object: DataT) -> "File[DataT]":
        return File(path="", object=object)  # type: ignore

    def get(self) -> DataT:
        return self.object  # type: ignore

    def load(self) -> DataT:
        return self.loader()  # type: ignore


def Field(
    *args,
    reader=None,
    writer=None,
    ext="pickle",
    path=None,
    protocol=None,
    json_schema_extra={},
    **kwargs,
):
    json_schema_extra = {
        "reader": reader,
        "writer": writer,
        "ext": ext,
        "path": path,
        "protocol": protocol,
        **json_schema_extra,
    }
    return pydantic.Field(*args, json_schema_extra=json_schema_extra, **kwargs)
