from typing import Generic, TypeVar

import pydantic


DataT = TypeVar("DataT")


class File(pydantic.BaseModel, Generic[DataT]):
    path: str

    def load(self) -> DataT:
        if "reader" not in self.json_schema_extra:
            raise RuntimeError("Can't read file without reader")
        return self.json_schema_extra["reader"](self.path)


def Field(*args, reader=None, writer=None, json_schema_extra={}, **kwargs):
    json_schema_extra = {"reader": reader, "writer": writer, **json_schema_extra}
    return pydantic.Field(*args, json_schema_extra=json_schema_extra, **kwargs)
