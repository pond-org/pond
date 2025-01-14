import os
from typing import List, Type, get_args, get_origin
from dataclasses import dataclass

import datetime

import lance
from parse import parse
from pydantic import BaseModel, NaiveDatetime

import pydantic_to_pyarrow
import pyarrow as pa

from pond.abstract_catalog import TypeField, LensPath, LanceCatalog

# NOTE: this does not require the root_type but we
# should probably add validation of the type
# This allows setting paths such as
# catalog.drives[0].navigation
# catalog.values
# catalog.values.value1
# NOTE: the type could be
BasicType = str | float | int | bool
EntryType = BaseModel | List[BaseModel] | BasicType | List[BasicType]


def get_pyarrow_schema(t: Type) -> pa.Schema:
    settings = pydantic_to_pyarrow.schema.Settings(
        allow_losing_tz=False,
        by_alias=False,
        exclude_fields=True,
    )
    metadata = {}
    # return pydantic_to_pyarrow.schema._get_pyarrow_type()
    print(t)
    if get_origin(t) == list:
        t = get_args(t)[0]

    if not issubclass(t, BaseModel):
        print("Getting base type")
        field_type = pydantic_to_pyarrow.schema._get_pyarrow_type(t, metadata, settings)
        return pa.schema({"value": field_type})
    else:
        print("Gettin pydantic type")
        return pydantic_to_pyarrow.get_pyarrow_schema(t)
    # elif t in pydantic_to_pyarrow.schema.FIELD_MAP:
    #     return pydantic_to_pyarrow.schema.FIELD_MAP[t]
    # else:
    #     raise RuntimeError(f"Can't convert type {t} to arrow schema")


def get_tree_type(path: list[TypeField], root_type: Type[BaseModel]) -> Type[BaseModel]:
    if not path:
        return root_type
    field = path.pop(0)
    field_type = root_type.model_fields[field.name].annotation
    if field.index is not None:
        print(field_type)
        assert get_origin(field_type) is list
        field_type = get_args(field_type)[0]
    type = get_tree_type(path, field_type) if path else field_type
    path.insert(0, field)
    return type


FIELD_MAP = {
    str: pa.string(),
    bytes: pa.binary(),
    bool: pa.bool_(),
    float: pa.float64(),
    int: pa.int64(),
    datetime.date: pa.date32(),
    NaiveDatetime: pa.timestamp("ms", tz=None),
    datetime.time: pa.time64("us"),
}


class Lens:
    def __init__(
        self,
        root_type: Type[BaseModel],
        path: str = "",
        root_path: str = "catalog",
        db_path: os.PathLike = "test_db",
    ):
        self.lens_path = LensPath.from_path(path, root_path)
        self.type = get_tree_type(self.lens_path.path[1:], root_type)
        self.db_path = db_path
        self.catalog = LanceCatalog(db_path)

    def get_type(self) -> Type:
        return self.type

    def get(self) -> BaseModel:
        table, is_query = self.catalog.load_table(self.lens_path)
        # TODO: not that these could be treated the same way
        # the scalar ones are just one element list, just
        # need to assert length 1 and get the first element on return
        if get_origin(self.type) == list:
            field_type = get_args(self.type)[0]
            print("LIST!")
            print(table.to_pylist())
            if issubclass(field_type, BaseModel):
                return [
                    field_type.parse_obj(t)
                    for t in (
                        table.to_pylist()[0]["value"] if is_query else table.to_pylist()
                    )
                ]
            else:
                return (
                    table.to_pylist()[0]["value"]
                    if is_query
                    else [t["value"] for t in table.to_pylist()]
                )
        elif not issubclass(self.type, BaseModel):
            print("SIMPLE TYPE!")
            print(table.to_pylist())
            return table.to_pylist()[0]["value"]
        else:
            return self.type.parse_obj(
                table.to_pylist()[0]["value"] if is_query else table.to_pylist()[0]
            )

    def set(self, value: EntryType) -> bool:
        # TODO: check that value is of type self.type
        print("FS path: ", self.lens_path.path)
        fs_path = self.lens_path.to_fspath(level=len(self.lens_path.path))
        print(f"Writing {fs_path} with value {value}")
        print(f"With type {type(value)}")
        print(f"Self type: {self.type}")
        # schema = get_pyarrow_schema(type(value))
        schema = get_pyarrow_schema(self.type)
        print(schema)
        field_type = self.type  # type(value)
        value_to_write = None
        per_row = False
        if get_origin(self.type) == list:
            field_type = get_args(field_type)[0]
            print(field_type)
            if len(value) == 0:
                raise RuntimeError("pond can not yet write empty lists")
            elif isinstance(value[0], BaseModel):
                value_to_write = [v.dict() for v in value]
                per_row = True
                print("WRITING FIRST VALUE: ", value_to_write)
            # elif field_type in pydantic_to_pyarrow.schema.FIELD_MAP:
            elif field_type in FIELD_MAP:
                value_to_write = [{"value": v} for v in value]
            else:
                raise RuntimeError(f"pond can not write type {type(value)}")
        else:
            if isinstance(value, BaseModel):
                value_to_write = [value.dict()]
            # elif field_type in pydantic_to_pyarrow.schema.FIELD_MAP:
            elif field_type in FIELD_MAP:
                print("Writing simple type that is not a list")
                value_to_write = [{"value": value}]
            else:
                raise RuntimeError(f"pond can not write type {type(value)}")

        print("Writing value: ", value_to_write)
        table = pa.Table.from_pylist(value_to_write, schema=schema)
        print("Table: ", table)
        return self.catalog.write_table(table, self.lens_path, schema, per_row=per_row)
