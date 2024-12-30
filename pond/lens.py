import os
from typing import List, Type, get_args, get_origin
from dataclasses import dataclass

import lance
from parse import parse
from pydantic import BaseModel

import pydantic_to_pyarrow
import pyarrow as pa

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
    if issubclass(t, BaseModel):
        print("Gettin pydantic type")
        return pydantic_to_pyarrow.get_pyarrow_schema(t)
    else:
        print("Getting base type")
        field_type = pydantic_to_pyarrow.schema._get_pyarrow_type(t, metadata, settings)
        return pa.schema({"value": field_type})
    # elif t in pydantic_to_pyarrow.schema.FIELD_MAP:
    #     return pydantic_to_pyarrow.schema.FIELD_MAP[t]
    # else:
    #     raise RuntimeError(f"Can't convert type {t} to arrow schema")


@dataclass
class TypeField:
    name: str
    index: int | None


@dataclass
class LensPath:
    dataset: str
    path: list[TypeField]

    @staticmethod
    def from_path(path: str) -> "LensPath":
        components = path.split(".")
        dataset = components.pop(0)
        parts = []
        for i, c in enumerate(components):
            if matches := parse("{:w}[{:d}]", c):
                name, index = matches
            elif matches := parse("{:w}", c):
                name = matches[0]
                index = None
            else:
                raise RuntimeError(f"Could not parse {c} as column")
            parts.append(TypeField(name, index))
        return LensPath(dataset, parts)

    def get_db_query(self) -> str:
        parts = []
        for i, field in enumerate(self.path):
            parts.append(field.name if i == 0 else f"['{field.name}']")
            if field.index is not None:
                parts.append(f"[{field.index+1}]")
        return "".join(parts)

    def to_fspath(self) -> os.PathLike:
        entries = [self.dataset]
        entries.extend(
            map(
                lambda p: p.name if p.index is None else f"{p.name}[{p.index}]",
                self.path,
            )
        )
        return "/".join(entries)


def get_tree_type(path: LensPath, root_type: Type[BaseModel]) -> Type[BaseModel]:
    if not path.path:
        return root_type
    field = path.path.pop(0)
    field_type = root_type.model_fields[field.name].annotation
    if field.index is not None:
        print(field_type)
        assert get_origin(field_type) is list
        field_type = get_args(field_type)[0]
    type = get_tree_type(path, field_type) if path else field_type
    path.path.insert(0, field)
    return type


def get_entry_with_type(type_path: LensPath, type: Type[BaseModel]) -> BaseModel:
    db_path = "test_db"
    ds = lance.dataset(f"./{db_path}/{type_path.dataset}.lance")
    if not type_path.path:
        table = ds.to_table()
        return type.parse_obj(table.to_pylist()[0])
    query = type_path.get_db_query()
    print(f"Getting {query} from {type_path.dataset}")
    table = ds.to_table(columns={"value": query})
    print(table)
    print(type)
    return type.parse_obj(table.to_pylist()[0]["value"])


def get_entry(path: str, root_type: Type[BaseModel]) -> BaseModel:
    type_path = LensPath.from_path(path)
    type = get_tree_type(type_path, root_type)
    return get_entry_with_type(type_path, type)


def set_entry(path: str, value: EntryType) -> bool:
    db_path = "test_db"
    type_path = LensPath.from_path(path)
    fs_path = type_path.to_fspath()
    print(f"Writing {fs_path}")
    schema = get_pyarrow_schema(type(value))
    print(schema)
    field_type = type(value)
    value_to_write = None
    remaining = []
    if get_origin(type) == list:
        field_type = get_args(field_type)[0]
        if isinstance(value, BaseModel):
            value_to_write = [value.pop(0).dict()]
            remaining = value
        elif field_type in pydantic_to_pyarrow.schema.FIELD_MAP:
            value_to_write = [{"value": value}]
        else:
            raise RuntimeError(f"pond can not write type {type(value)}")
    else:
        if isinstance(value, BaseModel):
            value_to_write = [value.dict()]
        elif field_type in pydantic_to_pyarrow.schema.FIELD_MAP:
            print("Writing simple type that is not a list")
            value_to_write = [{"value": value}]
        else:
            raise RuntimeError(f"pond can not write type {type(value)}")

    table = pa.Table.from_pylist(value_to_write, schema=schema)
    ds = lance.write_dataset(
        table, f"{db_path}/{fs_path}.lance", schema=schema, mode="overwrite"
    )
    for value in remaining:
        table = pa.Table.from_pylist([value.dict()], schema=schema)
        ds.insert(value)
