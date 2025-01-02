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


@dataclass
class TypeField:
    name: str
    index: int | None


@dataclass
class LensPath:
    path: list[TypeField]

    @staticmethod
    def from_path(path: str, root_path: str = "catalog") -> "LensPath":
        parts = [TypeField(root_path, None)]
        if path == "":
            return LensPath(parts)
        components = path.split(".")
        for i, c in enumerate(components):
            if matches := parse("{:w}[{:d}]", c):
                name, index = matches
            elif matches := parse("{:w}", c):
                name = matches[0]
                index = None
            else:
                raise RuntimeError(f"Could not parse {c} as column")
            parts.append(TypeField(name, index))
        return LensPath(parts)

    def get_db_query(self, level: int = 1) -> str:
        assert level >= 1 and level <= len(self.path)
        parts = []
        for i, field in enumerate(self.path[level:]):
            parts.append(field.name if i == 0 else f"['{field.name}']")
            if field.index is not None:
                parts.append(f"[{field.index+1}]")
        return "".join(parts)

    def to_fspath(self, level: int = 1) -> os.PathLike:
        assert level >= 1 and level <= len(self.path)
        entries = map(
            lambda p: p.name if p.index is None else f"{p.name}[{p.index}]",
            self.path[:level],
        )
        return "/".join(entries)

    def path_and_query(self, level: int = 1) -> tuple[os.PathLike, str]:
        return self.to_fspath(level), self.get_db_query(level)


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


def get_entry_with_type(type_path: LensPath, type: Type[BaseModel]) -> BaseModel:
    db_path = "test_db"
    for level in reversed(range(1, len(type_path.path) + 1)):
        field_path, query = type_path.path_and_query(level)
        path = f"./{db_path}/{field_path}.lance"
        if os.path.exists(path):
            break
    ds = lance.dataset(path)
    print(f"Getting {query} from {path}")
    if query:
        print("Table: ", ds.to_table())
        table = ds.to_table(columns={"value": query})
        return type.parse_obj(table.to_pylist()[0]["value"])
    else:
        table = ds.to_table()
        # TODO: not that these could be treated the same way
        # the scalar ones are just one element list, just
        # need to assert length 1 and get the first element on return
        if get_origin(type) == list:
            field_type = get_args(type)[0]
            print("LIST!")
            print(table.to_pylist())
            if issubclass(field_type, BaseModel):
                return [field_type.parse_obj(t) for t in table.to_pylist()]
            else:
                return [t["value"] for t in table.to_pylist()]
        elif not issubclass(type, BaseModel):
            print("SIMPLE TYPE!")
            print(table.to_pylist())
            return table.to_pylist()[0]["value"]
        else:
            return type.parse_obj(table.to_pylist()[0])


class Lens:
    def __init__(
        self, root_type: Type[BaseModel], path: str = "", root_path: str = "catalog"
    ):
        self.lens_path = LensPath.from_path(path, root_path)
        self.type = get_tree_type(self.lens_path.path[1:], root_type)

    def get(self) -> BaseModel:
        return get_entry_with_type(self.lens_path, self.type)

    def set(self, value: EntryType) -> bool:
        # TODO: check that value is of type self.type
        db_path = "test_db"
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
        remaining = []
        if get_origin(self.type) == list:
            field_type = get_args(field_type)[0]
            print(field_type)
            if len(value) == 0:
                raise RuntimeError("pond can not yet write empty lists")
            elif isinstance(value[0], BaseModel):
                value_to_write = [value[0].dict()]
                remaining = value[1:]
                # value_to_write = value
                print("WRITING FIRST VALUE: ", value_to_write)
            elif field_type in pydantic_to_pyarrow.schema.FIELD_MAP:
                value_to_write = [{"value": v} for v in value]
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

        print("Writing value: ", value_to_write)
        table = pa.Table.from_pylist(value_to_write, schema=schema)
        print("Table: ", table)
        ds = lance.write_dataset(
            table, f"{db_path}/{fs_path}.lance", schema=schema, mode="overwrite"
        )
        for value in remaining:
            print("WRITING REMAINING VALUE: ", value)
            table = pa.Table.from_pylist([value.dict()], schema=schema)
            ds.insert(table, schema=schema)
