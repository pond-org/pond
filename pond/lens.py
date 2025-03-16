import os
from typing import List, Type, Any, get_args, get_origin

import datetime

from pydantic import BaseModel, NaiveDatetime
from parse import parse

import pydantic_to_pyarrow
from pydantic._internal import _generics
import pyarrow as pa

from pond.abstract_catalog import (
    TypeField,
    LensPath,
    AbstractCatalog,
)
from pond.field import File

from fsspec.implementations.local import LocalFileSystem

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


def get_tree_type(
    path: list[TypeField], root_type: Type[BaseModel]
) -> tuple[Type[BaseModel], dict]:
    if not path:
        return root_type, {}
    field = path.pop(0)
    field_type = root_type.model_fields[field.name].annotation
    extra_args = root_type.model_fields[field.name].json_schema_extra
    print("Root extra args: ", extra_args)
    if field.index is not None:
        print(field_type)
        assert get_origin(field_type) is list
        field_type = get_args(field_type)[0]
    type, extra_args = (
        get_tree_type(path, field_type) if path else (field_type, extra_args)
    )
    path.insert(0, field)
    return type, extra_args


def get_cleaned_path(path: str, root_path: str) -> tuple[str, LensPath]:
    if matches := parse("{:l}:{}", path):
        variant, path = matches
    else:
        variant = "default"
    lens_path = LensPath.from_path(path, root_path)
    return variant, lens_path


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


class LensInfo:
    def __init__(
        self,
        root_type: Type[BaseModel],
        path: str,
        root_path: str = "catalog",
    ):
        self.variant, self.lens_path = get_cleaned_path(path, root_path)
        self.type, self.extra_args = get_tree_type(self.lens_path.path[1:], root_type)

    def get_type(self) -> Type:
        if self.variant == "default":
            return self.type
        elif self.variant == "file":
            assert issubclass(self.type, File)
            return _generics.get_args(self.type)[0]
        elif self.variant == "table":
            return pa.Table
        raise RuntimeError(f"pond does not support lens variant {self.variant}")


class Lens(LensInfo):
    def __init__(
        self,
        root_type: Type[BaseModel],
        path: str,
        catalog: AbstractCatalog,
        root_path: str = "catalog",
        storage_path: str = ".",
    ):
        super().__init__(root_type, path, root_path)
        self.catalog = catalog
        self.storage_path = storage_path
        self.fs = LocalFileSystem(auto_mkdir=True)

    def index_files(self):
        self.index_files_impl(
            self.lens_path.path,
            self.lens_path.to_volume_path(),
            self.type,
            self.extra_args,
        )

    def index_files_impl(
        self, path: list[TypeField], file_path: str, model_type: Type, extra_args: dict
    ):
        # lens_path = ""  # TODO
        per_row = False
        if _generics.get_origin(model_type) == File:
            print(f"Checking file: {path}")
            # NOTE: we need to check existence and extension here
            ext = extra_args["ext"]
            lens_path = LensPath(path=path)
            fs_path = f"{self.storage_path}/{file_path}.{ext}"
            if not self.fs.exists(fs_path):
                print("FILE DOES NOT EXIST: ", fs_path)
                return
            value = File(path=file_path)
            schema = get_pyarrow_schema(model_type)
            table = pa.Table.from_pylist([value.model_dump()], schema=schema)
            # writer(model.get(), self.fs, f"{self.storage_path}/{path}")
            print(f"Writing {lens_path} with values {value}")
            self.catalog.write_table(table, lens_path, schema, per_row=per_row)
        elif (
            get_origin(model_type) is list
            and _generics.get_origin(get_args(model_type)[0]) is File
        ):
            print(f"Checking file list: {path}")
            # for i, value in enumerate(model):
            #     self.index_files_impl(f"{path}/{i}", value)
            ext = extra_args["ext"]
            lens_path = LensPath(path=path)
            # fs_path = f"{self.storage_path}/{lens_path.to_volume_path()}"
            print(f"Checking fs path {file_path}")
            listing = self.fs.ls(f"{self.storage_path}/{file_path}", detail=True)
            values = []
            for info in listing:
                print(info["name"][-(len(ext) + 1) :])
                print(ext)
                if (
                    info["type"]
                    == "file"
                    # and self.fs.path.splitext(info["name"][-(len(ext) + 1) :] == f".{ext}"
                ):
                    # item_path = self.fs.path.relativefrom(
                    #     self.storage_path, info["name"]
                    # )
                    # item_name, item_ext = self.fs.splitext(item_path)
                    item_path = info["name"][len(str(self.storage_path)) + 1 :]
                    item_path, item_ext = item_path.split(".")
                    # assert (
                    #     len(parts) == 2
                    # ), f"pond only deals with files with 1 ext, found {parts}"
                    # item_path = item_path[0]
                    # item_ext = item_path[1]
                    # item_name, item_ext = os.path.splitext(item_path)
                    # item_ext = item_ext.strip(".").lower()
                    print("Found file: ", info["name"])
                    print("Parts: ", item_path, item_ext)
                    if item_ext == ext:
                        value = File(path=item_path)
                        values.append(value.model_dump())
            schema = get_pyarrow_schema(model_type)
            table = pa.Table.from_pylist(values, schema=schema)
            print(f"Writing {lens_path} with values {values}")
            self.catalog.write_table(table, lens_path, schema, per_row=per_row)
        elif get_origin(model_type) is list:
            print(f"Checking list: {path}")
            item_type = get_args(model_type)[0]
            # TODO: does this make sense, to add the dataset if
            # there are files available in the subfolder?
            # or should we just list all the folders here instead?
            # lens_path = LensPath(path=path)
            # fs_path = f"{self.storage_path}/{lens_path.to_volume_path()}"
            if not self.fs.exists(f"{self.storage_path}/{file_path}"):
                return
            listing = self.fs.ls(f"{self.storage_path}/{file_path}", detail=True)
            counter = 0
            for info in listing:
                if info["type"] == "directory":
                    item_name = info["name"].split("/")[-1]
                    item_path = list(path)
                    item_path[-1].index = counter
                    item_file_path = f"{file_path}/{item_name}"
                    self.index_files_impl(
                        item_path, item_file_path, item_type, extra_args
                    )
                    counter += 1
        elif issubclass(model_type, BaseModel):
            print(f"Checking model: {path}")
            for field in model_type.model_fields:
                extra_args = model_type.model_fields[field].json_schema_extra
                field_type = model_type.model_fields[field].annotation
                field_path = path + [TypeField(field, None)]
                field_file_path = f"{file_path}/{field}"
                # print(f"Trying to set {path}/{field} with extra args {extra_args}")
                self.index_files_impl(
                    field_path, field_file_path, field_type, extra_args
                )

    def get_file_paths(self, model: Any, extra_args: dict):
        if isinstance(model, File):
            reader = extra_args["reader"]
            ext = extra_args["ext"]
            model.object = reader(self.fs, f"{self.storage_path}/{model.path}.{ext}")
        elif isinstance(model, list):
            for i, value in enumerate(model):
                self.get_file_paths(value, extra_args)
        elif isinstance(model, BaseModel):
            for field, value in model:
                extra_args = model.model_fields[field].json_schema_extra
                self.get_file_paths(value, extra_args)

    def get(self) -> BaseModel:
        table, is_query = self.catalog.load_table(self.lens_path)
        # TODO: not that these could be treated the same way
        # the scalar ones are just one element list, just
        # need to assert length 1 and get the first element on return
        rtn: None | list[BaseModel] | BaseModel = None
        if get_origin(self.type) == list:
            field_type = get_args(self.type)[0]
            print("LIST!")
            print(table.to_pylist())
            if issubclass(field_type, BaseModel):
                sub_table = table["value"] if is_query else table
                if self.variant == "default" or self.variant == "file":
                    ts = sub_table.to_pylist()
                    if is_query:
                        ts = ts[0]
                    rtn = [field_type.model_validate(t) for t in ts]
                elif self.variant == "table":
                    return sub_table
                else:
                    raise RuntimeError(
                        f"pond does not support lens variant {self.variant}"
                    )
            else:
                return (
                    table.to_pylist()[0]["value"]
                    if is_query
                    else [t["value"] for t in table.to_pylist()]
                )
        elif not issubclass(self.type, BaseModel):
            print("SIMPLE TYPE!")
            print(table.to_pylist())
            rtn = table.to_pylist()[0]["value"]
        else:
            sub_table = table["value"] if is_query else table
            if self.variant == "default" or self.variant == "file":
                rtn = self.type.model_validate(sub_table.to_pylist()[0])
            elif self.variant == "table":
                return sub_table
            else:
                raise RuntimeError(f"pond does not support lens variant {self.variant}")

        assert rtn is not None
        self.get_file_paths(rtn, self.extra_args)

        if self.variant == "file":
            if _generics.get_origin(self.type) == File:
                return rtn.get()
            elif (
                get_origin(self.type) == list
                and _generics.get_origin(get_args(self.type)[0]) == File
            ):
                return [r.get() for r in rtn]
            else:
                raise RuntimeError("pond requires file variant to have type File")

        return rtn

    def set_file_paths(self, path: str, model: Any, extra_args: dict):
        if isinstance(model, File):
            model.path = path
            writer = extra_args["writer"]
            ext = extra_args["ext"]
            writer(model.get(), self.fs, f"{self.storage_path}/{path}.{ext}")
        elif isinstance(model, list):
            for i, value in enumerate(model):
                self.set_file_paths(f"{path}/{i}", value, extra_args)
        elif isinstance(model, BaseModel):
            for field, value in model:
                extra_args = model.model_fields[field].json_schema_extra
                print(f"Trying to set {path}/{field} with extra args {extra_args}")
                self.set_file_paths(f"{path}/{field}", value, extra_args)

    def set(self, value: EntryType, append: bool = False) -> bool:
        # TODO: check that value is of type self.type
        print("FS path: ", self.lens_path.path)
        fs_path = self.lens_path.to_fspath(level=len(self.lens_path.path))
        print("Extra args: ", self.extra_args)
        print(f"Writing {fs_path} with value {value}")
        print(f"With type {type(value)}")
        print(f"Self type: {self.type}")
        # schema = get_pyarrow_schema(type(value))
        schema = get_pyarrow_schema(self.type)
        print(schema)
        field_type = self.type  # type(value)
        value_to_write = None
        per_row = False

        if self.variant == "file":
            if _generics.get_origin(self.type) == File:
                value = File.set(value)
            elif (
                get_origin(self.type) == list
                and _generics.get_origin(get_args(self.type)[0]) == File
            ):
                value = [File.set(v) for v in value]
            else:
                raise RuntimeError("pond requires file variant to have type File")
        elif self.variant == "table":
            assert isinstance(
                value, pa.Table
            ), "pond requires table variant to use a pyarrow table"
            return self.catalog.write_table(
                value, self.lens_path, schema, per_row=per_row
            )

        # NOTE: table does not handle files
        self.set_file_paths(self.lens_path.to_volume_path(), value, self.extra_args)

        # TODO: this should be a recursive function instead
        if get_origin(self.type) == list:
            field_type = get_args(field_type)[0]
            print(field_type)
            if len(value) == 0:
                # raise RuntimeError("pond can not yet write empty lists")
                value_to_write = []
            elif isinstance(value[0], BaseModel):
                value_to_write = [v.model_dump() for v in value]
                # per_row = True
                print("WRITING FIRST VALUE: ", value_to_write)
            # elif field_type in pydantic_to_pyarrow.schema.FIELD_MAP:
            elif field_type in FIELD_MAP:
                value_to_write = [{"value": v} for v in value]
            else:
                raise RuntimeError(f"pond can not write type {type(value)}")
        # elif issubclass(self.type, File):
        #     if self.variant == "file":
        elif isinstance(value, BaseModel):
            value_to_write = [value.model_dump()]
        # elif field_type in pydantic_to_pyarrow.schema.FIELD_MAP:
        elif field_type in FIELD_MAP:
            print("Writing simple type that is not a list")
            value_to_write = [{"value": value}]
        else:
            raise RuntimeError(f"pond can not write type {type(value)}")

        print("Writing value: ", value_to_write)
        table = pa.Table.from_pylist(value_to_write, schema=schema)
        print("Table: ", table)
        write_path = self.lens_path
        if append and self.lens_path[-1].index is not None:
            write_path = self.lens_path[:-1] + [
                TypeField(self.lens_path[-1].name, None)
            ]
        return self.catalog.write_table(
            table, write_path, schema, per_row=per_row, append=append
        )
