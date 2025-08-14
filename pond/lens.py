import datetime
from typing import Any, List, Type, get_args, get_origin

import fsspec  # type: ignore
import pyarrow as pa  # type: ignore
import pydantic_to_pyarrow  # type: ignore
from parse import parse  # type: ignore
from pydantic import BaseModel, NaiveDatetime
from pydantic._internal import _generics

from pond.catalogs.abstract_catalog import AbstractCatalog, LensPath, TypeField
from pond.field import File

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
    """Convert a Python type to a PyArrow schema.

    Converts pydantic model types and basic Python types to equivalent
    PyArrow schemas for catalog storage. Handles both single types and
    list types appropriately.

    Args:
        t: The Python type to convert. Can be a pydantic BaseModel subclass,
            basic type (str, int, float, bool), or list type.

    Returns:
        PyArrow schema representing the type structure.

    Note:
        For list types, extracts the element type and creates schema for that.
        For non-BaseModel types, wraps in a 'value' field schema.
        Uses pydantic-to-pyarrow for BaseModel conversion.
    """
    settings = pydantic_to_pyarrow.schema.Settings(
        allow_losing_tz=False,
        by_alias=False,
        exclude_fields=True,
    )
    metadata: list[Any] = []
    # return pydantic_to_pyarrow.schema._get_pyarrow_type()
    if get_origin(t) is list:
        t = get_args(t)[0]

    if not issubclass(t, BaseModel):
        field_type = pydantic_to_pyarrow.schema._get_pyarrow_type(t, metadata, settings)
        return pa.schema({"value": field_type})
    else:
        return pydantic_to_pyarrow.get_pyarrow_schema(t)
    # elif t in pydantic_to_pyarrow.schema.FIELD_MAP:
    #     return pydantic_to_pyarrow.schema.FIELD_MAP[t]
    # else:
    #     raise RuntimeError(f"Can't convert type {t} to arrow schema")


def get_tree_type(
    path: list[TypeField], root_type: Type[BaseModel]
) -> tuple[Type[BaseModel], dict]:
    """Navigate a type hierarchy to find the type at a specific path.

    Traverses the pydantic model type hierarchy following the given path
    to determine the final type and associated metadata.

    Args:
        path: List of TypeField components representing the navigation path.
            Modified in-place during traversal.
        root_type: The root pydantic model class to start navigation from.

    Returns:
        Tuple containing:
        - The final type at the path location
        - Dictionary of extra arguments/metadata from field definitions

    Note:
        Handles array indexing by unwrapping list types when index is specified.
        Recursively processes nested model structures.
    """
    if not path:
        return root_type, {}
    field = path.pop(0)
    field_type: Type
    field_type = root_type.model_fields[field.name].annotation  # type: ignore
    extra_args: dict
    extra_args = root_type.model_fields[field.name].json_schema_extra  # type: ignore
    if field.index is not None:
        assert get_origin(field_type) is list
        field_type = get_args(field_type)[0]
    if path:
        assert issubclass(field_type, BaseModel)
        type, extra_args = get_tree_type(path, field_type)
    else:
        type = field_type
    path.insert(0, field)
    return type, extra_args


def get_cleaned_path(path: str, root_path: str) -> LensPath:
    """Parse and clean a path string into a LensPath object.

    Handles variant prefixes in path strings and creates appropriate
    LensPath objects with the correct variant annotation.

    Args:
        path: Path string, optionally prefixed with variant like 'table:path.field'.
        root_path: Root path name for the LensPath.

    Returns:
        LensPath object with the appropriate variant set.

    Note:
        Variant prefixes like 'table:', 'file:', etc. are extracted and
        set as the LensPath variant. Default variant is 'default'.
    """
    if matches := parse("{:l}:{}", path):
        variant, path = matches
    else:
        variant = "default"
    lens_path = LensPath.from_path(path, root_path, variant)
    return lens_path


"""Mapping of Python types to PyArrow types.

Defines the conversion mapping for basic Python types to their
equivalent PyArrow type representations for catalog storage.

Note:
    Used as fallback for types not handled by pydantic-to-pyarrow.
    Covers common Python types and datetime variants.
"""
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
    """Information about a lens path including type and metadata.

    Encapsulates the path information and resolved type for a specific
    location in the data catalog. Used by transforms for type validation
    and by the lens system for data access.

    Attributes:
        lens_path: The LensPath specifying the data location.
        type: The resolved Python type at the path location.
        extra_args: Metadata dictionary from field definitions.

    Note:
        The type resolution considers variants and array indexing to
        determine the actual data type that will be accessed.
    """

    def __init__(
        self,
        root_type: Type[BaseModel],
        lens_path: LensPath,
    ):
        """Initialize LensInfo by resolving path type information.

        Args:
            root_type: The root pydantic model class.
            lens_path: The path to resolve type information for.
        """
        self.lens_path = lens_path
        self.type, self.extra_args = get_tree_type(self.lens_path.path[1:], root_type)

    @staticmethod
    def from_path(
        root_type: Type[BaseModel],
        path: str,
        root_path: str = "catalog",
    ) -> "LensInfo":
        lens_path = get_cleaned_path(path, root_path)
        return LensInfo(root_type, lens_path)

    def set_index(self, index: int, value: int):
        assert index >= 0
        assert self.lens_path.path[index].index is not None, (
            "Lens only supports setting index for list item lenses"
        )
        assert value >= 0 or value == -1
        self.lens_path.path[index].index = value

    def get_type(self) -> Type:
        if self.lens_path.variant == "default":
            return self.type
        elif self.lens_path.variant == "file":
            if get_origin(self.type) is list:
                item_type = get_args(self.type)[0]
                assert issubclass(item_type, File)
                return list[_generics.get_args(item_type)[0]]  # type: ignore
            else:
                assert issubclass(self.type, File)
                return _generics.get_args(self.type)[0]
        elif self.lens_path.variant == "table":
            return pa.Table
        raise RuntimeError(
            f"pond does not support lens variant {self.lens_path.variant}"
        )


class Lens(LensInfo):
    """Primary interface for accessing and manipulating data in pond catalogs.

    The Lens provides a unified interface for reading and writing data at specific
    paths in the catalog hierarchy. It handles both structured data (stored in
    catalogs) and unstructured data (stored as files), with automatic conversion
    between Python objects and catalog storage formats.

    Key Features:
    - Path-based data access with variant support (default, table, file)
    - Automatic type conversion and validation
    - File system integration via fsspec
    - Lazy loading and efficient data access
    - Support for both scalar and array data

    Attributes:
        catalog: The catalog backend for structured data storage.
        volume_protocol_args: Configuration for filesystem protocols.
        default_volume_protocol: Default protocol when none specified.

    Note:
        Lenses are typically created via State.lens() rather than directly.
        The lens system is the primary way to interact with pond data.
    """

    def __init__(
        self,
        root_type: Type[BaseModel],
        path: str,
        catalog: AbstractCatalog,
        root_path: str = "catalog",
        volume_protocol_args: dict[str, Any] = {},
        default_volume_protocol: str = "dir",
    ):
        """Initialize a Lens for the specified path.

        Args:
            root_type: The root pydantic model class defining the schema.
            path: Dot-separated path string specifying the data location.
            catalog: Catalog backend for structured data operations.
            root_path: Root path name for path resolution.
            volume_protocol_args: Filesystem protocol configurations.
            default_volume_protocol: Default protocol for unspecified fields.

        Note:
            Automatically configures filesystem protocols and validates
            the path against the schema.
        """
        super().__init__(root_type, get_cleaned_path(path, root_path))
        self.catalog = catalog
        # self.storage_path = storage_path
        self.volume_protocol_args = volume_protocol_args
        self.default_volume_protocol = default_volume_protocol
        for name, args in self.volume_protocol_args.items():
            if "protocol" not in args:
                args["protocol"] = name
            if args["protocol"] == "dir":
                args["target_protocol"] = "file"
                args["target_options"] = {"auto_mkdir": True}
        # self.fs = LocalFileSystem(auto_mkdir=True)
        # self.fs = fsspec.filesystem(**self.volume_protocol_args)

    def len(self) -> int:
        return self.catalog.len(self.lens_path)

    def exists(self) -> bool:
        return self.catalog.exists(self.lens_path)

    def index_files(self):
        if self.extra_args and self.extra_args.get("path", None):
            file_path = self.extra_args["path"]
        else:
            file_path = self.lens_path.to_volume_path()
        self.index_files_impl(
            self.lens_path.path,
            file_path,
            self.type,
            self.extra_args,
        )

    def index_files_impl(
        self, path: list[TypeField], file_path: str, model_type: Type, extra_args: dict
    ):
        # lens_path = ""  # TODO
        per_row = False
        if _generics.get_origin(model_type) == File:
            # NOTE: we need to check existence and extension here
            ext = extra_args["ext"]
            protocol = extra_args["protocol"] or self.default_volume_protocol
            lens_path = LensPath(path=path)
            fs = fsspec.filesystem(**self.volume_protocol_args[protocol])
            fs_path = f"{file_path}.{ext}"
            if not fs.exists(fs_path):
                return
            value: File[Any] = File(path=file_path)
            schema = get_pyarrow_schema(model_type)
            table = pa.Table.from_pylist([value.model_dump()], schema=schema)
            # writer(model.get(), self.fs, f"{self.storage_path}/{path}")
            self.catalog.write_table(table, lens_path, schema, per_row=per_row)
        elif (
            get_origin(model_type) is list
            and _generics.get_origin(get_args(model_type)[0]) is File
        ):
            # for i, value in enumerate(model):
            #     self.index_files_impl(f"{path}/{i}", value)
            ext = extra_args["ext"]
            protocol = extra_args["protocol"] or self.default_volume_protocol
            lens_path = LensPath(path=path)
            # fs_path = f"{self.storage_path}/{lens_path.to_volume_path()}"
            fs = fsspec.filesystem(**self.volume_protocol_args[protocol])
            # if wildcard, use glob instead
            if "*" in file_path:
                listing_dict = fs.glob(file_path, detail=True)
                listing = list(listing_dict.values())
            else:
                listing = fs.ls(file_path, detail=True)
            values = []
            for info in listing:
                if (
                    info["type"] == "file"
                    # and self.fs.path.splitext(info["name"][-(len(ext) + 1) :] == f".{ext}"
                ):
                    item_path = info["name"]  # [len(str(self.storage_path)) + 1 :]
                    item_path, item_ext = item_path.split(".")
                    if item_ext == ext:
                        value = File(path=item_path)
                        values.append(value.model_dump())
            schema = get_pyarrow_schema(model_type)
            table = pa.Table.from_pylist(values, schema=schema)
            self.catalog.write_table(table, lens_path, schema, per_row=per_row)
        elif get_origin(model_type) is list:
            item_type = get_args(model_type)[0]
            # TODO: does this make sense, to add the dataset if
            # there are files available in the subfolder?
            # or should we just list all the folders here instead?
            # lens_path = LensPath(path=path)
            # fs_path = f"{self.storage_path}/{lens_path.to_volume_path()}"
            fs = fsspec.filesystem(
                **self.volume_protocol_args[self.default_volume_protocol]
            )
            if not fs.exists(file_path):
                return
            if "*" in file_path:
                listing_dict = fs.glob(file_path, detail=True)
                listing = list(listing_dict.values())
            else:
                listing = fs.ls(file_path, detail=True)
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
            for field in model_type.model_fields:
                extra_args = model_type.model_fields[field].json_schema_extra  # type: ignore
                field_type = model_type.model_fields[field].annotation
                assert field_type is not None
                field_path = path + [TypeField(field, None)]
                if extra_args and extra_args.get("path", None):
                    field_file_path = extra_args["path"]
                else:
                    field_file_path = f"{file_path}/{field}"
                self.index_files_impl(
                    field_path, field_file_path, field_type, extra_args
                )

    def get_file_paths(self, model: Any, extra_args: dict):
        if isinstance(model, File):
            reader = extra_args["reader"]
            ext = extra_args["ext"]
            protocol = extra_args["protocol"] or self.default_volume_protocol
            fs = fsspec.filesystem(**self.volume_protocol_args[protocol])
            model.object = reader(fs, f"{model.path}.{ext}")  # type: ignore[attr-defined]
        elif isinstance(model, list):
            for i, value in enumerate(model):
                self.get_file_paths(value, extra_args)
        elif isinstance(model, BaseModel):
            for field, value in model:
                extra_args = model.model_fields[field].json_schema_extra  # type: ignore[assignment]
                self.get_file_paths(value, extra_args)

    def get(self) -> None | list[BaseModel] | BaseModel:
        table, is_query = self.catalog.load_table(self.lens_path)
        if table is None:
            return None
        # TODO: not that these could be treated the same way
        # the scalar ones are just one element list, just
        # need to assert length 1 and get the first element on return
        rtn: None | list[BaseModel] | BaseModel = None
        if get_origin(self.type) is list:
            field_type = get_args(self.type)[0]
            if issubclass(field_type, BaseModel):
                sub_table = table["value"] if is_query else table
                if (
                    self.lens_path.variant == "default"
                    or self.lens_path.variant == "file"
                ):
                    ts = sub_table.to_pylist()
                    if is_query:
                        ts = ts[0]
                    rtn = [field_type.model_validate(t) for t in ts]
                elif self.lens_path.variant == "table":
                    if is_query:
                        return pa.Table.from_batches(
                            [
                                pa.RecordBatch.from_struct_array(s.flatten())
                                for s in sub_table.iterchunks()
                            ],
                            schema=get_pyarrow_schema(self.type),
                        )
                    else:
                        return sub_table
                else:
                    raise RuntimeError(
                        f"pond does not support lens variant {self.lens_path.variant}"
                    )
            else:
                return (
                    table.to_pylist()[0]["value"]
                    if is_query
                    else [t["value"] for t in table.to_pylist()]
                )
        elif not issubclass(self.type, BaseModel):
            rtn = table.to_pylist()[0]["value"]
        else:
            sub_table = table["value"] if is_query else table
            if self.lens_path.variant == "default" or self.lens_path.variant == "file":
                rtn = self.type.model_validate(sub_table.to_pylist()[0])
            elif self.lens_path.variant == "table":
                if is_query:
                    return pa.Table.from_batches(
                        [
                            pa.RecordBatch.from_struct_array(s.flatten())
                            for s in sub_table.iterchunks()
                        ],
                        schema=get_pyarrow_schema(self.type),
                    )
                else:
                    return sub_table
            else:
                raise RuntimeError(
                    f"pond does not support lens variant {self.lens_path.variant}"
                )

        assert rtn is not None
        self.get_file_paths(rtn, self.extra_args)

        if self.lens_path.variant == "file":
            if _generics.get_origin(self.type) == File:
                assert isinstance(rtn, File)
                return rtn.get()
            elif (
                get_origin(self.type) is list
                and _generics.get_origin(get_args(self.type)[0]) == File
            ):
                assert isinstance(rtn, list)
                return [r.get() for r in rtn if isinstance(r, File)]
            else:
                raise RuntimeError("pond requires file variant to have type File")

        return rtn

    def set_file_paths(self, path: str, model: Any, extra_args: dict) -> bool:
        if isinstance(model, File):
            model.path = path
            writer = extra_args["writer"]
            ext = extra_args["ext"]
            protocol = extra_args["protocol"] or self.default_volume_protocol
            fs = fsspec.filesystem(**self.volume_protocol_args[protocol])
            writer(model.get(), fs, f"{path}.{ext}")
            return True
        elif isinstance(model, list):
            if len(model) == 0 or not self.set_file_paths(
                f"{path}/0", model[0], extra_args
            ):
                return False
            for i, value in enumerate(model[1:]):
                self.set_file_paths(f"{path}/{i + 1}", value, extra_args)
            return True
        elif isinstance(model, BaseModel):
            found = False
            for field, value in model:
                extra_args = model.model_fields[field].json_schema_extra  # type: ignore[assignment]
                found = (
                    self.set_file_paths(f"{path}/{field}", value, extra_args) or found
                )
            return found
        return False

    def create_table(self, value: EntryType) -> pa.Table:
        # TODO: check that value is of type self.type
        schema = get_pyarrow_schema(self.type)
        field_type = self.type  # type(value)
        value_to_write = None

        if self.lens_path.variant == "file":
            if _generics.get_origin(self.type) == File:
                value = File.set(value)
            elif (
                get_origin(self.type) is list
                and _generics.get_origin(get_args(self.type)[0]) == File
            ):
                assert isinstance(value, list)
                value = [File.set(v) for v in value]  # type: ignore[assignment]
            else:
                raise RuntimeError("pond requires file variant to have type File")
        elif self.lens_path.variant == "table":
            assert isinstance(value, pa.Table), (
                "pond requires table variant to use a pyarrow table"
            )
            return value

        # NOTE: table does not handle files
        self.set_file_paths(self.lens_path.to_volume_path(), value, self.extra_args)

        # TODO: this should be a recursive function instead
        if get_origin(self.type) is list:
            assert isinstance(value, list)
            field_type = get_args(field_type)[0]
            if len(value) == 0:
                # raise RuntimeError("pond can not yet write empty lists")
                value_to_write = []
            elif isinstance(value[0], BaseModel):
                value_to_write = [
                    v.model_dump() for v in value if isinstance(v, BaseModel)
                ]
                # per_row = True
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
            value_to_write = [{"value": value}]
        else:
            raise RuntimeError(f"pond can not write type {type(value)}")

        table = pa.Table.from_pylist(value_to_write, schema=schema)
        return table

    def write_table(self, table: pa.Table, append: bool = False) -> bool:
        per_row = False
        # return self.catalog.write_table(value, self.lens_path, schema, per_row=per_row)
        write_path = self.lens_path.path
        schema = get_pyarrow_schema(self.type)
        if append and write_path[-1].index is not None:
            write_path = write_path[:-1] + [TypeField(write_path[-1].name, None)]
        return self.catalog.write_table(
            table, LensPath(write_path), schema, per_row=per_row, append=append
        )

    def set(self, value: EntryType, append: bool = False) -> bool:
        table = self.create_table(value)
        return self.write_table(table, append=append)
