import os
import copy

from typing import Self
from dataclasses import dataclass
from abc import ABC

from parse import parse
import pyarrow as pa


@dataclass
class TypeField:
    name: str
    index: int | None

    def __eq__(self, other: Self) -> bool:
        return self.name == other.name and self.index == other.index

    def subset_of(self, other: Self):
        return self.name == other.name and (
            other.index is None or self.index == other.index
        )


@dataclass
class LensPath:
    path: list[TypeField]
    variant: str = "default"

    @staticmethod
    def from_path(
        path: str, root_path: str = "catalog", variant="default"
    ) -> "LensPath":
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
        return LensPath(parts, variant)

    def to_path(self) -> str:
        path = ".".join(
            map(
                lambda t: t.name if t.index is None else f"{t.name}[{t.index}]",
                self.path[1:],
            )
        )
        return path if self.variant == "default" else f"{self.variant}:{path}"

    def clone(self) -> Self:
        return copy.deepcopy(self)

    def __eq__(self, other: Self) -> bool:
        return self.path == other.path

    def subset_of(self, other: Self) -> bool:
        if len(self.path) < len(other.path):
            return False
        return all(
            a.subset_of(b) for a, b in zip(self.path[: len(other.path)], other.path)
        )

    def get_db_query(self, level: int = 1, dot_accessor: bool = False) -> str:
        assert level >= 1 and level <= len(self.path)
        parts = []
        for i, field in enumerate(self.path[level:]):
            # parts.append(field.name if i == 0 else f"['{field.name}']")
            field_accessor = f".{field.name}" if dot_accessor else f"['{field.name}']"
            parts.append(field.name if i == 0 else field_accessor)
            if field.index is not None:
                parts.append(f"[{field.index+1}]")
        return "".join(parts)

    def to_fspath(self, level: int = 1, last_index: bool = True) -> os.PathLike:
        assert level >= 1 and level <= len(self.path)
        entries = list(
            map(
                lambda p: p.name if p.index is None else f"{p.name}__{p.index}",
                self.path[:level],
            )
        )
        if not last_index and self.path[level - 1].index is not None:
            entries[-1] = self.path[level - 1].name
        return "/".join(entries)

    def to_volume_path(self) -> os.PathLike:
        entries = map(
            lambda p: p.name if p.index is None else f"{p.name}/{p.index}",
            self.path,
        )
        return "/".join(entries)

    def path_and_query(
        self, level: int = 1, last_index: bool = True, dot_accessor: bool = False
    ) -> tuple[os.PathLike, str]:
        return self.to_fspath(level, last_index), self.get_db_query(level, dot_accessor)


class AbstractCatalog(ABC):
    def len(self, path: LensPath) -> int:
        pass

    def __getstate__(self):
        pass

    def __setstate__(self, state):
        pass

    def write_table(
        self,
        table: pa.Table,
        path: LensPath,
        schema: pa.Schema,
        per_row: bool,
        append: bool,
    ) -> bool:
        pass

    def load_table(self, path: LensPath) -> pa.Table | None:
        pass
