import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import pyarrow as pa  # type: ignore
from parse import parse  # type: ignore


@dataclass
class TypeField:
    """Represents a field in a hierarchical data path.

    A component of a data access path that includes both the field name and
    an optional array index. Used to construct paths through nested data
    structures like "clouds[0].points".

    Attributes:
        name: The field name in the data structure.
        index: Optional array index. None for non-array fields, -1 for wildcard
            array access ([:]), or a specific integer index for array elements.
    """

    name: str
    index: int | None

    def __eq__(self, other: Self) -> bool:  # type: ignore
        """Check equality between TypeField instances.

        Two TypeFields are equal if they have the same name and index, with
        special handling for wildcard indices (-1) which are treated as None.

        Args:
            other: Another TypeField instance to compare against.

        Returns:
            True if the fields represent the same path component.

        Note:
            Index value -1 (wildcard) is normalized to None for comparison.
        """
        self_index = self.index if self.index != -1 else None
        other_index = other.index if other.index != -1 else None
        return self.name == other.name and self_index == other_index

    def subset_of(self, other: Self) -> bool:
        """Check if this field is a subset of another field.

        A field is a subset if it has the same name and either the same index
        or the other field has no index constraint (None).

        Args:
            other: Another TypeField to check subset relationship against.

        Returns:
            True if this field is a subset of the other field.

        Note:
            This is used for path matching where None index acts as a wildcard.
            Index -1 is normalized to None before comparison.
        """
        self_index = self.index if self.index != -1 else None
        other_index = other.index if other.index != -1 else None
        return self.name == other.name and (
            other_index is None or self_index == other_index
        )


@dataclass
class LensPath:
    """Represents a hierarchical path to data in a catalog.

    A data access path consisting of TypeField components that form a
    dotted path like "clouds[0].points". Includes support for variants
    to handle different data representations.

    Attributes:
        path: List of TypeField components forming the access path.
            First element is typically the root (e.g., "catalog").
        variant: String identifier for data variant. Defaults to "default".
            Used to distinguish between different representations of the same data.

    Note:
        The path structure supports nested arrays and structures, with index
        values of -1 representing wildcard access patterns.
    """

    path: list[TypeField]
    variant: str = "default"

    @staticmethod
    def from_path(
        path: str, root_path: str = "catalog", variant="default"
    ) -> "LensPath":
        """Parse a path string into a LensPath object.

        Creates a LensPath from a dotted string path like "clouds[0].points" or
        "data[:].values[5]". Supports array indexing with specific indices and
        wildcard access patterns.

        Args:
            path: Dotted path string to parse. Empty string creates root-only path.
                Supports formats: "field", "field[index]", "field[:]" for wildcard.
            root_path: Root path component name. Defaults to "catalog".
            variant: Data variant identifier. Defaults to "default".

        Returns:
            LensPath object representing the parsed path structure.

        Raises:
            RuntimeError: If path component cannot be parsed or if more than one
                wildcard is present (currently unsupported).

        Note:
            Only one wildcard pattern "[:] " is currently supported per path.
            Indices in the path are zero-based.
        """
        parts = [TypeField(root_path, None)]
        if path == "":
            return LensPath(parts)
        components = path.split(".")
        wildcards = 0
        for i, c in enumerate(components):
            if matches := parse("{:w}[:]", c):
                name = matches[0]
                index = -1
                wildcards += 1
            elif matches := parse("{:w}[{:d}]", c):
                name, index = matches
            elif matches := parse("{:w}", c):
                name = matches[0]
                index = None
            else:
                raise RuntimeError(f"Could not parse {c} as column")
            parts.append(TypeField(name, index))
        if wildcards > 1:
            raise RuntimeError(
                f"Only one wildcard currently supported by pond, got {path}"
            )
        return LensPath(parts, variant)

    def to_path(self) -> str:
        """Convert the LensPath back to a string representation.

        Creates a dotted path string from the TypeField components, excluding
        the root component. Includes array indices in bracket notation.

        Returns:
            String path representation like "clouds[0].points" or "data".
            If variant is not "default", prefixes with "variant:" format.

        Note:
            The root path component (index 0) is excluded from the output.
            Array indices are rendered as [index] notation.
        """
        path = ".".join(
            map(
                lambda t: t.name if t.index is None else f"{t.name}[{t.index}]",
                self.path[1:],
            )
        )
        return path if self.variant == "default" else f"{self.variant}:{path}"

    def clone(self) -> Self:
        """Create a deep copy of this LensPath.

        Returns:
            A new LensPath instance with identical path and variant values.

        Note:
            Uses deep copy to ensure complete independence from original.
        """
        return copy.deepcopy(self)

    def __eq__(self, other: Self) -> bool:  # type: ignore
        """Check equality between LensPath instances.

        Args:
            other: Another LensPath instance to compare against.

        Returns:
            True if both paths have identical path components.

        Note:
            Comparison is based on path components only, variant is ignored.
        """
        equal = self.path == other.path
        return equal

    def subset_of(self, other: Self) -> bool:
        """Check if this path is a subset of another path.

        A path is a subset if it has at least as many components as the other
        path and each corresponding component is a subset of the other's component.

        Args:
            other: Another LensPath to check subset relationship against.

        Returns:
            True if this path is a subset of the other path.

        Note:
            Uses TypeField.subset_of() for component-wise comparison.
        """
        if len(self.path) < len(other.path):
            return False
        subset = all(
            a.subset_of(b) for a, b in zip(self.path[: len(other.path)], other.path)
        )
        return subset

    def get_db_query(self, level: int = 1, dot_accessor: bool = False) -> str:
        """Generate a database query string for accessing nested data.

        Creates a query string to access nested fields in database columns,
        handling both bracket and dot accessor notation.

        Args:
            level: Starting level in the path (1-indexed, minimum 1).
            dot_accessor: If True, use dot notation, otherwise use bracket notation.

        Returns:
            Query string for database access like "field['nested'][2]" or "field.nested[2]".

        Raises:
            AssertionError: If level is less than 1 or greater than path length.

        Note:
            Array indices are 1-indexed in the output for database compatibility.
        """
        assert level >= 1 and level <= len(self.path)
        parts = []
        for i, field in enumerate(self.path[level:]):
            # parts.append(field.name if i == 0 else f"['{field.name}']")
            field_accessor = f".{field.name}" if dot_accessor else f"['{field.name}']"
            parts.append(field.name if i == 0 else field_accessor)
            if field.index is not None:
                parts.append(f"[{field.index + 1}]")
        return "".join(parts)

    def to_fspath(self, level: int = 1, last_index: bool = True) -> str:
        """Convert path to filesystem-safe string representation.

        Creates a filesystem path by joining path components with '/' and
        replacing array indices with double underscore notation.

        Args:
            level: Number of path components to include (1-indexed, minimum 1).
            last_index: Whether to include index in the last component.

        Returns:
            Filesystem path string like "catalog/clouds__0/points".

        Raises:
            AssertionError: If level is less than 1 or greater than path length.

        Note:
            Array indices use double underscore separator for filesystem safety.
        """
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

    def to_volume_path(self) -> str:
        """Convert path to volume-style path representation.

        Creates a slash-separated path where array indices become separate
        path components, suitable for volume or hierarchical storage.

        Returns:
            Volume path string like "catalog/clouds/0/points".

        Note:
            Array indices become separate path segments rather than suffixes.
        """
        entries = map(
            lambda p: p.name if p.index is None else f"{p.name}/{p.index}",
            self.path,
        )
        return "/".join(entries)

    def path_and_query(
        self, level: int = 1, last_index: bool = True, dot_accessor: bool = False
    ) -> tuple[str, str]:
        """Get both filesystem path and database query representations.

        Convenience method that returns both to_fspath() and get_db_query()
        results with the same parameters.

        Args:
            level: Number of path components to include (1-indexed, minimum 1).
            last_index: Whether to include index in the last filesystem component.
            dot_accessor: If True, use dot notation in query, otherwise bracket notation.

        Returns:
            Tuple of (filesystem_path, database_query) strings.

        Raises:
            AssertionError: If level is less than 1 or greater than path length.
        """
        return self.to_fspath(level, last_index), self.get_db_query(level, dot_accessor)


class AbstractCatalog(ABC):
    """Abstract base class for data catalog implementations.

    Defines the interface for catalog systems that manage hierarchical data
    storage and retrieval. Catalogs handle table operations, path resolution,
    and data persistence across different storage backends.

    The catalog system provides:
    - Hierarchical path-based data access
    - Table read/write operations with Apache Arrow
    - Existence checking at different path levels
    - Serialization support for distributed processing

    Note:
        All methods are abstract and must be implemented by concrete subclasses.
        Thread safety and concurrency handling are implementation-specific.
    """

    @abstractmethod
    def len(self, path: LensPath) -> int:
        """Get the number of elements at the specified path.

        Returns the count of items accessible at the given path location.
        For array paths, returns the array length. For table paths, returns
        the number of rows.

        Args:
            path: LensPath specifying the location to measure.

        Returns:
            Integer count of elements at the path location.

        Raises:
            Implementation-specific exceptions for invalid paths or access errors.

        Note:
            Behavior for non-existent paths is implementation-specific.
        """
        pass

    @abstractmethod
    def __getstate__(self):
        """Prepare catalog state for serialization.

        Returns the state dictionary needed to serialize this catalog instance.
        Used by pickle and other serialization mechanisms for distributed processing.

        Returns:
            Dictionary containing serializable state information.

        Note:
            Must include all information needed to reconstruct the catalog.
            Should exclude non-serializable objects like database connections.
        """
        pass

    @abstractmethod
    def __setstate__(self, state):
        """Restore catalog state from serialization.

        Reconstructs the catalog instance from a state dictionary created by
        __getstate__. Used by pickle and other serialization mechanisms.

        Args:
            state: Dictionary containing the serialized catalog state.

        Note:
            Should recreate all necessary connections and initialize resources.
            Must be compatible with state format from __getstate__.
        """
        pass

    @abstractmethod
    def write_table(
        self,
        table: pa.Table,
        path: LensPath,
        schema: pa.Schema,
        per_row: bool = False,
        append: bool = False,
    ) -> bool:
        """Write an Apache Arrow table to the specified path.

        Stores table data at the given path location with optional per-row
        or append modes. Schema validation ensures data consistency.

        Args:
            table: Apache Arrow table containing the data to write.
            path: LensPath specifying the destination location.
            schema: Expected schema for validation and storage.
            per_row: If True, write each row as a separate entry.
            append: If True, append to existing data instead of overwriting.

        Returns:
            True if write operation was successful, False otherwise.

        Raises:
            Implementation-specific exceptions for schema mismatches,
            path errors, or storage failures.

        Note:
            Schema validation behavior and conflict resolution strategies
            are implementation-specific.
        """
        pass

    @abstractmethod
    def exists_at_level(self, path: LensPath) -> bool:
        """Check if data exists at the exact path level specified.

        Tests for data existence at the precise path location without
        checking parent or child paths. Used for exact path validation.

        Args:
            path: LensPath specifying the exact location to check.

        Returns:
            True if data exists at the specified path level.

        Note:
            This method checks only the exact path level, not parent paths.
            Use exists() for more flexible existence checking.
        """
        pass

    def exists(self, path: LensPath) -> bool:
        """Check if data exists at the specified path or any parent path.

        Performs hierarchical existence checking, first testing the exact path,
        then trying with wildcard indices, and finally recursively checking
        parent paths. This provides flexible path resolution.

        Args:
            path: LensPath to check for data existence.

        Returns:
            True if data exists at the path or any accessible parent path.

        Note:
            Uses exists_at_level() for exact checks and recursively searches
            up the path hierarchy. Handles array index resolution automatically.
        """
        if not path.path:  # len(path.path) <= 1:
            return False
        if self.exists_at_level(path):
            return True
        if path.path[-1].index is not None:
            path = path.clone()
            path.path[-1].index = None
            if self.exists_at_level(path):
                return True
        return self.exists(LensPath(path.path[:-1]))

    @abstractmethod
    def load_table(self, path: LensPath) -> tuple[pa.Table | None, bool]:
        """Load an Apache Arrow table from the specified path.

        Retrieves table data from the given path location. Returns both
        the table data and a success indicator for error handling.

        Args:
            path: LensPath specifying the location to load from.

        Returns:
            Tuple of (table, success) where:
            - table: Apache Arrow table with the data, or None if failed
            - success: Boolean indicating if the load operation succeeded

        Note:
            Implementation-specific behavior for missing paths, schema evolution,
            and error conditions. The success flag helps distinguish between
            empty data and load failures.
        """
        pass
