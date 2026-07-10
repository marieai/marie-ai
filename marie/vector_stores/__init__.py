"""Marie vector stores module."""

from marie.vector_stores.pgvector import PGVectorStore
from marie.vector_stores.types import (
    BasePydanticVectorStore,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)

__all__ = [
    # Base types
    "BasePydanticVectorStore",
    "VectorStore",
    "VectorStoreQuery",
    "VectorStoreQueryResult",
    "VectorStoreQueryMode",
    # Filters
    "MetadataFilter",
    "MetadataFilters",
    "FilterOperator",
    "FilterCondition",
    # Implementations
    "SimpleVectorStore",
    "PGVectorStore",
]


def __getattr__(name):
    # Lazy: simple.py drags in marie.core.indices -> marie.core.node_parser,
    # which is absent from this checkout (incompletely vendored fork). No
    # in-repo consumer imports SimpleVectorStore from this package today;
    # resolve it on demand so PGVectorStore users don't pay for the break.
    if name == "SimpleVectorStore":
        from marie.vector_stores.simple import SimpleVectorStore

        return SimpleVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
