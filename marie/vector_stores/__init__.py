"""Marie vector stores module."""

from marie.vector_stores.pgvector import PGVectorStore
from marie.vector_stores.simple import SimpleVectorStore
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
