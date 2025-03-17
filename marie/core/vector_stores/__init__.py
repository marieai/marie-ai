"""Vector stores."""

from marie.core.vector_stores.simple import SimpleVectorStore
from marie.core.vector_stores.types import (
    ExactMatchFilter,
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    MetadataInfo,
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStoreInfo,
)

__all__ = [
    "VectorStoreQuery",
    "VectorStoreQueryResult",
    "MetadataFilters",
    "MetadataFilter",
    "MetadataInfo",
    "ExactMatchFilter",
    "FilterCondition",
    "FilterOperator",
    "SimpleVectorStore",
    "VectorStoreInfo",
]
