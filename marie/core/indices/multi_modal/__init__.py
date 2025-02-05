"""Vector-store based data structures."""

from marie.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from marie.core.indices.multi_modal.retriever import (
    MultiModalVectorIndexRetriever,
)

__all__ = [
    "MultiModalVectorStoreIndex",
    "MultiModalVectorIndexRetriever",
]
