"""Empty Index."""

from marie.core.indices.empty.base import EmptyIndex, GPTEmptyIndex
from marie.core.indices.empty.retrievers import EmptyIndexRetriever

__all__ = ["EmptyIndex", "EmptyIndexRetriever", "GPTEmptyIndex"]
