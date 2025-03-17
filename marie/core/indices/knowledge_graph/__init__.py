"""KG-based data structures."""

from marie.core.indices.knowledge_graph.base import (
    KnowledgeGraphIndex,
)
from marie.core.indices.knowledge_graph.retrievers import (
    KGTableRetriever,
    KnowledgeGraphRAGRetriever,
)

__all__ = [
    "KnowledgeGraphIndex",
    "KGTableRetriever",
    "KnowledgeGraphRAGRetriever",
]
