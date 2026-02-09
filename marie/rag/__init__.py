"""RAG (Retrieval-Augmented Generation) module for Marie.

This module provides document retrieval and citation tracking capabilities
for building document Q&A and knowledge-grounded conversational AI.
"""

from marie.rag.models import (
    DocumentSource,
    RAGNode,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievalResult,
    SourceCitation,
)
from marie.rag.retriever import MultiSourceRetriever, RAGRetriever

__all__ = [
    "DocumentSource",
    "SourceCitation",
    "RetrievalResult",
    "RAGNode",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "RAGRetriever",
    "MultiSourceRetriever",
]
