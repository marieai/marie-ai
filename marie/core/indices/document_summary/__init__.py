"""Document summary index."""


from marie.core.indices.document_summary.base import (
    DocumentSummaryIndex,
    GPTDocumentSummaryIndex,
)
from marie.core.indices.document_summary.retrievers import (
    DocumentSummaryIndexEmbeddingRetriever,
    DocumentSummaryIndexLLMRetriever,
    DocumentSummaryIndexRetriever,
)

__all__ = [
    "DocumentSummaryIndex",
    "DocumentSummaryIndexLLMRetriever",
    "DocumentSummaryIndexEmbeddingRetriever",
    # legacy
    "GPTDocumentSummaryIndex",
    "DocumentSummaryIndexRetriever",
]
