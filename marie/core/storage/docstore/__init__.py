# alias for backwards compatibility
from marie.core.storage.docstore.simple_docstore import (
    DocumentStore,
    SimpleDocumentStore,
)
from marie.core.storage.docstore.types import BaseDocumentStore

__all__ = [
    "BaseDocumentStore",
    "DocumentStore",
    "SimpleDocumentStore",
]
