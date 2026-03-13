"""Marie AI embeddings implementations."""

from marie.embeddings.base import EmbeddingsBase
from marie.embeddings.embeddings_object import EmbeddingsObject
from marie.embeddings.gemini import GeminiEmbeddings
from marie.embeddings.jina import JinaEmbeddings, JinaEmbeddingsV4

__all__ = [
    "EmbeddingsBase",
    "EmbeddingsObject",
    "GeminiEmbeddings",
    "JinaEmbeddings",
    "JinaEmbeddingsV4",
]
