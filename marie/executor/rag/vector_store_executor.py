"""Vector Store Executor for DAG workflows.

This executor provides embed and store operations as nodes in Marie's
DAG workflow system. It integrates JinaEmbeddingsV4 for unified text+image
embeddings with PGVectorStore for persistence.

Usage in workflow:
    EXTRACT_DOCUMENTS (existing)
        ↓
    OCR/TABLE_EXTRACTION (existing)
        ↓
    EMBED_AND_STORE (this executor)
        ↓
    UPDATE_SOURCE_STATUS (existing DB update)
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional, Union

from docarray import DocList
from docarray.documents import TextDoc

from marie import Executor, requests
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from marie.rag.models import SourceCitation

logger = MarieLogger("marie.executor.rag.vector_store_executor").logger


class VectorStoreExecutor(MarieExecutor):
    """Executor for vector store operations in DAG workflows.

    Uses jina-embeddings-v4 for unified text+image embeddings and
    PGVectorStore for persistence. This executor can be added to
    existing document processing workflows to enable RAG capabilities.

    Endpoints:
        /embed_and_store: Embed documents and store in vector store
        /search: Semantic search over stored documents
        /delete_source: Delete all nodes for a source

    Example workflow configuration:
        ```yaml
        - name: vector_store
          uses: VectorStoreExecutor
          with:
            connection_string: postgresql://user:pass@localhost/db
            embedding_model: jinaai/jina-embeddings-v4
            embedding_dim: 2048
        ```
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        embedding_model: str = "jinaai/jina-embeddings-v4",
        embedding_dim: int = 2048,
        embedding_task: str = "retrieval",
        table_name: str = "rag_vectors",
        use_gpu: bool = True,
        batch_size: int = 4,
        **kwargs,
    ):
        """Initialize VectorStoreExecutor.

        Args:
            connection_string: PostgreSQL connection string. If not provided,
                uses PGVECTOR_CONNECTION_STRING environment variable.
            embedding_model: Model name for embeddings.
            embedding_dim: Embedding dimension (Matryoshka: 128/256/512/1024/2048).
            embedding_task: Task adapter (retrieval/text-matching/code).
            table_name: Name of the vector store table.
            use_gpu: Whether to use GPU for embeddings.
            batch_size: Batch size for embedding operations.
            **kwargs: Additional MarieExecutor arguments.
        """
        super().__init__(**kwargs)
        self.logger = MarieLogger(self.__class__.__name__).logger
        self.logger.info(f"Initializing VectorStoreExecutor")

        self._connection_string = connection_string or os.environ.get(
            "PGVECTOR_CONNECTION_STRING"
        )
        if not self._connection_string:
            raise ValueError(
                "connection_string required. Provide via argument or "
                "PGVECTOR_CONNECTION_STRING environment variable."
            )

        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        self._embedding_task = embedding_task
        self._table_name = table_name
        self._use_gpu = use_gpu
        self._batch_size = batch_size

        # Lazy initialization
        self._vector_store = None
        self._embeddings = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazily initialize vector store and embeddings."""
        if self._initialized:
            return

        self.logger.info("Initializing vector store and embeddings...")

        # Initialize embeddings
        from marie.embeddings.jina import JinaEmbeddingsV4

        self._embeddings = JinaEmbeddingsV4(
            model_name_or_path=self._embedding_model,
            task=self._embedding_task,
            truncate_dim=self._embedding_dim,
            use_gpu=self._use_gpu,
            batch_size=self._batch_size,
        )

        # Initialize vector store
        from marie.vector_stores.pgvector import PGVectorStore

        self._vector_store = PGVectorStore(
            connection_string=self._connection_string,
            table_name=self._table_name,
            embedding_dim=self._embedding_dim,
        )
        await self._vector_store.initialize()

        self._initialized = True
        self.logger.info(
            f"VectorStoreExecutor initialized: model={self._embedding_model}, "
            f"dim={self._embedding_dim}, table={self._table_name}"
        )

    @requests(on="/embed_and_store")
    async def embed_and_store(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> DocList[TextDoc]:
        """Embed documents and store in vector store.

        Called from DAG workflow after document extraction.

        Args:
            docs: DocList of documents to embed and store.
            parameters: Must include:
                - source_id: DocumentSource ID
                Optional:
                - node_type: Type of nodes (text/image/document)
                - ref_doc_id: Reference document ID

        Returns:
            Input docs (passthrough for pipeline chaining).
        """
        await self._ensure_initialized()

        source_id = parameters.get("source_id")
        if not source_id:
            raise ValueError("source_id is required in parameters")

        node_type = parameters.get("node_type", "text")
        ref_doc_id = parameters.get("ref_doc_id")

        self.logger.info(f"Processing {len(docs)} documents for source_id={source_id}")

        stored_count = 0
        for doc in docs:
            try:
                # Determine content type and embed accordingly
                content = None
                embedding = None

                # Check if this is an image document
                if hasattr(doc, "image_path") and doc.image_path:
                    # Embed image
                    embedding = self._embeddings.embed_images([doc.image_path])[0]
                    content = f"[Image: {doc.image_path}]"
                    if hasattr(doc, "text") and doc.text:
                        content = doc.text  # Use description if available
                    actual_node_type = "image"
                elif hasattr(doc, "image_url") and doc.image_url:
                    # Embed image from URL
                    embedding = self._embeddings.embed_images([doc.image_url])[0]
                    content = f"[Image: {doc.image_url}]"
                    if hasattr(doc, "text") and doc.text:
                        content = doc.text
                    actual_node_type = "image"
                else:
                    # Embed text
                    content = doc.text if hasattr(doc, "text") else str(doc)
                    if content:
                        embedding = self._embeddings.embed_text(
                            [content], is_query=False
                        )[0]
                    actual_node_type = node_type

                if embedding is None or content is None:
                    self.logger.warning(f"Skipping doc {doc.id}: no content to embed")
                    continue

                # Get metadata
                metadata = {}
                if hasattr(doc, "metadata"):
                    metadata = dict(doc.metadata) if doc.metadata else {}

                # Store in vector store
                await self._vector_store.add_node(
                    node_id=str(doc.id),
                    embedding=embedding.tolist(),
                    content=content,
                    node_type=actual_node_type,
                    source_id=source_id,
                    metadata=metadata,
                    ref_doc_id=ref_doc_id,
                )
                stored_count += 1

            except Exception as e:
                self.logger.error(f"Failed to process doc {doc.id}: {e}")

        self.logger.info(f"Stored {stored_count}/{len(docs)} documents")
        return docs

    @requests(on="/search")
    async def search(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> DocList[TextDoc]:
        """Semantic search over stored documents.

        Args:
            docs: DocList with query document (uses first doc's text as query).
            parameters: Search parameters:
                - query: Override query text
                - source_ids: Filter to specific sources
                - top_k: Number of results (default: 10)
                - node_type: Filter by node type

        Returns:
            DocList of matching documents with similarity scores in metadata.
        """
        await self._ensure_initialized()

        # Get query
        query = parameters.get("query")
        if not query and docs:
            query = docs[0].text if hasattr(docs[0], "text") else str(docs[0])

        if not query:
            raise ValueError("Query required: provide in parameters or docs")

        source_ids = parameters.get("source_ids", [])
        top_k = parameters.get("top_k", 10)
        node_type = parameters.get("node_type")

        self.logger.info(f"Searching: query='{query[:50]}...', top_k={top_k}")

        # Embed query
        query_embedding = self._embeddings.embed_text([query], is_query=True)[0]

        # Search
        results = await self._vector_store.search(
            query_embedding=query_embedding.tolist(),
            source_ids=source_ids if source_ids else None,
            top_k=top_k,
            node_type=node_type,
        )

        # Convert to DocList
        result_docs = DocList[TextDoc]()
        for r in results:
            doc = TextDoc(
                id=r["node_id"],
                text=r["content"] or "",
            )
            # Store search metadata
            doc.metadata = r.get("metadata") or {}
            doc.metadata["_similarity"] = r["similarity"]
            doc.metadata["_source_id"] = r["source_id"]
            doc.metadata["_node_type"] = r["node_type"]
            doc.metadata["_ref_doc_id"] = r.get("ref_doc_id")
            result_docs.append(doc)

        self.logger.info(f"Found {len(result_docs)} results")
        return result_docs

    @requests(on="/search_with_citations")
    async def search_with_citations(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Search and return results with citation information.

        Args:
            docs: DocList with query document.
            parameters: Search parameters (same as /search).

        Returns:
            Dictionary with:
                - results: List of result dictionaries
                - citations: List of SourceCitation objects
        """
        await self._ensure_initialized()

        # Get query
        query = parameters.get("query")
        if not query and docs:
            query = docs[0].text if hasattr(docs[0], "text") else str(docs[0])

        if not query:
            raise ValueError("Query required")

        source_ids = parameters.get("source_ids", [])
        top_k = parameters.get("top_k", 10)
        node_type = parameters.get("node_type")

        # Embed query
        query_embedding = self._embeddings.embed_text([query], is_query=True)[0]

        # Search
        results = await self._vector_store.search(
            query_embedding=query_embedding.tolist(),
            source_ids=source_ids if source_ids else None,
            top_k=top_k,
            node_type=node_type,
        )

        # Build citations
        citations = []
        for r in results:
            metadata = r.get("metadata") or {}
            citation = SourceCitation(
                source_id=r["source_id"],
                node_id=r["node_id"],
                node_type=r["node_type"],
                title=metadata.get("title", r["node_id"]),
                filename=metadata.get("filename", "unknown"),
                content_preview=(r["content"] or "")[:200],
                page=metadata.get("page"),
                similarity=r["similarity"],
                image_url=metadata.get("image_url"),
                metadata=metadata,
            )
            citations.append(citation)

        return {
            "results": results,
            "citations": [c.model_dump() for c in citations],
            "query": query,
        }

    @requests(on="/delete_source")
    async def delete_source(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete all nodes for a DocumentSource.

        Args:
            docs: Ignored.
            parameters: Must include:
                - source_id: DocumentSource ID to delete

        Returns:
            Dictionary with deleted count.
        """
        await self._ensure_initialized()

        source_id = parameters.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")

        count = await self._vector_store.delete_by_source(source_id)
        self.logger.info(f"Deleted {count} nodes for source_id={source_id}")

        return {"deleted_count": count, "source_id": source_id}

    @requests(on="/source_stats")
    async def source_stats(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Get statistics for a DocumentSource.

        Args:
            docs: Ignored.
            parameters: Must include:
                - source_id: DocumentSource ID

        Returns:
            Dictionary with node counts by type.
        """
        await self._ensure_initialized()

        source_id = parameters.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")

        stats = await self._vector_store.get_source_stats(source_id)
        return {"source_id": source_id, **stats}

    async def close(self) -> None:
        """Close executor resources."""
        if self._vector_store:
            await self._vector_store.close()
        self.logger.info("VectorStoreExecutor closed")
