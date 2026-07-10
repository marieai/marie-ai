"""Vector Store Executor for DAG workflows.

This executor provides embed and store operations as nodes in Marie's
DAG workflow system. It integrates QwenVLEmbeddings for unified text+image
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
import json
import os
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from docarray import DocList
from docarray.documents import TextDoc

from marie import Executor, requests
from marie.executor.kb.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextChunk,
    chunk_text,
)
from marie.executor.marie_executor import MarieExecutor
from marie.logging_core.logger import MarieLogger
from marie.models.utils import initialize_device_settings
from marie.rag.models import SourceCitation
from marie.storage import StorageManager
from marie.utils.asset_util import s3_asset_path, split_filename

logger = MarieLogger("marie.executor.kb.vector_store_executor").logger


def _has_usable_text(doc: Any) -> bool:
    """True if ``doc`` already carries real text content (e.g. a TextDoc from
    a caller other than the DAG distributor)."""
    text = getattr(doc, "text", None)
    return bool(text and str(text).strip())


def _run_param(parameters: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Look up ``key`` in ``parameters["run_params"]`` (nested, studio-sent)
    first, falling back to a top-level ``parameters[key]``."""
    run_params = parameters.get("run_params") or {}
    if key in run_params:
        return run_params[key]
    return parameters.get(key, default)


def _connection_string_from_storage(storage: Optional[Dict[str, Any]]) -> Optional[str]:
    """Build a PostgreSQL DSN from the structured ``storage.psql`` block
    (provider/hostname/port/username/password/database) that Marie service
    configs pass to executors via ``with:``."""
    psql = (storage or {}).get("psql") or {}
    hostname = psql.get("hostname")
    if not hostname:
        return None
    auth = ""
    username = psql.get("username")
    if username:
        auth = quote(str(username), safe="")
        password = psql.get("password")
        if password:
            auth += f":{quote(str(password), safe='')}"
        auth += "@"
    port = psql.get("port", 5432)
    database = psql.get("database", "postgres")
    return f"postgresql://{auth}{hostname}:{port}/{database}"


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
            storage:
              psql:
                <<: *psql_conf_shared
              s3:
                <<: *s3_conf_shared
                enabled: True
            embedding_model: Qwen/Qwen3-VL-Embedding-2B
            embedding_dim: 1024
        ```
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        embedding_model: str = "Qwen/Qwen3-VL-Embedding-2B",
        embedding_dim: int = 1024,
        embedding_task: str = "retrieval",
        table_name: str = "kb_vectors",
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

        self._connection_string = (
            connection_string
            or _connection_string_from_storage(kwargs.get("storage"))
            or os.environ.get("PGVECTOR_CONNECTION_STRING")
        )
        if not self._connection_string:
            raise ValueError(
                "connection_string required. Provide a storage.psql config "
                "block, a connection_string argument, or the "
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
        from marie.embeddings.qwen import QwenVLEmbeddings

        self._embeddings = QwenVLEmbeddings(
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

        Called from DAG workflow after document extraction. The DAG framework
        is document-level and passes every node an AssetKeyDoc(uri) by
        reference rather than chaining a prior node's output - so when the
        incoming docs carry no usable text (the AssetKeyDoc case), this reads
        the extraction output persisted by TextExtractionExecutor for
        ref_id/ref_type, chunks it, and embeds the chunks directly. Callers
        that pass real TextDocs with content already set keep working as
        before.

        Args:
            docs: DocList of documents to embed and store, or a single
                AssetKeyDoc-like placeholder to resolve from storage.
            parameters: Must include:
                - source_id: DocumentSource ID
                Optional:
                - index_name: Index/collection name (default: 'default')
                - node_type: Type of nodes (text/image/document)
                - ref_doc_id: Reference document ID
                - ref_id / ref_type: required when resolving extraction
                  output from storage (AssetKeyDoc case)
                - chunk_size / chunk_overlap: character chunk window,
                  looked up in parameters["run_params"] first, then
                  top-level parameters. Defaults 1024/200.

        Returns:
            Input docs (passthrough for pipeline chaining).
        """
        await self._ensure_initialized()

        source_id = parameters.get("source_id")
        if not source_id:
            raise ValueError("source_id is required in parameters")

        index_name = parameters.get("index_name", "default")
        node_type = parameters.get("node_type", "text")
        ref_doc_id = parameters.get("ref_doc_id")

        if docs and not any(_has_usable_text(d) for d in docs):
            ref_id = parameters.get("ref_id")
            ref_type = parameters.get("ref_type") or "document"
            if not ref_id:
                raise ValueError(
                    "ref_id is required in parameters to resolve extraction "
                    "output for embedding (docs carried no usable text)"
                )

            full_text, page_ranges = await asyncio.to_thread(
                self._resolve_extraction_output, ref_id, ref_type
            )

            chunk_size = int(_run_param(parameters, "chunk_size", DEFAULT_CHUNK_SIZE))
            chunk_overlap = int(
                _run_param(parameters, "chunk_overlap", DEFAULT_CHUNK_OVERLAP)
            )
            segmentation_mode = _run_param(parameters, "segmentation_mode", "character")
            if segmentation_mode not in (None, "character"):
                self.logger.warning(
                    f"Unsupported segmentation_mode={segmentation_mode!r}, "
                    "falling back to character-based chunking"
                )

            chunks = chunk_text(
                full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            if not chunks:
                raise ValueError(
                    f"Extraction output for ref_id={ref_id} produced no "
                    "chunkable text"
                )

            stored_count = await self._embed_and_store_chunks(
                chunks=chunks,
                page_ranges=page_ranges,
                ref_id=ref_id,
                source_id=source_id,
                index_name=index_name,
            )
            self.logger.info(
                f"Stored {stored_count}/{len(chunks)} chunks for ref_id={ref_id}"
            )
            await self._update_submission_status(parameters, stored_count, len(chunks))
            return docs

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
                    index_name=index_name,
                )
                stored_count += 1

            except Exception as e:
                self.logger.error(f"Failed to process doc {doc.id}: {e}")

        self.logger.info(f"Stored {stored_count}/{len(docs)} documents")

        # Update submission document status if this is a submission workflow
        await self._update_submission_status(parameters, stored_count, len(docs))

        return docs

    def _resolve_extraction_output(
        self, ref_id: str, ref_type: str
    ) -> Tuple[str, List[Tuple[int, int, int]]]:
        """Read the extraction output TextExtractionExecutor persisted for
        ref_id/ref_type and flatten it to plain text with page offsets.

        Mirrors the read path document_annotator_executor.py uses via
        prepare_asset_directory()/download_asset() to fetch the same
        ``{ref_id}.meta.json`` written by ExtractPipeline.store_metadata() +
        store_assets() (marie/pipe/extract_pipeline.py), but reads the S3
        object directly into memory instead of restoring a full working
        directory - the EMBED stage only needs the OCR text, not the frames.

        The S3 key is rebuilt with the same split_filename()/s3_asset_path()
        helpers the writer uses (rather than download_asset's
        f"{ref_id}.meta.json", which mis-resolves when ref_id is a
        slash-containing S3 key, as it is for KB documents).
        """
        filename, _, _ = split_filename(ref_id)
        uri = f"{s3_asset_path(ref_id, ref_type)}/{filename}.meta.json"

        connected = StorageManager.ensure_connection("s3://", silence_exceptions=True)
        if not connected:
            raise RuntimeError(
                f"Unable to connect to S3 to resolve extraction output for "
                f"ref_id={ref_id}"
            )
        if not StorageManager.exists(uri):
            raise ValueError(
                f"No extraction output found for ref_id={ref_id}, "
                f"ref_type={ref_type} (expected {uri}). Has the EXTRACT stage "
                "run for this document?"
            )

        raw = StorageManager.read(uri)
        try:
            metadata = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Extraction output at {uri} is not valid JSON: {e}"
            ) from e

        ocr_results = metadata.get("ocr")
        if not ocr_results:
            raise ValueError(
                f"Extraction output at {uri} has no OCR results to embed for "
                f"ref_id={ref_id}"
            )

        return self._flatten_ocr_pages(ocr_results)

    @staticmethod
    def _flatten_ocr_pages(
        ocr_results: List[Dict[str, Any]],
    ) -> Tuple[str, List[Tuple[int, int, int]]]:
        """Concatenate per-page OCR line text (same flattening
        TextExtractionExecutor.persist() uses for asset tracking) and return
        (full_text, page_ranges), where page_ranges holds
        (page_index, start, end) character offsets into full_text."""
        parts: List[str] = []
        page_ranges: List[Tuple[int, int, int]] = []
        cursor = 0
        for page_idx, page_result in enumerate(ocr_results):
            lines = page_result.get("lines") or []
            page_text = "\n".join(line["text"] for line in lines if "text" in line)
            if parts:
                parts.append("\n")
                cursor += 1
            start = cursor
            parts.append(page_text)
            cursor += len(page_text)
            page_ranges.append((page_idx, start, cursor))
        return "".join(parts), page_ranges

    @staticmethod
    def _page_for_offset(
        page_ranges: List[Tuple[int, int, int]], offset: int
    ) -> Optional[int]:
        """Return the page index whose [start, end) range contains offset."""
        for page_idx, start, end in page_ranges:
            if start <= offset < end:
                return page_idx
        return page_ranges[-1][0] if page_ranges else None

    async def _embed_and_store_chunks(
        self,
        chunks: List[TextChunk],
        page_ranges: List[Tuple[int, int, int]],
        ref_id: str,
        source_id: str,
        index_name: str,
    ) -> int:
        """Embed chunk texts and batch-insert them as text nodes, ref'd to
        the source document (ref_doc_id = ref_id)."""
        texts = [c.text for c in chunks]
        embeddings = self._embeddings.embed_text(texts, is_query=False)

        nodes = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata: Dict[str, Any] = {"chunk_index": idx}
            page = self._page_for_offset(page_ranges, chunk.start)
            if page is not None:
                metadata["page"] = page

            nodes.append(
                {
                    "node_id": f"{ref_id}_{idx}",
                    "embedding": embedding.tolist(),
                    "content": chunk.text,
                    "node_type": "text",
                    "metadata": metadata,
                    "ref_doc_id": ref_id,
                }
            )

        return await self._vector_store.add_nodes_batch(
            nodes=nodes,
            source_id=source_id,
            index_name=index_name,
        )

    async def _update_submission_status(
        self,
        parameters: Dict[str, Any],
        stored_count: int,
        total_count: int,
    ) -> None:
        """Update submission document indexing status after embedding.

        Only updates if workflow_id and ref_id are present in parameters,
        indicating this is part of a submission document workflow.
        """
        workflow_id = parameters.get("workflow_id")
        ref_id = parameters.get("ref_id")

        if not workflow_id or not ref_id:
            return

        try:
            import os

            postgres_url = os.getenv("DATABASE_URL")
            if not postgres_url:
                self.logger.warning(
                    "DATABASE_URL not set, skipping submission status update"
                )
                return

            from marie.storage.submission import SubmissionStorage

            storage = SubmissionStorage(postgres_url)

            if stored_count > 0:
                # Success - update both workflow and document status
                storage.update_workflow_status(
                    workflow_id=workflow_id,
                    status="completed",
                )
                storage.update_document_indexing_status(
                    document_id=ref_id,
                    status="indexed",
                )
                self.logger.info(
                    f"Updated submission document {ref_id} status to 'indexed'"
                )
            else:
                # No documents stored - mark as failed
                storage.update_workflow_status(
                    workflow_id=workflow_id,
                    status="failed",
                    error="No documents were stored",
                )
                storage.update_document_indexing_status(
                    document_id=ref_id,
                    status="failed",
                    error="No documents were stored",
                )
                self.logger.warning(
                    f"No documents stored for {ref_id}, marked as failed"
                )

            storage.stop()

        except Exception as e:
            self.logger.error(f"Failed to update submission status: {e}")

    @requests(on="/search")
    async def search(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Semantic search over stored documents.

        Args:
            docs: DocList with query document (uses first doc's text as query).
            parameters: Search parameters:
                - query: Override query text
                - index_name: Index/collection to search (default: 'default')
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

        index_name = parameters.get("index_name", "default")
        source_ids = parameters.get("source_ids", [])
        top_k = parameters.get("top_k", 10)
        node_type = parameters.get("node_type")

        self.logger.info(
            f"Searching: query='{query[:50]}...', index={index_name}, top_k={top_k}"
        )

        # Embed query
        query_embedding = self._embeddings.embed_text([query], is_query=True)[0]

        # Search
        results = await self._vector_store.search(
            query_embedding=query_embedding.tolist(),
            source_ids=source_ids if source_ids else None,
            top_k=top_k,
            node_type=node_type,
            index_name=index_name,
        )

        # Convert to DocList
        rows = [
            {
                "id": r["node_id"],
                "content": r["content"] or "",
                "score": r["similarity"],
                "source_id": r["source_id"],
                "node_type": r["node_type"],
                "index_name": r.get("index_name", index_name),
                "ref_doc_id": r.get("ref_doc_id"),
            }
            for r in results
        ]

        self.logger.info(f"Found {len(rows)} results")
        return {"results": rows, "count": len(rows), "index_name": index_name}

    @requests(on="/embed")
    async def embed(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Embed texts and return the vectors in the response parameters.

        This is a pure model-execution endpoint: the gateway's ``kb`` command
        calls it to embed a query before searching pgvector itself. Returning
        the vectors as a small dict (which the worker places in the response
        ``parameters``) avoids serializing embeddings through a doc schema.

        Args:
            docs: Ignored.
            parameters:
                - texts: List of strings to embed (required).
                - is_query: Whether to use the query adapter. Default: False.

        Returns:
            Dictionary with ``embeddings`` (list of vectors) and ``dim``.
        """
        await self._ensure_initialized()

        texts = parameters.get("texts")
        if not texts:
            raise ValueError("texts list is required in parameters")

        is_query = bool(parameters.get("is_query", False))
        embeddings = self._embeddings.embed_text(texts, is_query=is_query)
        embeddings = [
            e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings
        ]

        return {"embeddings": embeddings, "dim": self._embedding_dim}

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

        index_name = parameters.get("index_name", "default")
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
            index_name=index_name,
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
                Optional:
                - index_name: Delete only from specific index

        Returns:
            Dictionary with deleted count.
        """
        await self._ensure_initialized()

        source_id = parameters.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")

        index_name = parameters.get("index_name")
        count = await self._vector_store.delete_by_source(
            source_id, index_name=index_name
        )
        self.logger.info(f"Deleted {count} nodes for source_id={source_id}")

        return {
            "deleted_count": count,
            "source_id": source_id,
            "index_name": index_name,
        }

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
                Optional:
                - index_name: Filter to specific index

        Returns:
            Dictionary with node counts by type.
        """
        await self._ensure_initialized()

        source_id = parameters.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")

        index_name = parameters.get("index_name")
        stats = await self._vector_store.get_source_stats(
            source_id, index_name=index_name
        )
        return {"source_id": source_id, "index_name": index_name, **stats}

    @requests(on="/index_stats")
    async def index_stats(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Get statistics for an index/collection.

        Args:
            docs: Ignored.
            parameters: Must include:
                - index_name: Index name

        Returns:
            Dictionary with node counts by type and source.
        """
        await self._ensure_initialized()

        index_name = parameters.get("index_name")
        if not index_name:
            raise ValueError("index_name is required")

        stats = await self._vector_store.get_index_stats(index_name)
        return {"index_name": index_name, **stats}

    @requests(on="/delete_index")
    async def delete_index(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete all nodes for an index/collection.

        Args:
            docs: Ignored.
            parameters: Must include:
                - index_name: Index name to delete
                - confirm: Must be True to proceed

        Returns:
            Dictionary with deleted count.
        """
        await self._ensure_initialized()

        index_name = parameters.get("index_name")
        if not index_name:
            raise ValueError("index_name is required")

        if index_name == "default":
            confirm = parameters.get("confirm", False)
            if not confirm:
                raise ValueError("Set confirm=True to delete the default index")

        count = await self._vector_store.delete_by_index(index_name)
        self.logger.info(f"Deleted {count} nodes for index_name={index_name}")

        return {"deleted_count": count, "index_name": index_name}

    @requests(on="/hybrid_search")
    async def hybrid_search(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Hybrid search combining vector and full-text search.

        Uses RRF (Reciprocal Rank Fusion) to combine results.

        Args:
            docs: DocList with query document.
            parameters: Search parameters:
                - query: Query text
                - index_name: Index/collection to search (default: 'default')
                - source_ids: Filter to specific sources
                - top_k: Number of results (default: 10)
                - alpha: Weight for vector vs text (0-1, default: 0.5)

        Returns:
            DocList of matching documents with combined scores.
        """
        await self._ensure_initialized()

        query = parameters.get("query")
        if not query and docs:
            query = docs[0].text if hasattr(docs[0], "text") else str(docs[0])

        if not query:
            raise ValueError("Query required: provide in parameters or docs")

        index_name = parameters.get("index_name", "default")
        source_ids = parameters.get("source_ids", [])
        top_k = parameters.get("top_k", 10)
        alpha = parameters.get("alpha", 0.5)
        node_type = parameters.get("node_type")

        self.logger.info(
            f"Hybrid search: query='{query[:50]}...', index={index_name}, top_k={top_k}, alpha={alpha}"
        )

        # Embed query
        query_embedding = self._embeddings.embed_text([query], is_query=True)[0]

        # Hybrid search
        results = await self._vector_store.hybrid_search(
            query_embedding=query_embedding.tolist(),
            query_text=query,
            source_ids=source_ids if source_ids else None,
            top_k=top_k,
            alpha=alpha,
            node_type=node_type,
            index_name=index_name,
        )

        rows = [
            {
                "id": r["node_id"],
                "content": r["content"] or "",
                "score": r.get("rrf_score", r["similarity"]),
                "text_score": r.get("text_score", 0),
                "rrf_score": r.get("rrf_score", 0),
                "source_id": r["source_id"],
                "node_type": r["node_type"],
                "index_name": r.get("index_name", index_name),
                "ref_doc_id": r.get("ref_doc_id"),
            }
            for r in results
        ]

        self.logger.info(f"Found {len(rows)} results via hybrid search")
        return {"results": rows, "count": len(rows), "index_name": index_name}

    @requests(on="/ingest_batch")
    async def ingest_batch(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Batch ingest documents for high throughput.

        Args:
            docs: Ignored (uses parameters).
            parameters: Must include:
                - source_id: DocumentSource ID
                - texts: List of text strings to ingest
                Optional:
                - index_name: Index/collection name (default: 'default')
                - metadata: Shared metadata for all texts
                - ref_doc_id: Reference document ID
                - batch_size: Batch size for insert (default: 100)

        Returns:
            Dictionary with ingestion count.
        """
        await self._ensure_initialized()

        source_id = parameters.get("source_id")
        if not source_id:
            raise ValueError("source_id is required")

        texts = parameters.get("texts", [])
        if not texts:
            raise ValueError("texts list is required")

        index_name = parameters.get("index_name", "default")
        metadata = parameters.get("metadata", {})
        ref_doc_id = parameters.get("ref_doc_id")
        batch_size = parameters.get("batch_size", 100)

        self.logger.info(
            f"Batch ingesting {len(texts)} texts for source_id={source_id}, index={index_name}"
        )

        # Embed all texts
        embeddings = self._embeddings.embed_text(texts, is_query=False)

        # Prepare nodes
        import uuid

        nodes = []
        for text, embedding in zip(texts, embeddings):
            node_id = f"{source_id}_{uuid.uuid4().hex[:8]}"
            nodes.append(
                {
                    "node_id": node_id,
                    "embedding": embedding.tolist(),
                    "content": text,
                    "node_type": "text",
                    "metadata": metadata,
                    "ref_doc_id": ref_doc_id,
                }
            )

        # Batch insert
        count = await self._vector_store.add_nodes_batch(
            nodes=nodes,
            source_id=source_id,
            batch_size=batch_size,
            index_name=index_name,
        )

        self.logger.info(f"Batch ingested {count} documents")
        return {"count": count, "source_id": source_id, "index_name": index_name}

    @requests(on="/get_nodes")
    async def get_nodes(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> DocList[TextDoc]:
        """Retrieve nodes by IDs.

        Args:
            docs: Ignored.
            parameters:
                - node_ids: List of node IDs to retrieve
                - index_name: Optional index filter

        Returns:
            DocList of matching nodes.
        """
        await self._ensure_initialized()

        node_ids = parameters.get("node_ids", [])
        if not node_ids:
            raise ValueError("node_ids list is required")

        index_name = parameters.get("index_name")
        nodes = await self._vector_store.aget_nodes(
            node_ids=node_ids, index_name=index_name
        )

        result_docs = DocList[TextDoc]()
        for node in nodes:
            doc = TextDoc(
                id=node.node_id,
                text=node.text or "",
            )
            doc.metadata = node.metadata
            result_docs.append(doc)

        return result_docs

    @requests(on="/delete_nodes")
    async def delete_nodes(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Delete nodes by IDs.

        Args:
            docs: Ignored.
            parameters:
                - node_ids: List of node IDs to delete
                - index_name: Optional index filter

        Returns:
            Dictionary with deletion status.
        """
        await self._ensure_initialized()

        node_ids = parameters.get("node_ids", [])
        if not node_ids:
            raise ValueError("node_ids list is required")

        index_name = parameters.get("index_name")
        await self._vector_store.adelete_nodes(node_ids=node_ids, index_name=index_name)
        self.logger.info(f"Deleted {len(node_ids)} nodes")

        return {"deleted_count": len(node_ids), "index_name": index_name}

    @requests(on="/count")
    async def count_nodes(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Count nodes in the vector store.

        Args:
            docs: Ignored.
            parameters:
                - index_name: Optional index filter
                - source_id: Optional source filter
                - node_type: Optional type filter

        Returns:
            Dictionary with node count.
        """
        await self._ensure_initialized()

        index_name = parameters.get("index_name")
        source_id = parameters.get("source_id")
        node_type = parameters.get("node_type")

        count = await self._vector_store.count_nodes(
            source_id=source_id,
            node_type=node_type,
            index_name=index_name,
        )

        return {
            "count": count,
            "index_name": index_name,
            "source_id": source_id,
            "node_type": node_type,
        }

    @requests(on="/clear")
    async def clear_store(
        self,
        docs: DocList[TextDoc],
        parameters: Dict[str, Any],
        **kwargs,
    ) -> Dict[str, Any]:
        """Clear all nodes from the vector store.

        WARNING: This deletes ALL data. Use with caution.

        Args:
            docs: Ignored.
            parameters:
                - confirm: Must be True to proceed

        Returns:
            Dictionary with status.
        """
        await self._ensure_initialized()

        confirm = parameters.get("confirm", False)
        if not confirm:
            raise ValueError("Set confirm=True to clear all data")

        await self._vector_store.aclear()
        self.logger.warning(f"Cleared all nodes from {self._table_name}")

        return {"cleared": True, "table": self._table_name}

    async def close(self) -> None:
        """Close executor resources."""
        if self._vector_store:
            await self._vector_store.close()
        self.logger.info("VectorStoreExecutor closed")
