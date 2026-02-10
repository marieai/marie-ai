"""RAG API routes for document retrieval and management.

This module provides REST API endpoints for the RAG system, including
document source management and query execution.

Endpoints:
    POST /api/rag/sources              - Create source, start ingestion
    GET  /api/rag/sources              - List user's sources
    GET  /api/rag/sources/{id}         - Get source details + status
    DELETE /api/rag/sources/{id}       - Delete source and nodes
    POST /api/rag/sources/{id}/reingest - Re-process source
    POST /api/rag/query                - RAG query with citations
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.logging_core.logger import MarieLogger
from marie.rag.models import (
    DocumentSource,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievalResult,
    SourceCitation,
)

logger = MarieLogger("marie.api.routes.rag").logger


# -------------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------------


class CreateSourceRequest(BaseModel):
    """Request to create a new document source."""

    name: str = Field(..., description="Human-readable name for the source")
    workspace_id: Optional[str] = Field(
        default=None, description="Optional workspace ID"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class CreateSourceResponse(BaseModel):
    """Response after creating a source."""

    source: DocumentSource
    upload_url: Optional[str] = Field(
        default=None, description="Pre-signed URL for file upload (if applicable)"
    )


class ListSourcesResponse(BaseModel):
    """Response listing document sources."""

    sources: List[DocumentSource]
    total: int


class SourceStatsResponse(BaseModel):
    """Statistics for a document source."""

    source_id: str
    total_nodes: int
    by_type: Dict[str, int]


class ReingestResponse(BaseModel):
    """Response after triggering re-ingestion."""

    source_id: str
    status: str
    message: str


class IngestRequest(BaseModel):
    """Request for direct text ingestion."""

    texts: List[str] = Field(..., description="Text chunks to ingest", min_length=1)
    source_id: str = Field(..., description="Source ID for the documents")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata to attach to all chunks"
    )
    ref_doc_id: Optional[str] = Field(default=None, description="Reference document ID")


class IngestResponse(BaseModel):
    """Response after ingestion."""

    source_id: str
    count: int
    message: str


class SearchRequest(BaseModel):
    """Request for semantic search (simpler than full RAG query)."""

    query: str = Field(..., description="Search query")
    source_ids: Optional[List[str]] = Field(
        default=None, description="Filter to specific sources"
    )
    top_k: int = Field(default=10, description="Number of results")
    use_hybrid: bool = Field(
        default=False, description="Use hybrid search (vector + keyword)"
    )


class SearchResponse(BaseModel):
    """Response from semantic search."""

    results: List[Dict[str, Any]]
    query: str
    total_results: int


class DeleteRequest(BaseModel):
    """Request for deletion."""

    source_id: Optional[str] = Field(
        default=None, description="Delete all nodes from this source"
    )
    node_ids: Optional[List[str]] = Field(
        default=None, description="Delete specific node IDs"
    )
    ref_doc_id: Optional[str] = Field(
        default=None, description="Delete by reference document ID"
    )


class DeleteResponse(BaseModel):
    """Response after deletion."""

    deleted_count: int
    message: str


class QueryRequest(BaseModel):
    """Request for RAG query."""

    query: str = Field(..., description="Search query")
    source_ids: Optional[List[str]] = Field(
        default=None, description="Filter to specific sources"
    )
    top_k: int = Field(default=10, description="Number of results")
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Conversation history for context"
    )


class QueryResponse(BaseModel):
    """Response from RAG query."""

    results: List[Dict[str, Any]]
    sources: List[SourceCitation]
    query: str
    rephrased_query: Optional[str] = None
    total_results: int


# -------------------------------------------------------------------------
# Router Class
# -------------------------------------------------------------------------


class RAGRouter:
    """RAG API router.

    Provides methods that can be integrated with FastAPI, Flask, or other
    web frameworks. Each method corresponds to an API endpoint.

    Example with FastAPI:
        ```python
        from fastapi import FastAPI, Depends, HTTPException
        from marie.api.routes.rag import RAGRouter, CreateSourceRequest

        app = FastAPI()
        rag_router = RAGRouter(
            vector_store=store,
            embeddings=embeddings,
            retriever=retriever,
        )


        @app.post("/api/rag/sources")
        async def create_source(
            request: CreateSourceRequest,
            user_id: str = Depends(get_current_user),
        ):
            return await rag_router.create_source(request, user_id)


        @app.get("/api/rag/sources")
        async def list_sources(user_id: str = Depends(get_current_user)):
            return await rag_router.list_sources(user_id)
        ```
    """

    def __init__(
        self,
        vector_store: Any = None,
        embeddings: Any = None,
        retriever: Any = None,
        source_storage: Optional[Dict[str, DocumentSource]] = None,
    ):
        """Initialize RAG router.

        Args:
            vector_store: PGVectorStore instance.
            embeddings: Embeddings instance.
            retriever: RAGRetriever instance.
            source_storage: Optional dict for in-memory source storage.
                           In production, use a database.
        """
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._retriever = retriever
        # In-memory storage for demo; replace with database in production
        self._sources: Dict[str, DocumentSource] = source_storage or {}

    async def create_source(
        self,
        request: CreateSourceRequest,
        user_id: str,
    ) -> CreateSourceResponse:
        """Create a new document source.

        Args:
            request: Source creation request.
            user_id: ID of the user creating the source.

        Returns:
            CreateSourceResponse with the new source.
        """
        source_id = str(uuid.uuid4())
        now = datetime.utcnow()

        source = DocumentSource(
            id=source_id,
            name=request.name,
            user_id=user_id,
            workspace_id=request.workspace_id,
            status="pending",
            file_count=0,
            node_count=0,
            metadata=request.metadata or {},
            created_at=now,
            updated_at=now,
        )

        self._sources[source_id] = source
        logger.info(f"Created source: {source_id} for user {user_id}")

        return CreateSourceResponse(
            source=source,
            upload_url=None,  # Implement pre-signed URL generation if needed
        )

    async def list_sources(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> ListSourcesResponse:
        """List document sources for a user.

        Args:
            user_id: User ID to filter by.
            workspace_id: Optional workspace filter.

        Returns:
            ListSourcesResponse with matching sources.
        """
        sources = [
            s
            for s in self._sources.values()
            if s.user_id == user_id
            and (workspace_id is None or s.workspace_id == workspace_id)
        ]

        return ListSourcesResponse(
            sources=sources,
            total=len(sources),
        )

    async def get_source(
        self,
        source_id: str,
        user_id: str,
    ) -> DocumentSource:
        """Get a specific document source.

        Args:
            source_id: Source ID.
            user_id: User ID for authorization.

        Returns:
            DocumentSource if found.

        Raises:
            ValueError: If source not found or unauthorized.
        """
        source = self._sources.get(source_id)
        if not source:
            raise ValueError(f"Source not found: {source_id}")
        if source.user_id != user_id:
            raise ValueError(f"Unauthorized access to source: {source_id}")

        # Get node count from vector store if available
        if self._vector_store:
            try:
                stats = await self._vector_store.get_source_stats(source_id)
                source.node_count = stats.get("total", 0)
            except Exception as e:
                logger.warning(f"Could not get source stats: {e}")

        return source

    async def delete_source(
        self,
        source_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Delete a document source and its nodes.

        Args:
            source_id: Source ID to delete.
            user_id: User ID for authorization.

        Returns:
            Dictionary with deletion status.
        """
        source = self._sources.get(source_id)
        if not source:
            raise ValueError(f"Source not found: {source_id}")
        if source.user_id != user_id:
            raise ValueError(f"Unauthorized access to source: {source_id}")

        # Delete from vector store
        deleted_nodes = 0
        if self._vector_store:
            try:
                deleted_nodes = await self._vector_store.delete_by_source(source_id)
            except Exception as e:
                logger.error(f"Error deleting source nodes: {e}")

        # Remove from storage
        del self._sources[source_id]
        logger.info(f"Deleted source: {source_id}, {deleted_nodes} nodes removed")

        return {
            "source_id": source_id,
            "deleted": True,
            "nodes_deleted": deleted_nodes,
        }

    async def reingest_source(
        self,
        source_id: str,
        user_id: str,
    ) -> ReingestResponse:
        """Trigger re-ingestion of a document source.

        Args:
            source_id: Source ID to re-ingest.
            user_id: User ID for authorization.

        Returns:
            ReingestResponse with status.
        """
        source = self._sources.get(source_id)
        if not source:
            raise ValueError(f"Source not found: {source_id}")
        if source.user_id != user_id:
            raise ValueError(f"Unauthorized access to source: {source_id}")

        # Update status
        source.status = "processing"
        source.updated_at = datetime.utcnow()

        # In production, trigger async ingestion job here
        # For now, just return processing status
        logger.info(f"Triggered re-ingestion for source: {source_id}")

        return ReingestResponse(
            source_id=source_id,
            status="processing",
            message="Re-ingestion started. Check status endpoint for progress.",
        )

    async def get_source_stats(
        self,
        source_id: str,
        user_id: str,
    ) -> SourceStatsResponse:
        """Get statistics for a document source.

        Args:
            source_id: Source ID.
            user_id: User ID for authorization.

        Returns:
            SourceStatsResponse with node counts.
        """
        source = self._sources.get(source_id)
        if not source:
            raise ValueError(f"Source not found: {source_id}")
        if source.user_id != user_id:
            raise ValueError(f"Unauthorized access to source: {source_id}")

        stats = {"total": 0, "by_type": {}}
        if self._vector_store:
            try:
                stats = await self._vector_store.get_source_stats(source_id)
            except Exception as e:
                logger.warning(f"Could not get source stats: {e}")

        return SourceStatsResponse(
            source_id=source_id,
            total_nodes=stats.get("total", 0),
            by_type=stats.get("by_type", {}),
        )

    async def ingest(
        self,
        request: IngestRequest,
        user_id: str,
    ) -> IngestResponse:
        """Ingest text documents directly.

        Args:
            request: Ingest request with texts.
            user_id: User ID for authorization.

        Returns:
            IngestResponse with count.
        """
        source = self._sources.get(request.source_id)
        if source and source.user_id != user_id:
            raise ValueError(f"Unauthorized access to source: {request.source_id}")

        if not self._embeddings:
            raise ValueError("Embeddings not configured")

        if not self._vector_store:
            raise ValueError("Vector store not configured")

        logger.info(
            f"Ingesting {len(request.texts)} texts to source {request.source_id}"
        )

        # Embed texts
        embeddings = self._embeddings.embed_text(request.texts, is_query=False)

        # Prepare nodes
        import uuid as uuid_module

        nodes = []
        for text, embedding in zip(request.texts, embeddings):
            node_id = f"{request.source_id}_{uuid_module.uuid4().hex[:8]}"
            nodes.append(
                {
                    "node_id": node_id,
                    "embedding": embedding.tolist(),
                    "content": text,
                    "node_type": "text",
                    "metadata": request.metadata or {},
                    "ref_doc_id": request.ref_doc_id,
                }
            )

        # Batch insert
        count = await self._vector_store.add_nodes_batch(
            nodes=nodes,
            source_id=request.source_id,
        )

        # Update source if it exists
        if source:
            source.node_count += count
            source.updated_at = datetime.utcnow()

        return IngestResponse(
            source_id=request.source_id,
            count=count,
            message=f"Successfully ingested {count} documents.",
        )

    async def search(
        self,
        request: SearchRequest,
        user_id: str,
    ) -> SearchResponse:
        """Semantic search over documents.

        Args:
            request: Search request.
            user_id: User ID for authorization.

        Returns:
            SearchResponse with results.
        """
        if not self._embeddings:
            raise ValueError("Embeddings not configured")

        if not self._vector_store:
            raise ValueError("Vector store not configured")

        # Validate source access
        if request.source_ids:
            for sid in request.source_ids:
                source = self._sources.get(sid)
                if source and source.user_id != user_id:
                    raise ValueError(f"Unauthorized access to source: {sid}")

        logger.info(
            f"Searching: query='{request.query[:50]}...', top_k={request.top_k}"
        )

        # Embed query
        query_embedding = self._embeddings.embed_text([request.query], is_query=True)[0]

        # Search
        if request.use_hybrid:
            results = await self._vector_store.hybrid_search(
                query_embedding=query_embedding.tolist(),
                query_text=request.query,
                source_ids=request.source_ids,
                top_k=request.top_k,
            )
        else:
            results = await self._vector_store.search(
                query_embedding=query_embedding.tolist(),
                source_ids=request.source_ids,
                top_k=request.top_k,
            )

        return SearchResponse(
            results=results,
            query=request.query,
            total_results=len(results),
        )

    async def delete(
        self,
        request: DeleteRequest,
        user_id: str,
    ) -> DeleteResponse:
        """Delete documents from the store.

        Args:
            request: Delete request.
            user_id: User ID for authorization.

        Returns:
            DeleteResponse with count.
        """
        if not self._vector_store:
            raise ValueError("Vector store not configured")

        if not any([request.source_id, request.node_ids, request.ref_doc_id]):
            raise ValueError(
                "At least one of source_id, node_ids, or ref_doc_id required"
            )

        # Check authorization
        if request.source_id:
            source = self._sources.get(request.source_id)
            if source and source.user_id != user_id:
                raise ValueError(f"Unauthorized access to source: {request.source_id}")

        deleted = 0

        if request.source_id and not request.node_ids and not request.ref_doc_id:
            # Delete entire source
            deleted = await self._vector_store.delete_by_source(request.source_id)
        elif request.ref_doc_id:
            # Delete by ref_doc_id
            await self._vector_store.adelete(request.ref_doc_id)
            deleted = -1
        elif request.node_ids:
            # Delete specific nodes
            await self._vector_store.adelete_nodes(node_ids=request.node_ids)
            deleted = len(request.node_ids)

        if deleted == -1:
            message = f"Deleted documents with ref_doc_id '{request.ref_doc_id}'."
        else:
            message = f"Deleted {deleted} documents."

        return DeleteResponse(
            deleted_count=deleted if deleted >= 0 else 0,
            message=message,
        )

    async def query(
        self,
        request: QueryRequest,
        user_id: str,
    ) -> QueryResponse:
        """Execute a RAG query.

        Args:
            request: Query request.
            user_id: User ID for authorization.

        Returns:
            QueryResponse with results and citations.
        """
        if not self._retriever:
            raise ValueError("Retriever not configured")

        # Validate source access if specific sources requested
        if request.source_ids:
            for sid in request.source_ids:
                source = self._sources.get(sid)
                if source and source.user_id != user_id:
                    raise ValueError(f"Unauthorized access to source: {sid}")

        # Perform retrieval
        result = await self._retriever.aretrieve(
            query=request.query,
            source_ids=request.source_ids,
            top_k=request.top_k,
            chat_history=request.chat_history,
        )

        return QueryResponse(
            results=result.nodes,
            sources=result.sources,
            query=result.query,
            rephrased_query=result.rephrased_query,
            total_results=len(result.nodes),
        )


# -------------------------------------------------------------------------
# FastAPI Router Factory
# -------------------------------------------------------------------------


def create_fastapi_router(
    vector_store: Any = None,
    embeddings: Any = None,
    retriever: Any = None,
    prefix: str = "/api/rag",
):
    """Create a FastAPI router with RAG endpoints.

    Args:
        vector_store: PGVectorStore instance.
        embeddings: Embeddings instance.
        retriever: RAGRetriever instance.
        prefix: URL prefix for routes.

    Returns:
        FastAPI APIRouter instance.

    Example:
        ```python
        from fastapi import FastAPI
        from marie.api.routes.rag import create_fastapi_router
        from marie.rag import RAGRetriever
        from marie.vector_stores.pgvector import PGVectorStore
        from marie.embeddings.jina import JinaEmbeddingsV4

        # Initialize components
        store = PGVectorStore(connection_string="...")
        embeddings = JinaEmbeddingsV4()
        retriever = RAGRetriever(vector_store=store, embeddings=embeddings)

        # Create router
        rag_router = create_fastapi_router(
            vector_store=store,
            embeddings=embeddings,
            retriever=retriever,
        )

        # Mount in app
        app = FastAPI()
        app.include_router(rag_router)
        ```
    """
    try:
        from fastapi import APIRouter, Depends, HTTPException, Query

        api_router = APIRouter(prefix=prefix, tags=["rag"])
        rag = RAGRouter(
            vector_store=vector_store,
            embeddings=embeddings,
            retriever=retriever,
        )

        # Placeholder for user auth - implement based on your auth system
        async def get_current_user() -> str:
            # In production, extract user from JWT/session
            return "default_user"

        @api_router.post("/sources", response_model=CreateSourceResponse)
        async def create_source(
            request: CreateSourceRequest,
            user_id: str = Depends(get_current_user),
        ):
            return await rag.create_source(request, user_id)

        @api_router.get("/sources", response_model=ListSourcesResponse)
        async def list_sources(
            workspace_id: Optional[str] = Query(None),
            user_id: str = Depends(get_current_user),
        ):
            return await rag.list_sources(user_id, workspace_id)

        @api_router.get("/sources/{source_id}", response_model=DocumentSource)
        async def get_source(
            source_id: str,
            user_id: str = Depends(get_current_user),
        ):
            try:
                return await rag.get_source(source_id, user_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api_router.delete("/sources/{source_id}")
        async def delete_source(
            source_id: str,
            user_id: str = Depends(get_current_user),
        ):
            try:
                return await rag.delete_source(source_id, user_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api_router.post(
            "/sources/{source_id}/reingest", response_model=ReingestResponse
        )
        async def reingest_source(
            source_id: str,
            user_id: str = Depends(get_current_user),
        ):
            try:
                return await rag.reingest_source(source_id, user_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api_router.get(
            "/sources/{source_id}/stats", response_model=SourceStatsResponse
        )
        async def get_source_stats(
            source_id: str,
            user_id: str = Depends(get_current_user),
        ):
            try:
                return await rag.get_source_stats(source_id, user_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api_router.post("/query", response_model=QueryResponse)
        async def query(
            request: QueryRequest,
            user_id: str = Depends(get_current_user),
        ):
            try:
                return await rag.query(request, user_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @api_router.post("/ingest", response_model=IngestResponse)
        async def ingest(
            request: IngestRequest,
            user_id: str = Depends(get_current_user),
        ):
            """Ingest text documents directly into the vector store."""
            try:
                return await rag.ingest(request, user_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @api_router.post("/search", response_model=SearchResponse)
        async def search(
            request: SearchRequest,
            user_id: str = Depends(get_current_user),
        ):
            """Semantic search over documents."""
            try:
                return await rag.search(request, user_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @api_router.post("/delete", response_model=DeleteResponse)
        async def delete(
            request: DeleteRequest,
            user_id: str = Depends(get_current_user),
        ):
            """Delete documents from the vector store."""
            try:
                return await rag.delete(request, user_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        return api_router

    except ImportError:
        logger.warning("FastAPI not installed, cannot create router")
        return None


# Convenience export
router = RAGRouter
