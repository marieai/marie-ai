"""RAG Index API routes for managing collections/indexes.

This module provides REST API endpoints for managing RAG indexes,
which are named collections with their own configuration for embedding,
chunking, and retrieval.

Endpoints:
    POST   /api/rag/indexes          - Create index
    GET    /api/rag/indexes          - List indexes
    GET    /api/rag/indexes/{name}   - Get index details
    PUT    /api/rag/indexes/{name}   - Update config
    DELETE /api/rag/indexes/{name}   - Delete index + all data
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from marie.logging_core.logger import MarieLogger
from marie.rag.models import RAGIndex

logger = MarieLogger("marie.api.routes.rag_index").logger


# -------------------------------------------------------------------------
# Request/Response Models
# -------------------------------------------------------------------------


class CreateIndexRequest(BaseModel):
    """Request to create a new RAG index."""

    name: str = Field(..., description="Human-readable name, unique per workspace")
    workspace_id: str = Field(..., description="Workspace this index belongs to")

    # Embedding configuration (optional, uses defaults)
    embedding_model: str = Field(
        default="jinaai/jina-embeddings-v4",
        description="Embedding model to use",
    )
    embedding_dim: int = Field(
        default=2048,
        description="Embedding dimension (128/256/512/1024/2048)",
    )

    # Transform configuration
    chunk_size: int = Field(default=1024, description="Maximum chunk size")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    segmentation_mode: Literal["page", "paragraph", "semantic"] = Field(
        default="semantic", description="Document segmentation strategy"
    )

    # Search configuration
    hybrid_enabled: bool = Field(default=True, description="Enable hybrid search")
    hybrid_alpha: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Dense/sparse balance"
    )

    # Multimodal
    multimodal_enabled: bool = Field(default=True, description="Enable image embedding")

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class UpdateIndexRequest(BaseModel):
    """Request to update an index configuration."""

    # Embedding config (cannot change model/dim after creation in most cases)
    chunk_size: Optional[int] = Field(default=None, description="Maximum chunk size")
    chunk_overlap: Optional[int] = Field(
        default=None, description="Overlap between chunks"
    )
    segmentation_mode: Optional[Literal["page", "paragraph", "semantic"]] = Field(
        default=None, description="Document segmentation strategy"
    )

    # Search configuration
    hybrid_enabled: Optional[bool] = Field(
        default=None, description="Enable hybrid search"
    )
    hybrid_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Dense/sparse balance"
    )

    # Multimodal
    multimodal_enabled: Optional[bool] = Field(
        default=None, description="Enable image embedding"
    )

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata (merged with existing)"
    )


class IndexResponse(BaseModel):
    """Response containing an index."""

    index: RAGIndex


class ListIndexesResponse(BaseModel):
    """Response listing indexes."""

    indexes: List[RAGIndex]
    total: int


class IndexStatsResponse(BaseModel):
    """Statistics for an index."""

    index_name: str
    total_nodes: int
    source_count: int
    by_type: Dict[str, int]
    by_source: Dict[str, int]


class DeleteIndexResponse(BaseModel):
    """Response after deleting an index."""

    index_name: str
    deleted: bool
    nodes_deleted: int
    message: str


# -------------------------------------------------------------------------
# Router Class
# -------------------------------------------------------------------------


class RAGIndexRouter:
    """RAG Index API router.

    Provides CRUD operations for managing RAG indexes (collections).

    Example with FastAPI:
        ```python
        from fastapi import FastAPI
        from marie.api.routes.rag_index import create_fastapi_router

        app = FastAPI()
        index_router = create_fastapi_router(vector_store=store)
        app.include_router(index_router)
        ```
    """

    def __init__(
        self,
        vector_store: Any = None,
        index_storage: Optional[Dict[str, RAGIndex]] = None,
    ):
        """Initialize RAG Index router.

        Args:
            vector_store: PGVectorStore instance.
            index_storage: Optional dict for in-memory index storage.
                          In production, use a database.
        """
        self._vector_store = vector_store
        # In-memory storage; replace with database in production
        self._indexes: Dict[str, RAGIndex] = index_storage or {}

        # Create default index if it doesn't exist
        if "default" not in self._indexes:
            self._indexes["default"] = RAGIndex(
                id=str(uuid.uuid4()),
                name="default",
                workspace_id="system",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

    def _get_index_key(self, workspace_id: str, name: str) -> str:
        """Get the storage key for an index."""
        return f"{workspace_id}:{name}"

    async def create_index(
        self,
        request: CreateIndexRequest,
        user_id: str,
    ) -> IndexResponse:
        """Create a new RAG index.

        Args:
            request: Index creation request.
            user_id: ID of the user creating the index.

        Returns:
            IndexResponse with the new index.
        """
        index_key = self._get_index_key(request.workspace_id, request.name)

        if index_key in self._indexes:
            raise ValueError(f"Index '{request.name}' already exists in workspace")

        now = datetime.utcnow()
        index = RAGIndex(
            id=str(uuid.uuid4()),
            name=request.name,
            workspace_id=request.workspace_id,
            embedding_model=request.embedding_model,
            embedding_dim=request.embedding_dim,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            segmentation_mode=request.segmentation_mode,
            hybrid_enabled=request.hybrid_enabled,
            hybrid_alpha=request.hybrid_alpha,
            multimodal_enabled=request.multimodal_enabled,
            metadata=request.metadata or {},
            created_at=now,
            updated_at=now,
        )

        self._indexes[index_key] = index
        # Also store by name for backward compatibility
        self._indexes[request.name] = index

        logger.info(
            f"Created index: {request.name} in workspace {request.workspace_id}"
        )

        return IndexResponse(index=index)

    async def list_indexes(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> ListIndexesResponse:
        """List RAG indexes.

        Args:
            user_id: User ID for authorization.
            workspace_id: Optional workspace filter.

        Returns:
            ListIndexesResponse with matching indexes.
        """
        # Filter by workspace if specified
        indexes = []
        seen_ids = set()

        for index in self._indexes.values():
            if index.id in seen_ids:
                continue
            if workspace_id is None or index.workspace_id == workspace_id:
                indexes.append(index)
                seen_ids.add(index.id)

        return ListIndexesResponse(
            indexes=indexes,
            total=len(indexes),
        )

    async def get_index(
        self,
        name: str,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> RAGIndex:
        """Get a specific index.

        Args:
            name: Index name.
            user_id: User ID for authorization.
            workspace_id: Optional workspace filter.

        Returns:
            RAGIndex if found.

        Raises:
            ValueError: If index not found.
        """
        # Try workspace-specific key first
        if workspace_id:
            index_key = self._get_index_key(workspace_id, name)
            if index_key in self._indexes:
                return self._indexes[index_key]

        # Fall back to name-only lookup
        if name in self._indexes:
            return self._indexes[name]

        raise ValueError(f"Index not found: {name}")

    async def update_index(
        self,
        name: str,
        request: UpdateIndexRequest,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> IndexResponse:
        """Update an index configuration.

        Args:
            name: Index name to update.
            request: Update request.
            user_id: User ID for authorization.
            workspace_id: Optional workspace filter.

        Returns:
            IndexResponse with updated index.
        """
        index = await self.get_index(name, user_id, workspace_id)

        # Update fields if provided
        if request.chunk_size is not None:
            index.chunk_size = request.chunk_size
        if request.chunk_overlap is not None:
            index.chunk_overlap = request.chunk_overlap
        if request.segmentation_mode is not None:
            index.segmentation_mode = request.segmentation_mode
        if request.hybrid_enabled is not None:
            index.hybrid_enabled = request.hybrid_enabled
        if request.hybrid_alpha is not None:
            index.hybrid_alpha = request.hybrid_alpha
        if request.multimodal_enabled is not None:
            index.multimodal_enabled = request.multimodal_enabled
        if request.metadata is not None:
            index.metadata = {**index.metadata, **request.metadata}

        index.updated_at = datetime.utcnow()

        logger.info(f"Updated index: {name}")

        return IndexResponse(index=index)

    async def delete_index(
        self,
        name: str,
        user_id: str,
        workspace_id: Optional[str] = None,
        confirm: bool = False,
    ) -> DeleteIndexResponse:
        """Delete an index and all its data.

        Args:
            name: Index name to delete.
            user_id: User ID for authorization.
            workspace_id: Optional workspace filter.
            confirm: Must be True to delete.

        Returns:
            DeleteIndexResponse with deletion status.
        """
        if name == "default" and not confirm:
            raise ValueError("Cannot delete 'default' index without confirm=True")

        index = await self.get_index(name, user_id, workspace_id)

        # Delete all nodes in the index from vector store
        deleted_nodes = 0
        if self._vector_store:
            try:
                deleted_nodes = await self._vector_store.delete_by_index(name)
            except Exception as e:
                logger.error(f"Error deleting index nodes: {e}")

        # Remove from storage
        if workspace_id:
            index_key = self._get_index_key(workspace_id, name)
            if index_key in self._indexes:
                del self._indexes[index_key]

        if name in self._indexes:
            del self._indexes[name]

        logger.info(f"Deleted index: {name}, {deleted_nodes} nodes removed")

        return DeleteIndexResponse(
            index_name=name,
            deleted=True,
            nodes_deleted=deleted_nodes,
            message=f"Index '{name}' deleted with {deleted_nodes} nodes.",
        )

    async def get_index_stats(
        self,
        name: str,
        user_id: str,
        workspace_id: Optional[str] = None,
    ) -> IndexStatsResponse:
        """Get statistics for an index.

        Args:
            name: Index name.
            user_id: User ID for authorization.
            workspace_id: Optional workspace filter.

        Returns:
            IndexStatsResponse with node counts.
        """
        # Verify index exists
        await self.get_index(name, user_id, workspace_id)

        stats = {"total": 0, "by_type": {}, "by_source": {}, "source_count": 0}

        if self._vector_store:
            try:
                stats = await self._vector_store.get_index_stats(name)
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")

        return IndexStatsResponse(
            index_name=name,
            total_nodes=stats.get("total", 0),
            source_count=stats.get("source_count", 0),
            by_type=stats.get("by_type", {}),
            by_source=stats.get("by_source", {}),
        )


# -------------------------------------------------------------------------
# FastAPI Router Factory
# -------------------------------------------------------------------------


def create_fastapi_router(
    vector_store: Any = None,
    prefix: str = "/api/rag",
):
    """Create a FastAPI router with RAG Index endpoints.

    Args:
        vector_store: PGVectorStore instance.
        prefix: URL prefix for routes.

    Returns:
        FastAPI APIRouter instance.
    """
    try:
        from fastapi import APIRouter, Depends, HTTPException, Query

        api_router = APIRouter(prefix=prefix, tags=["rag-indexes"])
        index_router = RAGIndexRouter(vector_store=vector_store)

        # Placeholder for user auth
        async def get_current_user() -> str:
            return "default_user"

        @api_router.post("/indexes", response_model=IndexResponse)
        async def create_index(
            request: CreateIndexRequest,
            user_id: str = Depends(get_current_user),
        ):
            """Create a new RAG index."""
            try:
                return await index_router.create_index(request, user_id)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @api_router.get("/indexes", response_model=ListIndexesResponse)
        async def list_indexes(
            workspace_id: Optional[str] = Query(None),
            user_id: str = Depends(get_current_user),
        ):
            """List RAG indexes."""
            return await index_router.list_indexes(user_id, workspace_id)

        @api_router.get("/indexes/{name}", response_model=RAGIndex)
        async def get_index(
            name: str,
            workspace_id: Optional[str] = Query(None),
            user_id: str = Depends(get_current_user),
        ):
            """Get a specific index."""
            try:
                return await index_router.get_index(name, user_id, workspace_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api_router.put("/indexes/{name}", response_model=IndexResponse)
        async def update_index(
            name: str,
            request: UpdateIndexRequest,
            workspace_id: Optional[str] = Query(None),
            user_id: str = Depends(get_current_user),
        ):
            """Update an index configuration."""
            try:
                return await index_router.update_index(
                    name, request, user_id, workspace_id
                )
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        @api_router.delete("/indexes/{name}", response_model=DeleteIndexResponse)
        async def delete_index(
            name: str,
            workspace_id: Optional[str] = Query(None),
            confirm: bool = Query(False),
            user_id: str = Depends(get_current_user),
        ):
            """Delete an index and all its data."""
            try:
                return await index_router.delete_index(
                    name, user_id, workspace_id, confirm
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))

        @api_router.get("/indexes/{name}/stats", response_model=IndexStatsResponse)
        async def get_index_stats(
            name: str,
            workspace_id: Optional[str] = Query(None),
            user_id: str = Depends(get_current_user),
        ):
            """Get statistics for an index."""
            try:
                return await index_router.get_index_stats(name, user_id, workspace_id)
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        return api_router

    except ImportError:
        logger.warning("FastAPI not installed, cannot create router")
        return None


# Convenience export
router = RAGIndexRouter
