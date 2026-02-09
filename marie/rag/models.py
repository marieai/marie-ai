"""RAG models for document retrieval and citations.

This module defines the core data models for the RAG (Retrieval-Augmented Generation)
system, including document sources, citations, and retrieval results.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DocumentSource(BaseModel):
    """Tracks a document collection for RAG.

    A DocumentSource represents a collection of documents that have been
    ingested and indexed for retrieval. It tracks the ingestion status
    and metadata about the source.
    """

    id: str = Field(..., description="Unique identifier for the source")
    name: str = Field(..., description="Human-readable name for the source")
    user_id: str = Field(..., description="Owner user ID")
    workspace_id: Optional[str] = Field(
        default=None, description="Optional workspace ID for organization"
    )
    status: str = Field(
        default="pending",
        description="Ingestion status: pending, processing, ready, error",
    )
    file_count: int = Field(default=0, description="Number of files in the source")
    node_count: int = Field(default=0, description="Number of indexed nodes")
    error_message: Optional[str] = Field(
        default=None, description="Error message if status is 'error'"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional source metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class SourceCitation(BaseModel):
    """Citation for RAG responses.

    Represents a reference to a specific piece of content that was used
    to answer a user's query. Compatible with ChatQuery.sources from
    the chat-assistants plan.
    """

    source_id: str = Field(..., description="ID of the DocumentSource")
    node_id: str = Field(..., description="ID of the specific node/chunk")
    node_type: Literal["text", "image", "document"] = Field(
        ..., description="Type of content"
    )
    title: str = Field(..., description="Title or identifier for the citation")
    filename: str = Field(..., description="Original filename")
    content_preview: str = Field(
        ..., description="Preview of the cited content (first ~200 chars)"
    )
    page: Optional[int] = Field(default=None, description="Page number if applicable")
    similarity: float = Field(..., description="Similarity score (0-1)")
    image_url: Optional[str] = Field(
        default=None, description="Image URL for image nodes"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional citation metadata"
    )


class RetrievalResult(BaseModel):
    """Result from unified RAG retrieval.

    Contains the retrieved nodes along with citation information
    for display and tracking purposes.
    """

    nodes: List[Dict[str, Any]] = Field(
        default_factory=list, description="Retrieved nodes (text + images)"
    )
    sources: List[SourceCitation] = Field(
        default_factory=list, description="Citations for the retrieved content"
    )
    query: str = Field(..., description="Original query string")
    rephrased_query: Optional[str] = Field(
        default=None, description="Query after rephrasing with context"
    )
    total_tokens: int = Field(
        default=0, description="Total tokens in retrieved content"
    )


class RAGNode(BaseModel):
    """A node stored in the vector store for RAG.

    Represents a single indexed unit (text chunk or image) that can
    be retrieved and used to answer queries.
    """

    id: str = Field(..., description="Unique node identifier")
    node_id: str = Field(..., description="External node identifier")
    source_id: str = Field(..., description="Parent DocumentSource ID")
    ref_doc_id: Optional[str] = Field(
        default=None, description="Reference to original document"
    )
    content: str = Field(..., description="Text content or image description")
    node_type: Literal["text", "image", "document"] = Field(
        default="text", description="Type of node"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Node metadata (filename, page, etc.)"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""

    query: str = Field(..., description="Search query")
    source_ids: Optional[List[str]] = Field(
        default=None, description="Filter to specific sources"
    )
    top_k: int = Field(default=10, description="Number of results to return")
    node_type: Optional[Literal["text", "image", "document"]] = Field(
        default=None, description="Filter by node type"
    )
    chat_history: Optional[List[Dict[str, str]]] = Field(
        default=None, description="Chat history for query rephrasing"
    )


class RAGQueryResponse(BaseModel):
    """Response model for RAG queries."""

    results: RetrievalResult = Field(..., description="Retrieval results")
    processing_time_ms: float = Field(
        default=0.0, description="Query processing time in milliseconds"
    )
