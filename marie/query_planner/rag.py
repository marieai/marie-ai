"""RAG Query Definitions for workflow nodes.

Provides workflow nodes for Retrieval-Augmented Generation, supporting:
- Agentic retrieval patterns (dynamic index selection, routing)
- Hybrid search (vector + keyword)
- Reranking (Cohere, cross-encoder)
- Cache-Augmented Generation (CAG) for small knowledge bases

References:
- Vellum Search Node: https://docs.vellum.ai/product/workflows/nodes/search-node
- Agentic Retrieval: https://www.llamaindex.ai/blog/rag-is-dead-long-live-agentic-retrieval
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field

from marie.query_planner.base import QueryDefinition, QueryTypeRegistry


@QueryTypeRegistry.register("RAG_SEARCH")
class RAGSearchQueryDefinition(QueryDefinition):
    """Search documents in vector store.

    Unified search node supporting vector, hybrid, and reranked retrieval.
    Routes to VectorStoreExecutor endpoints based on configuration.

    Features:
    - Vector search (semantic similarity)
    - Hybrid search (vector + keyword with RRF fusion)
    - Metadata filtering
    - Reranking (cross-encoder or Cohere)
    - Citation generation

    Example workflows:
        Basic RAG:     START → RAG_SEARCH → PROMPT → END
        With rerank:   START → RAG_SEARCH(rerank=true) → PROMPT → END
        Multi-index:   START → SWITCH → [RAG_SEARCH(index_a) | RAG_SEARCH(index_b)] → MERGE → PROMPT → END
    """

    method: str = "RAG_SEARCH"
    endpoint: str = Field(
        default="vector_store_executor://rag/search",
        description="VectorStoreExecutor endpoint (auto-selected based on options)",
    )
    params: dict = Field(default_factory=dict)

    # Index/collection selection
    index_name: str = Field(
        default="default",
        description="Index/collection to search within",
    )

    # Source filtering (supports dynamic via upstream node output)
    source_ids: Optional[List[str]] = Field(
        default=None,
        description="Document source IDs to filter. If None, searches all sources in index.",
    )

    # Search configuration
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    score_threshold: Optional[float] = Field(
        default=None, ge=0, le=1, description="Minimum similarity score filter"
    )

    # Search mode
    hybrid: bool = Field(
        default=False,
        description="Use hybrid search (vector + keyword with RRF fusion)",
    )
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Hybrid balance: 0=pure vector, 1=pure keyword",
    )

    # Reranking
    rerank: bool = Field(default=False, description="Apply reranking to results")
    rerank_model: Optional[str] = Field(
        default=None,
        description="Rerank model (e.g., 'cohere-rerank-v3', 'cross-encoder')",
    )
    rerank_top_k: Optional[int] = Field(
        default=None, description="Return top-k after reranking (defaults to top_k)"
    )

    # Metadata filtering (like Vellum's metadata filters)
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter by document metadata (e.g., {'policy_type': 'hr'})",
    )

    # Output options
    include_citations: bool = Field(
        default=True, description="Include source citations in output"
    )
    include_content: bool = Field(
        default=True, description="Include full content (vs just metadata)"
    )

    def validate_params(self):
        pass


@QueryTypeRegistry.register("RAG_INGEST")
class RAGIngestQueryDefinition(QueryDefinition):
    """Ingest documents into vector store.

    Embeds and stores documents for later retrieval. Supports both
    single document and batch ingestion modes.

    Features:
    - Multimodal embeddings (text + images via JinaEmbeddingsV4)
    - Batch mode for high throughput
    - Metadata attachment
    - Reference document grouping

    Example workflows:
        Single:  DOCUMENT_PARSER → RAG_INGEST → UPDATE_STATUS
        Batch:   CHUNKER → RAG_INGEST(batch=true) → NOTIFY
    """

    method: str = "RAG_INGEST"
    endpoint: str = Field(
        default="vector_store_executor://rag/embed_and_store",
        description="VectorStoreExecutor ingest endpoint",
    )
    params: dict = Field(default_factory=dict)

    # Index/collection target
    index_name: str = Field(
        default="default",
        description="Index/collection to ingest documents into",
    )

    # Target configuration
    source_id: str = Field(..., description="Document source identifier")
    ref_doc_id: Optional[str] = Field(
        default=None, description="Reference document ID for grouping chunks"
    )

    # Content type
    node_type: Literal["text", "image", "document"] = Field(
        default="text", description="Type of content being ingested"
    )

    # Batch mode
    batch_mode: bool = Field(
        default=False, description="Use batch ingestion for multiple documents"
    )
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Batch size for bulk insert"
    )

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata to attach to ingested documents"
    )

    def validate_params(self):
        if not self.source_id:
            raise ValueError("RAG_INGEST requires a source_id")


@QueryTypeRegistry.register("RAG_DELETE")
class RAGDeleteQueryDefinition(QueryDefinition):
    """Delete documents from vector store.

    Remove documents by source, node IDs, or reference document.

    Example workflow:
        DELETE_REQUEST → RAG_DELETE → CONFIRM
    """

    method: str = "RAG_DELETE"
    endpoint: str = Field(
        default="vector_store_executor://rag/delete_source",
        description="VectorStoreExecutor delete endpoint",
    )
    params: dict = Field(default_factory=dict)

    # Index/collection scope
    index_name: Optional[str] = Field(
        default=None,
        description="Index/collection to delete from. If None, deletes from all indexes.",
    )

    # Delete targets (at least one required)
    source_id: Optional[str] = Field(
        default=None, description="Delete all documents from source"
    )
    node_ids: Optional[List[str]] = Field(
        default=None, description="Delete specific node IDs"
    )
    ref_doc_id: Optional[str] = Field(
        default=None, description="Delete by reference document ID"
    )

    def validate_params(self):
        if not any([self.source_id, self.node_ids, self.ref_doc_id]):
            raise ValueError("RAG_DELETE requires source_id, node_ids, or ref_doc_id")


@QueryTypeRegistry.register("CONTEXT_CACHE")
class ContextCacheQueryDefinition(QueryDefinition):
    """Cache-Augmented Generation (CAG) preloading.

    Pre-loads documents into LLM context cache for retrieval-free generation.
    Use when knowledge base fits in context window (<128k tokens) and
    low latency is critical.

    CAG vs RAG tradeoffs:
    - CAG: ~9x faster, no retrieval errors, simpler architecture
    - RAG: Scales to large knowledge bases, fresher data

    When to use CAG:
    - Static knowledge bases (FAQs, manuals, policies)
    - Latency-sensitive applications
    - Small document collections

    Example workflow:
        LOAD_DOCS → CONTEXT_CACHE → PROMPT(with cached context) → RESPONSE
    """

    method: str = "CONTEXT_CACHE"
    endpoint: str = Field(
        default="context_cache://preload",
        description="Context cache preload endpoint",
    )
    params: dict = Field(default_factory=dict)

    # Index/collection selection
    index_name: str = Field(
        default="default",
        description="Index/collection to cache documents from",
    )

    # What to cache
    source_ids: List[str] = Field(
        default_factory=list,
        description="Document sources to preload into context",
    )

    # Cache limits
    max_tokens: int = Field(
        default=100000,
        ge=1000,
        le=200000,
        description="Maximum tokens to cache (model-dependent)",
    )

    # Cache management
    cache_key: Optional[str] = Field(
        default=None,
        description="Cache key for reuse across requests (auto-generated if None)",
    )
    ttl_seconds: int = Field(
        default=3600, ge=60, le=86400, description="Cache TTL in seconds"
    )

    # Content selection
    include_metadata: bool = Field(
        default=False, description="Include document metadata in cache"
    )
    truncation_strategy: Literal["end", "middle", "semantic"] = Field(
        default="end",
        description="How to truncate if content exceeds max_tokens",
    )

    def validate_params(self):
        if not self.source_ids:
            raise ValueError("CONTEXT_CACHE requires at least one source_id")
