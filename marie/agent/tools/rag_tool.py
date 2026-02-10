"""RAG Tool for agentic document operations.

This module provides comprehensive tools for RAG operations including
ingestion, search, and deletion - enabling agents to manage document
knowledge bases autonomously.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.embeddings.jina import JinaEmbeddingsV4
    from marie.rag.retriever import RAGRetriever
    from marie.vector_stores.pgvector import PGVectorStore

logger = MarieLogger("marie.agent.tools.rag_tool").logger


# -------------------------------------------------------------------------
# Input Schemas
# -------------------------------------------------------------------------


class RAGSearchInput(BaseModel):
    """Input schema for RAG search."""

    query: str = Field(
        ...,
        description="Search query to find relevant documents. Be specific and include key terms.",
    )
    source_ids: Optional[List[str]] = Field(
        default=None,
        description="Filter to specific document source IDs. If not provided, searches all sources.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of results to return (1-50).",
    )
    use_hybrid: bool = Field(
        default=False,
        description="Use hybrid search (vector + keyword) for better recall on keyword-heavy queries.",
    )


class RAGIngestInput(BaseModel):
    """Input schema for RAG ingestion."""

    texts: List[str] = Field(
        ...,
        description="List of text chunks to ingest into the knowledge base.",
        min_length=1,
    )
    source_id: str = Field(
        ...,
        description="Unique identifier for the document source (e.g., 'user_docs_123').",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata to attach to all ingested chunks (e.g., {'filename': 'doc.pdf'}).",
    )
    ref_doc_id: Optional[str] = Field(
        default=None,
        description="Optional reference document ID for grouping related chunks.",
    )


class RAGDeleteInput(BaseModel):
    """Input schema for RAG deletion."""

    source_id: Optional[str] = Field(
        default=None,
        description="Delete all documents from this source ID.",
    )
    node_ids: Optional[List[str]] = Field(
        default=None,
        description="Delete specific node IDs.",
    )
    ref_doc_id: Optional[str] = Field(
        default=None,
        description="Delete all nodes with this reference document ID.",
    )


class RAGStatsInput(BaseModel):
    """Input schema for RAG statistics."""

    source_id: Optional[str] = Field(
        default=None,
        description="Get statistics for a specific source. If not provided, returns global stats.",
    )


# -------------------------------------------------------------------------
# RAG Tool Implementation
# -------------------------------------------------------------------------


class RAGTool(AgentTool):
    """Comprehensive RAG tool for document knowledge management.

    This tool enables agents to:
    - Search documents for relevant information
    - Ingest new documents into the knowledge base
    - Delete documents or sources
    - Get statistics about the knowledge base

    Unlike simpler search-only tools, RAGTool gives agents full control
    over the document lifecycle, enabling autonomous knowledge management.

    Example usage:
        ```python
        from marie.agent.tools import RAGTool
        from marie.vector_stores.pgvector import PGVectorStore
        from marie.embeddings.jina import JinaEmbeddingsV4

        # Initialize components
        store = PGVectorStore(connection_string="...")
        embeddings = JinaEmbeddingsV4()

        # Create tool
        rag_tool = RAGTool(
            vector_store=store,
            embeddings=embeddings,
            default_source_id="agent_knowledge",
        )

        # Use in agent
        agent = ReactAgent(llm=llm, function_list=[rag_tool])
        ```
    """

    def __init__(
        self,
        vector_store: "PGVectorStore",
        embeddings: "JinaEmbeddingsV4",
        retriever: Optional["RAGRetriever"] = None,
        default_source_id: Optional[str] = None,
        allowed_sources: Optional[List[str]] = None,
        name: str = "rag",
        description: Optional[str] = None,
        enable_ingest: bool = True,
        enable_delete: bool = True,
    ):
        """Initialize RAGTool.

        Args:
            vector_store: PGVectorStore instance for storage.
            embeddings: JinaEmbeddingsV4 instance for creating embeddings.
            retriever: Optional RAGRetriever for advanced search features.
            default_source_id: Default source ID for operations.
            allowed_sources: Limit operations to specific sources (security).
            name: Tool name.
            description: Custom tool description.
            enable_ingest: Allow ingestion operations.
            enable_delete: Allow deletion operations.
        """
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._retriever = retriever
        self._default_source_id = default_source_id
        self._allowed_sources = allowed_sources
        self._name = name
        self._enable_ingest = enable_ingest
        self._enable_delete = enable_delete

        # Build description based on enabled features
        features = ["search documents"]
        if enable_ingest:
            features.append("ingest new content")
        if enable_delete:
            features.append("delete documents")

        self._description = description or (
            f"Manage the document knowledge base. Can {', '.join(features)}. "
            "Use 'action' parameter to specify operation: 'search', 'ingest', 'delete', or 'stats'."
        )

    @property
    def metadata(self) -> ToolMetadata:
        """Return tool metadata with dynamic schema."""
        return ToolMetadata(
            name=self._name,
            description=self._description,
            fn_schema=self._build_schema(),
        )

    def _build_schema(self) -> type:
        """Build dynamic schema based on enabled features."""

        class RAGInput(BaseModel):
            action: str = Field(
                ...,
                description="Operation to perform: 'search', 'ingest', 'delete', or 'stats'",
            )
            # Search parameters
            query: Optional[str] = Field(
                default=None,
                description="Search query (required for 'search' action)",
            )
            source_ids: Optional[List[str]] = Field(
                default=None,
                description="Filter to specific source IDs",
            )
            top_k: int = Field(
                default=5,
                description="Number of results for search",
            )
            use_hybrid: bool = Field(
                default=False,
                description="Use hybrid search (vector + keyword)",
            )
            # Ingest parameters
            texts: Optional[List[str]] = Field(
                default=None,
                description="Texts to ingest (required for 'ingest' action)",
            )
            source_id: Optional[str] = Field(
                default=None,
                description="Source ID for ingest/delete operations",
            )
            metadata: Optional[Dict[str, Any]] = Field(
                default=None,
                description="Metadata for ingested documents",
            )
            ref_doc_id: Optional[str] = Field(
                default=None,
                description="Reference document ID",
            )
            # Delete parameters
            node_ids: Optional[List[str]] = Field(
                default=None,
                description="Specific node IDs to delete",
            )

        return RAGInput

    def _validate_source_access(self, source_ids: Optional[List[str]]) -> None:
        """Validate source access if allowed_sources is set."""
        if self._allowed_sources is None:
            return

        if source_ids:
            for sid in source_ids:
                if sid not in self._allowed_sources:
                    raise ValueError(f"Access denied to source: {sid}")

    def call(self, **kwargs: Any) -> ToolOutput:
        """Execute RAG operation synchronously."""
        return asyncio.get_event_loop().run_until_complete(self.acall(**kwargs))

    async def acall(self, **kwargs: Any) -> ToolOutput:
        """Execute RAG operation asynchronously.

        Routes to the appropriate handler based on 'action' parameter.
        """
        action = kwargs.get("action", "search").lower()

        try:
            if action == "search":
                return await self._handle_search(kwargs)
            elif action == "ingest":
                if not self._enable_ingest:
                    return ToolOutput(
                        content="Ingest operation is disabled for this tool.",
                        tool_name=self._name,
                        raw_input=kwargs,
                        is_error=True,
                    )
                return await self._handle_ingest(kwargs)
            elif action == "delete":
                if not self._enable_delete:
                    return ToolOutput(
                        content="Delete operation is disabled for this tool.",
                        tool_name=self._name,
                        raw_input=kwargs,
                        is_error=True,
                    )
                return await self._handle_delete(kwargs)
            elif action == "stats":
                return await self._handle_stats(kwargs)
            else:
                return ToolOutput(
                    content=f"Unknown action: {action}. Use 'search', 'ingest', 'delete', or 'stats'.",
                    tool_name=self._name,
                    raw_input=kwargs,
                    is_error=True,
                )

        except Exception as e:
            logger.error(f"RAG tool error: {e}")
            return ToolOutput(
                content=f"Operation failed: {str(e)}",
                tool_name=self._name,
                raw_input=kwargs,
                is_error=True,
            )

    async def _handle_search(self, kwargs: Dict[str, Any]) -> ToolOutput:
        """Handle search operation."""
        query = kwargs.get("query")
        if not query:
            return ToolOutput(
                content="Search requires a 'query' parameter.",
                tool_name=self._name,
                raw_input=kwargs,
                is_error=True,
            )

        source_ids = kwargs.get("source_ids")
        self._validate_source_access(source_ids)

        top_k = kwargs.get("top_k", 5)
        use_hybrid = kwargs.get("use_hybrid", False)

        logger.info(f"RAG search: query='{query[:50]}...', top_k={top_k}")

        # Get query embedding
        query_embedding = self._embeddings.embed_text([query], is_query=True)[0]

        # Perform search
        if use_hybrid:
            results = await self._vector_store.hybrid_search(
                query_embedding=query_embedding.tolist(),
                query_text=query,
                source_ids=source_ids,
                top_k=top_k,
            )
        else:
            results = await self._vector_store.search(
                query_embedding=query_embedding.tolist(),
                source_ids=source_ids,
                top_k=top_k,
            )

        # Format results
        formatted = self._format_search_results(results)

        return ToolOutput(
            content=formatted,
            tool_name=self._name,
            raw_input=kwargs,
            raw_output={"results": results, "count": len(results)},
        )

    async def _handle_ingest(self, kwargs: Dict[str, Any]) -> ToolOutput:
        """Handle ingest operation."""
        texts = kwargs.get("texts")
        if not texts:
            return ToolOutput(
                content="Ingest requires a 'texts' parameter with list of texts.",
                tool_name=self._name,
                raw_input=kwargs,
                is_error=True,
            )

        source_id = kwargs.get("source_id") or self._default_source_id
        if not source_id:
            return ToolOutput(
                content="Ingest requires a 'source_id' parameter.",
                tool_name=self._name,
                raw_input=kwargs,
                is_error=True,
            )

        self._validate_source_access([source_id])

        metadata = kwargs.get("metadata", {})
        ref_doc_id = kwargs.get("ref_doc_id")

        logger.info(f"RAG ingest: {len(texts)} texts to source={source_id}")

        # Embed texts
        embeddings = self._embeddings.embed_text(texts, is_query=False)

        # Prepare nodes for batch insert
        nodes = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
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
        )

        return ToolOutput(
            content=f"Successfully ingested {count} documents to source '{source_id}'.",
            tool_name=self._name,
            raw_input=kwargs,
            raw_output={"count": count, "source_id": source_id},
        )

    async def _handle_delete(self, kwargs: Dict[str, Any]) -> ToolOutput:
        """Handle delete operation."""
        source_id = kwargs.get("source_id")
        node_ids = kwargs.get("node_ids")
        ref_doc_id = kwargs.get("ref_doc_id")

        if not any([source_id, node_ids, ref_doc_id]):
            return ToolOutput(
                content="Delete requires 'source_id', 'node_ids', or 'ref_doc_id'.",
                tool_name=self._name,
                raw_input=kwargs,
                is_error=True,
            )

        if source_id:
            self._validate_source_access([source_id])

        logger.info(
            f"RAG delete: source_id={source_id}, node_ids={node_ids}, ref_doc_id={ref_doc_id}"
        )

        deleted = 0

        if source_id and not node_ids and not ref_doc_id:
            # Delete entire source
            deleted = await self._vector_store.delete_by_source(source_id)
        elif ref_doc_id:
            # Delete by reference document
            await self._vector_store.adelete(ref_doc_id)
            deleted = -1  # Unknown count from adelete
        elif node_ids:
            # Delete specific nodes
            await self._vector_store.adelete_nodes(node_ids=node_ids)
            deleted = len(node_ids)

        if deleted == -1:
            message = f"Deleted documents with ref_doc_id '{ref_doc_id}'."
        else:
            message = f"Deleted {deleted} documents."

        return ToolOutput(
            content=message,
            tool_name=self._name,
            raw_input=kwargs,
            raw_output={"deleted": deleted},
        )

    async def _handle_stats(self, kwargs: Dict[str, Any]) -> ToolOutput:
        """Handle stats operation."""
        source_id = kwargs.get("source_id")

        if source_id:
            self._validate_source_access([source_id])
            stats = await self._vector_store.get_source_stats(source_id)
            message = f"Source '{source_id}' statistics:\n"
        else:
            # Get global count
            total = await self._vector_store.count_nodes()
            stats = {"total": total}
            message = "Knowledge base statistics:\n"

        message += f"- Total nodes: {stats.get('total', 0)}\n"
        if "by_type" in stats:
            for node_type, count in stats["by_type"].items():
                message += f"- {node_type}: {count}\n"

        return ToolOutput(
            content=message,
            tool_name=self._name,
            raw_input=kwargs,
            raw_output=stats,
        )

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for agent consumption."""
        if not results:
            return "No relevant documents found."

        parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", "")
            similarity = result.get("similarity", 0)
            metadata = result.get("metadata", {})

            # Build citation
            source_info = []
            if metadata.get("filename"):
                source_info.append(metadata["filename"])
            if metadata.get("page"):
                source_info.append(f"p.{metadata['page']}")

            citation = f"[{', '.join(source_info)}]" if source_info else f"[node_{i}]"

            # Truncate long content
            if len(content) > 800:
                content = content[:800] + "..."

            # Format with score
            similarity_pct = int(similarity * 100)

            # Include RRF score if available (hybrid search)
            score_str = f"relevance: {similarity_pct}%"
            if "rrf_score" in result:
                score_str += f", hybrid: {result['rrf_score']:.3f}"

            parts.append(f"**Result {i}** {citation} ({score_str})\n{content}")

        return "\n\n---\n\n".join(parts)


# -------------------------------------------------------------------------
# Specialized Tools
# -------------------------------------------------------------------------


class RAGSearchTool(RAGTool):
    """Search-only RAG tool for retrieval operations.

    A simpler variant of RAGTool that only allows search operations,
    suitable for read-only agent scenarios.
    """

    def __init__(
        self,
        vector_store: "PGVectorStore",
        embeddings: "JinaEmbeddingsV4",
        retriever: Optional["RAGRetriever"] = None,
        allowed_sources: Optional[List[str]] = None,
        name: str = "rag_search",
    ):
        super().__init__(
            vector_store=vector_store,
            embeddings=embeddings,
            retriever=retriever,
            allowed_sources=allowed_sources,
            name=name,
            description=(
                "Search the document knowledge base for relevant information. "
                "Returns matching excerpts with citations and relevance scores."
            ),
            enable_ingest=False,
            enable_delete=False,
        )

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._name,
            description=self._description,
            fn_schema=RAGSearchInput,
        )

    async def acall(
        self,
        query: str,
        source_ids: Optional[List[str]] = None,
        top_k: int = 5,
        use_hybrid: bool = False,
        **kwargs,
    ) -> ToolOutput:
        """Execute search operation."""
        return await self._handle_search(
            {
                "query": query,
                "source_ids": source_ids,
                "top_k": top_k,
                "use_hybrid": use_hybrid,
            }
        )


class RAGIngestTool(RAGTool):
    """Ingest-only RAG tool for adding documents.

    Allows agents to add content to the knowledge base without
    search or delete capabilities.
    """

    def __init__(
        self,
        vector_store: "PGVectorStore",
        embeddings: "JinaEmbeddingsV4",
        default_source_id: str,
        allowed_sources: Optional[List[str]] = None,
        name: str = "rag_ingest",
    ):
        super().__init__(
            vector_store=vector_store,
            embeddings=embeddings,
            default_source_id=default_source_id,
            allowed_sources=allowed_sources,
            name=name,
            description=(
                "Add new documents to the knowledge base. "
                "Provide a list of text chunks to be embedded and stored."
            ),
            enable_ingest=True,
            enable_delete=False,
        )

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._name,
            description=self._description,
            fn_schema=RAGIngestInput,
        )

    async def acall(
        self,
        texts: List[str],
        source_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ref_doc_id: Optional[str] = None,
        **kwargs,
    ) -> ToolOutput:
        """Execute ingest operation."""
        return await self._handle_ingest(
            {
                "texts": texts,
                "source_id": source_id or self._default_source_id,
                "metadata": metadata,
                "ref_doc_id": ref_doc_id,
            }
        )
