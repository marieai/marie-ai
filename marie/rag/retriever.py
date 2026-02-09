"""RAG Retriever with citation tracking.

This module provides a retriever that performs semantic search over
document collections and returns results with citation information
for use in RAG pipelines.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from marie.logging_core.logger import MarieLogger
from marie.rag.models import RetrievalResult, SourceCitation

if TYPE_CHECKING:
    from marie.embeddings.base import EmbeddingsBase
    from marie.vector_stores.pgvector import PGVectorStore

logger = MarieLogger("marie.rag.retriever").logger


class RAGRetriever:
    """RAG retriever with citation tracking.

    Performs semantic search over document collections stored in
    PGVectorStore and returns results with citation metadata for
    display and tracking.

    Features:
    - Query rephrasing with conversation context (optional)
    - Multi-source filtering
    - Citation tracking for responses
    - Token budget management
    - Both sync and async interfaces

    Example:
        ```python
        retriever = RAGRetriever(
            vector_store=store,
            embeddings=embeddings,
        )

        result = await retriever.aretrieve(
            query="What is machine learning?",
            source_ids=["source_1", "source_2"],
            top_k=5,
        )

        # Access results
        for citation in result.sources:
            print(f"[{citation.filename}]: {citation.content_preview}")
        ```
    """

    def __init__(
        self,
        vector_store: "PGVectorStore",
        embeddings: "EmbeddingsBase",
        default_top_k: int = 10,
        max_content_length: int = 200,
    ):
        """Initialize RAGRetriever.

        Args:
            vector_store: PGVectorStore instance for document storage.
            embeddings: Embeddings instance for query encoding.
            default_top_k: Default number of results to return.
            max_content_length: Max length for content preview in citations.
        """
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._default_top_k = default_top_k
        self._max_content_length = max_content_length

    def retrieve(
        self,
        query: str,
        source_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        node_type: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> RetrievalResult:
        """Retrieve documents synchronously.

        Args:
            query: Search query string.
            source_ids: Filter to specific DocumentSource IDs.
            top_k: Number of results to return.
            node_type: Filter by node type (text/image/document).
            chat_history: Optional conversation history for context.

        Returns:
            RetrievalResult with nodes and citations.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.aretrieve(
                query=query,
                source_ids=source_ids,
                top_k=top_k,
                node_type=node_type,
                chat_history=chat_history,
            )
        )

    async def aretrieve(
        self,
        query: str,
        source_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        node_type: Optional[str] = None,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> RetrievalResult:
        """Retrieve documents asynchronously.

        Args:
            query: Search query string.
            source_ids: Filter to specific DocumentSource IDs.
            top_k: Number of results to return.
            node_type: Filter by node type (text/image/document).
            chat_history: Optional conversation history for context.

        Returns:
            RetrievalResult with nodes and citations.
        """
        top_k = top_k or self._default_top_k
        rephrased_query = None

        # Optional query rephrasing with chat history
        if chat_history:
            rephrased_query = self._rephrase_query(query, chat_history)
            search_query = rephrased_query
        else:
            search_query = query

        logger.info(f"Retrieving: query='{search_query[:50]}...', top_k={top_k}")

        # Embed query (is_query=True for retrieval optimization)
        if hasattr(self._embeddings, "embed_text"):
            query_embedding = self._embeddings.embed_text(
                [search_query], is_query=True
            )[0]
        else:
            result = self._embeddings.get_embeddings([search_query])
            query_embedding = result.embeddings[0]

        # Search vector store
        results = await self._vector_store.search(
            query_embedding=query_embedding.tolist(),
            source_ids=source_ids,
            top_k=top_k,
            node_type=node_type,
        )

        # Build citations
        nodes = []
        citations = []

        for r in results:
            # Add to nodes list
            nodes.append(r)

            # Build citation
            metadata = r.get("metadata") or {}
            citation = SourceCitation(
                source_id=r["source_id"],
                node_id=r["node_id"],
                node_type=r["node_type"],
                title=metadata.get("title", self._extract_title(r)),
                filename=metadata.get("filename", "unknown"),
                content_preview=self._truncate_content(r.get("content", "")),
                page=metadata.get("page"),
                similarity=r["similarity"],
                image_url=metadata.get("image_url"),
                metadata=metadata,
            )
            citations.append(citation)

        logger.info(f"Retrieved {len(nodes)} documents")

        return RetrievalResult(
            nodes=nodes,
            sources=citations,
            query=query,
            rephrased_query=rephrased_query,
            total_tokens=self._estimate_tokens(nodes),
        )

    def _rephrase_query(
        self,
        query: str,
        chat_history: List[Dict[str, str]],
    ) -> str:
        """Rephrase query using conversation context.

        This is a simple implementation that appends recent context.
        For more sophisticated rephrasing, integrate with an LLM.

        Args:
            query: Original query.
            chat_history: List of {"role": "...", "content": "..."} messages.

        Returns:
            Rephrased query string.
        """
        # Simple implementation: append last user message context
        if not chat_history:
            return query

        # Get last few messages for context
        recent_context = []
        for msg in chat_history[-3:]:  # Last 3 messages
            if msg.get("role") in ("user", "assistant"):
                content = msg.get("content", "")[:100]  # Truncate
                recent_context.append(content)

        if recent_context:
            context_str = " | ".join(recent_context)
            return f"Context: {context_str} | Query: {query}"

        return query

    def _truncate_content(self, content: str) -> str:
        """Truncate content for preview."""
        if not content:
            return ""
        if len(content) <= self._max_content_length:
            return content
        return content[: self._max_content_length] + "..."

    def _extract_title(self, result: Dict[str, Any]) -> str:
        """Extract a title from result."""
        metadata = result.get("metadata") or {}
        if metadata.get("title"):
            return metadata["title"]
        if metadata.get("filename"):
            return metadata["filename"]
        # Use first line of content as title
        content = result.get("content", "")
        if content:
            first_line = content.split("\n")[0][:50]
            return first_line
        return result.get("node_id", "Unknown")

    def _estimate_tokens(self, nodes: List[Dict[str, Any]]) -> int:
        """Estimate total tokens in retrieved content.

        Simple estimation: ~4 characters per token.
        """
        total_chars = sum(len(n.get("content", "")) for n in nodes)
        return total_chars // 4

    def format_context(
        self,
        result: RetrievalResult,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Format retrieval results as context for LLM.

        Args:
            result: RetrievalResult from retrieve/aretrieve.
            max_tokens: Optional token budget.

        Returns:
            Formatted context string with citations.
        """
        if not result.nodes:
            return "No relevant documents found."

        context_parts = []
        current_tokens = 0

        for i, (node, citation) in enumerate(zip(result.nodes, result.sources), 1):
            content = node.get("content", "")

            # Check token budget
            content_tokens = len(content) // 4
            if max_tokens and current_tokens + content_tokens > max_tokens:
                # Truncate to fit budget
                available = (max_tokens - current_tokens) * 4
                content = content[:available] + "..."
                content_tokens = len(content) // 4

            # Format with citation
            citation_ref = f"[{citation.filename}]"
            if citation.page:
                citation_ref = f"[{citation.filename}, p.{citation.page}]"

            context_parts.append(f"{citation_ref}\n{content}")
            current_tokens += content_tokens

            if max_tokens and current_tokens >= max_tokens:
                break

        return "\n\n---\n\n".join(context_parts)

    def format_sources_for_display(
        self,
        citations: List[SourceCitation],
    ) -> str:
        """Format citations for user display.

        Args:
            citations: List of SourceCitation objects.

        Returns:
            Formatted string listing sources.
        """
        if not citations:
            return ""

        lines = ["**Sources:**"]
        for i, c in enumerate(citations, 1):
            line = f"{i}. {c.filename}"
            if c.page:
                line += f" (p.{c.page})"
            line += f" - {c.content_preview[:50]}..."
            lines.append(line)

        return "\n".join(lines)


class MultiSourceRetriever:
    """Retriever that searches across multiple vector stores.

    Useful when documents are stored in different databases or
    when implementing federated search.
    """

    def __init__(
        self,
        retrievers: Dict[str, RAGRetriever],
    ):
        """Initialize with named retrievers.

        Args:
            retrievers: Dictionary mapping source names to RAGRetriever instances.
        """
        self._retrievers = retrievers

    async def aretrieve(
        self,
        query: str,
        source_names: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Search across multiple retrievers.

        Args:
            query: Search query.
            source_names: Which retrievers to search (None = all).
            top_k: Results per retriever.

        Returns:
            Combined RetrievalResult.
        """
        names = source_names or list(self._retrievers.keys())

        # Search all retrievers concurrently
        tasks = [
            self._retrievers[name].aretrieve(query=query, top_k=top_k)
            for name in names
            if name in self._retrievers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        all_nodes = []
        all_citations = []

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Retriever failed: {result}")
                continue
            all_nodes.extend(result.nodes)
            all_citations.extend(result.sources)

        # Sort by similarity
        combined = list(zip(all_nodes, all_citations))
        combined.sort(key=lambda x: x[1].similarity, reverse=True)

        # Take top_k overall
        combined = combined[:top_k]
        nodes = [c[0] for c in combined]
        citations = [c[1] for c in combined]

        return RetrievalResult(
            nodes=nodes,
            sources=citations,
            query=query,
            total_tokens=sum(len(n.get("content", "")) for n in nodes) // 4,
        )
