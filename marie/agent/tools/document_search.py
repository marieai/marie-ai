"""Document Search Tool for agentic RAG.

This module provides a tool that enables agents to search through
uploaded documents for relevant information. It follows the modern
agentic pattern where the agent decides when to call the search tool.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    from marie.rag.retriever import RAGRetriever

logger = MarieLogger("marie.agent.tools.document_search").logger


class DocumentSearchInput(BaseModel):
    """Input schema for document search tool."""

    query: str = Field(
        ...,
        description="Search query to find relevant documents. Be specific and include key terms.",
    )
    source_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific document source IDs to search. If not provided, searches all available sources.",
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return. Use more for broad questions, fewer for specific lookups.",
    )


class DocumentSearchTool(AgentTool):
    """Modern document search tool for agentic RAG.

    This tool enables agents to search through uploaded documents to find
    relevant information. Like ChatGPT's file_search or Claude's retrieval
    tool, the agent decides when to call this based on the user's question.

    The tool returns formatted results that the agent can use to synthesize
    an answer with proper citations.

    Example agent reasoning:
        User: "What's in my API docs about authentication?"
        Agent: I need to search the documents for auth info
        Agent: → calls document_search(query="authentication API methods")
        Agent: Got results about JWT, OAuth, etc.
        Agent: → synthesizes answer with citations

    Example usage:
        ```python
        from marie.rag import RAGRetriever
        from marie.agent.tools import DocumentSearchTool

        retriever = RAGRetriever(vector_store=store, embeddings=embeddings)
        tool = DocumentSearchTool(
            retriever=retriever,
            available_sources=["api_docs", "user_guide"],
        )

        # Use in agent
        agent = ReactAgent(
            llm=llm,
            function_list=[tool],
        )
        ```
    """

    def __init__(
        self,
        retriever: "RAGRetriever",
        available_sources: Optional[List[str]] = None,
        name: str = "document_search",
        description: Optional[str] = None,
    ):
        """Initialize DocumentSearchTool.

        Args:
            retriever: RAGRetriever instance for searching.
            available_sources: Limit searches to specific source IDs.
                If None, searches all sources the retriever has access to.
            name: Tool name for registration.
            description: Custom description. If None, uses default.
        """
        self._retriever = retriever
        self._available_sources = available_sources
        self._name = name
        self._description = description or (
            "Search through uploaded documents to find relevant information. "
            "Use this when you need to answer questions about the user's documents, "
            "find specific information, or cite document sources. "
            "Returns relevant excerpts with source citations."
        )

    @property
    def metadata(self) -> ToolMetadata:
        """Return tool metadata."""
        return ToolMetadata(
            name=self._name,
            description=self._description,
            fn_schema=DocumentSearchInput,
        )

    def call(
        self,
        query: str,
        source_ids: Optional[List[str]] = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> ToolOutput:
        """Execute document search synchronously.

        Args:
            query: Search query string.
            source_ids: Filter to specific sources.
            top_k: Number of results.
            **kwargs: Additional arguments.

        Returns:
            ToolOutput with formatted results and citations.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.acall(query=query, source_ids=source_ids, top_k=top_k, **kwargs)
        )

    async def acall(
        self,
        query: str,
        source_ids: Optional[List[str]] = None,
        top_k: int = 5,
        **kwargs: Any,
    ) -> ToolOutput:
        """Execute document search asynchronously.

        Args:
            query: Search query string.
            source_ids: Filter to specific sources.
            top_k: Number of results.
            **kwargs: Additional arguments.

        Returns:
            ToolOutput with formatted results and citations.
        """
        # Determine which sources to search
        search_sources = source_ids
        if search_sources is None:
            search_sources = self._available_sources

        logger.info(f"Document search: query='{query[:50]}...', top_k={top_k}")

        try:
            # Perform retrieval
            result = await self._retriever.aretrieve(
                query=query,
                source_ids=search_sources,
                top_k=top_k,
            )

            # Format results for agent consumption
            formatted = self._format_results(result)

            return ToolOutput(
                content=formatted,
                tool_name=self._name,
                raw_input={
                    "query": query,
                    "source_ids": search_sources,
                    "top_k": top_k,
                },
                raw_output={
                    "nodes": result.nodes,
                    "sources": [s.model_dump() for s in result.sources],
                    "query": result.query,
                    "rephrased_query": result.rephrased_query,
                },
            )

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return ToolOutput(
                content=f"Search failed: {str(e)}",
                tool_name=self._name,
                raw_input={
                    "query": query,
                    "source_ids": search_sources,
                    "top_k": top_k,
                },
                raw_output=None,
                is_error=True,
            )

    def _format_results(self, result) -> str:
        """Format search results for agent to process.

        Creates a readable format that agents can use to synthesize
        answers with proper citations.
        """
        if not result.nodes:
            return "No relevant documents found for this query."

        formatted_parts = []
        for i, (node, source) in enumerate(zip(result.nodes, result.sources), 1):
            content = node.get("content", "")

            # Build citation reference
            citation = f"[{source.filename}]"
            if source.page:
                citation = f"[{source.filename}, p.{source.page}]"

            # Truncate very long content
            if len(content) > 1000:
                content = content[:1000] + "..."

            # Format with similarity score for context
            similarity_pct = int(source.similarity * 100)
            formatted_parts.append(
                f"**Source {i}** {citation} (relevance: {similarity_pct}%)\n{content}"
            )

        return "\n\n---\n\n".join(formatted_parts)


class MultiDocumentSearchTool(AgentTool):
    """Tool for searching multiple document collections with one call.

    Useful when the agent needs to compare information across different
    document sources.
    """

    def __init__(
        self,
        retriever: "RAGRetriever",
        source_configs: Dict[str, Dict[str, Any]],
        name: str = "multi_document_search",
    ):
        """Initialize MultiDocumentSearchTool.

        Args:
            retriever: RAGRetriever instance.
            source_configs: Dictionary mapping source names to config:
                {
                    "api_docs": {"source_ids": ["source_1"], "top_k": 3},
                    "user_guide": {"source_ids": ["source_2"], "top_k": 3},
                }
            name: Tool name.
        """
        self._retriever = retriever
        self._source_configs = source_configs
        self._name = name

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self._name,
            description=(
                "Search multiple document collections simultaneously. "
                f"Available collections: {', '.join(self._source_configs.keys())}. "
                "Returns results from each collection with citations."
            ),
            fn_schema=self._build_schema(),
        )

    def _build_schema(self):
        """Build dynamic schema based on source configs."""

        class MultiSearchInput(BaseModel):
            query: str = Field(..., description="Search query")
            collections: Optional[List[str]] = Field(
                default=None,
                description=f"Collections to search. Options: {', '.join(self._source_configs.keys())}",
            )

        return MultiSearchInput

    def call(
        self, query: str, collections: Optional[List[str]] = None, **kwargs
    ) -> ToolOutput:
        return asyncio.get_event_loop().run_until_complete(
            self.acall(query=query, collections=collections, **kwargs)
        )

    async def acall(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolOutput:
        """Search multiple collections."""
        collections = collections or list(self._source_configs.keys())

        all_results = {}
        all_sources = []

        for coll_name in collections:
            if coll_name not in self._source_configs:
                continue

            config = self._source_configs[coll_name]
            result = await self._retriever.aretrieve(
                query=query,
                source_ids=config.get("source_ids"),
                top_k=config.get("top_k", 3),
            )

            all_results[coll_name] = {
                "nodes": result.nodes,
                "sources": [s.model_dump() for s in result.sources],
            }
            all_sources.extend(result.sources)

        # Format for agent
        formatted = self._format_multi_results(all_results)

        return ToolOutput(
            content=formatted,
            tool_name=self._name,
            raw_input={"query": query, "collections": collections},
            raw_output={
                "results_by_collection": all_results,
                "sources": [s.model_dump() for s in all_sources],
            },
        )

    def _format_multi_results(self, results: Dict[str, Any]) -> str:
        """Format multi-collection results."""
        parts = []
        for coll_name, data in results.items():
            nodes = data.get("nodes", [])
            sources = data.get("sources", [])

            if not nodes:
                parts.append(f"## {coll_name}\nNo relevant documents found.")
                continue

            coll_parts = [f"## {coll_name}"]
            for node, source in zip(nodes, sources):
                content = node.get("content", "")[:500]
                filename = source.get("filename", "unknown")
                coll_parts.append(f"**[{filename}]**\n{content}...")

            parts.append("\n".join(coll_parts))

        return "\n\n" + "\n\n---\n\n".join(parts)
