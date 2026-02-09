"""Tests for DocumentSearchTool and MultiDocumentSearchTool."""
import json
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from marie.agent.tools import ToolOutput
from marie.agent.tools.document_search import (
    DocumentSearchInput,
    DocumentSearchTool,
    MultiDocumentSearchTool,
)


@dataclass
class MockSource:
    """Mock source for testing."""
    filename: str
    page: Optional[int] = None
    similarity: float = 0.85

    def model_dump(self):
        return {"filename": self.filename, "page": self.page, "similarity": self.similarity}


@dataclass
class MockRetrievalResult:
    """Mock retrieval result."""
    nodes: List[dict]
    sources: List[MockSource]
    query: str
    rephrased_query: Optional[str] = None


class MockRetriever:
    """Mock RAGRetriever for testing."""

    def __init__(self, results: Optional[MockRetrievalResult] = None):
        self._results = results or MockRetrievalResult(
            nodes=[{"content": "Test content"}],
            sources=[MockSource(filename="test.pdf", page=1)],
            query="test query",
        )
        self.aretrieve = AsyncMock(return_value=self._results)


class TestDocumentSearchToolMetadata:
    """Tests for DocumentSearchTool metadata."""

    def test_default_name(self):
        """Tool should have default name 'document_search'."""
        retriever = MockRetriever()
        tool = DocumentSearchTool(retriever=retriever)
        assert tool.name == "document_search"

    def test_custom_name(self):
        """Tool should accept custom name."""
        retriever = MockRetriever()
        tool = DocumentSearchTool(retriever=retriever, name="my_search")
        assert tool.name == "my_search"

    def test_default_description(self):
        """Tool should have default description."""
        retriever = MockRetriever()
        tool = DocumentSearchTool(retriever=retriever)
        assert "Search through uploaded documents" in tool.description

    def test_custom_description(self):
        """Tool should accept custom description."""
        retriever = MockRetriever()
        tool = DocumentSearchTool(
            retriever=retriever,
            description="Custom search description"
        )
        assert tool.description == "Custom search description"

    def test_input_schema(self):
        """Tool should have DocumentSearchInput schema."""
        retriever = MockRetriever()
        tool = DocumentSearchTool(retriever=retriever)
        assert tool.metadata.fn_schema is DocumentSearchInput


class TestDocumentSearchToolAcall:
    """Tests for DocumentSearchTool async execution."""

    @pytest.mark.asyncio
    async def test_basic_search(self):
        """acall should return search results."""
        result = MockRetrievalResult(
            nodes=[{"content": "Authentication uses JWT tokens."}],
            sources=[MockSource(filename="api_docs.pdf", page=5, similarity=0.92)],
            query="authentication",
        )
        retriever = MockRetriever(result)
        tool = DocumentSearchTool(retriever=retriever)

        output = await tool.acall(query="authentication")

        assert isinstance(output, ToolOutput)
        assert output.is_error is False
        assert "api_docs.pdf" in output.content
        assert "JWT tokens" in output.content
        retriever.aretrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_source_filter(self):
        """acall should pass source_ids to retriever."""
        retriever = MockRetriever()
        tool = DocumentSearchTool(retriever=retriever)

        await tool.acall(query="test", source_ids=["doc1", "doc2"])

        retriever.aretrieve.assert_called_once()
        call_kwargs = retriever.aretrieve.call_args.kwargs
        assert call_kwargs["source_ids"] == ["doc1", "doc2"]

    @pytest.mark.asyncio
    async def test_search_with_available_sources(self):
        """acall should use available_sources when source_ids not provided."""
        retriever = MockRetriever()
        tool = DocumentSearchTool(
            retriever=retriever,
            available_sources=["api_docs", "user_guide"]
        )

        await tool.acall(query="test")

        call_kwargs = retriever.aretrieve.call_args.kwargs
        assert call_kwargs["source_ids"] == ["api_docs", "user_guide"]

    @pytest.mark.asyncio
    async def test_search_with_top_k(self):
        """acall should pass top_k to retriever."""
        retriever = MockRetriever()
        tool = DocumentSearchTool(retriever=retriever)

        await tool.acall(query="test", top_k=10)

        call_kwargs = retriever.aretrieve.call_args.kwargs
        assert call_kwargs["top_k"] == 10

    @pytest.mark.asyncio
    async def test_search_no_results(self):
        """acall should handle no results gracefully."""
        result = MockRetrievalResult(nodes=[], sources=[], query="nothing")
        retriever = MockRetriever(result)
        tool = DocumentSearchTool(retriever=retriever)

        output = await tool.acall(query="nothing")

        assert output.is_error is False
        assert "No relevant documents found" in output.content

    @pytest.mark.asyncio
    async def test_search_error_handling(self):
        """acall should handle retriever errors."""
        retriever = MockRetriever()
        retriever.aretrieve = AsyncMock(side_effect=Exception("Connection failed"))
        tool = DocumentSearchTool(retriever=retriever)

        output = await tool.acall(query="test")

        assert output.is_error is True
        assert "failed" in output.content.lower()

    @pytest.mark.asyncio
    async def test_raw_output_structure(self):
        """acall should include structured raw_output."""
        result = MockRetrievalResult(
            nodes=[{"content": "Test"}],
            sources=[MockSource(filename="test.pdf")],
            query="test",
            rephrased_query="expanded test query",
        )
        retriever = MockRetriever(result)
        tool = DocumentSearchTool(retriever=retriever)

        output = await tool.acall(query="test")

        assert "nodes" in output.raw_output
        assert "sources" in output.raw_output
        assert "query" in output.raw_output


class TestDocumentSearchToolFormatting:
    """Tests for result formatting."""

    @pytest.mark.asyncio
    async def test_format_with_page_number(self):
        """Results should include page numbers when available."""
        result = MockRetrievalResult(
            nodes=[{"content": "Page content"}],
            sources=[MockSource(filename="doc.pdf", page=42)],
            query="test",
        )
        retriever = MockRetriever(result)
        tool = DocumentSearchTool(retriever=retriever)

        output = await tool.acall(query="test")

        assert "p.42" in output.content

    @pytest.mark.asyncio
    async def test_format_without_page_number(self):
        """Results should work without page numbers."""
        result = MockRetrievalResult(
            nodes=[{"content": "Content"}],
            sources=[MockSource(filename="doc.pdf", page=None)],
            query="test",
        )
        retriever = MockRetriever(result)
        tool = DocumentSearchTool(retriever=retriever)

        output = await tool.acall(query="test")

        assert "[doc.pdf]" in output.content
        assert "p." not in output.content

    @pytest.mark.asyncio
    async def test_format_truncates_long_content(self):
        """Very long content should be truncated."""
        long_content = "A" * 2000
        result = MockRetrievalResult(
            nodes=[{"content": long_content}],
            sources=[MockSource(filename="doc.pdf")],
            query="test",
        )
        retriever = MockRetriever(result)
        tool = DocumentSearchTool(retriever=retriever)

        output = await tool.acall(query="test")

        assert "..." in output.content
        assert len(output.content) < len(long_content) + 200


class TestMultiDocumentSearchToolMetadata:
    """Tests for MultiDocumentSearchTool metadata."""

    def test_default_name(self):
        """Tool should have default name."""
        retriever = MockRetriever()
        tool = MultiDocumentSearchTool(
            retriever=retriever,
            source_configs={"api": {"source_ids": ["s1"]}},
        )
        assert tool.name == "multi_document_search"

    def test_description_includes_collections(self):
        """Description should list available collections."""
        retriever = MockRetriever()
        tool = MultiDocumentSearchTool(
            retriever=retriever,
            source_configs={
                "api_docs": {"source_ids": ["s1"]},
                "user_guide": {"source_ids": ["s2"]},
            },
        )
        assert "api_docs" in tool.description
        assert "user_guide" in tool.description


class TestMultiDocumentSearchToolAcall:
    """Tests for MultiDocumentSearchTool async execution."""

    @pytest.mark.asyncio
    async def test_search_all_collections(self):
        """acall should search all collections by default."""
        retriever = MockRetriever()
        tool = MultiDocumentSearchTool(
            retriever=retriever,
            source_configs={
                "api": {"source_ids": ["s1"], "top_k": 3},
                "guide": {"source_ids": ["s2"], "top_k": 3},
            },
        )

        await tool.acall(query="test")

        # Should call aretrieve for each collection
        assert retriever.aretrieve.call_count == 2

    @pytest.mark.asyncio
    async def test_search_specific_collections(self):
        """acall should filter to specific collections."""
        retriever = MockRetriever()
        tool = MultiDocumentSearchTool(
            retriever=retriever,
            source_configs={
                "api": {"source_ids": ["s1"]},
                "guide": {"source_ids": ["s2"]},
                "faq": {"source_ids": ["s3"]},
            },
        )

        await tool.acall(query="test", collections=["api", "faq"])

        # Should only call for specified collections
        assert retriever.aretrieve.call_count == 2

    @pytest.mark.asyncio
    async def test_search_ignores_unknown_collections(self):
        """acall should ignore unknown collection names."""
        retriever = MockRetriever()
        tool = MultiDocumentSearchTool(
            retriever=retriever,
            source_configs={"api": {"source_ids": ["s1"]}},
        )

        await tool.acall(query="test", collections=["api", "unknown"])

        # Should only call for known collection
        assert retriever.aretrieve.call_count == 1

    @pytest.mark.asyncio
    async def test_output_structure(self):
        """acall should return properly structured output."""
        retriever = MockRetriever()
        tool = MultiDocumentSearchTool(
            retriever=retriever,
            source_configs={"api": {"source_ids": ["s1"]}},
        )

        output = await tool.acall(query="test")

        assert isinstance(output, ToolOutput)
        assert output.is_error is False
        assert "results_by_collection" in output.raw_output


class TestDocumentSearchToolSync:
    """Tests for synchronous call method."""

    def test_call_bridges_to_acall(self):
        """call() should bridge to acall()."""
        result = MockRetrievalResult(
            nodes=[{"content": "Test"}],
            sources=[MockSource(filename="test.pdf")],
            query="test",
        )
        retriever = MockRetriever(result)
        tool = DocumentSearchTool(retriever=retriever)

        output = tool.call(query="test")

        assert isinstance(output, ToolOutput)
        assert output.is_error is False
