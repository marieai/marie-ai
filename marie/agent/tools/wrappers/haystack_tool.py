"""HaystackPipelineTool - Wrap Haystack pipelines as agent tools.

This module provides tools that wrap Haystack pipelines, enabling agents
to use RAG and other Haystack capabilities as callable tools.
"""

from __future__ import annotations

import asyncio
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from pydantic import BaseModel, Field

from marie.agent.tools.base import AgentTool, ToolMetadata, ToolOutput
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    try:
        from haystack import Pipeline
        from haystack.components.retrievers import InMemoryBM25Retriever
    except ImportError:
        Pipeline = Any

logger = MarieLogger("marie.agent.tools.wrappers.haystack")


class HaystackToolInput(BaseModel):
    """Default input schema for Haystack tools."""

    query: str = Field(..., description="Query string for the pipeline")
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional filters for retrieval",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Number of results to return",
    )


class HaystackPipelineTool(AgentTool):
    """Tool that wraps a Haystack pipeline.

    Enables agents to use Haystack pipelines (RAG, QA, etc.) as tools
    for information retrieval and processing.

    Example:
        ```python
        from haystack import Pipeline
        from haystack.components.retrievers import InMemoryBM25Retriever
        from haystack.components.readers import ExtractiveReader

        # Build Haystack pipeline
        pipeline = Pipeline()
        pipeline.add_component("retriever", InMemoryBM25Retriever(document_store))
        pipeline.add_component("reader", ExtractiveReader())
        pipeline.connect("retriever", "reader")

        # Wrap as tool
        tool = HaystackPipelineTool.from_pipeline(
            pipeline=pipeline,
            name="search_documents",
            description="Search and extract answers from documents",
            input_mapping={"query": "retriever.query"},
            output_key="reader.answers",
        )

        # Use in agent
        result = tool.call(query="What is machine learning?")
        ```
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        metadata: ToolMetadata,
        input_mapping: Optional[Dict[str, str]] = None,
        output_key: Optional[str] = None,
        output_transformer: Optional[Callable[[Any], str]] = None,
    ):
        """Initialize HaystackPipelineTool.

        Args:
            pipeline: Haystack Pipeline instance
            metadata: Tool metadata
            input_mapping: Maps input field names to pipeline component inputs
                e.g., {"query": "retriever.query", "top_k": "retriever.top_k"}
            output_key: Key to extract from pipeline output (e.g., "reader.answers")
            output_transformer: Custom function to transform output to string
        """
        self._pipeline = pipeline
        self._metadata = metadata
        self._input_mapping = input_mapping or {"query": "query"}
        self._output_key = output_key
        self._output_transformer = (
            output_transformer or self._default_output_transformer
        )

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def _prepare_input(self, **kwargs: Any) -> Dict[str, Any]:
        """Prepare pipeline input from tool arguments.

        Args:
            **kwargs: Tool input arguments

        Returns:
            Dict formatted for pipeline.run()
        """
        pipeline_input: Dict[str, Any] = {}

        for input_key, pipeline_path in self._input_mapping.items():
            if input_key in kwargs and kwargs[input_key] is not None:
                # Parse pipeline path (e.g., "retriever.query")
                if "." in pipeline_path:
                    component, param = pipeline_path.split(".", 1)
                    if component not in pipeline_input:
                        pipeline_input[component] = {}
                    pipeline_input[component][param] = kwargs[input_key]
                else:
                    pipeline_input[pipeline_path] = kwargs[input_key]

        return pipeline_input

    def _extract_output(self, result: Dict[str, Any]) -> Any:
        """Extract relevant output from pipeline result.

        Args:
            result: Pipeline output dict

        Returns:
            Extracted output value
        """
        if self._output_key is None:
            return result

        # Parse output key (e.g., "reader.answers")
        if "." in self._output_key:
            component, key = self._output_key.split(".", 1)
            if component in result and key in result[component]:
                return result[component][key]
            # Try flat access
            if self._output_key in result:
                return result[self._output_key]
        elif self._output_key in result:
            return result[self._output_key]

        # Return full result if key not found
        return result

    def _default_output_transformer(self, output: Any) -> str:
        """Transform pipeline output to string.

        Args:
            output: Pipeline output

        Returns:
            String representation
        """
        if output is None:
            return "No results found."

        if isinstance(output, str):
            return output

        if isinstance(output, list):
            # Handle Haystack Answer objects
            results = []
            for item in output:
                if hasattr(item, "answer"):
                    # Haystack Answer object
                    answer_str = item.answer
                    if hasattr(item, "score") and item.score is not None:
                        answer_str += f" (score: {item.score:.3f})"
                    if hasattr(item, "document") and item.document:
                        doc = item.document
                        if hasattr(doc, "meta") and doc.meta:
                            source = doc.meta.get("source", doc.meta.get("name", ""))
                            if source:
                                answer_str += f" [source: {source}]"
                    results.append(answer_str)
                elif hasattr(item, "content"):
                    # Haystack Document object
                    results.append(item.content)
                else:
                    results.append(str(item))

            return "\n\n".join(results) if results else "No results found."

        # Try JSON serialization
        try:
            return json.dumps(output, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            return str(output)

    def call(self, **kwargs: Any) -> ToolOutput:
        """Execute the Haystack pipeline synchronously.

        Args:
            **kwargs: Input arguments (query, filters, top_k, etc.)

        Returns:
            ToolOutput with results
        """
        query = kwargs.get("query", kwargs.get("input", ""))

        try:
            # Prepare input
            pipeline_input = self._prepare_input(**kwargs)

            # Run pipeline
            result = self._pipeline.run(pipeline_input)

            # Extract and transform output
            extracted = self._extract_output(result)
            output_str = self._output_transformer(extracted)

            return ToolOutput(
                content=output_str,
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
            )

        except Exception as e:
            logger.error(f"Haystack pipeline '{self.name}' failed: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=None,
                is_error=True,
            )

    async def acall(self, **kwargs: Any) -> ToolOutput:
        """Execute the Haystack pipeline asynchronously.

        Args:
            **kwargs: Input arguments

        Returns:
            ToolOutput with results
        """
        query = kwargs.get("query", kwargs.get("input", ""))

        try:
            pipeline_input = self._prepare_input(**kwargs)

            # Check for async run method
            if hasattr(self._pipeline, "run_async"):
                result = await self._pipeline.run_async(pipeline_input)
            else:
                # Run sync in thread pool
                result = await asyncio.to_thread(self._pipeline.run, pipeline_input)

            extracted = self._extract_output(result)
            output_str = self._output_transformer(extracted)

            return ToolOutput(
                content=output_str,
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=result,
            )

        except Exception as e:
            logger.error(f"Haystack pipeline '{self.name}' failed: {e}")
            return ToolOutput(
                content=f"Error: {str(e)}",
                tool_name=self.name,
                raw_input=kwargs,
                raw_output=None,
                is_error=True,
            )

    @classmethod
    def from_pipeline(
        cls,
        pipeline: "Pipeline",
        name: str,
        description: str,
        input_mapping: Optional[Dict[str, str]] = None,
        output_key: Optional[str] = None,
        fn_schema: Optional[Type[BaseModel]] = None,
        output_transformer: Optional[Callable] = None,
    ) -> "HaystackPipelineTool":
        """Create a HaystackPipelineTool from a pipeline.

        Args:
            pipeline: Haystack Pipeline instance
            name: Tool name
            description: Tool description
            input_mapping: Maps input fields to pipeline component inputs
            output_key: Key to extract from pipeline output
            fn_schema: Custom input schema
            output_transformer: Custom output transformer

        Returns:
            Configured HaystackPipelineTool
        """
        metadata = ToolMetadata(
            name=name,
            description=description,
            fn_schema=fn_schema or HaystackToolInput,
        )

        return cls(
            pipeline=pipeline,
            metadata=metadata,
            input_mapping=input_mapping,
            output_key=output_key,
            output_transformer=output_transformer,
        )


class RAGTool(HaystackPipelineTool):
    """Specialized Haystack tool for RAG (Retrieval-Augmented Generation).

    Provides a convenient interface for RAG pipelines with common defaults.
    """

    class InputSchema(BaseModel):
        """Input schema for RAG queries."""

        query: str = Field(..., description="Question or query to answer")
        top_k: int = Field(
            default=5,
            description="Number of documents to retrieve",
        )
        filters: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Metadata filters for retrieval",
        )

    @classmethod
    def from_pipeline(
        cls,
        pipeline: "Pipeline",
        name: str = "rag_search",
        description: str = "Search documents and generate answers using RAG",
        retriever_component: str = "retriever",
        generator_component: str = "generator",
        **kwargs: Any,
    ) -> "RAGTool":
        """Create a RAG tool from a Haystack RAG pipeline.

        Args:
            pipeline: Haystack RAG pipeline
            name: Tool name
            description: Tool description
            retriever_component: Name of retriever component in pipeline
            generator_component: Name of generator component in pipeline
            **kwargs: Additional arguments

        Returns:
            Configured RAGTool
        """
        input_mapping = {
            "query": f"{retriever_component}.query",
            "top_k": f"{retriever_component}.top_k",
            "filters": f"{retriever_component}.filters",
        }

        # Try to find the generator output key
        output_key = None
        if generator_component:
            # Common output keys for generators
            for key in ["replies", "answers", "generated_text"]:
                output_key = f"{generator_component}.{key}"
                break

        return super().from_pipeline(
            pipeline=pipeline,
            name=name,
            description=description,
            input_mapping=input_mapping,
            output_key=output_key,
            fn_schema=cls.InputSchema,
            **kwargs,
        )


class DocumentSearchTool(HaystackPipelineTool):
    """Specialized tool for document search/retrieval.

    Wraps Haystack retrieval pipelines for semantic or keyword search.
    """

    class InputSchema(BaseModel):
        """Input schema for document search."""

        query: str = Field(..., description="Search query")
        top_k: int = Field(default=10, description="Number of documents to return")
        filters: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Metadata filters",
        )

    @classmethod
    def from_document_store(
        cls,
        document_store: Any,
        name: str = "search_documents",
        description: str = "Search for relevant documents",
        retriever_type: str = "bm25",
        **kwargs: Any,
    ) -> "DocumentSearchTool":
        """Create a document search tool from a document store.

        Args:
            document_store: Haystack DocumentStore instance
            name: Tool name
            description: Tool description
            retriever_type: Type of retriever ('bm25', 'embedding', 'hybrid')
            **kwargs: Additional retriever arguments

        Returns:
            Configured DocumentSearchTool
        """
        try:
            from haystack import Pipeline

            # Create appropriate retriever
            if retriever_type == "bm25":
                from haystack.components.retrievers import InMemoryBM25Retriever

                retriever = InMemoryBM25Retriever(
                    document_store=document_store, **kwargs
                )
            elif retriever_type == "embedding":
                from haystack.components.retrievers import InMemoryEmbeddingRetriever

                retriever = InMemoryEmbeddingRetriever(
                    document_store=document_store, **kwargs
                )
            else:
                raise ValueError(f"Unknown retriever type: {retriever_type}")

            # Build simple retrieval pipeline
            pipeline = Pipeline()
            pipeline.add_component("retriever", retriever)

            input_mapping = {
                "query": "retriever.query",
                "top_k": "retriever.top_k",
                "filters": "retriever.filters",
            }

            metadata = ToolMetadata(
                name=name,
                description=description,
                fn_schema=cls.InputSchema,
            )

            return cls(
                pipeline=pipeline,
                metadata=metadata,
                input_mapping=input_mapping,
                output_key="retriever.documents",
            )

        except ImportError as e:
            raise ImportError(
                f"Haystack is required for DocumentSearchTool: {e}. "
                "Install with: pip install haystack-ai"
            )
