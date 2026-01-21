"""Haystack agent backend for Marie agent framework.

This module provides a backend that wraps Haystack pipelines,
enabling Haystack-based agents and RAG systems to be used
within the Marie agent framework.
"""

from __future__ import annotations

import asyncio
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from pydantic import Field

from marie.agent.backends.base import (
    AgentBackend,
    AgentResult,
    AgentStatus,
    BackendConfig,
    ToolCallRecord,
)
from marie.agent.message import Message
from marie.agent.tools.base import AgentTool
from marie.logging_core.logger import MarieLogger

if TYPE_CHECKING:
    try:
        from haystack import Pipeline
    except ImportError:
        Pipeline = Any

logger = MarieLogger("marie.agent.backends.haystack")


class HaystackBackendConfig(BackendConfig):
    """Configuration for Haystack backend."""

    pipeline_type: str = Field(
        default="rag",
        description="Type of pipeline (rag, extractive_qa, generative_qa, custom)",
    )
    retriever_component: str = Field(
        default="retriever",
        description="Name of retriever component in pipeline",
    )
    generator_component: Optional[str] = Field(
        default="generator",
        description="Name of generator component (if any)",
    )
    reader_component: Optional[str] = Field(
        default=None,
        description="Name of reader component (for extractive QA)",
    )
    top_k: int = Field(default=5, description="Default number of documents to retrieve")
    return_documents: bool = Field(
        default=True,
        description="Whether to include retrieved documents in result",
    )


class HaystackAgentBackend(AgentBackend):
    """Backend that wraps Haystack pipelines.

    Enables using Haystack pipelines (RAG, QA, etc.) as agent backends
    for information retrieval and question answering tasks.

    Example:
        ```python
        from haystack import Pipeline
        from haystack.components.retrievers import InMemoryBM25Retriever
        from haystack.components.generators import OpenAIGenerator

        # Build RAG pipeline
        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("generator", generator)
        pipeline.connect("retriever", "generator")

        # Create backend
        backend = HaystackAgentBackend(
            pipeline=pipeline,
            config=HaystackBackendConfig(
                pipeline_type="rag",
                retriever_component="retriever",
                generator_component="generator",
            ),
        )

        # Run
        messages = [Message.user("What is machine learning?")]
        result = await backend.run(messages)
        print(result.output_text)
        ```
    """

    def __init__(
        self,
        pipeline: "Pipeline",
        config: Optional[HaystackBackendConfig] = None,
        input_builder: Optional[Callable[[List[Message]], Dict[str, Any]]] = None,
        output_extractor: Optional[Callable[[Dict[str, Any]], str]] = None,
        **kwargs: Any,
    ):
        """Initialize Haystack backend.

        Args:
            pipeline: Haystack Pipeline instance
            config: Backend configuration
            input_builder: Custom function to build pipeline input from messages
            output_extractor: Custom function to extract output from pipeline result
            **kwargs: Additional arguments
        """
        if config is None:
            config = HaystackBackendConfig(**kwargs)

        super().__init__(config=config)

        self._pipeline = pipeline
        self._input_builder = input_builder or self._default_input_builder
        self._output_extractor = output_extractor or self._default_output_extractor

    @property
    def haystack_config(self) -> HaystackBackendConfig:
        """Get typed configuration."""
        return self.config  # type: ignore

    def _default_input_builder(self, messages: List[Message]) -> Dict[str, Any]:
        """Build pipeline input from messages.

        Extracts the last user message as the query.

        Args:
            messages: Conversation messages

        Returns:
            Dict for pipeline.run()
        """
        # Find the last user message
        query = ""
        for msg in reversed(messages):
            if msg.role == "user":
                query = msg.text_content
                break

        if not query:
            # Fallback to last message content
            query = messages[-1].text_content if messages else ""

        # Build input for retriever
        retriever = self.haystack_config.retriever_component
        pipeline_input = {
            retriever: {
                "query": query,
                "top_k": self.haystack_config.top_k,
            }
        }

        return pipeline_input

    def _default_output_extractor(self, result: Dict[str, Any]) -> str:
        """Extract output from pipeline result.

        Args:
            result: Pipeline output dict

        Returns:
            Extracted string output
        """
        config = self.haystack_config

        # Try generator output first (RAG)
        if config.generator_component and config.generator_component in result:
            gen_result = result[config.generator_component]
            if "replies" in gen_result:
                replies = gen_result["replies"]
                if replies:
                    return (
                        replies[0] if isinstance(replies[0], str) else str(replies[0])
                    )

        # Try reader output (extractive QA)
        if config.reader_component and config.reader_component in result:
            reader_result = result[config.reader_component]
            if "answers" in reader_result:
                answers = reader_result["answers"]
                if answers:
                    # Format answers
                    formatted = []
                    for ans in answers[:3]:  # Top 3 answers
                        if hasattr(ans, "answer"):
                            text = ans.answer
                            if hasattr(ans, "score"):
                                text += f" (confidence: {ans.score:.2f})"
                            formatted.append(text)
                        else:
                            formatted.append(str(ans))
                    return "\n".join(formatted)

        # Try retriever output (documents)
        if config.retriever_component in result:
            ret_result = result[config.retriever_component]
            if "documents" in ret_result:
                docs = ret_result["documents"]
                if docs:
                    # Format documents
                    formatted = []
                    for i, doc in enumerate(docs[:5], 1):
                        content = doc.content if hasattr(doc, "content") else str(doc)
                        formatted.append(f"{i}. {content[:500]}...")
                    return "\n\n".join(formatted)

        # Fallback: return raw result
        return str(result)

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute the Haystack pipeline.

        Args:
            messages: Input messages (last user message used as query)
            tools: Not used by Haystack backend
            config: Optional override configuration
            **kwargs: Additional pipeline arguments

        Returns:
            AgentResult with pipeline output
        """
        start_time = time.time()

        try:
            # Build pipeline input
            pipeline_input = self._input_builder(messages)

            # Merge any additional kwargs
            for key, value in kwargs.items():
                if key in pipeline_input:
                    pipeline_input[key].update(value)
                else:
                    pipeline_input[key] = value

            # Run pipeline
            if hasattr(self._pipeline, "run_async"):
                result = await self._pipeline.run_async(pipeline_input)
            else:
                result = await asyncio.to_thread(self._pipeline.run, pipeline_input)

            # Extract output
            output_text = self._output_extractor(result)

            # Build response message
            response_msg = Message.assistant(content=output_text)

            duration_ms = (time.time() - start_time) * 1000

            return AgentResult(
                output=response_msg,
                messages=[response_msg.model_dump()],
                status=AgentStatus.COMPLETED,
                is_complete=True,
                metadata={
                    "duration_ms": duration_ms,
                    "pipeline_type": self.haystack_config.pipeline_type,
                    "raw_result": (
                        result if self.haystack_config.return_documents else None
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Haystack backend failed: {e}")
            return AgentResult(
                output="",
                status=AgentStatus.FAILED,
                error=str(e),
                is_complete=False,
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Haystack backend doesn't expose tools directly."""
        return []

    @classmethod
    def from_rag_pipeline(
        cls,
        document_store: Any,
        retriever_type: str = "bm25",
        generator_model: str = "gpt-3.5-turbo",
        top_k: int = 5,
        **kwargs: Any,
    ) -> "HaystackAgentBackend":
        """Create a RAG backend from components.

        Args:
            document_store: Haystack DocumentStore
            retriever_type: Type of retriever (bm25, embedding)
            generator_model: Model for generation
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments

        Returns:
            Configured HaystackAgentBackend
        """
        try:
            from haystack import Pipeline
            from haystack.components.builders import PromptBuilder
            from haystack.components.generators import OpenAIGenerator
        except ImportError:
            raise ImportError(
                "Haystack is required. Install with: pip install haystack-ai"
            )

        # Create retriever
        if retriever_type == "bm25":
            from haystack.components.retrievers import InMemoryBM25Retriever

            retriever = InMemoryBM25Retriever(
                document_store=document_store, top_k=top_k
            )
        elif retriever_type == "embedding":
            from haystack.components.retrievers import InMemoryEmbeddingRetriever

            retriever = InMemoryEmbeddingRetriever(
                document_store=document_store, top_k=top_k
            )
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

        # Create prompt builder
        prompt_template = """
        Answer the question based on the provided context.

        Context:
        {% for doc in documents %}
        {{ doc.content }}
        {% endfor %}

        Question: {{ query }}
        Answer:
        """
        prompt_builder = PromptBuilder(template=prompt_template)

        # Create generator
        generator = OpenAIGenerator(model=generator_model)

        # Build pipeline
        pipeline = Pipeline()
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("generator", generator)

        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder", "generator")

        config = HaystackBackendConfig(
            pipeline_type="rag",
            retriever_component="retriever",
            generator_component="generator",
            top_k=top_k,
        )

        # Custom input builder for this pipeline structure
        def rag_input_builder(messages: List[Message]) -> Dict[str, Any]:
            query = ""
            for msg in reversed(messages):
                if msg.role == "user":
                    query = msg.text_content
                    break
            return {
                "retriever": {"query": query},
                "prompt_builder": {"query": query},
            }

        return cls(
            pipeline=pipeline,
            config=config,
            input_builder=rag_input_builder,
            **kwargs,
        )


class SimpleHaystackBackend(AgentBackend):
    """Simplified Haystack backend for document retrieval only.

    Wraps a retriever for simple search without generation.
    """

    def __init__(
        self,
        document_store: Any,
        retriever_type: str = "bm25",
        top_k: int = 10,
        **kwargs: Any,
    ):
        """Initialize simple retrieval backend.

        Args:
            document_store: Haystack DocumentStore
            retriever_type: Type of retriever
            top_k: Number of results
            **kwargs: Additional arguments
        """
        super().__init__(config=BackendConfig(**kwargs))

        self.document_store = document_store
        self.retriever_type = retriever_type
        self.top_k = top_k
        self._retriever = self._create_retriever()

    def _create_retriever(self) -> Any:
        """Create the retriever component."""
        try:
            if self.retriever_type == "bm25":
                from haystack.components.retrievers import InMemoryBM25Retriever

                return InMemoryBM25Retriever(
                    document_store=self.document_store,
                    top_k=self.top_k,
                )
            elif self.retriever_type == "embedding":
                from haystack.components.retrievers import InMemoryEmbeddingRetriever

                return InMemoryEmbeddingRetriever(
                    document_store=self.document_store,
                    top_k=self.top_k,
                )
            else:
                raise ValueError(f"Unknown retriever type: {self.retriever_type}")
        except ImportError:
            raise ImportError(
                "Haystack is required. Install with: pip install haystack-ai"
            )

    async def run(
        self,
        messages: List[Message],
        tools: Optional[Dict[str, AgentTool]] = None,
        config: Optional[BackendConfig] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """Execute document retrieval.

        Args:
            messages: Input messages
            tools: Not used
            config: Not used
            **kwargs: Additional arguments

        Returns:
            AgentResult with retrieved documents
        """
        try:
            # Get query from last user message
            query = ""
            for msg in reversed(messages):
                if msg.role == "user":
                    query = msg.text_content
                    break

            # Run retriever
            result = await asyncio.to_thread(
                self._retriever.run,
                query=query,
                top_k=kwargs.get("top_k", self.top_k),
            )

            # Format documents
            documents = result.get("documents", [])
            formatted = []
            for i, doc in enumerate(documents, 1):
                content = doc.content if hasattr(doc, "content") else str(doc)
                meta = ""
                if hasattr(doc, "meta") and doc.meta:
                    source = doc.meta.get("source", doc.meta.get("file_path", ""))
                    if source:
                        meta = f" [Source: {source}]"
                formatted.append(f"{i}. {content[:500]}{meta}")

            output_text = "\n\n".join(formatted) if formatted else "No documents found."

            return AgentResult(
                output=Message.assistant(content=output_text),
                messages=[{"role": "assistant", "content": output_text}],
                status=AgentStatus.COMPLETED,
                is_complete=True,
                metadata={"document_count": len(documents)},
            )

        except Exception as e:
            logger.error(f"Simple Haystack backend failed: {e}")
            return AgentResult(
                output="",
                status=AgentStatus.FAILED,
                error=str(e),
                is_complete=False,
            )

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """No tools for simple backend."""
        return []
