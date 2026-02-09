"""Document Assistant for tool-based RAG.

This module provides a specialized assistant for document Q&A that uses
the DocumentSearchTool for agentic retrieval. The assistant decides when
to search documents and synthesizes answers with citations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Iterator,
    List,
    Optional,
    Union,
)

from marie.agent.agents.assistant import ReactAgent
from marie.agent.message import Message
from marie.agent.tools.base import AgentTool
from marie.agent.tools.document_search import DocumentSearchTool
from marie.logging_core.logger import MarieLogger
from marie.rag.models import SourceCitation

if TYPE_CHECKING:
    from marie.agent.llm_wrapper import BaseLLMWrapper
    from marie.rag.retriever import RAGRetriever

logger = MarieLogger("marie.agent.assistants.document_assistant").logger


class StreamEventType(str, Enum):
    """Types of events emitted during streaming chat."""

    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SOURCES = "sources"
    CONTENT = "content"
    DONE = "done"
    ERROR = "error"


@dataclass
class StreamEvent:
    """Event emitted during streaming chat."""

    type: StreamEventType
    data: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type.value,
            "data": self.data,
        }


class DocumentAssistant(ReactAgent):
    """Document Q&A assistant with tool-based RAG.

    A specialized assistant that provides document Q&A capabilities using
    the modern agentic RAG pattern. The agent has access to a document_search
    tool and decides when to search based on the user's question.

    Features:
    - Tool-based retrieval (agent decides when to search)
    - Multi-query support (agent can search multiple times)
    - Citation tracking and formatting
    - Streaming responses with tool call events
    - Configurable system prompt

    Example:
        ```python
        from marie.rag import RAGRetriever
        from marie.agent.assistants import DocumentAssistant
        from marie.agent.llm_wrapper import MarieEngineLLMWrapper

        # Setup retriever
        retriever = RAGRetriever(vector_store=store, embeddings=embeddings)

        # Create assistant
        assistant = DocumentAssistant(
            llm=MarieEngineLLMWrapper(engine_name="qwen2_5_vl_7b"),
            retriever=retriever,
            source_ids=["api_docs", "user_guide"],
        )

        # Chat with streaming
        async for event in assistant.chat_stream(
            message="What authentication methods are supported?",
            history=[],
        ):
            if event.type == StreamEventType.CONTENT:
                print(event.data, end="", flush=True)
            elif event.type == StreamEventType.SOURCES:
                print(f"\\n\\nSources: {event.data}")
        ```

    Streaming Events (SSE pattern):
        - tool_call: Agent is calling a tool {"name": "...", "arguments": {...}}
        - tool_result: Tool returned results {"name": "...", "result": "..."}
        - sources: Citations for the response [SourceCitation, ...]
        - content: Text content chunk (for streaming output)
        - done: Chat complete {"conversation_id": "...", "tool_calls": [...]}
    """

    DOCUMENT_ASSISTANT_PROMPT = """You are a helpful document assistant. You have access to the user's uploaded documents through the document_search tool.

When answering questions:
1. Use document_search to find relevant information in the documents
2. Cite your sources using the filenames provided in the search results
3. If you can't find information in the documents, say so clearly
4. You can search multiple times with different queries if needed
5. Synthesize information from multiple sources when relevant

Always provide accurate citations in your response. When citing, use the format [filename] or [filename, p.X] for specific pages.

If the user's question doesn't require searching documents (e.g., general greetings, clarifying questions), respond directly without searching."""

    def __init__(
        self,
        retriever: "RAGRetriever",
        source_ids: Optional[List[str]] = None,
        llm: Optional["BaseLLMWrapper"] = None,
        system_message: Optional[str] = None,
        additional_tools: Optional[List[AgentTool]] = None,
        name: str = "document_assistant",
        description: str = "Assistant for answering questions about documents",
        max_iterations: int = 10,
        search_top_k: int = 5,
        **kwargs: Any,
    ):
        """Initialize DocumentAssistant.

        Args:
            retriever: RAGRetriever for document search.
            source_ids: Limit searches to specific document sources.
            llm: LLM wrapper for generating responses.
            system_message: Custom system prompt. If None, uses default.
            additional_tools: Extra tools beyond document_search.
            name: Assistant name.
            description: Assistant description.
            max_iterations: Max reasoning iterations.
            search_top_k: Default number of search results.
            **kwargs: Additional ReactAgent arguments.
        """
        # Create document search tool
        search_tool = DocumentSearchTool(
            retriever=retriever,
            available_sources=source_ids,
            name="document_search",
        )

        # Build tool list
        tools: List[AgentTool] = [search_tool]
        if additional_tools:
            tools.extend(additional_tools)

        # Use custom system message or default
        prompt = system_message or self.DOCUMENT_ASSISTANT_PROMPT

        super().__init__(
            function_list=tools,
            llm=llm,
            system_message=prompt,
            name=name,
            description=description,
            max_iterations=max_iterations,
            **kwargs,
        )

        self._retriever = retriever
        self._source_ids = source_ids
        self._search_top_k = search_top_k

    def chat(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Synchronous chat with the assistant.

        Args:
            message: User message.
            history: Conversation history as list of {"role": "...", "content": "..."}.
            **kwargs: Additional arguments.

        Returns:
            Dictionary with response and metadata:
            {
                "response": "...",
                "sources": [SourceCitation, ...],
                "tool_calls": [...],
            }
        """
        # Build messages
        messages = self._build_messages(message, history)

        # Collect tool calls and sources
        tool_calls = []
        sources = []
        final_response = ""

        # Run agent
        for responses in self.run(messages, **kwargs):
            if responses:
                last_msg = responses[-1]
                if last_msg.content:
                    final_response = last_msg.content

                # Check for tool calls in the response
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tool_calls.append(
                            {
                                "name": (
                                    tc.function.name
                                    if hasattr(tc.function, "name")
                                    else tc.get("name")
                                ),
                                "arguments": (
                                    tc.function.arguments
                                    if hasattr(tc.function, "arguments")
                                    else tc.get("arguments")
                                ),
                            }
                        )

                # Extract sources from tool results
                if last_msg.role == "tool" or last_msg.role == "function":
                    sources.extend(self._extract_sources_from_tool_result(last_msg))

        return {
            "response": final_response,
            "sources": sources,
            "tool_calls": tool_calls,
        }

    async def chat_stream(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Streaming chat with SSE-style events.

        Yields events as the assistant processes the request, including
        tool calls, tool results, sources, and content chunks.

        Args:
            message: User message.
            history: Conversation history.
            **kwargs: Additional arguments.

        Yields:
            StreamEvent objects with type and data.
        """
        messages = self._build_messages(message, history)
        collected_sources: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []

        try:
            for responses in self.run(messages, **kwargs):
                if not responses:
                    continue

                last_msg = responses[-1]

                # Handle tool calls
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        tc_info = {
                            "name": (
                                tc.function.name
                                if hasattr(tc.function, "name")
                                else tc.get("name")
                            ),
                            "arguments": (
                                tc.function.arguments
                                if hasattr(tc.function, "arguments")
                                else tc.get("arguments")
                            ),
                        }
                        tool_calls.append(tc_info)
                        yield StreamEvent(
                            type=StreamEventType.TOOL_CALL,
                            data=tc_info,
                        )

                # Handle tool results
                if last_msg.role == "tool" or last_msg.role == "function":
                    # Extract sources from the tool result
                    new_sources = self._extract_sources_from_tool_result(last_msg)
                    collected_sources.extend(new_sources)

                    yield StreamEvent(
                        type=StreamEventType.TOOL_RESULT,
                        data={
                            "name": (
                                last_msg.name
                                if hasattr(last_msg, "name")
                                else "document_search"
                            ),
                            "result": (
                                (last_msg.content[:500] + "...")
                                if len(last_msg.content or "") > 500
                                else last_msg.content
                            ),
                        },
                    )

                # Handle content
                if last_msg.role == "assistant" and last_msg.content:
                    # Check if this is a final response (no tool calls pending)
                    if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
                        # Emit sources before final content
                        if collected_sources:
                            yield StreamEvent(
                                type=StreamEventType.SOURCES,
                                data=collected_sources,
                            )

                        yield StreamEvent(
                            type=StreamEventType.CONTENT,
                            data=last_msg.content,
                        )

            # Done event
            yield StreamEvent(
                type=StreamEventType.DONE,
                data={
                    "tool_calls": tool_calls,
                    "source_count": len(collected_sources),
                },
            )

        except Exception as e:
            logger.error(f"Chat stream error: {e}")
            yield StreamEvent(
                type=StreamEventType.ERROR,
                data={"error": str(e)},
            )

    def _build_messages(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> List[Message]:
        """Build message list for agent."""
        messages = []

        # Add history
        if history:
            for h in history:
                messages.append(
                    Message(
                        role=h.get("role", "user"),
                        content=h.get("content", ""),
                    )
                )

        # Add current message
        messages.append(Message(role="user", content=message))

        return messages

    def _extract_sources_from_tool_result(
        self,
        msg: Message,
    ) -> List[Dict[str, Any]]:
        """Extract source citations from tool result message."""
        sources = []

        # Try to parse tool result content as JSON to get raw_output
        try:
            if hasattr(msg, "content") and msg.content:
                # Check if this is a search result with sources
                # The DocumentSearchTool stores sources in raw_output
                # but the message content has the formatted results

                # For now, extract basic citation info from the formatted content
                # In practice, the raw_output would be accessible if we track it

                # Simple extraction from formatted content
                content = msg.content
                if "**Source" in content and "[" in content:
                    # Extract citations like [filename.pdf]
                    import re

                    citations = re.findall(
                        r'\[([^\]]+\.(?:pdf|docx?|txt|md)(?:, p\.\d+)?)\]',
                        content,
                        re.IGNORECASE,
                    )
                    for cite in citations:
                        parts = cite.split(", p.")
                        filename = parts[0]
                        page = int(parts[1]) if len(parts) > 1 else None
                        sources.append(
                            {
                                "filename": filename,
                                "page": page,
                                "source": "document_search",
                            }
                        )

        except Exception as e:
            logger.debug(f"Could not extract sources from tool result: {e}")

        return sources

    def get_source_ids(self) -> Optional[List[str]]:
        """Get configured source IDs."""
        return self._source_ids

    def set_source_ids(self, source_ids: List[str]) -> None:
        """Update the source IDs for search filtering.

        Args:
            source_ids: New list of source IDs to search.
        """
        self._source_ids = source_ids
        # Update the search tool
        for tool in self.tools:
            if isinstance(tool, DocumentSearchTool):
                tool._available_sources = source_ids


class ConversationalDocumentAssistant(DocumentAssistant):
    """Document assistant with persistent conversation memory.

    Extends DocumentAssistant with automatic conversation tracking,
    making it easier to maintain multi-turn conversations.
    """

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        max_history: int = 20,
        **kwargs: Any,
    ):
        """Initialize with conversation tracking.

        Args:
            conversation_id: Unique ID for this conversation.
            max_history: Maximum messages to keep in history.
            **kwargs: DocumentAssistant arguments.
        """
        super().__init__(**kwargs)
        self._conversation_id = conversation_id
        self._max_history = max_history
        self._history: List[Dict[str, str]] = []

    @property
    def conversation_id(self) -> Optional[str]:
        return self._conversation_id

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)

    def chat(
        self,
        message: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Chat with automatic history management.

        Args:
            message: User message.
            **kwargs: Additional arguments.

        Returns:
            Response dictionary with sources.
        """
        # Use internal history
        result = super().chat(message, history=self._history, **kwargs)

        # Update history
        self._history.append({"role": "user", "content": message})
        self._history.append({"role": "assistant", "content": result["response"]})

        # Trim if needed
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-(self._max_history * 2) :]

        return result

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history = []

    def add_to_history(self, role: str, content: str) -> None:
        """Manually add a message to history."""
        self._history.append({"role": role, "content": content})
