"""Streaming types for chunk-based async LLM output.

Provides ``StreamChunk`` — a lightweight delta model that accumulates into a
full ``Message`` — plus event dataclasses for future emitter integration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from marie.agent.message import FunctionCall, Message, ToolCall


class StreamUsage(BaseModel):
    """Token usage reported with a streaming response."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StreamChunk(BaseModel):
    """Single streaming delta from an LLM.

    Chunks are designed to be merged: ``StreamChunk.from_chunks(list_of_chunks)``
    reconstructs the complete response.
    """

    content: Optional[str] = Field(default=None, description="Text delta")
    finish_reason: Optional[str] = Field(
        default=None,
        description="Finish reason when generation is complete",
    )
    tool_calls: Optional[List[ToolCall]] = Field(
        default=None,
        description="Complete tool calls (only set when valid)",
    )
    usage: Optional[StreamUsage] = Field(
        default=None,
        description="Token usage (typically on last chunk)",
    )
    # Metadata for consumers
    event_type: str = Field(
        default="token",
        description="Event type: token, tool_start, tool_result, error, done",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def merge(self, other: "StreamChunk") -> "StreamChunk":
        """Return a new chunk that is the accumulation of self + other."""
        merged_content: Optional[str] = None
        if self.content is not None or other.content is not None:
            merged_content = (self.content or "") + (other.content or "")

        merged_tool_calls = other.tool_calls if other.tool_calls else self.tool_calls

        return StreamChunk(
            content=merged_content,
            finish_reason=other.finish_reason or self.finish_reason,
            tool_calls=merged_tool_calls,
            usage=other.usage or self.usage,
            event_type=other.event_type,
            metadata={**self.metadata, **other.metadata},
        )

    @classmethod
    def from_chunks(cls, chunks: List["StreamChunk"]) -> "StreamChunk":
        """Reconstruct a full response from a list of deltas."""
        if not chunks:
            return cls(content="", finish_reason="stop")

        accumulated = chunks[0]
        for chunk in chunks[1:]:
            accumulated = accumulated.merge(chunk)
        return accumulated

    def to_message(self) -> Message:
        """Convert the accumulated chunk into a ``Message``."""
        return Message.assistant(
            content=self.content,
            tool_calls=self.tool_calls,
        )

    @classmethod
    def error(cls, error: str) -> "StreamChunk":
        """Create an error chunk."""
        return cls(
            content=f"Error: {error}",
            finish_reason="error",
            event_type="error",
        )

    @classmethod
    def tool_result_chunk(
        cls, tool_name: str, tool_call_id: str, result: str
    ) -> "StreamChunk":
        """Create a chunk representing a tool execution result."""
        return cls(
            content=None,
            event_type="tool_result",
            metadata={
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "tool_result": result,
            },
        )

    @classmethod
    def done(cls, usage: Optional[StreamUsage] = None) -> "StreamChunk":
        """Create a terminal 'done' chunk."""
        return cls(
            content=None,
            finish_reason="stop",
            event_type="done",
            usage=usage,
        )


class ToolCallAccumulator:
    """Accumulates partial tool-call deltas from OpenAI's indexed protocol.

    OpenAI streams tool calls as indexed fragments::

        delta.tool_calls = [
            {"index": 0, "id": "call_abc", "function": {"name": "search", "arguments": ""}},
        ]
        delta.tool_calls = [
            {"index": 0, "function": {"arguments": '{"qu'}},
        ]
        delta.tool_calls = [
            {"index": 0, "function": {"arguments": 'ery": "hello"}'}},
        ]

    This class buffers fragments by index and produces complete ``ToolCall``
    objects once the JSON arguments are parseable.
    """

    def __init__(self) -> None:
        self._calls: Dict[int, Dict[str, Any]] = {}

    def feed(self, delta_tool_calls: List[Any]) -> None:
        """Feed a list of tool-call deltas (from one SSE chunk)."""
        for tc_delta in delta_tool_calls:
            idx = getattr(tc_delta, "index", None)
            if idx is None:
                idx = tc_delta.get("index", 0) if isinstance(tc_delta, dict) else 0

            if idx not in self._calls:
                self._calls[idx] = {"id": None, "name": "", "arguments": ""}

            entry = self._calls[idx]

            # Extract id
            tc_id = (
                getattr(tc_delta, "id", None)
                if not isinstance(tc_delta, dict)
                else tc_delta.get("id")
            )
            if tc_id:
                entry["id"] = tc_id

            # Extract function fields
            fn = (
                getattr(tc_delta, "function", None)
                if not isinstance(tc_delta, dict)
                else tc_delta.get("function")
            )
            if fn is not None:
                fn_name = (
                    getattr(fn, "name", None)
                    if not isinstance(fn, dict)
                    else fn.get("name")
                )
                fn_args = (
                    getattr(fn, "arguments", None)
                    if not isinstance(fn, dict)
                    else fn.get("arguments")
                )
                if fn_name:
                    entry["name"] = fn_name
                if fn_args:
                    entry["arguments"] += fn_args

    def get_complete_calls(self) -> Optional[List[ToolCall]]:
        """Return tool calls if ALL accumulated calls have valid JSON arguments.

        Returns None if any call's arguments are not yet parseable.
        """
        if not self._calls:
            return None

        result: List[ToolCall] = []
        for idx in sorted(self._calls):
            entry = self._calls[idx]
            args = entry["arguments"]
            if not args:
                return None
            try:
                json.loads(args)
            except json.JSONDecodeError:
                return None
            result.append(
                ToolCall(
                    id=entry["id"] or f"call_{idx}",
                    type="function",
                    function=FunctionCall(
                        name=entry["name"],
                        arguments=args,
                    ),
                )
            )
        return result

    def reset(self) -> None:
        self._calls.clear()


# ---------------------------------------------------------------------------
# Event dataclasses for future emitter integration
# ---------------------------------------------------------------------------


@dataclass
class StreamStartEvent:
    """Emitted when streaming begins."""

    messages: List[Any]
    functions: Optional[List[Dict[str, Any]]] = None


@dataclass
class StreamTokenEvent:
    """Emitted for each token/chunk."""

    chunk: StreamChunk
    abort_fn: Optional[Callable[[], None]] = None


@dataclass
class StreamCompleteEvent:
    """Emitted when streaming completes successfully."""

    result: StreamChunk


@dataclass
class StreamErrorEvent:
    """Emitted on streaming error."""

    error: Exception
    partial_result: Optional[StreamChunk] = None
