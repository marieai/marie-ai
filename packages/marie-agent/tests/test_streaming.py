"""Tests for streaming types and tool-call accumulation."""

import json

from marie.agent.message import FunctionCall, Message, ToolCall
from marie.agent.streaming import StreamChunk, StreamUsage, ToolCallAccumulator


class TestStreamChunk:
    def test_merge_content(self):
        a = StreamChunk(content="Hello")
        b = StreamChunk(content=" world")
        merged = a.merge(b)
        assert merged.content == "Hello world"

    def test_merge_finish_reason(self):
        a = StreamChunk(content="Hi")
        b = StreamChunk(content="", finish_reason="stop")
        merged = a.merge(b)
        assert merged.finish_reason == "stop"

    def test_merge_none_content(self):
        a = StreamChunk(content=None)
        b = StreamChunk(content="text")
        merged = a.merge(b)
        assert merged.content == "text"

    def test_merge_both_none_content(self):
        a = StreamChunk(content=None)
        b = StreamChunk(content=None)
        merged = a.merge(b)
        assert merged.content is None

    def test_from_chunks_empty(self):
        result = StreamChunk.from_chunks([])
        assert result.content == ""
        assert result.finish_reason == "stop"

    def test_from_chunks_reconstructs_content(self):
        chunks = [
            StreamChunk(content="Hello"),
            StreamChunk(content=", "),
            StreamChunk(content="world"),
            StreamChunk(content="!", finish_reason="stop"),
        ]
        result = StreamChunk.from_chunks(chunks)
        assert result.content == "Hello, world!"
        assert result.finish_reason == "stop"

    def test_from_chunks_with_tool_calls(self):
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="search", arguments='{"q": "test"}'),
        )
        chunks = [
            StreamChunk(content="Let me search"),
            StreamChunk(content=None, tool_calls=[tc], finish_reason="stop"),
        ]
        result = StreamChunk.from_chunks(chunks)
        assert result.content == "Let me search"
        assert result.tool_calls is not None
        assert result.tool_calls[0].function.name == "search"

    def test_to_message(self):
        chunk = StreamChunk(content="Hello world", finish_reason="stop")
        msg = chunk.to_message()
        assert isinstance(msg, Message)
        assert msg.role == "assistant"
        assert msg.content == "Hello world"

    def test_to_message_with_tool_calls(self):
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="calc", arguments='{"expr": "2+2"}'),
        )
        chunk = StreamChunk(tool_calls=[tc])
        msg = chunk.to_message()
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].function.name == "calc"

    def test_error_chunk(self):
        chunk = StreamChunk.error("Something broke")
        assert chunk.event_type == "error"
        assert "Something broke" in chunk.content
        assert chunk.finish_reason == "error"

    def test_tool_result_chunk(self):
        chunk = StreamChunk.tool_result_chunk(
            tool_name="search",
            tool_call_id="call_1",
            result='{"results": []}',
        )
        assert chunk.event_type == "tool_result"
        assert chunk.metadata["tool_name"] == "search"
        assert chunk.metadata["tool_call_id"] == "call_1"

    def test_done_chunk(self):
        usage = StreamUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        chunk = StreamChunk.done(usage=usage)
        assert chunk.event_type == "done"
        assert chunk.finish_reason == "stop"
        assert chunk.usage.total_tokens == 30


class TestToolCallAccumulator:
    def _make_delta(self, index, id=None, name=None, arguments=None):
        """Create a mock tool-call delta matching OpenAI's protocol."""
        return {
            "index": index,
            "id": id,
            "function": {
                "name": name,
                "arguments": arguments,
            },
        }

    def test_single_complete_call(self):
        acc = ToolCallAccumulator()
        acc.feed(
            [
                self._make_delta(
                    0, id="call_1", name="search", arguments='{"q": "hello"}'
                )
            ]
        )
        calls = acc.get_complete_calls()
        assert calls is not None
        assert len(calls) == 1
        assert calls[0].function.name == "search"
        assert calls[0].id == "call_1"

    def test_partial_then_complete(self):
        acc = ToolCallAccumulator()

        # First chunk: id + name + start of arguments
        acc.feed([self._make_delta(0, id="call_1", name="search", arguments='{"q"')])
        assert acc.get_complete_calls() is None  # Not yet valid JSON

        # Second chunk: rest of arguments
        acc.feed([self._make_delta(0, arguments=': "hello"}')])
        calls = acc.get_complete_calls()
        assert calls is not None
        assert len(calls) == 1
        parsed_args = json.loads(calls[0].function.get_arguments_str())
        assert parsed_args == {"q": "hello"}

    def test_multiple_parallel_calls(self):
        acc = ToolCallAccumulator()

        # Two parallel tool calls
        acc.feed(
            [
                self._make_delta(0, id="call_1", name="search", arguments='{"q": "a"}'),
                self._make_delta(1, id="call_2", name="calc", arguments='{"x": 1}'),
            ]
        )

        calls = acc.get_complete_calls()
        assert calls is not None
        assert len(calls) == 2
        assert calls[0].function.name == "search"
        assert calls[1].function.name == "calc"

    def test_incomplete_blocks_all(self):
        """All calls must have valid JSON for get_complete_calls to return."""
        acc = ToolCallAccumulator()
        acc.feed(
            [
                self._make_delta(
                    0, id="call_1", name="search", arguments='{"q": "hello"}'
                ),
                self._make_delta(1, id="call_2", name="calc", arguments='{"x":'),
            ]
        )
        assert acc.get_complete_calls() is None  # call_2 is incomplete

        acc.feed([self._make_delta(1, arguments=' 42}')])
        calls = acc.get_complete_calls()
        assert calls is not None
        assert len(calls) == 2

    def test_reset(self):
        acc = ToolCallAccumulator()
        acc.feed([self._make_delta(0, id="c", name="f", arguments="{}")])
        assert acc.get_complete_calls() is not None
        acc.reset()
        assert acc.get_complete_calls() is None

    def test_empty_arguments_returns_none(self):
        acc = ToolCallAccumulator()
        acc.feed([self._make_delta(0, id="c", name="f", arguments="")])
        assert acc.get_complete_calls() is None
