"""Tests for marie.agent.tool_call_parser."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from marie.agent.message import FunctionCall, Message, ToolCall
from marie.agent.tool_call_parser import (
    ActionInputToolCallParser,
    HermesToolCallParser,
    LlamaJsonToolCallParser,
    ParsedToolCalls,
    ToolCallTextParser,
    generate_tool_call_id,
)

# ---------------------------------------------------------------------------
# generate_tool_call_id
# ---------------------------------------------------------------------------


class TestGenerateToolCallId:
    def test_format(self):
        tid = generate_tool_call_id()
        assert tid.startswith("call_")
        assert len(tid) == len("call_") + 24

    def test_unique(self):
        ids = {generate_tool_call_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# HermesToolCallParser
# ---------------------------------------------------------------------------


class TestHermesToolCallParser:
    parser = HermesToolCallParser()

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
        result = self.parser.parse(text)
        assert result is not None
        assert len(result.tool_calls) == 1
        tc = result.tool_calls[0]
        assert tc.function.name == "get_weather"
        assert tc.function.get_arguments_dict() == {"city": "NYC"}
        assert tc.id.startswith("call_")

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "fn1", "arguments": {"a": 1}}</tool_call>'
            '<tool_call>{"name": "fn2", "arguments": {"b": 2}}</tool_call>'
        )
        result = self.parser.parse(text)
        assert result is not None
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].function.name == "fn1"
        assert result.tool_calls[1].function.name == "fn2"

    def test_incomplete_tag_matches_to_end(self):
        text = '<tool_call>{"name": "fn", "arguments": {}}'
        result = self.parser.parse(text)
        assert result is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "fn"

    def test_content_before_tags(self):
        text = 'I will call the tool now.\n<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = self.parser.parse(text)
        assert result is not None
        assert result.clean_content == "I will call the tool now."

    def test_no_content_before_tags(self):
        text = '<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = self.parser.parse(text)
        assert result is not None
        assert result.clean_content is None

    def test_malformed_json_returns_none(self):
        text = "<tool_call>not json</tool_call>"
        result = self.parser.parse(text)
        assert result is None

    def test_no_tool_call_tag(self):
        text = "Just regular text."
        result = self.parser.parse(text)
        assert result is None

    def test_missing_name_skipped(self):
        text = '<tool_call>{"arguments": {"a": 1}}</tool_call>'
        result = self.parser.parse(text)
        assert result is None

    def test_whitespace_in_tag(self):
        text = '<tool_call>\n  {"name": "fn", "arguments": {"x": 1}}\n</tool_call>'
        result = self.parser.parse(text)
        assert result is not None
        assert result.tool_calls[0].function.name == "fn"


# ---------------------------------------------------------------------------
# LlamaJsonToolCallParser
# ---------------------------------------------------------------------------


class TestLlamaJsonToolCallParser:
    parser = LlamaJsonToolCallParser()

    def test_raw_json_with_arguments(self):
        text = '{"name": "search", "arguments": {"query": "hello"}}'
        result = self.parser.parse(text)
        assert result is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"

    def test_parameters_variant(self):
        text = '{"name": "search", "parameters": {"query": "hello"}}'
        result = self.parser.parse(text)
        assert result is not None
        assert result.tool_calls[0].function.get_arguments_dict() == {"query": "hello"}

    def test_multiple_json_objects(self):
        text = '{"name": "fn1", "arguments": {}} {"name": "fn2", "arguments": {"x": 1}}'
        result = self.parser.parse(text)
        assert result is not None
        assert len(result.tool_calls) == 2

    def test_skips_inner_braces(self):
        text = '{"name": "fn", "arguments": {"nested": {"deep": true}}}'
        result = self.parser.parse(text)
        assert result is not None
        assert len(result.tool_calls) == 1
        args = result.tool_calls[0].function.get_arguments_dict()
        assert args == {"nested": {"deep": True}}

    def test_no_valid_json(self):
        text = "No JSON here at all."
        result = self.parser.parse(text)
        assert result is None

    def test_json_without_name_key(self):
        text = '{"key": "value", "other": 123}'
        result = self.parser.parse(text)
        assert result is None

    def test_json_with_name_but_no_arguments(self):
        text = '{"name": "fn", "description": "something"}'
        result = self.parser.parse(text)
        assert result is None

    def test_surrounding_text(self):
        text = 'Here is the call: {"name": "fn", "arguments": {}} done.'
        result = self.parser.parse(text)
        assert result is not None
        assert len(result.tool_calls) == 1

    def test_no_brace(self):
        result = self.parser.parse("nothing")
        assert result is None


# ---------------------------------------------------------------------------
# ActionInputToolCallParser
# ---------------------------------------------------------------------------


class TestActionInputToolCallParser:
    parser = ActionInputToolCallParser()

    def test_standard_format(self):
        text = 'Action: search\nAction Input: {"query": "hello"}\n'
        result = self.parser.parse(text)
        assert result is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "search"
        assert result.tool_calls[0].function.get_arguments_dict() == {"query": "hello"}

    def test_non_json_input(self):
        text = "Action: search\nAction Input: hello world\n"
        result = self.parser.parse(text)
        assert result is not None
        assert result.tool_calls[0].function.get_arguments_dict() == {
            "input": "hello world"
        }

    def test_content_before_action(self):
        text = "Let me search for that.\nAction: search\nAction Input: {}\n"
        result = self.parser.parse(text)
        assert result is not None
        assert result.clean_content == "Let me search for that."

    def test_no_match(self):
        text = "Regular text with no actions."
        result = self.parser.parse(text)
        assert result is None


# ---------------------------------------------------------------------------
# ToolCallTextParser (composite)
# ---------------------------------------------------------------------------


class TestToolCallTextParser:
    def test_auto_detects_hermes(self):
        parser = ToolCallTextParser(format="auto")
        text = '<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = parser.parse(text)
        assert result is not None
        assert result.tool_calls[0].function.name == "fn"

    def test_auto_detects_llama_json(self):
        parser = ToolCallTextParser(format="auto")
        text = '{"name": "fn", "arguments": {}}'
        result = parser.parse(text)
        assert result is not None
        assert result.tool_calls[0].function.name == "fn"

    def test_auto_detects_action(self):
        parser = ToolCallTextParser(format="auto")
        text = "Action: fn\nAction Input: {}\n"
        result = parser.parse(text)
        assert result is not None
        assert result.tool_calls[0].function.name == "fn"

    def test_strips_think_tags(self):
        parser = ToolCallTextParser(format="auto")
        text = '<think>reasoning...</think><tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = parser.parse(text)
        assert result is not None
        assert result.tool_calls[0].function.name == "fn"

    def test_strips_standalone_closing_think_tag(self):
        parser = ToolCallTextParser(format="auto")
        text = '</think>\n<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = parser.parse(text)
        assert result is not None

    def test_format_none_returns_none(self):
        parser = ToolCallTextParser(format="none")
        text = '<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        result = parser.parse(text)
        assert result is None

    def test_empty_text(self):
        parser = ToolCallTextParser(format="auto")
        assert parser.parse("") is None
        assert parser.parse("   ") is None

    def test_no_match_returns_none(self):
        parser = ToolCallTextParser(format="auto")
        result = parser.parse("Just some plain text.")
        assert result is None

    def test_specific_format_hermes(self):
        parser = ToolCallTextParser(format="hermes")
        # Should match hermes
        text = '<tool_call>{"name": "fn", "arguments": {}}</tool_call>'
        assert parser.parse(text) is not None
        # Should NOT match action format (only hermes parser active)
        text2 = "Action: fn\nAction Input: {}\n"
        assert parser.parse(text2) is None

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Unknown tool call format"):
            ToolCallTextParser(format="invalid")


# ---------------------------------------------------------------------------
# Integration: OpenAICompatibleWrapper._openai_to_message fallback
# ---------------------------------------------------------------------------


class TestOpenAIToMessageFallback:
    """Test that _openai_to_message extracts tool calls from content text."""

    def _make_wrapper(self, tool_call_format="auto"):
        """Create an OpenAICompatibleWrapper without a real OpenAI client."""
        with patch("openai.OpenAI"):
            from marie.agent.llm_wrapper import OpenAICompatibleWrapper

            wrapper = OpenAICompatibleWrapper(
                api_key="fake",
                model="test",
                tool_call_format=tool_call_format,
            )
        return wrapper

    def test_fallback_parses_hermes_from_content(self):
        wrapper = self._make_wrapper()
        openai_msg = SimpleNamespace(
            role="assistant",
            content='<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>',
            function_call=None,
            tool_calls=None,
        )
        msg = wrapper._openai_to_message(openai_msg)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "get_weather"
        # Content should be cleaned (nothing before the tag)
        assert msg.content is None

    def test_fallback_preserves_content_before_tag(self):
        wrapper = self._make_wrapper()
        openai_msg = SimpleNamespace(
            role="assistant",
            content='Here is the result:\n<tool_call>{"name": "fn", "arguments": {}}</tool_call>',
            function_call=None,
            tool_calls=None,
        )
        msg = wrapper._openai_to_message(openai_msg)
        assert msg.tool_calls is not None
        assert msg.content == "Here is the result:"

    def test_structured_tool_calls_take_precedence(self):
        """When real tool_calls exist, fallback should NOT fire."""
        wrapper = self._make_wrapper()
        mock_tc = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(name="real_fn", arguments='{"x": 1}'),
        )
        openai_msg = SimpleNamespace(
            role="assistant",
            content='<tool_call>{"name": "text_fn", "arguments": {}}</tool_call>',
            function_call=None,
            tool_calls=[mock_tc],
        )
        msg = wrapper._openai_to_message(openai_msg)
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function.name == "real_fn"
        # Content should be the raw text (not cleaned by fallback)
        assert "<tool_call>" in msg.content

    def test_no_fallback_when_format_none(self):
        wrapper = self._make_wrapper(tool_call_format="none")
        openai_msg = SimpleNamespace(
            role="assistant",
            content='<tool_call>{"name": "fn", "arguments": {}}</tool_call>',
            function_call=None,
            tool_calls=None,
        )
        msg = wrapper._openai_to_message(openai_msg)
        assert msg.tool_calls is None
        assert "<tool_call>" in msg.content

    def test_no_content_no_error(self):
        wrapper = self._make_wrapper()
        openai_msg = SimpleNamespace(
            role="assistant",
            content=None,
            function_call=None,
            tool_calls=None,
        )
        msg = wrapper._openai_to_message(openai_msg)
        assert msg.tool_calls is None
        assert msg.content is None

    def test_fallback_with_think_tags(self):
        wrapper = self._make_wrapper()
        openai_msg = SimpleNamespace(
            role="assistant",
            content='<think>Let me think...</think><tool_call>{"name": "fn", "arguments": {"a": 1}}</tool_call>',
            function_call=None,
            tool_calls=None,
        )
        msg = wrapper._openai_to_message(openai_msg)
        assert msg.tool_calls is not None
        assert msg.tool_calls[0].function.name == "fn"
