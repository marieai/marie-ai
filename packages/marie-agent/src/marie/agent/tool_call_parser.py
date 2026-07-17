"""Tool call text parsers for extracting tool calls from LLM text output.

Ported from vLLM's tool_parsers/ (hermes, llama) into standalone,
dependency-free parsers. These handle the case where vLLM (or similar
OpenAI-compatible endpoints) return tool calls as text in the content
field instead of structured tool_calls.
"""

from __future__ import annotations

import json
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from marie.agent.message import FunctionCall, ToolCall

# Regex to strip <think>...</think> blocks (including incomplete closing tags)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>|</think>", re.DOTALL)


def generate_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


@dataclass
class ParsedToolCalls:
    tool_calls: List[ToolCall] = field(default_factory=list)
    clean_content: Optional[str] = None


class BaseToolCallParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> Optional[ParsedToolCalls]: ...


class HermesToolCallParser(BaseToolCallParser):
    """Parse Hermes-style <tool_call>...</tool_call> markup.

    Ported from vLLM hermes_tool_parser.py extract_tool_calls().
    Regex matches either between tags or from tag to end-of-string
    (for incomplete output).
    """

    _RE = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)

    def parse(self, text: str) -> Optional[ParsedToolCalls]:
        if "<tool_call>" not in text:
            return None

        matches = self._RE.findall(text)
        if not matches:
            return None

        tool_calls: List[ToolCall] = []
        for match in matches:
            raw = (match[0] if match[0] else match[1]).strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            name = obj.get("name")
            arguments = obj.get("arguments", {})
            if not name:
                continue
            tool_calls.append(
                ToolCall(
                    id=generate_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=arguments,
                    ),
                )
            )

        if not tool_calls:
            return None

        # Content is everything before the first <tool_call> tag
        content = text[: text.find("<tool_call>")].strip() or None
        return ParsedToolCalls(tool_calls=tool_calls, clean_content=content)


class LlamaJsonToolCallParser(BaseToolCallParser):
    """Parse raw JSON tool calls from text.

    Ported from vLLM llama_tool_parser.py extract_tool_calls().
    Uses json.JSONDecoder.raw_decode() to find JSON objects containing
    "name" and either "arguments" or "parameters" keys.
    """

    _BRACE_RE = re.compile(r"\{")

    def __init__(self) -> None:
        self._decoder = json.JSONDecoder()

    def parse(self, text: str) -> Optional[ParsedToolCalls]:
        if "{" not in text:
            return None

        tool_calls: List[ToolCall] = []
        end_index = -1

        for match in self._BRACE_RE.finditer(text):
            start = match.start()
            if start <= end_index:
                continue
            try:
                obj, json_len = self._decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue

            end_index = start + json_len

            if not isinstance(obj, dict) or "name" not in obj:
                continue
            if "arguments" not in obj and "parameters" not in obj:
                continue

            name = obj["name"]
            arguments = obj.get("arguments", obj.get("parameters", {}))
            tool_calls.append(
                ToolCall(
                    id=generate_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=arguments,
                    ),
                )
            )

        if not tool_calls:
            return None

        # No reliable way to separate content from JSON for raw-JSON format;
        # return full text as content so callers can decide.
        return ParsedToolCalls(tool_calls=tool_calls, clean_content=None)


class ActionInputToolCallParser(BaseToolCallParser):
    """Parse Action/Action Input format.

    Matches patterns like:
        Action: tool_name
        Action Input: {"key": "value"}
    """

    _RE = re.compile(
        r"Action:\s*(\w+)[^\S\n]*\nAction Input:\s*(.+?)(?=\n(?:Observation|Action|$)|\Z)",
        re.DOTALL,
    )

    def parse(self, text: str) -> Optional[ParsedToolCalls]:
        matches = self._RE.findall(text)
        if not matches:
            return None

        tool_calls: List[ToolCall] = []
        first_start = None
        for m in self._RE.finditer(text):
            if first_start is None:
                first_start = m.start()
            action_name = m.group(1).strip()
            action_input = m.group(2).strip()
            try:
                arguments = json.loads(action_input)
            except json.JSONDecodeError:
                arguments = {"input": action_input}
            tool_calls.append(
                ToolCall(
                    id=generate_tool_call_id(),
                    type="function",
                    function=FunctionCall(
                        name=action_name,
                        arguments=arguments,
                    ),
                )
            )

        if not tool_calls:
            return None

        content = text[:first_start].strip() or None if first_start else None
        return ParsedToolCalls(tool_calls=tool_calls, clean_content=content)


class ToolCallTextParser:
    """Composite parser that tries multiple formats.

    Args:
        format: Parser format key. One of "hermes", "llama3_json",
            "action", "auto", or "none".
        strip_think_tags: Remove <think>...</think> blocks before parsing.
    """

    FORMAT_MAP = {
        "hermes": [HermesToolCallParser],
        "llama3_json": [LlamaJsonToolCallParser],
        "action": [ActionInputToolCallParser],
        "tool_call": [HermesToolCallParser],
        "auto": [
            HermesToolCallParser,
            LlamaJsonToolCallParser,
            ActionInputToolCallParser,
        ],
    }

    def __init__(self, format: str = "auto", strip_think_tags: bool = True) -> None:
        if format == "none":
            self._parsers: List[BaseToolCallParser] = []
            self._enabled = False
            return
        parser_classes = self.FORMAT_MAP.get(format)
        if parser_classes is None:
            raise ValueError(
                f"Unknown tool call format: {format!r}. "
                f"Valid formats: {list(self.FORMAT_MAP.keys()) + ['none']}"
            )
        self._parsers = [cls() for cls in parser_classes]
        self._enabled = True
        self._strip_think_tags = strip_think_tags

    def parse(self, text: str) -> Optional[ParsedToolCalls]:
        if not self._enabled or not text:
            return None

        if self._strip_think_tags:
            text = _THINK_TAG_RE.sub("", text).strip()

        if not text:
            return None

        for parser in self._parsers:
            result = parser.parse(text)
            if result is not None:
                return result
        return None
