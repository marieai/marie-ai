import json
import re
from typing import Any

import json5
import json_repair

_JSON_FENCE_START = re.compile(r"```[ \t]*json\b", re.IGNORECASE)
_JSON_FENCE = re.compile(
    r"```[ \t]*json\b[ \t]*(?:\r?\n)?(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_BARE_FENCE = re.compile(r"```[ \t]*\r?\n(.*?)```", re.DOTALL)
_CONTAINER_PAIRS = {"{": "}", "[": "]"}
_JSON_CONTAINERS = (dict, list)


class JSONOutputParserError(ValueError):
    """Raised when model output cannot be decoded into one JSON container."""


def _fenced_sections(text: str) -> list[str]:
    sections = _JSON_FENCE.findall(text)
    if sections:
        return sections
    return _BARE_FENCE.findall(text) or [text]


def _container_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack: list[str] = []
    start: int | None = None
    quote: str | None = None
    escaped = False

    for index, char in enumerate(text):
        if not stack:
            if char in _CONTAINER_PAIRS:
                start = index
                stack.append(char)
            continue

        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue

        if char in ('"', "'"):
            quote = char
        elif char in _CONTAINER_PAIRS:
            stack.append(char)
        elif char == _CONTAINER_PAIRS[stack[-1]]:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : index + 1])
                start = None

    if start is not None:
        candidates.append(text[start:])

    return candidates


def _parse_candidate(candidate: str, expected_type: type | tuple[type, ...]) -> Any:
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            value = json5.loads(candidate)
        except ValueError:
            try:
                value = json_repair.loads(candidate, strict=True)
            except ValueError as exc:
                raise JSONOutputParserError("JSON repair failed") from exc

    if not isinstance(value, expected_type):
        raise JSONOutputParserError(
            f"JSON output has unsupported root type {type(value).__name__}"
        )
    return value


def parse_json_markdown(
    text: str,
    *,
    expected_type: type | tuple[type, ...] = _JSON_CONTAINERS,
) -> Any:
    """Parse one JSON object or array from plain, fenced, or damaged model output."""
    if not text.strip():
        raise JSONOutputParserError("JSON output is empty")

    values: list[Any] = []
    last_error: JSONOutputParserError | None = None

    for section in _fenced_sections(text):
        section = section.strip()
        if not section:
            continue

        try:
            value = json.loads(section)
        except json.JSONDecodeError:
            candidates = _container_candidates(section)
        else:
            if isinstance(value, expected_type):
                values.append(value)
            else:
                last_error = JSONOutputParserError(
                    f"JSON output has unsupported root type {type(value).__name__}"
                )
            continue

        for candidate in candidates:
            try:
                values.append(_parse_candidate(candidate, expected_type))
            except JSONOutputParserError as exc:
                last_error = exc

    if not values:
        if last_error:
            raise last_error
        raise JSONOutputParserError(
            "JSON output does not contain a valid object or array"
        )
    if len(values) > 1:
        raise JSONOutputParserError("JSON output contains multiple candidate values")
    return values[0]


def parse_markdown_markdown(text: str, return_content=True) -> str:
    """
    Extracts the content enclosed in the first ```markdown ... ``` code block
    from the given text, even if there's a preceding block labeled ```plain text
    or similar.

    If no ```markdown block is found, returns an empty string.
    """
    if not text:
        return "" if return_content else ""

    if "```markdown" not in text:
        return text if return_content else ""
    text = text.split("```markdown")[1].strip().strip("```").strip()
    return text


def check_content_type(text: str) -> str:
    """
    Checks if the given text has code blocks with
    ```json or
    ```markdown.

    Returns:
        "json" if a code block labeled ```json is found.
        "markdown" if ```markdown is found.
        "none" otherwise.
    """
    if not text:
        return "none"

    if _JSON_FENCE_START.search(text):
        return "json"
    elif "```markdown" in text:
        return "markdown"
    else:
        return "none"
