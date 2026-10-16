from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class CompletionAttempt:
    task_id: str
    attempt_number: int
    raw_response: str | None
    finish_reason: str | None
    response_model: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: float
    error: str | None
    cost_usd: float | None = None
    execution_sequence: int | None = None
    physical_attempt: int | None = None
    claim_id: str | None = None


CompletionAttemptObserver = Callable[[CompletionAttempt], None]


def _response_value(response: Any, key: str) -> Any:
    if isinstance(response, dict):
        return response.get(key)
    return getattr(response, key, None)


def _raw_completion_text(completion: Any) -> str | None:
    if isinstance(completion, str):
        return completion

    choices = _response_value(completion, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return None
    message = _response_value(choices[0], "message")
    content = _response_value(message, "content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    parts = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    if not all(isinstance(part, str) for part in parts):
        return None
    return "".join(parts) or None


def _token_count(usage: Any, *names: str) -> int | None:
    for name in names:
        value = _response_value(usage, name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None
