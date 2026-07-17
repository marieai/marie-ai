from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

COMPLETION_QUEUE_CONTRACT_VERSION = "v2"


def normalize_guided_json(
    guided_json: Optional[dict[str, Any] | str | Any],
) -> Optional[dict[str, Any] | str]:
    if guided_json is None or isinstance(guided_json, (dict, str)):
        return guided_json

    model_json_schema = getattr(guided_json, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()

    schema = getattr(guided_json, "schema", None)
    if callable(schema):
        return schema()

    return str(guided_json)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Source provenance carried with one model request for tracing.

    requested_pages=None means the request covers all available pages.
    """

    ref_id: str | None = None
    ref_type: str | None = None
    page_number: int | None = None
    requested_pages: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.ref_id is not None:
            data["ref_id"] = self.ref_id
        if self.ref_type is not None:
            data["ref_type"] = self.ref_type
        if self.page_number is not None:
            data["page_number"] = self.page_number
        if self.requested_pages is not None:
            data["requested_pages"] = list(self.requested_pages)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RequestContext | None":
        if not data:
            return None

        requested_pages = data.get("requested_pages")
        if requested_pages is not None:
            requested_pages = tuple(int(page) for page in requested_pages)

        page_number = data.get("page_number")
        if page_number is not None:
            page_number = int(page_number)

        return cls(
            ref_id=data.get("ref_id"),
            ref_type=data.get("ref_type"),
            page_number=page_number,
            requested_pages=requested_pages,
        )


@dataclass
class CompletionCallParams:
    model: str
    messages: list[dict[str, Any]]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    max_tokens: Optional[int] = None
    n: int = 1
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[Any]] = None
    tool_choice: Optional[Any] = None
    stream: bool = False
    stream_options: Optional[dict[str, Any]] = None
    extra_body: Optional[dict[str, Any]] = None
    extra_create_kwargs: dict[str, Any] = field(default_factory=dict)
    context: RequestContext | None = None

    def to_create_kwargs(self) -> dict[str, Any]:
        create_kwargs = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "stop": self.stop,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "response_format": self.response_format,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "stream": self.stream,
            "stream_options": self.stream_options,
            "extra_body": self.extra_body,
        }
        create_kwargs.update(self.extra_create_kwargs)
        return {key: value for key, value in create_kwargs.items() if value is not None}

    def to_dict(self) -> dict[str, Any]:
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        if self.context is not None:
            data["context"] = self.context.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompletionCallParams":
        values = dict(data)
        values["context"] = RequestContext.from_dict(values.get("context"))
        return cls(**values)


def build_completion_call(
    *,
    model: str,
    messages: list[dict[str, Any]],
    default_completion_params: Optional[dict[str, Any]] = None,
    completion_params: Optional[dict[str, Any]] = None,
    guided_json: Optional[dict[str, Any] | str | Any] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[list[str]] = None,
    n: int = 1,
    stream: bool = False,
    stream_options: Optional[dict[str, Any]] = None,
    context: RequestContext | None = None,
) -> CompletionCallParams:
    effective = dict(default_completion_params or {})
    if completion_params:
        effective.update(completion_params)

    normalized_guided_json = normalize_guided_json(guided_json)
    effective.pop("context", None)
    response_format = effective.pop("response_format", None)
    extra_body = effective.pop("extra_body", None)
    if extra_body is not None and not isinstance(extra_body, dict):
        raise ValueError("completion_params.extra_body must be a dict when provided")

    extra_body_dict = dict(extra_body or {})
    if normalized_guided_json is not None and "guided_json" not in extra_body_dict:
        extra_body_dict["guided_json"] = normalized_guided_json

    return CompletionCallParams(
        model=model,
        messages=messages,
        temperature=effective.pop("temperature", 0.0),
        top_p=effective.pop("top_p", 1.0),
        frequency_penalty=effective.pop("frequency_penalty", 0.0),
        presence_penalty=effective.pop("presence_penalty", 0.0),
        stop=effective.pop("stop", stop),
        max_tokens=effective.pop("max_tokens", max_tokens),
        n=effective.pop("n", n),
        response_format=response_format,
        tools=effective.pop("tools", None),
        tool_choice=effective.pop("tool_choice", None),
        stream=effective.pop("stream", stream),
        stream_options=effective.pop("stream_options", stream_options),
        extra_body=extra_body_dict or None,
        extra_create_kwargs=effective,
        context=context,
    )


def build_dispatch_profile_key(call: CompletionCallParams) -> str:
    return call.model


@dataclass
class QueuedCompletionEnvelope:
    request_id: str
    producer_id: str
    pool_id: str
    submitted_at: float
    call: CompletionCallParams
    contract_version: str = COMPLETION_QUEUE_CONTRACT_VERSION
    timeout_seconds: Optional[float] = None
    traceparent: Optional[str] = None
    tracestate: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    dispatch_profile_key: Optional[str] = None
    estimated_cost_units: Optional[int] = None

    @property
    def model(self) -> str:
        return self.call.model

    @property
    def batch_key(self) -> str:
        return self.dispatch_profile_key or self.call.model

    def to_json(self) -> str:
        data = {item.name: getattr(self, item.name) for item in fields(self)}
        data["call"] = self.call.to_dict()
        return json.dumps(data, separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "QueuedCompletionEnvelope":
        data = json.loads(payload)
        _validate_contract_version(data, envelope_type="request")
        data["call"] = CompletionCallParams.from_dict(data["call"])
        return cls(**data)


@dataclass
class CompletionReplyEnvelope:
    request_id: str
    producer_id: str
    pool_id: str
    status: str
    completed_at: float
    contract_version: str = COMPLETION_QUEUE_CONTRACT_VERSION
    completion: Optional[Any] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_source: Optional[str] = None
    dispatcher_id: Optional[str] = None
    execution_backend_address: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "CompletionReplyEnvelope":
        data = json.loads(payload)
        _validate_contract_version(data, envelope_type="reply")
        return cls(**data)


def completion_payload_to_text(completion: Optional[Any]) -> Optional[str]:
    return extract_completion_text(completion)[1]


def summarize_completion_call(
    call: CompletionCallParams, *, max_chars: int = 160
) -> str:
    return summarize_completion_messages(call.messages, max_chars=max_chars)


def summarize_completion_messages(
    messages: list[dict[str, Any]], *, max_chars: int = 160
) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role") or "message")
        content = _normalize_message_content(message.get("content"))
        if not content:
            continue

        normalized = " ".join(content.split())
        if not normalized:
            continue

        parts.append(f"{role}: {normalized}")
        if len(" | ".join(parts)) >= max_chars * 2:
            break

    if not parts:
        count = len(messages)
        noun = "message" if count == 1 else "messages"
        return f"{count} {noun}"

    return _truncate_text(" | ".join(parts), max_chars=max_chars)


def extract_reasoning_content(model_output: str) -> tuple[Optional[str], Optional[str]]:
    think_start_token = "<think>"
    think_end_token = "</think>"
    reasoning_regex = re.compile(
        rf"{think_start_token}(.*?){think_end_token}",
        re.DOTALL,
    )

    if think_end_token not in model_output:
        return None, model_output

    if think_start_token not in model_output:
        model_output = f"{think_start_token}{model_output}"

    reasoning_content = reasoning_regex.findall(model_output)[0]
    end_index = len(f"{think_start_token}{reasoning_content}{think_end_token}")
    final_output = model_output[end_index:]
    if not final_output:
        return reasoning_content, None
    return reasoning_content, final_output


def completion_finish_reason(completion: Optional[Any]) -> Optional[str]:
    first_choice = _first_choice(completion)
    return _read_value(first_choice, "finish_reason")


def extract_completion_text(
    completion: Optional[Any],
) -> tuple[Optional[str], Optional[str]]:
    if completion is None:
        raise ValueError("No completion payload available")

    if isinstance(completion, str):
        return extract_reasoning_content(completion.strip())

    first_choice = _first_choice(completion)
    message = _read_value(first_choice, "message")
    content = _read_value(message, "content")
    if content is None:
        raise ValueError(f"No text extracted from response. : {completion}")

    extracted_text = _normalize_message_content(content)
    if extracted_text is None:
        raise ValueError(f"No text extracted from response. : {completion}")

    extracted_text = extracted_text.strip()
    reasoning_content = _read_value(message, "reasoning_content")
    if reasoning_content is None:
        reasoning_content, extracted_text = extract_reasoning_content(extracted_text)
    return reasoning_content, extracted_text


def _first_choice(completion: Any) -> Any:
    choices = _read_value(completion, "choices") or []
    if not choices:
        raise ValueError(f"No text extracted from response. : {completion}")
    return choices[0]


def _normalize_message_content(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    parts = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    normalized = "".join(parts)
    return normalized or None


def _truncate_text(value: str, *, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _read_value(payload: Any, key: str) -> Any:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def _validate_contract_version(data: dict[str, Any], *, envelope_type: str) -> None:
    actual = data.get("contract_version")
    if actual != COMPLETION_QUEUE_CONTRACT_VERSION:
        raise ValueError(
            f"Unsupported {envelope_type} contract version: {actual!r}; "
            f"expected {COMPLETION_QUEUE_CONTRACT_VERSION!r}"
        )
