"""Debug capture middleware for Marie agent runs.

Writes opt-in run artifacts to a folder so callers can inspect prompts,
tool schemas, streamed response batches, tool calls, tool results, and
middleware events after an agent run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable
from uuid import uuid4

from marie.agent.middleware.protocol import BaseMiddleware

if TYPE_CHECKING:
    from marie.agent.emitter import Emitter


class DebugCaptureMiddleware(BaseMiddleware):
    """Persist agent debug artifacts for one run.

    The middleware is intentionally generic: it records framework-level events
    and message batches. Domain agents can add their own files with
    ``write_json`` and ``write_text`` using the same debug directory.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        create_run_dir: bool = True,
        run_id: str | None = None,
        include_content: bool = True,
    ) -> None:
        super().__init__(name="DebugCaptureMiddleware", priority=20)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        root = Path(output_dir).expanduser().resolve()
        self.debug_dir = root / self.run_id if create_run_dir else root
        self.include_content = include_content
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_path = self.debug_dir / "debug.txt"
        self.dump_prompts_path = self.debug_dir / "dump-prompts.jsonl"
        self.transcript_path = self.debug_dir / "transcript.jsonl"
        self.events_path = self.debug_dir / "events.jsonl"
        self.messages_path = self.debug_dir / "messages.jsonl"
        self.tool_calls_path = self.debug_dir / "tool_calls.jsonl"
        self._parent_uuid: str | None = None
        self._init_written = False
        self._seen_tool_call_ids: set[str] = set()
        self._symlink_latest(root)
        self.write_json(
            "run.json",
            {
                "run_id": self.run_id,
                "debug_dir": str(self.debug_dir),
                "debug_log_path": str(self.debug_log_path),
                "dump_prompts_path": str(self.dump_prompts_path),
                "transcript_path": str(self.transcript_path),
                "created_at": _utc_now(),
                "include_content": self.include_content,
            },
        )

    def bind(self, emitter: "Emitter") -> None:
        for event_name in [
            "agent.start",
            "agent.input",
            "agent.response",
            "agent.success",
            "agent.error",
            "agent.finish",
            "tool.start",
            "tool.success",
            "tool.error",
            "tool.finish",
            "llm.start",
            "llm.success",
            "llm.error",
            "llm.finish",
        ]:
            self._listener_ids.append(
                emitter.on(
                    event_name,
                    self._event_handler(event_name),
                    priority=self.priority,
                    is_blocking=True,
                )
            )

    def write_json(self, filename: str, payload: Any) -> None:
        (self.debug_dir / filename).write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_text(self, filename: str, text: str) -> None:
        (self.debug_dir / filename).write_text(text, encoding="utf-8")

    def _event_handler(self, event_name: str):
        def handler(data: Dict[str, Any]) -> None:
            self._record_event(event_name, data)

        return handler

    def _record_event(self, event_name: str, data: Dict[str, Any]) -> None:
        self._log_debug(event_name, data)
        payload = {
            "timestamp": _utc_now(),
            "event": event_name,
            "data": _json_safe(data) if self.include_content else _summarize(data),
        }
        _append_jsonl(self.events_path, payload)
        if event_name == "agent.input":
            self._record_agent_input(data)
        elif event_name == "agent.response":
            self._record_agent_response(data)
        elif event_name.startswith("tool."):
            self._record_tool_event(event_name, data)

    def _log_debug(self, event_name: str, data: Dict[str, Any]) -> None:
        level = "ERROR" if event_name.endswith(".error") else "DEBUG"
        summary = _event_summary(event_name, data)
        with self.debug_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{_utc_now()} [{level}] {summary}\n")

    def _record_agent_input(self, data: Dict[str, Any]) -> None:
        messages = list(data.get("messages") or [])
        tools = list(data.get("tools") or [])
        timestamp = _utc_now()
        self.write_json(
            "input_messages.json",
            (
                messages
                if self.include_content
                else [_summarize(item) for item in messages]
            ),
        )
        self.write_json("tools.json", tools)
        system_prompt = _first_message_content(messages, "system")
        user_prompt = _last_message_content(messages, "user")
        if system_prompt:
            self.write_text("system_prompt.txt", system_prompt)
        if user_prompt:
            self.write_text("task_prompt.txt", user_prompt)
        init_data = {
            "agent_name": data.get("agent_name"),
            "model": data.get("model_name"),
            "system": system_prompt,
            "tools": tools,
            "message_count": len(messages),
        }
        dump_type = "init" if not self._init_written else "system_update"
        self._init_written = True
        _append_jsonl(
            self.dump_prompts_path,
            {"type": dump_type, "timestamp": timestamp, "data": init_data},
        )
        for message in messages:
            role = _get(message, "role")
            if role == "user":
                _append_jsonl(
                    self.dump_prompts_path,
                    {"type": "message", "timestamp": timestamp, "data": message},
                )
            self._append_transcript_message(message, timestamp=timestamp)

    def _record_agent_response(self, data: Dict[str, Any]) -> None:
        messages = list(data.get("messages") or [])
        timestamp = _utc_now()
        payload = {
            "timestamp": timestamp,
            "iteration": data.get("iteration"),
            "agent_name": data.get("agent_name"),
            "messages": (
                messages
                if self.include_content
                else [_summarize(item) for item in messages]
            ),
        }
        _append_jsonl(self.messages_path, payload)
        _append_jsonl(
            self.dump_prompts_path,
            {
                "type": "response",
                "timestamp": timestamp,
                "data": {
                    "iteration": data.get("iteration"),
                    "messages": messages,
                },
            },
        )
        for message_index, message in enumerate(messages):
            self._append_transcript_message(
                message,
                timestamp=timestamp,
                iteration=data.get("iteration"),
            )
            for tool_event in _extract_message_tool_events(message):
                if tool_event.get("event_type") == "tool_call":
                    tool_call_id = str(tool_event.get("tool_call_id") or "")
                    if tool_call_id and tool_call_id in self._seen_tool_call_ids:
                        continue
                    if tool_call_id:
                        self._seen_tool_call_ids.add(tool_call_id)
                tool_event.update(
                    {
                        "timestamp": timestamp,
                        "iteration": data.get("iteration"),
                        "message_index": message_index,
                    }
                )
                _append_jsonl(self.tool_calls_path, tool_event)

    def _record_tool_event(self, event_name: str, data: Dict[str, Any]) -> None:
        payload = {
            "timestamp": _utc_now(),
            "event_type": event_name.replace(".", "_"),
            "tool_name": data.get("tool_name"),
            "data": _json_safe(data) if self.include_content else _summarize(data),
        }
        _append_jsonl(self.tool_calls_path, payload)

    def _append_transcript_message(
        self,
        message: Any,
        *,
        timestamp: str,
        iteration: Any = None,
    ) -> None:
        role = str(_get(message, "role") or "unknown")
        entry_uuid = str(uuid4())
        entry = {
            "type": role,
            "uuid": entry_uuid,
            "parentUuid": self._parent_uuid,
            "isSidechain": False,
            "sessionId": self.run_id,
            "timestamp": timestamp,
            "version": "marie-agent",
            "iteration": iteration,
            "message": (
                _json_safe(message) if self.include_content else _summarize(message)
            ),
        }
        _append_jsonl(self.transcript_path, entry)
        if role in {"system", "user", "assistant", "tool", "function"}:
            self._parent_uuid = entry_uuid

    def _symlink_latest(self, root: Path) -> None:
        if self.debug_dir.parent != root:
            return
        latest = root / "latest"
        try:
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            latest.symlink_to(self.debug_dir, target_is_directory=True)
        except OSError:
            pass


def _append_jsonl(path: Path, payload: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(payload), sort_keys=True) + "\n")


def _event_summary(event_name: str, data: Dict[str, Any]) -> str:
    if event_name == "agent.input":
        return (
            f"agent.input agent={data.get('agent_name')} model={data.get('model_name')} "
            f"messages={len(data.get('messages') or [])} tools={len(data.get('tools') or [])}"
        )
    if event_name == "agent.response":
        return (
            f"agent.response agent={data.get('agent_name')} iteration={data.get('iteration')} "
            f"messages={len(data.get('messages') or [])}"
        )
    if event_name.startswith("tool."):
        suffix = event_name.split(".", 1)[1]
        return f"{event_name} name={data.get('tool_name')} {suffix}"
    if event_name.startswith("llm."):
        suffix = event_name.split(".", 1)[1]
        return f"{event_name} model={data.get('model_name')} {suffix}"
    if event_name.endswith(".error"):
        return f"{event_name} {data.get('error_message') or data.get('error') or ''}"
    return f"{event_name} {_summary_text(data)}"


def _summary_text(data: Dict[str, Any]) -> str:
    parts = []
    for key in ["agent_name", "success", "duration_ms", "result_count"]:
        if key in data:
            parts.append(f"{key}={data[key]}")
    return " ".join(parts)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _first_message_content(messages: Iterable[Any], role: str) -> str:
    for message in messages:
        if _get(message, "role") == role:
            content = _get(message, "content")
            return (
                content
                if isinstance(content, str)
                else json.dumps(_json_safe(content), indent=2)
            )
    return ""


def _last_message_content(messages: Iterable[Any], role: str) -> str:
    content = ""
    for message in messages:
        if _get(message, "role") == role:
            value = _get(message, "content")
            content = (
                value
                if isinstance(value, str)
                else json.dumps(_json_safe(value), indent=2)
            )
    return content


def _extract_message_tool_events(message: Any) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    role = str(_get(message, "role") or "")
    for tool_call in _get(message, "tool_calls") or []:
        function = _get(tool_call, "function") or {}
        events.append(
            {
                "event_type": "tool_call",
                "role": role,
                "tool_call_id": _get(tool_call, "id"),
                "tool_name": _get(function, "name"),
                "arguments": _get(function, "arguments"),
                "raw": _json_safe(tool_call),
            }
        )
    if role in {"tool", "function"} or _get(message, "tool_call_id"):
        events.append(
            {
                "event_type": "tool_result",
                "role": role,
                "tool_call_id": _get(message, "tool_call_id"),
                "tool_name": _get(message, "name"),
                "content": _get(message, "content"),
            }
        )
    return events


def _summarize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _summarize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_summarize(item) for item in value]
    if isinstance(value, str):
        return {"type": "str", "length": len(value), "preview": value[:240]}
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_info.name: _json_safe(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_safe(item) for item in value]
    if isinstance(value, set | frozenset):
        return [_json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)
