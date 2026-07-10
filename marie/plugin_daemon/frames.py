"""Parsing of marie-plugin-daemon response streams (SSE / NDJSON / JSON)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def parse_daemon_frames(raw: str, request_id: str) -> list[dict[str, Any]]:
    value = parse_json(raw)
    if isinstance(value, list):
        return [
            normalize_daemon_frame(item, request_id, sequence)
            for sequence, item in enumerate(value)
        ]
    if value is not None:
        return [normalize_daemon_frame(value, request_id, 0)]

    frames: list[dict[str, Any]] = []
    sequence = 0
    for line in raw.splitlines():
        item = parse_daemon_line(line)
        if item is None:
            continue
        frame = normalize_daemon_frame(item, request_id, sequence)
        frames.append(frame)
        sequence = int(frame.get("sequence", sequence)) + 1

    if frames:
        return frames
    text = raw.strip()
    return [normalize_daemon_frame(text, request_id, 0)] if text else []


def parse_daemon_line(line: str) -> Any | None:
    text = line.strip()
    if not text or text == "[DONE]" or text.startswith(":"):
        return None
    if text.startswith("data:"):
        text = text[5:].strip()
    value = parse_json(text)
    return value if value is not None else text


def parse_json(value: str) -> Any | None:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def normalize_daemon_frame(raw: Any, request_id: str, sequence: int) -> dict[str, Any]:
    if isinstance(raw, dict):
        frame = dict(raw)
        frame["requestId"] = (
            first_text(
                as_text(frame.get("requestId")),
                as_text(frame.get("request_id")),
                request_id,
            )
            or request_id
        )
        if not isinstance(frame.get("sequence"), int):
            frame["sequence"] = sequence
        frame["type"] = (
            first_text(as_text(frame.get("type")), as_text(frame.get("frameType")))
            or "event"
        )
        frame.setdefault("createdAt", now_utc().isoformat())
        frame.setdefault("final", False)
        if frame["type"] == "error" and not isinstance(frame.get("error"), dict):
            payload = frame.get("payload")
            payload_map = payload if isinstance(payload, dict) else {}
            code = (
                first_text(as_text(frame.get("code")), as_text(payload_map.get("code")))
                or "runtime_error"
            )
            message = (
                first_text(
                    as_text(frame.get("message")), as_text(payload_map.get("message"))
                )
                or "Runtime error"
            )
            frame["error"] = {"code": code, "message": message}
        return frame

    return {
        "requestId": request_id,
        "sequence": sequence,
        "type": "text",
        "payload": str(raw),
        "contentType": "text/plain",
        "artifactId": None,
        "final": False,
        "error": None,
        "createdAt": now_utc().isoformat(),
    }


def runtime_error_frame(
    request_id: str,
    message: str,
    code: str = "runtime_unavailable",
    sequence: int = 0,
) -> dict[str, Any]:
    return {
        "requestId": request_id,
        "sequence": sequence,
        "type": "error",
        "payload": {"code": code, "message": message},
        "contentType": "application/json",
        "artifactId": None,
        "final": True,
        "error": {"code": code, "message": message},
        "createdAt": now_utc().isoformat(),
    }


def first_text(*values: str | None) -> str | None:
    for value in values:
        if value and value.strip():
            return value.strip()
    return None


def as_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
