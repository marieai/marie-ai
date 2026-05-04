from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class QueueRequest:
    request_id: str
    producer_id: str
    pool_id: str
    route_key: str
    submitted_at: float
    messages: list[dict[str, Any]]
    completion_params: Optional[dict[str, Any]] = None
    metadata: Optional[dict[str, str]] = None
    traceparent: Optional[str] = None
    tracestate: Optional[str] = None
    timeout_seconds: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "QueueRequest":
        data = json.loads(payload)
        return cls(**data)


@dataclass
class QueueReply:
    request_id: str
    producer_id: str
    pool_id: str
    route_key: str
    status: str
    completed_at: float
    response: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "QueueReply":
        data = json.loads(payload)
        return cls(**data)
