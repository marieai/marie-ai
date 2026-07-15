"""Typed session-scoped daemon protocol frames."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

SessionMessageType = Literal['end', 'error', 'invoke', 'stream']


class SessionMessage(TypedDict):
    type: SessionMessageType
    data: Any


class SessionFrame(TypedDict):
    session_id: str
    event: Literal['session']
    data: SessionMessage


def session_frame(
    session_id: str, message_type: SessionMessageType, data: Any
) -> SessionFrame:
    """Create a session-scoped daemon frame."""
    return {
        'session_id': session_id,
        'event': 'session',
        'data': {'type': message_type, 'data': data},
    }


def error_frame(
    session_id: str,
    *,
    code: str,
    message: str,
    retryable: bool = False,
) -> SessionFrame:
    """Create a classified session error frame."""
    return session_frame(
        session_id,
        'error',
        {'code': code, 'message': message, 'retryable': retryable},
    )
