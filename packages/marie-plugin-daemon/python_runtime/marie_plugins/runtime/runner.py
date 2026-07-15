"""Shared stdin/stdout runner for daemon-managed Marie plugins."""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable, Iterable
from typing import Any, Literal, TextIO, TypedDict

from .session import SessionFrame, error_frame

RequestHandler = Callable[[dict[str, Any]], Iterable[SessionFrame]]


class _HeartbeatFrame(TypedDict):
    session_id: str
    event: Literal['heartbeat']
    data: None


class _LogData(TypedDict):
    level: str
    message: str


class _LogFrame(TypedDict):
    session_id: str
    event: Literal['log']
    data: _LogData


_OutboundFrame = SessionFrame | _HeartbeatFrame | _LogFrame


class StdioRunner:
    """Run one plugin request handler over the daemon's line protocol."""

    def __init__(
        self,
        handler: RequestHandler,
        *,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
        heartbeat_interval: float | None = 2.0,
    ) -> None:
        self._handler = handler
        self._stdin = stdin if stdin is not None else sys.stdin
        self._stdout = stdout if stdout is not None else sys.stdout
        self._heartbeat_interval = heartbeat_interval
        self._emit_lock = threading.Lock()
        self._stopped = threading.Event()

    def run(self) -> None:
        """Process request frames until stdin closes."""
        if self._heartbeat_interval is not None:
            threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        try:
            for line in self._stdin:
                if line.strip():
                    self._handle_line(line)
        finally:
            self._stopped.set()

    def emit(self, frame: _OutboundFrame) -> None:
        """Write one complete protocol frame atomically."""
        encoded = json.dumps(frame, separators=(',', ':'))
        with self._emit_lock:
            self._stdout.write(encoded + '\n')
            self._stdout.flush()

    def _handle_line(self, line: str) -> None:
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            self.emit(_log_frame('error', 'invalid json line'))
            return
        if not isinstance(request, dict):
            self.emit(_log_frame('error', 'request frame must be an object'))
            return

        session_id = request.get('session_id')
        if not isinstance(session_id, str) or not session_id:
            self.emit(_log_frame('error', 'missing session_id'))
            return
        if request.get('event') != 'request':
            self.emit(
                _log_frame('error', f"unsupported event: {request.get('event')!r}")
            )
            return

        try:
            for frame in self._handler(request):
                self.emit(frame)
        except Exception:
            self.emit(
                error_frame(
                    session_id,
                    code='internal_error',
                    message='plugin request handler failed',
                )
            )

    def _heartbeat_loop(self) -> None:
        interval = self._heartbeat_interval
        if interval is None:
            return
        while not self._stopped.is_set():
            self.emit(_heartbeat_frame())
            self._stopped.wait(interval)


def _heartbeat_frame() -> _HeartbeatFrame:
    return {'session_id': '', 'event': 'heartbeat', 'data': None}


def _log_frame(level: str, message: str) -> _LogFrame:
    return {
        'session_id': '',
        'event': 'log',
        'data': {'level': level, 'message': message},
    }


def run(handler: RequestHandler) -> None:
    """Run a plugin handler with the standard daemon stdin/stdout transport."""
    StdioRunner(handler).run()
