"""Reusable subprocess client for packaged plugin protocol tests."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO, cast

from ..session import SessionFrame


class _EndOfStream:
    pass


_EOF = _EndOfStream()


class StdioPluginTestClient:
    """Start a plugin process and invoke actions over the stdio session protocol."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | os.PathLike[str],
        env: Mapping[str, str] | None = None,
        timeout: float = 10.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        if not command:
            raise ValueError('command must not be empty')
        if timeout <= 0:
            raise ValueError('timeout must be positive')
        if shutdown_timeout <= 0:
            raise ValueError('shutdown_timeout must be positive')

        self._command = tuple(command)
        self._cwd = os.fspath(cwd)
        self._env = dict(env or {})
        self._timeout = timeout
        self._shutdown_timeout = shutdown_timeout
        self._process: subprocess.Popen[str] | None = None
        self._events: queue.Queue[dict[str, Any] | Exception | _EndOfStream] = (
            queue.Queue()
        )
        self._stderr_tail: deque[str] = deque(maxlen=50)
        self._reader_threads: list[threading.Thread] = []
        self._invoke_lock = threading.Lock()
        self._session_sequence = 0

    def __enter__(self) -> StdioPluginTestClient:
        return self.start()

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

    def start(self) -> StdioPluginTestClient:
        """Start the configured plugin process."""
        if self._process is not None:
            raise RuntimeError('plugin process is already started')

        self._events = queue.Queue()
        self._stderr_tail.clear()

        self._session_sequence = 0
        process = subprocess.Popen(
            self._command,
            cwd=self._cwd,
            env=self._subprocess_env(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise RuntimeError('plugin process streams are unavailable')

        self._process = process
        self._reader_threads = [
            threading.Thread(
                target=self._read_stdout,
                args=(process.stdout,),
                daemon=True,
            ),
            threading.Thread(
                target=self._read_stderr,
                args=(process.stderr,),
                daemon=True,
            ),
        ]
        for thread in self._reader_threads:
            thread.start()
        return self

    def invoke(
        self,
        action: str,
        *,
        session_id: str | None = None,
        **parameters: Any,
    ) -> SessionFrame:
        """Invoke one action and return its first session-scoped response frame."""
        with self._invoke_lock:
            process = self._require_process()
            if session_id is None:
                self._session_sequence += 1
                session_id = f'test-{self._session_sequence}'

            request = {
                'session_id': session_id,
                'event': 'request',
                'data': {'action': action, **parameters},
            }
            stdin = process.stdin
            if stdin is None:
                raise RuntimeError('plugin process stdin is unavailable')
            try:
                stdin.write(json.dumps(request) + '\n')
                stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise RuntimeError(
                    f'plugin process rejected request{self._diagnostics()}'
                ) from error

            deadline = time.monotonic() + self._timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f'timed out waiting for session {session_id!r}'
                        f'{self._diagnostics()}'
                    )
                try:
                    event = self._events.get(timeout=remaining)
                except queue.Empty as error:
                    raise TimeoutError(
                        f'timed out waiting for session {session_id!r}'
                        f'{self._diagnostics()}'
                    ) from error

                if isinstance(event, _EndOfStream):
                    raise RuntimeError(
                        'plugin process closed stdout before responding'
                        f'{self._diagnostics()}'
                    )
                if isinstance(event, Exception):
                    raise RuntimeError(
                        f'plugin process emitted an invalid frame: {event}'
                        f'{self._diagnostics()}'
                    ) from event
                if event.get('session_id') != session_id:
                    continue
                if event.get('event') != 'session':
                    raise RuntimeError(
                        f'plugin process emitted an invalid session event: {event!r}'
                    )
                return cast(SessionFrame, event)

    def close(self) -> None:
        """Stop the plugin process and release its streams."""
        process = self._process
        if process is None:
            return

        if process.stdin is not None and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass

        try:
            process.wait(timeout=self._shutdown_timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=self._shutdown_timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

        for stream in (process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        for thread in self._reader_threads:
            thread.join(timeout=self._shutdown_timeout)

        self._reader_threads = []
        self._process = None

    def _require_process(self) -> subprocess.Popen[str]:
        process = self._process
        if process is None:
            raise RuntimeError('plugin process is not started')
        return process

    def _subprocess_env(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(self._env)

        runtime_root = str(Path(__file__).resolve().parents[3])
        python_path = environment.get('PYTHONPATH')
        environment['PYTHONPATH'] = (
            os.pathsep.join((runtime_root, python_path))
            if python_path
            else runtime_root
        )
        return environment

    def _read_stdout(self, stdout: TextIO) -> None:
        try:
            for line in stdout:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError('protocol frame must be a JSON object')
                self._events.put(event)
        except Exception as error:
            self._events.put(error)
        finally:
            self._events.put(_EOF)

    def _read_stderr(self, stderr: TextIO) -> None:
        for line in stderr:
            stripped = line.rstrip()
            if stripped:
                self._stderr_tail.append(stripped)

    def _diagnostics(self) -> str:
        process = self._process
        status = process.poll() if process is not None else None
        details = [f'exit_code={status}']
        if self._stderr_tail:
            details.append(f"stderr={' | '.join(self._stderr_tail)}")
        return f" ({'; '.join(details)})"
