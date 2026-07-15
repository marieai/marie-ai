"""Tests for the reusable packaged-plugin stdio client."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from marie_plugins.runtime.testing import StdioPluginTestClient


def _write_script(path: Path, source: str) -> Path:
    path.write_text(textwrap.dedent(source))
    return path


def test_client_starts_runtime_plugin_and_invokes_sessions(tmp_path):
    script = _write_script(
        tmp_path / 'plugin.py',
        """\
        from marie_plugins.runtime import run, session_frame

        def dispatch(request):
            session_id = request['session_id']
            return [
                session_frame(session_id, 'stream', request['data']),
                session_frame(session_id, 'end', {}),
            ]

        run(dispatch)
        """,
    )

    with StdioPluginTestClient([sys.executable, str(script)], cwd=tmp_path) as plugin:
        first = plugin.invoke('echo', value=1)
        second = plugin.invoke('echo', value=2)

    assert first['data'] == {
        'type': 'stream',
        'data': {'action': 'echo', 'value': 1},
    }
    assert second['session_id'] == 'test-2'
    assert second['data']['data']['value'] == 2


def test_client_reports_invalid_plugin_output(tmp_path):
    script = _write_script(
        tmp_path / 'invalid.py',
        """\
        import sys

        sys.stdin.readline()
        print('not-json', flush=True)
        """,
    )

    with StdioPluginTestClient([sys.executable, str(script)], cwd=tmp_path) as plugin:
        with pytest.raises(RuntimeError, match='invalid frame'):
            plugin.invoke('invalid')


def test_client_times_out_and_stops_unresponsive_plugin(tmp_path):
    script = _write_script(
        tmp_path / 'blocked.py',
        """\
        import sys
        import time

        sys.stdin.readline()
        time.sleep(10)
        """,
    )

    with StdioPluginTestClient(
        [sys.executable, str(script)],
        cwd=tmp_path,
        timeout=0.05,
        shutdown_timeout=0.05,
    ) as plugin:
        with pytest.raises(TimeoutError, match='timed out waiting for session'):
            plugin.invoke('blocked')
