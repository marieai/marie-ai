"""Tests for the public Marie plugin Python runtime."""

import json
from io import StringIO

from marie_plugins.runtime import StdioRunner, session_frame


def _frames(output: StringIO) -> list[dict]:
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_runner_validates_envelopes_and_dispatches_requests():
    stdin = StringIO(
        'not-json\n'
        '[]\n'
        '{"event":"request","data":{}}\n'
        '{"session_id":"s1","event":"response","data":{}}\n'
        '{"session_id":"s1","event":"request","data":{"value":1}}\n'
    )
    stdout = StringIO()

    def dispatch(request):
        return [
            session_frame(request['session_id'], 'stream', request['data']),
            session_frame(request['session_id'], 'end', {}),
        ]

    StdioRunner(
        dispatch,
        stdin=stdin,
        stdout=stdout,
        heartbeat_interval=None,
    ).run()

    frames = _frames(stdout)
    assert [frame['event'] for frame in frames] == [
        'log',
        'log',
        'log',
        'log',
        'session',
        'session',
    ]
    assert frames[-2]['data'] == {'type': 'stream', 'data': {'value': 1}}
    assert frames[-1]['data'] == {'type': 'end', 'data': {}}


def test_runner_classifies_unhandled_handler_errors():
    stdin = StringIO('{"session_id":"s1","event":"request","data":{}}\n')
    stdout = StringIO()

    def dispatch(_request):
        raise RuntimeError('private failure detail')

    StdioRunner(
        dispatch,
        stdin=stdin,
        stdout=stdout,
        heartbeat_interval=None,
    ).run()

    assert _frames(stdout) == [
        {
            'session_id': 's1',
            'event': 'session',
            'data': {
                'type': 'error',
                'data': {
                    'code': 'internal_error',
                    'message': 'plugin request handler failed',
                    'retryable': False,
                },
            },
        }
    ]


def test_runner_emits_session_frames_as_handler_yields_them():
    stdin = StringIO('{"session_id":"s1","event":"request","data":{}}\n')
    stdout = StringIO()

    def dispatch(_request):
        yield session_frame('s1', 'stream', {'sequence': 1})
        assert _frames(stdout) == [
            {
                'session_id': 's1',
                'event': 'session',
                'data': {'type': 'stream', 'data': {'sequence': 1}},
            }
        ]
        yield session_frame('s1', 'end', {})

    StdioRunner(
        dispatch,
        stdin=stdin,
        stdout=stdout,
        heartbeat_interval=None,
    ).run()

    assert [frame['data']['type'] for frame in _frames(stdout)] == [
        'stream',
        'end',
    ]


def test_runner_preserves_yielded_frames_before_generator_error():
    stdin = StringIO('{"session_id":"s1","event":"request","data":{}}\n')
    stdout = StringIO()

    def dispatch(_request):
        yield session_frame('s1', 'stream', {'sequence': 1})
        raise RuntimeError('private failure detail')

    StdioRunner(
        dispatch,
        stdin=stdin,
        stdout=stdout,
        heartbeat_interval=None,
    ).run()

    frames = _frames(stdout)
    assert [frame['data']['type'] for frame in frames] == ['stream', 'error']
    assert frames[-1]['data']['data'] == {
        'code': 'internal_error',
        'message': 'plugin request handler failed',
        'retryable': False,
    }
