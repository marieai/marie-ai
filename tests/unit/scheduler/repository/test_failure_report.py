from contextlib import asynccontextmanager

import pytest

from marie.scheduler.repository.failure_report import read_failure_report


class ReportConnection:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def fetchrow(self, query, *args):
        self.calls.append((query, args))
        return (self.result,)


class ReportPool:
    def __init__(self, result):
        self.connection = ReportConnection(result)

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


async def test_failure_report_keeps_selected_attempt_and_full_traceback():
    pool = ReportPool(
        {
            'job': {'job_id': 'job-a', 'scheduler_state': 'completed'},
            'selected_event': {
                'history_id': 912,
                'run_attempt_id': 'failed-attempt',
                'error': {
                    'type': 'QueueTaskError',
                    'message': 'call_timeout',
                    'traceback': 'Traceback\n  worker.py:412\nQueueTaskError: call_timeout',
                },
            },
            'worker_history': [{'history_id': 913, 'status': 'SUCCEEDED'}],
            'scheduler_history': [{'status': 'retry', 'retry_count': 1}],
            'attempts': [
                {'run_attempt_id': 'failed-attempt', 'terminal_accepted': False}
            ],
        }
    )
    report = await read_failure_report(pool, 'job-a', 912)
    assert report['selected_event']['run_attempt_id'] == 'failed-attempt'
    assert report['selected_event']['error']['traceback'].endswith('call_timeout')
    assert report['job']['scheduler_state'] == 'completed'
    assert report['attempts'][0]['terminal_accepted'] is False
    assert pool.connection.calls[0][1] == ('job-a', 912)


async def test_failure_report_marks_limits_and_redacts_credential_strings():
    pool = ReportPool(
        {
            'job': {},
            'selected_event': {
                'error': {
                    'message': 'Authorization: Bearer private-token\napi_key=private-key\n'
                    '"api_key": "private-json-key"\npassword=private-password\n'
                    'https://user:private-url-password@example.test/',
                    'traceback': 'é' * 80_000,
                },
            },
            'worker_history': [{'history_id': i} for i in range(251)],
            'scheduler_history': [],
            'attempts': [],
        }
    )
    report = await read_failure_report(pool, 'job-a', 912)
    assert len(report['worker_history']) == 250
    assert report['truncated']['worker_history'] is True
    assert report['truncated']['text'] is True
    assert len(report['selected_event']['error']['traceback'].encode('utf-8')) < 70_000
    assert 'private-token' not in repr(report)
    assert 'private-key' not in repr(report)
    assert 'private-json-key' not in repr(report)
    assert 'private-url-password' not in repr(report)


async def test_failure_report_refuses_an_event_missing_from_selected_job():
    pool = ReportPool({'job': {}, 'selected_event': None})
    with pytest.raises(LookupError):
        await read_failure_report(pool, 'job-a', 999)
    assert await read_failure_report(ReportPool(None), 'missing', 999) is None
