import json
from unittest.mock import Mock

from marie.constants import (
    JOB_INFO_KEY_PREFIX,
    JOB_STATUS_NOTIFICATION_CHANNEL,
    KV_NAMESPACE_JOB,
)
from marie.storage.kv.psql import PostgreSQLKV, _terminal_job_notification


def test_terminal_job_notification_filters_non_terminal_values() -> None:
    key = f'{JOB_INFO_KEY_PREFIX}job-1'

    assert (
        _terminal_job_notification(
            KV_NAMESPACE_JOB.decode(), key, json.dumps({'status': 'RUNNING'})
        )
        is None
    )
    assert (
        _terminal_job_notification('other', key, json.dumps({'status': 'SUCCEEDED'}))
        is None
    )


async def test_terminal_put_commits_status_and_notification_together() -> None:
    storage = object.__new__(PostgreSQLKV)
    storage.logger = Mock()
    storage.schema = None
    storage.table = 'kv_store_worker'
    storage._db_executor = None
    connection = Mock()
    cursor = Mock(rowcount=1)
    storage._get_connection = Mock(return_value=connection)
    storage._close_cursor = Mock()
    storage._close_connection = Mock()
    calls: list[tuple[object, object, dict]] = []

    def execute(statement: object, data: object = tuple(), **kwargs: object) -> object:
        calls.append((statement, data, kwargs))
        if kwargs.get('return_cursor'):
            return cursor
        return []

    storage._execute_sql_gracefully = Mock(side_effect=execute)
    value = json.dumps(
        {
            'status': 'SUCCEEDED',
            'run_owner': 'owner-1',
            'run_attempt_id': 'attempt-1',
        }
    ).encode()

    result = await storage.internal_kv_put(
        f'{JOB_INFO_KEY_PREFIX}job-1'.encode(),
        value,
        overwrite=True,
        namespace=KV_NAMESPACE_JOB,
    )

    assert result == 1
    assert calls[0][2]['commit'] is False
    assert calls[0][2]['connection'] is connection
    assert calls[1][0] == 'SELECT pg_notify(%s, %s)'
    assert calls[1][1][0] == JOB_STATUS_NOTIFICATION_CHANNEL
    assert json.loads(calls[1][1][1]) == {
        'job_id': 'job-1',
        'status': 'SUCCEEDED',
        'run_owner': 'owner-1',
        'run_attempt_id': 'attempt-1',
    }
    assert calls[1][2]['connection'] is connection
