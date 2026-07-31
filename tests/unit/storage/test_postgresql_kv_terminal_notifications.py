import asyncio
import json
from contextlib import asynccontextmanager
from unittest.mock import Mock

import pytest

from marie.constants import (
    JOB_INFO_KEY_PREFIX,
    JOB_STATUS_NOTIFICATION_CHANNEL,
    KV_NAMESPACE_JOB,
)
from marie.storage.kv.psql import PostgreSQLKV, _terminal_job_notification


class RecordingTransaction:
    def __init__(self, events: list[tuple]) -> None:
        self.events = events

    async def __aenter__(self) -> None:
        self.events.append(('begin',))

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        self.events.append(('rollback' if exc_type else 'commit',))


class RecordingConnection:
    def __init__(
        self, *, fail_notification: bool = False, inserted: bool = True
    ) -> None:
        self.events: list[tuple] = []
        self.fail_notification = fail_notification
        self.inserted = inserted

    def transaction(self) -> RecordingTransaction:
        return RecordingTransaction(self.events)

    async def fetchrow(self, query: str, *args: object) -> dict[str, str] | None:
        self.events.append(('upsert', query, args))
        return {'id': 'row-1'} if self.inserted else None

    async def execute(self, query: str, *args: object) -> str:
        self.events.append(('notify', query, args))
        if self.fail_notification:
            raise RuntimeError('notification failed')
        return 'SELECT 1'


class RecordingPool:
    def __init__(
        self, connection: RecordingConnection, *, acquire_delay: float = 0.0
    ) -> None:
        self.connection = connection
        self.acquire_delay = acquire_delay
        self.initialize_calls: list[tuple[dict, dict]] = []
        self.closed = False

    async def initialize(self, config: dict, **kwargs: object) -> None:
        self.initialize_calls.append((config, kwargs))

    @asynccontextmanager
    async def acquire(self):
        if self.acquire_delay:
            await asyncio.sleep(self.acquire_delay)
        yield self.connection

    async def close(self) -> None:
        self.closed = True


class RecordingSyncPool:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def make_storage(pool: RecordingPool) -> PostgreSQLKV:
    storage = object.__new__(PostgreSQLKV)
    storage.logger = Mock()
    storage.schema = None
    storage.table = 'kv_store_worker'
    storage._config = {'max_connections': 4}
    storage._async_pool = pool
    storage._async_pool_lock = asyncio.Lock()
    storage._async_pool_initialized = True
    storage._closed = False
    storage.postgreSQL_pool = RecordingSyncPool()
    return storage


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
    connection = RecordingConnection()
    storage = make_storage(RecordingPool(connection))
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
    assert [event[0] for event in connection.events] == [
        'begin',
        'upsert',
        'notify',
        'commit',
    ]
    notify = connection.events[2]
    assert notify[1] == 'SELECT pg_notify($1, $2)'
    assert notify[2][0] == JOB_STATUS_NOTIFICATION_CHANNEL
    assert json.loads(notify[2][1]) == {
        'job_id': 'job-1',
        'status': 'SUCCEEDED',
        'run_owner': 'owner-1',
        'run_attempt_id': 'attempt-1',
    }


async def test_terminal_put_rolls_back_when_notification_fails() -> None:
    connection = RecordingConnection(fail_notification=True)
    storage = make_storage(RecordingPool(connection))
    value = json.dumps({'status': 'SUCCEEDED'}).encode()

    with pytest.raises(RuntimeError, match='notification failed'):
        await storage.internal_kv_put(
            f'{JOB_INFO_KEY_PREFIX}job-1'.encode(),
            value,
            overwrite=True,
            namespace=KV_NAMESPACE_JOB,
        )

    assert [event[0] for event in connection.events] == [
        'begin',
        'upsert',
        'notify',
        'rollback',
    ]


async def test_non_overwrite_put_ignores_existing_key() -> None:
    connection = RecordingConnection(inserted=False)
    storage = make_storage(RecordingPool(connection))

    result = await storage.internal_kv_put(
        b'key-1',
        b'{}',
        overwrite=False,
        namespace=b'test',
    )

    assert result == 0
    assert [event[0] for event in connection.events] == [
        'begin',
        'upsert',
        'commit',
    ]
    query = connection.events[1][1]
    assert 'ON CONFLICT (namespace, key) DO NOTHING' in query


async def test_db_operation_reports_async_pool_and_database_time(
    monkeypatch,
) -> None:
    storage = make_storage(RecordingPool(RecordingConnection(), acquire_delay=0.01))
    events: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        'marie.storage.kv.psql.scheduler_trace',
        lambda event, **fields: events.append((event, fields)),
    )

    async def operation(connection) -> str:
        return 'value'

    assert await storage._run_db_operation('get', operation, job_id='job-1') == 'value'

    event, fields = events[-1]
    assert event == 'postgres_kv_operation_completed'
    assert fields['operation'] == 'get'
    assert fields['job_id'] == 'job-1'
    assert fields['mode'] == 'async'
    assert fields['succeeded'] is True
    assert fields['pool_wait_ms'] >= 5.0
    assert fields['database_operation_ms'] >= 0.0
    assert fields['total_ms'] >= fields['pool_wait_ms']


async def test_async_pool_is_initialized_with_bounded_config() -> None:
    pool = RecordingPool(RecordingConnection())
    storage = make_storage(pool)
    storage._async_pool_initialized = False

    await storage._ensure_async_pool()

    assert pool.initialize_calls == [
        (
            {'max_connections': 4},
            {'autocommit': True, 'trace_name': 'async_kv'},
        )
    ]


async def test_close_releases_sync_and_async_pools() -> None:
    pool = RecordingPool(RecordingConnection())
    storage = make_storage(pool)

    await storage.close()
    await storage.close()

    assert storage._closed is True
    assert pool.closed is True
    assert storage.postgreSQL_pool.closed is True
