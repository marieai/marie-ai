from types import SimpleNamespace
from unittest.mock import AsyncMock

from marie.job.common import JobInfo, JobStatus
from marie.job.job_storage_client_proxy import JobInfoStorageClientProxy
from marie.storage.kv.in_memory import InMemoryKV


async def test_terminal_status_uses_deduplicating_callback() -> None:
    storage = InMemoryKV()
    storage.kv_store = {}
    event_publisher = SimpleNamespace(publish=AsyncMock())
    terminal_event_callback = AsyncMock(return_value=True)
    client = JobInfoStorageClientProxy(
        event_publisher,
        storage,
        terminal_event_callback=terminal_event_callback,
    )
    await client.put_info(
        'job-1',
        JobInfo(
            status=JobStatus.RUNNING,
            entrypoint='mock_executor_a:///document/extract',
            run_owner='owner-1',
            run_attempt_id='attempt-1',
        ),
    )
    storage.internal_kv_get = AsyncMock(wraps=storage.internal_kv_get)

    await client.put_status('job-1', JobStatus.SUCCEEDED)

    storage.internal_kv_get.assert_awaited_once()
    terminal_event_callback.assert_awaited_once_with(
        'job-1',
        JobStatus.SUCCEEDED,
        'owner-1',
        'attempt-1',
        'job_info_proxy',
    )
    event_publisher.publish.assert_not_awaited()
