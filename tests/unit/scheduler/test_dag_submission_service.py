import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import marie.scheduler.services.dag_submission_service as submission_module
from marie.scheduler.services.dag_submission_service import DagSubmissionService


def build_service(
    *,
    running: list[bool] | None = None,
    queue_size: int = 10,
) -> DagSubmissionService:
    running = running or [True]
    return DagSubmissionService(
        repository=SimpleNamespace(
            create_queue=AsyncMock(),
            get_job_by_policy=AsyncMock(return_value=None),
            create_dag_with_jobs=AsyncMock(return_value=(True, 'dag-1')),
        ),
        frontier=SimpleNamespace(add_dag=AsyncMock()),
        known_queues={'extract'},
        notify_callback=AsyncMock(return_value=True),
        is_running=lambda: running[0],
        submission_processed_callback=AsyncMock(),
        logger=MagicMock(),
        queue_size=queue_size,
    )


def work_item(job_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=job_id,
        name='extract',
        policy='ALLOW_ALL',
        data={'metadata': {}},
    )


@pytest.mark.asyncio
async def test_submission_queue_applies_backpressure_and_is_abortable() -> None:
    service = build_service(queue_size=1)

    assert await service.submit(work_item('job-1')) == 'job-1'
    blocked_submit = asyncio.create_task(service.submit(work_item('job-2')))
    async with asyncio.timeout(1):
        while service.pending_count < 2:
            await asyncio.sleep(0)

    assert service.queue_size == 1
    assert service.pending_count == 2
    assert not blocked_submit.done()

    first_request = service._queue.get_nowait()
    service._queue.task_done()
    service._pending.pop(first_request.request_id)
    assert first_request.work_info.id == 'job-1'
    assert await blocked_submit == 'job-2'

    assert service.queue_size == 1
    assert service.pending_count == 1

    service.abort_pending()

    assert service.queue_size == 0
    assert service.pending_count == 0


@pytest.mark.asyncio
async def test_cancelled_submission_waiter_is_removed_from_pending() -> None:
    service = build_service(queue_size=1)

    await service.submit(work_item('job-1'))
    blocked_submit = asyncio.create_task(service.submit(work_item('job-2')))
    async with asyncio.timeout(1):
        while service.pending_count < 2:
            await asyncio.sleep(0)

    blocked_submit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked_submit

    assert service.queue_size == 1
    assert service.pending_count == 1
    service.abort_pending()


@pytest.mark.asyncio
async def test_submission_worker_accounts_for_successful_persistence() -> None:
    running = [True]
    service = build_service(running=running)
    service.persist = AsyncMock(return_value='job-1')

    await service.submit(work_item('job-1'))
    worker = asyncio.create_task(service.run_worker(0))
    async with asyncio.timeout(1):
        while service.submission_count == 0:
            await asyncio.sleep(0)

    running[0] = False
    worker.cancel()
    await worker

    service.persist.assert_awaited_once()
    service._submission_processed_callback.assert_awaited_once_with(1)
    assert service.pending_count == 0


@pytest.mark.asyncio
async def test_submission_worker_publishes_scheduled_after_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = [True]
    service = build_service(running=running)
    submitted = work_item('job-1')
    submitted.data = {
        'api_key': 'project-1',
        'name': 'extract',
        'metadata': {'ref_type': 'invoice', 'ref_id': 'document-1'},
    }
    calls: list[str] = []

    async def persist(*_: object) -> str:
        calls.append('persist')
        return 'job-1'

    async def publish(**kwargs: object) -> bool:
        calls.append('scheduled')
        timestamp = kwargs.pop('timestamp')
        assert isinstance(timestamp, int)
        assert kwargs == {
            'api_key': 'project-1',
            'job_id': 'job-1',
            'event_name': 'extract',
            'job_tag': 'invoice',
            'status': 'OK',
            'payload': {'ref_type': 'invoice', 'ref_id': 'document-1'},
        }
        return True

    service.persist = persist
    monkeypatch.setattr(submission_module, 'mark_as_scheduled_toast', publish)

    await service.submit(submitted)
    worker = asyncio.create_task(service.run_worker(0))
    async with asyncio.timeout(1):
        while service.submission_count == 0:
            await asyncio.sleep(0)

    running[0] = False
    worker.cancel()
    await worker

    assert calls == ['persist', 'scheduled']


@pytest.mark.asyncio
async def test_submission_worker_does_not_publish_scheduled_when_persistence_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = [True]
    service = build_service(running=running)
    submitted = work_item('job-1')
    submitted.data = {
        'api_key': 'project-1',
        'name': 'extract',
        'metadata': {'ref_type': 'invoice'},
    }
    service.persist = AsyncMock(side_effect=RuntimeError('database unavailable'))
    scheduled = AsyncMock(return_value=True)
    failure = AsyncMock(return_value=True)
    monkeypatch.setattr(submission_module, 'mark_as_scheduled_toast', scheduled)
    monkeypatch.setattr(submission_module, 'mark_as_failed_toast', failure)

    await service.submit(submitted)
    worker = asyncio.create_task(service.run_worker(0))
    async with asyncio.timeout(1):
        while service.pending_count:
            await asyncio.sleep(0)

    running[0] = False
    worker.cancel()
    await worker

    scheduled.assert_not_awaited()
    failure.assert_awaited_once()
    assert service.submission_count == 0


@pytest.mark.asyncio
async def test_scheduled_publication_failure_does_not_fail_persisted_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    running = [True]
    service = build_service(running=running)
    submitted = work_item('job-1')
    submitted.data = {
        'api_key': 'project-1',
        'name': 'extract',
        'metadata': {'ref_type': 'invoice'},
    }
    service.persist = AsyncMock(return_value='job-1')
    monkeypatch.setattr(
        submission_module,
        'mark_as_scheduled_toast',
        AsyncMock(return_value=False),
    )

    await service.submit(submitted)
    worker = asyncio.create_task(service.run_worker(0))
    async with asyncio.timeout(1):
        while service.submission_count == 0:
            await asyncio.sleep(0)

    running[0] = False
    worker.cancel()
    await worker

    assert service.submission_count == 1
    service._submission_processed_callback.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_persist_builds_and_commits_the_dag_before_frontier_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service()
    plan = object()
    nodes = [SimpleNamespace(dag_id=None), SimpleNamespace(dag_id=None)]
    monkeypatch.setattr(
        submission_module,
        'query_plan_work_items',
        MagicMock(return_value=(plan, nodes)),
    )
    submitted = work_item('dag-1')

    result = await service.persist(submitted)

    assert result == 'dag-1'
    assert [node.dag_id for node in nodes] == ['dag-1', 'dag-1']
    service.repository.create_dag_with_jobs.assert_awaited_once_with(
        dag_id='dag-1',
        plan=plan,
        dag_nodes=nodes,
        work_info=submitted,
    )
    service.frontier.add_dag.assert_awaited_once_with(plan, nodes)
    service._notify_callback.assert_awaited_once_with()
