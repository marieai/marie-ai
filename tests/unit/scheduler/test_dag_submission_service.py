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
async def test_submission_queue_is_bounded_and_abortable() -> None:
    service = build_service(queue_size=1)

    assert await service.submit(work_item('job-1')) == 'job-1'
    with pytest.raises(asyncio.QueueFull):
        await service.submit(work_item('job-2'))

    assert service.queue_size == 1
    assert service.pending_count == 1

    service.abort_pending()

    assert service.queue_size == 0
    assert service.pending_count == 0


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
