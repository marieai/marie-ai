import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from marie.engine.llm_queue.admission_policy import AdmissionPolicy

import marie.scheduler.services.dag_submission_service as submission_module
from marie.query_planner.base import LlmQueryDefinition, Query, QueryPlan, QueryType
from marie.scheduler.llm_routing import (
    RoutingSubmissionError,
    TrustedRoutingContext,
    TrustedRoutingFacts,
    TrustedRoutingOverride,
)
from marie.scheduler.models import ExistingWorkPolicy
from marie.scheduler.services.dag_submission_service import DagSubmissionService


def build_service(*, running: list[bool] | None = None) -> DagSubmissionService:
    running = running or [True]
    return DagSubmissionService(
        repository=SimpleNamespace(
            create_queue=AsyncMock(),
            get_job_by_policy=AsyncMock(return_value=None),
            create_dag_with_jobs=AsyncMock(return_value=(True, 'dag-1')),
        ),
        dag_admission_callback=AsyncMock(return_value=True),
        known_queues={'extract'},
        notify_callback=AsyncMock(return_value=True),
        is_running=lambda: running[0],
        submission_processed_callback=AsyncMock(),
        logger=MagicMock(),
    )


def work_item(job_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=job_id,
        name='extract',
        policy='ALLOW_ALL',
        data={'metadata': {}},
        routing_override=None,
    )


def llm_plan(node_id: str) -> QueryPlan:
    return QueryPlan(
        nodes=[
            Query(
                task_id=node_id,
                query_str='Extract',
                dependencies=[],
                node_type=QueryType.COMPUTE,
                definition=LlmQueryDefinition(
                    model_name='mock',
                    endpoint='annotator_llm://extract',
                    params={'layout': 'mock'},
                ),
            )
        ]
    )


def admission_policy() -> AdmissionPolicy:
    def pool(pool_id: str, priority: int, match: dict) -> dict:
        return {
            'pool_id': pool_id,
            'enabled': True,
            'metadata': {
                'admission': {
                    'schema_version': 1,
                    'priority': priority,
                    'accepting': True,
                    'match': match,
                },
                'llm_dispatch': {
                    'schema_version': 1,
                    'endpoint_id': 'primary',
                    'revision': 'r1',
                },
            },
        }

    return AdmissionPolicy.from_rows(
        'default',
        1,
        [
            pool('default', 1_000_000, {}),
            pool(
                'document-medium',
                20,
                {'document.effective_page_count': {'gte': 5, 'lte': 16}},
            ),
            pool(
                'document-small',
                10,
                {'document.effective_page_count': {'gte': 1, 'lte': 4}},
            ),
        ],
    )


@pytest.mark.asyncio
async def test_llm_plan_passes_immutable_routes_to_submission_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true')
    service = build_service()
    policy = admission_policy()
    root = work_item('018fa1f1-0000-7000-8000-000000000001')
    root.routing_context = TrustedRoutingContext(
        base_facts=TrustedRoutingFacts.from_values(
            {
                'workload.kind': 'text',
                'workload.mode': 'batch',
                'request.source': 'workflow',
            }
        ),
        policy=policy,
    )
    node_id = '018fa1f1-0000-7000-8000-000000000002'
    plan = llm_plan(node_id)

    routes = await service._plan_llm_routes(
        root,
        plan,
        [SimpleNamespace(id=node_id)],
    )

    assert len(routes) == 1
    assert routes[0].work_unit_id == node_id
    assert routes[0].pool_id == 'default'
    assert routes[0].logical_endpoint_group_id == 'primary'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('metadata', 'default_count', 'expected_pool', 'expected_count'),
    [
        (
            {'uri': 's3://marie/stress/document.tif', 'page_count': 3.0},
            '8',
            'document-small',
            3,
        ),
        ({'uri': 's3://marie/stress/document.tif'}, None, 'document-small', 1),
        ({'uri': 's3://marie/stress/document.tif'}, '8', 'document-medium', 8),
        ({'uri': 's3://marie/stress/document.tif'}, 'invalid', 'document-small', 1),
        (
            {'uri': 's3://marie/stress/document.tif', 'pages': [2, 8, 2]},
            '8',
            'document-small',
            2,
        ),
        (
            {
                'document_id': '018fa1f1-0000-7000-8000-000000000099',
                'uri': 's3://marie/stress/document.tif',
                'page_count': 3.0,
            },
            '8',
            None,
            None,
        ),
        ({'document_id': '018fa1f1-0000-7000-8000-000000000099'}, '8', None, None),
    ],
)
async def test_queued_document_routing(
    metadata: dict[str, object],
    default_count: str | None,
    expected_pool: str | None,
    expected_count: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'true')
    if default_count is None:
        monkeypatch.delenv('LLM_QUEUE_DEFAULT_PAGE_COUNT', raising=False)
    else:
        monkeypatch.setenv('LLM_QUEUE_DEFAULT_PAGE_COUNT', default_count)
    service = build_service()
    policy = admission_policy()
    service.repository.load_active_admission_policy = AsyncMock(return_value=policy)
    service.repository.get_submission_document_for_routing = AsyncMock(
        return_value=None
    )
    root = work_item('018fa1f1-0000-7000-8000-000000000011')
    root.routing_context = None
    root.routing_fabric_group_id = 'default'
    root.routing_request_source = 'workflow'
    root.data['metadata'] = metadata
    node_id = '018fa1f1-0000-7000-8000-000000000012'
    plan = llm_plan(node_id)

    if expected_pool is None:
        with pytest.raises(RoutingSubmissionError, match='^routing_facts_missing$'):
            await service._plan_llm_routes(root, plan, [SimpleNamespace(id=node_id)])
        service.repository.get_submission_document_for_routing.assert_awaited_once_with(
            document_id=metadata.get('document_id'), storage_key=metadata.get('uri')
        )
        assert root.routing_context is None
        return

    routes = await service._plan_llm_routes(
        root,
        plan,
        [SimpleNamespace(id=node_id)],
    )

    assert len(routes) == 1
    assert routes[0].work_unit_id == node_id
    assert routes[0].pool_id == expected_pool
    assert routes[0].logical_endpoint_group_id == 'primary'
    assert routes[0].effective_page_count == expected_count
    if 'page_count' not in metadata and 'pages' not in metadata:
        assert 'document.total_page_count' not in root.routing_context.base_facts.values
        service.logger.warning.assert_called_once()
        warning = service.logger.warning.call_args.args[0]
        assert root.id in warning
        assert 'page count' in warning
        assert str(expected_count) in warning
        assert expected_pool in warning
        assert metadata['uri'] not in warning
        await service._plan_llm_routes(root, plan, [SimpleNamespace(id=node_id)])
        assert service.logger.warning.call_count == 2
    else:
        service.logger.warning.assert_not_called()
        service.logger.info.assert_called_once()
        assert expected_pool in service.logger.info.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('enabled', [None, 'false'])
async def test_direct_batch_persists_uri_only_llm_job_without_routing(
    monkeypatch: pytest.MonkeyPatch, enabled: str | None
) -> None:
    if enabled is None:
        monkeypatch.delenv('LLM_QUEUE_ENABLED', raising=False)
    else:
        monkeypatch.setenv('LLM_QUEUE_ENABLED', enabled)
    service = build_service()
    service.repository.load_active_admission_policy = AsyncMock()
    service.repository.get_submission_document_for_routing = AsyncMock(
        return_value=None
    )
    root = work_item('018fa1f1-0000-7000-8000-000000000011')
    root.routing_context = None
    root.routing_fabric_group_id = 'default'
    root.data['metadata'] = {'uri': 's3://marie/stress/document.tif'}
    node = work_item('018fa1f1-0000-7000-8000-000000000012')
    plan = llm_plan(node.id)
    monkeypatch.setattr(
        submission_module, 'query_plan_work_items', lambda _: (plan, [node])
    )

    assert await service.submit(root) == root.id

    service.logger.info.assert_any_call(
        f'Job submission received job_id={root.id} queue=extract'
    )
    service.repository.load_active_admission_policy.assert_not_awaited()
    service.repository.get_submission_document_for_routing.assert_not_awaited()
    service.repository.create_dag_with_jobs.assert_awaited_once_with(
        dag_id=root.id, plan=plan, dag_nodes=[node], work_info=root, llm_routes=()
    )


@pytest.mark.asyncio
async def test_direct_batch_rejects_an_explicit_queue_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('LLM_QUEUE_ENABLED', 'false')
    service = build_service()
    root = work_item('018fa1f1-0000-7000-8000-000000000011')
    root.routing_context = None
    root.routing_fabric_group_id = 'default'
    root.routing_override = TrustedRoutingOverride(
        pool_id='document-small', actor='operator', reason='routing test'
    )
    service.repository.load_active_admission_policy = AsyncMock(return_value=None)
    node = work_item('018fa1f1-0000-7000-8000-000000000012')
    with pytest.raises(RoutingSubmissionError, match='^routing_override_unavailable$'):
        await service._plan_llm_routes(root, llm_plan(node.id), [node])


@pytest.mark.asyncio
async def test_submit_waits_for_durable_persistence_before_returning() -> None:
    service = build_service()
    persistence_started = asyncio.Event()
    allow_commit = asyncio.Event()

    async def persist(*_: object) -> str:
        persistence_started.set()
        await allow_commit.wait()
        return 'job-1'

    service.persist = persist

    submission = asyncio.create_task(service.submit(work_item('job-1')))
    await persistence_started.wait()

    assert not submission.done()

    allow_commit.set()
    assert await submission == 'job-1'
    assert service.submission_count == 1
    service._submission_processed_callback.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_submit_rejects_when_scheduler_is_not_running() -> None:
    service = build_service(running=[False])

    with pytest.raises(RuntimeError, match='not running'):
        await service.submit(work_item('job-1'))

    service.logger.info.assert_any_call(
        'Job submission received job_id=job-1 queue=extract'
    )
    service.repository.create_dag_with_jobs.assert_not_awaited()


@pytest.mark.asyncio
async def test_submit_creates_job_queues_before_persistence() -> None:
    service = build_service()
    service.known_queues.clear()
    calls: list[str] = []

    async def create_queue(queue_name: str) -> None:
        calls.append(queue_name)

    async def persist(*_: object) -> str:
        calls.append('persist')
        return 'job-1'

    service.repository.create_queue = create_queue
    service.persist = persist

    assert await service.submit(work_item('job-1')) == 'job-1'
    assert calls == ['extract', '$extract_dlq', 'persist']
    assert service.known_queues == {'extract'}


@pytest.mark.asyncio
async def test_submit_propagates_persistence_failure_and_publishes_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service()
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

    with pytest.raises(RuntimeError, match='database unavailable'):
        await service.submit(submitted)

    scheduled.assert_not_awaited()
    failure.assert_awaited_once()
    assert service.submission_count == 0


@pytest.mark.asyncio
async def test_submit_publishes_scheduled_only_after_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service()
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

    assert await service.submit(submitted) == 'job-1'

    assert calls == ['persist', 'scheduled']


@pytest.mark.asyncio
async def test_post_commit_reporting_failures_do_not_fail_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service()
    submitted = work_item('job-1')
    submitted.data = {
        'api_key': 'project-1',
        'name': 'extract',
        'metadata': {'ref_type': 'invoice'},
    }
    service.persist = AsyncMock(return_value='job-1')
    service._submission_processed_callback.side_effect = RuntimeError('counter failed')
    monkeypatch.setattr(
        submission_module,
        'mark_as_scheduled_toast',
        AsyncMock(side_effect=RuntimeError('toast failed')),
    )

    assert await service.submit(submitted) == 'job-1'

    assert service.submission_count == 1
    service.logger.error.assert_any_call(
        'Failed to send scheduled toast for job-1: toast failed'
    )
    service.logger.error.assert_any_call(
        'Failed to record submission count for job-1: counter failed'
    )


@pytest.mark.asyncio
async def test_persist_commits_the_dag_before_post_commit_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service()
    plan = QueryPlan(nodes=[])
    nodes = [SimpleNamespace(dag_id=None), SimpleNamespace(dag_id=None)]
    calls: list[str] = []

    async def create_dag_with_jobs(**_: object) -> tuple[bool, str]:
        calls.append('commit')
        return True, 'dag-1'

    async def request_admission(_: str) -> bool:
        calls.append('admission')
        return True

    async def notify() -> bool:
        calls.append('notify')
        return True

    service.repository.create_dag_with_jobs = create_dag_with_jobs
    service._dag_admission_callback = request_admission
    service._notify_callback = notify
    monkeypatch.setattr(
        submission_module,
        'query_plan_work_items',
        MagicMock(return_value=(plan, nodes)),
    )
    submitted = work_item('dag-1')

    result = await service.persist(submitted)

    assert result == 'dag-1'
    assert calls == ['commit', 'admission', 'notify']
    assert [node.dag_id for node in nodes] == ['dag-1', 'dag-1']


@pytest.mark.asyncio
async def test_persist_keeps_dag_successful_when_post_commit_effects_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service()
    service._dag_admission_callback.side_effect = RuntimeError('admission failed')
    service._notify_callback.side_effect = RuntimeError('notify failed')
    monkeypatch.setattr(
        submission_module,
        'query_plan_work_items',
        MagicMock(return_value=(QueryPlan(nodes=[]), [SimpleNamespace(dag_id=None)])),
    )

    assert await service.persist(work_item('dag-1')) == 'dag-1'

    service.repository.create_dag_with_jobs.assert_awaited_once()
    service._dag_admission_callback.assert_awaited_once_with('submission')
    service._notify_callback.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_submission_validation_propagates_repository_outage() -> None:
    service = build_service()
    service.repository.get_job_by_policy.side_effect = RuntimeError(
        'database unavailable'
    )

    with pytest.raises(RuntimeError, match='database unavailable'):
        await service.is_valid_submission(
            work_item('dag-1'),
            ExistingWorkPolicy.REJECT_DUPLICATE,
        )


@pytest.mark.asyncio
async def test_concurrent_duplicate_id_has_one_durable_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = build_service()
    lock = asyncio.Lock()
    inserted = False

    async def create_dag_with_jobs(**_: object) -> tuple[bool, str]:
        nonlocal inserted
        async with lock:
            if inserted:
                return False, 'dag-1'
            inserted = True
            return True, 'dag-1'

    service.repository.create_dag_with_jobs = create_dag_with_jobs
    monkeypatch.setattr(
        submission_module,
        'query_plan_work_items',
        MagicMock(return_value=(QueryPlan(nodes=[]), [SimpleNamespace(dag_id=None)])),
    )

    results = await asyncio.gather(
        service.submit(work_item('dag-1')),
        service.submit(work_item('dag-1')),
        return_exceptions=True,
    )

    assert sum(result == 'dag-1' for result in results) == 1
    errors = [result for result in results if isinstance(result, Exception)]
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert service.submission_count == 1
