import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from marie.engine.llm_queue.store import (
    AdmissionConflict,
    RoutingManifestProjection,
    StoreUnavailable,
)

from marie.scheduler.services.llm_routing_projection_service import (
    LlmRoutingProjectionService,
)


def _event(route_digest: str = 'd' * 64) -> dict:
    return {
        'event_id': '018fa1f1-0000-7000-8000-000000000010',
        'work_unit_id': '018fa1f1-0000-7000-8000-000000000002',
        'route_digest': route_digest,
        'attempt_count': 1,
        'payload': {
            'job_id': '018fa1f1-0000-7000-8000-000000000001',
            'work_unit_id': '018fa1f1-0000-7000-8000-000000000002',
            'fabric_group_id': 'default',
            'policy_generation': 2,
            'policy_digest': 'a' * 64,
            'rule_digest': 'b' * 64,
            'normalized_fact_digest': 'c' * 64,
            'effective_page_count': 3,
            'pool_id': 'document-small',
            'logical_endpoint_group_id': 'primary',
            'endpoint_revision': 'r1',
            'estimator_version': 'page-count-v1',
            'routing_source': 'automatic',
            'routing_actor': None,
            'routing_reason': None,
        },
    }


class _Repository:
    def __init__(self) -> None:
        self.pending = [_event()]
        self.fail_first_ack = False
        self.ack_calls = 0
        self.failures = []
        self.failure_recording_error: BaseException | None = None

    async def claim_pending_routing_outbox(self, *, limit):
        return self.pending[:limit]

    async def acknowledge_routing_projection(self, *, event_id, route_digest):
        self.ack_calls += 1
        if self.fail_first_ack and self.ack_calls == 1:
            raise RuntimeError('database unavailable after projection')
        index = next(
            index
            for index, event in enumerate(self.pending)
            if isinstance(event, dict)
            and event.get('event_id') == event_id
            and event.get('route_digest') == route_digest
        )
        self.pending.pop(index)
        return True

    async def record_routing_projection_failure(self, **values):
        self.failures.append(values)
        if self.failure_recording_error is not None:
            raise self.failure_recording_error
        return True


class _Store:
    def __init__(self) -> None:
        self.routes = {}

    def project_routing_manifest(self, projection):
        identity = (projection.job_id, projection.work_unit_id)
        prior = self.routes.get(identity)
        if prior is not None and prior != projection.route_digest:
            raise AdmissionConflict('routing_binding_conflict')
        self.routes[identity] = projection.route_digest
        return SimpleNamespace(disposition='existing' if prior else 'projected')


@pytest.mark.asyncio
async def test_projector_restart_replays_projection_and_acknowledges() -> None:
    repository = _Repository()
    repository.fail_first_ack = True
    store = _Store()
    request_admission = AsyncMock(return_value=True)
    service = LlmRoutingProjectionService(
        repository=repository,
        store=store,
        logger=SimpleNamespace(warning=lambda *args: None),
        admission_callback=request_admission,
    )

    first = await service.run_once()
    second = await service.run_once()

    assert first.projected == 1
    assert first.failed == 1
    assert second.existing == 1
    assert second.acknowledged == 1
    assert repository.pending == []
    request_admission.assert_awaited_once_with('llm_routing_projection')


@pytest.mark.asyncio
async def test_projection_conflict_stays_pending_with_bounded_category() -> None:
    repository = _Repository()
    store = _Store()
    event = repository.pending[0]
    identity = (event['payload']['job_id'], event['payload']['work_unit_id'])
    store.routes[identity] = 'e' * 64
    service = LlmRoutingProjectionService(
        repository=repository,
        store=store,
        logger=SimpleNamespace(warning=lambda *args: None),
        admission_callback=AsyncMock(return_value=True),
    )

    result = await service.run_once()

    assert result.failed == 1
    assert result.acknowledged == 0
    assert repository.failures[0]['category'] == 'routing_binding_conflict'


@pytest.mark.asyncio
async def test_admission_wakeup_retries_after_projection_is_acknowledged() -> None:
    repository = _Repository()
    request_admission = AsyncMock(
        side_effect=[RuntimeError('admission worker unavailable'), True]
    )
    service = LlmRoutingProjectionService(
        repository=repository,
        store=_Store(),
        logger=SimpleNamespace(
            warning=lambda *args: None,
            exception=lambda *args: None,
        ),
        admission_callback=request_admission,
    )

    first = await service.run_once()
    second = await service.run_once()

    assert first.acknowledged == 1
    assert second.attempted == 0
    assert request_admission.await_count == 2


@pytest.mark.asyncio
async def test_store_unavailable_does_not_acknowledge() -> None:
    repository = _Repository()

    class Store:
        def project_routing_manifest(self, projection):
            raise StoreUnavailable('down')

    service = LlmRoutingProjectionService(
        repository=repository,
        store=Store(),
        logger=SimpleNamespace(warning=lambda *args: None),
        admission_callback=AsyncMock(return_value=True),
    )

    result = await service.run_once()

    assert result.failed == 1
    assert repository.ack_calls == 0
    assert repository.failures[0]['category'] == 'routing_store_unavailable'


@pytest.mark.parametrize(
    'malformed_event',
    [
        {},
        {'event_id': '018fa1f1-0000-7000-8000-000000000020'},
        {'route_digest': 'e' * 64},
        None,
        [],
    ],
)
async def test_malformed_event_does_not_abort_claimed_batch(malformed_event: Any) -> None:
    repository = _Repository()
    first = _event()
    last = _event()
    last['event_id'] = '018fa1f1-0000-7000-8000-000000000030'
    repository.pending = [first, malformed_event, last]
    request_admission = AsyncMock(return_value=True)
    logger = Mock()
    service = LlmRoutingProjectionService(
        repository=repository,
        store=_Store(),
        logger=logger,
        admission_callback=request_admission,
    )

    result = await service.run_once()

    assert result.attempted == 3
    assert result.projected == 1
    assert result.existing == 1
    assert result.acknowledged == 2
    assert result.failed == 1
    assert repository.pending == [malformed_event]
    assert repository.failures == []
    assert any(
        'routing_projection_invalid' in call.args
        for call in logger.warning.call_args_list
    )
    request_admission.assert_awaited_once_with('llm_routing_projection')


@pytest.mark.parametrize(
    ('projection_error', 'category'),
    [
        (AdmissionConflict('conflict'), 'routing_binding_conflict'),
        (StoreUnavailable('store down'), 'routing_store_unavailable'),
        (ValueError('invalid payload'), 'routing_projection_invalid'),
        (RuntimeError('projection rejected'), 'routing_projection_failed'),
    ],
)
@pytest.mark.parametrize(
    'recording_error', [StoreUnavailable('repository down'), RuntimeError('database down')]
)
async def test_failure_recording_error_does_not_abort_claimed_batch(
    projection_error: Exception, category: str, recording_error: Exception
) -> None:
    repository = _Repository()
    first = _event()
    last = _event()
    last['event_id'] = '018fa1f1-0000-7000-8000-000000000030'
    last['payload']['work_unit_id'] = '018fa1f1-0000-7000-8000-000000000003'
    repository.pending = [first, last]
    repository.failure_recording_error = recording_error

    class Store(_Store):
        def project_routing_manifest(
            self, projection: RoutingManifestProjection
        ) -> SimpleNamespace:
            if projection.work_unit_id == first['payload']['work_unit_id']:
                raise projection_error
            return super().project_routing_manifest(projection)

    store = Store()
    request_admission = AsyncMock(return_value=True)
    logger = Mock()
    service = LlmRoutingProjectionService(
        repository=repository,
        store=store,
        logger=logger,
        admission_callback=request_admission,
    )

    result = await service.run_once()

    assert result.attempted == 2
    assert result.projected == 1
    assert result.acknowledged == 1
    assert result.failed == 1
    assert repository.pending == [first]
    assert len(store.routes) == 1
    assert repository.failures == [
        {
            'event_id': first['event_id'],
            'route_digest': first['route_digest'],
            'category': category,
            'retry_seconds': 5,
        }
    ]
    assert any(category in call.args for call in logger.warning.call_args_list)
    assert any(call.kwargs.get('exc_info') for call in logger.warning.call_args_list)
    request_admission.assert_awaited_once_with('llm_routing_projection')


async def test_failure_recording_preserves_cancellation() -> None:
    repository = _Repository()
    repository.pending[0]['payload'] = {}
    repository.failure_recording_error = asyncio.CancelledError()
    service = LlmRoutingProjectionService(
        repository=repository,
        store=_Store(),
        logger=Mock(),
        admission_callback=AsyncMock(return_value=True),
    )

    with pytest.raises(asyncio.CancelledError):
        await service.run_once()

    assert repository.ack_calls == 0
