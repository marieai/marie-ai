from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from marie.engine.llm_queue.store import AdmissionConflict, StoreUnavailable

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

    async def claim_pending_routing_outbox(self, *, limit):
        return self.pending[:limit]

    async def acknowledge_routing_projection(self, *, event_id, route_digest):
        self.ack_calls += 1
        if self.fail_first_ack and self.ack_calls == 1:
            raise RuntimeError('database unavailable after projection')
        assert event_id == self.pending[0]['event_id']
        assert route_digest == self.pending[0]['route_digest']
        self.pending.clear()
        return True

    async def record_routing_projection_failure(self, **values):
        self.failures.append(values)
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
