from importlib import import_module
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from marie.scheduler.llm_routing import TrustedRoutingOverride
from marie.serve.runtimes.gateway.marie.llm_dispatch_runtime import (  # noqa: F401
    GatewayLlmDispatchRuntime,
)
from marie.serve.runtimes.servers.marie_gateway import MarieServerGateway

gateway_module = import_module(MarieServerGateway.__module__)


def build_gateway() -> tuple[MarieServerGateway, list[Any]]:
    gateway = object.__new__(MarieServerGateway)
    submitted: list[Any] = []

    async def submit(work_info: Any) -> str:
        submitted.append(work_info)
        return work_info.id

    gateway.job_scheduler = SimpleNamespace(submit_job=submit)
    gateway.llm_dispatch_runtime = SimpleNamespace(
        config=SimpleNamespace(fabric_group_id='default')
    )
    gateway.gateway_instance_id = 'gateway-1'
    gateway.logger = MagicMock()
    return gateway, submitted


def submission_message() -> dict[str, object]:
    return {
        'api_key': 'project-1',
        'action_type': 'invoke',
        'command': 'submit',
        'action': 'submit',
        'name': 'extract',
        'metadata': {
            'project_id': 'project-1',
            'ref_type': 'invoice',
            'ref_id': 'document-1',
            'planner': 'extract',
        },
    }


@pytest.mark.asyncio
async def test_gateway_marks_operator_override_as_trusted_scheduler_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, submitted = build_gateway()
    monkeypatch.delenv('MARIE_GATEWAY_PUBLISH_ACCEPTED_EVENT', raising=False)
    override = TrustedRoutingOverride(
        pool_id='document-small',
        actor='routing-admin',
        reason='qualification replay',
    )

    response = await gateway.handle_job_submit_command(
        submission_message(), routing_override=override
    )

    assert response.parameters['status'] == 'ok'
    assert submitted[0].routing_request_source == 'operator'
    assert submitted[0].routing_override == override
    assert submitted[0].data['metadata'].get('pool_id') is None


@pytest.mark.asyncio
async def test_gateway_generates_job_id_without_publishing_acceptance_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, submitted = build_gateway()
    accepted = AsyncMock(return_value=True)
    monkeypatch.delenv('MARIE_GATEWAY_PUBLISH_ACCEPTED_EVENT', raising=False)
    monkeypatch.setattr(gateway_module, 'mark_as_accepted', accepted)

    response = await gateway.handle_job_submit_command(submission_message())

    assert len(submitted) == 1
    job_id = submitted[0].id
    assert UUID(job_id).version == 7
    assert response.parameters['job_id'] == job_id
    accepted.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_can_publish_optional_accepted_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, submitted = build_gateway()
    accepted = AsyncMock(return_value=True)
    monkeypatch.setenv('MARIE_GATEWAY_PUBLISH_ACCEPTED_EVENT', 'true')
    monkeypatch.setattr(gateway_module, 'mark_as_accepted', accepted)

    response = await gateway.handle_job_submit_command(submission_message())

    job_id = submitted[0].id
    assert response.parameters['job_id'] == job_id
    accepted.assert_awaited_once()
    accepted_kwargs = dict(accepted.await_args.kwargs)
    timestamp = accepted_kwargs.pop('timestamp')
    assert isinstance(timestamp, int)
    assert accepted_kwargs == {
        'api_key': 'project-1',
        'job_id': job_id,
        'event_name': 'extract',
        'job_tag': 'invoice',
        'status': 'OK',
        'payload': {
            'project_id': 'project-1',
            'ref_type': 'invoice',
            'ref_id': 'document-1',
            'planner': 'extract',
        },
    }


@pytest.mark.asyncio
async def test_gateway_keeps_durable_success_when_accepted_event_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, submitted = build_gateway()
    monkeypatch.setenv('MARIE_GATEWAY_PUBLISH_ACCEPTED_EVENT', 'true')
    monkeypatch.setattr(
        gateway_module,
        'mark_as_accepted',
        AsyncMock(side_effect=RuntimeError('toast unavailable')),
    )

    response = await gateway.handle_job_submit_command(submission_message())

    assert response.parameters['status'] == 'ok'
    assert response.parameters['job_id'] == submitted[0].id
    gateway.logger.error.assert_called_once_with(
        f'Failed to publish accepted event for durable job '
        f'{submitted[0].id}: toast unavailable'
    )


@pytest.mark.asyncio
async def test_gateway_does_not_duplicate_delegated_submission_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, _ = build_gateway()
    gateway.job_scheduler.submit_job = AsyncMock(
        side_effect=RuntimeError('database unavailable')
    )
    failed = AsyncMock(return_value=True)
    monkeypatch.setattr(gateway_module, 'mark_as_failed', failed)

    response = await gateway.handle_job_submit_command(submission_message())

    assert response.parameters['status'] == 'error'
    failed.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_rejects_invalid_accepted_event_flag_before_submission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, submitted = build_gateway()
    monkeypatch.setenv('MARIE_GATEWAY_PUBLISH_ACCEPTED_EVENT', 'invalid')

    response = await gateway.handle_job_submit_command(submission_message())

    assert submitted == []
    assert response.parameters['status'] == 'error'
