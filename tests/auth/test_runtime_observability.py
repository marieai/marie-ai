import asyncio
from dataclasses import replace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from marie.auth.api_key_manager import APIKeyManager
from marie.auth.auth_bearer import TokenBearer
from marie.messaging.events import EventMessage
from marie.messaging.grpc_event_broker import GrpcEventBroker
from marie.messaging.grpc_event_service import EventStreamServicer


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setattr(APIKeyManager, '_keys', {})


def key(name='operator', **policy):
    value = 'mas_' + name[0] * 54
    APIKeyManager.add_key(dict(name=name, api_key=value, **policy))
    return value


def test_runtime_key_policy_is_explicit_and_fabric_scoped():
    token = key(scopes=['runtime-observability'], allowed_fabrics=['fabric-a'])
    assert APIKeyManager.can_observe_runtime(token, 'fabric-a')
    assert not APIKeyManager.can_observe_runtime(token, 'fabric-b')
    assert not APIKeyManager.can_observe_runtime(token, 'fabric-a', tenant_id='org')
    assert not APIKeyManager.can_observe_runtime(key('inference'), 'fabric-a')


def test_duplicate_key_does_not_disclose_token():
    token = key()
    with pytest.raises(ValueError) as caught:
        APIKeyManager.add_key(dict(name='other', api_key=token))
    assert token not in str(caught.value)


@pytest.mark.asyncio
async def test_missing_token_is_401():
    with pytest.raises(HTTPException) as caught:
        await TokenBearer()(Request({'type': 'http', 'headers': []}))
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_runtime_event_policy_covers_live_replay_and_revocation():
    token = key(scopes=['runtime-observability'], allowed_fabrics=['fabric-a'])
    broker = GrpcEventBroker(ack_timeout_s=0.001, redelivery_check_interval_s=0.01)
    allowed = await broker.register_connection('allowed', api_key=token)
    denied = await broker.register_connection('denied')
    event = EventMessage(
        id='e',
        jobid='gateway',
        jobtag='',
        status='',
        timestamp=1,
        api_key='internal-topic',
        event='llm.dispatch.runtime.snapshot',
        payload={'fabric_group_id': 'fabric-a'},
        source='gateway://control-plane',
    )
    await broker.publish_event_message(event)
    for connection in ['allowed', 'denied']:
        await broker.subscribe(
            connection, connection, {'internal-topic'}, set(), {}, last_sequence_num=0
        )
    assert denied.empty()
    assert not allowed.empty()
    envelope = allowed.get_nowait()
    assert (
        EventStreamServicer(broker)._build_event_message(envelope).event.event.api_key
        == ''
    )
    replay, _, _ = await broker.get_replay_buffer('internal-topic', 0, 10)
    assert replay == []
    replay, _, _ = await broker.get_replay_buffer(
        'internal-topic', 0, 10, api_key=token
    )
    assert len(replay) == 1
    await broker.publish_event_message(
        replace(event, payload={'fabric_group_id': 'fabric-b'})
    )
    assert allowed.empty() and denied.empty()
    APIKeyManager._keys[token]['enabled'] = False
    await broker.start()
    await asyncio.sleep(0.04)
    await broker.stop()
    assert allowed.empty()


@pytest.mark.asyncio
async def test_heartbeat_does_not_emit_credential_topics():
    broker = GrpcEventBroker()
    token = key()
    event = EventMessage(
        id='e',
        jobid='',
        jobtag='',
        status='',
        timestamp=1,
        api_key=token,
        event='ordinary',
        payload={},
        source='gateway://control-plane',
    )
    await broker.publish_event_message(event)
    queue = asyncio.Queue()
    service = EventStreamServicer(broker, heartbeat_interval_s=0.001)
    task = asyncio.create_task(service._heartbeat_loop('fixture', queue))
    _, message = await asyncio.wait_for(queue.get(), 0.1)
    task.cancel()
    await task
    assert token not in str(message)


@pytest.mark.asyncio
async def test_grpc_standalone_replay_auth_and_subscription_event_filters():
    from marie.proto import event_stream_pb2 as pb2

    token = key(scopes=['runtime-observability'], allowed_fabrics=['fabric-a'])
    broker = GrpcEventBroker()
    event = EventMessage(
        id='e',
        jobid='',
        jobtag='',
        status='',
        timestamp=1,
        api_key='system:gateway',
        event='engine.event',
        payload={
            'event_type': 'llm.dispatch.runtime.snapshot',
            'fabric_group_id': 'fabric-a',
        },
        source='gateway://control-plane',
    )
    await broker.publish_event_message(event)
    service = EventStreamServicer(broker)
    for credential, count in [(token, 1), ('', 0)]:
        context = type(
            'Context',
            (),
            {
                'invocation_metadata': lambda self: [
                    ('authorization', 'Bearer ' + credential)
                ]
            },
        )()
        result = await service.GetReplayBuffer(
            pb2.ReplayRequest(
                topic='system:gateway', from_sequence_num=0, max_events=10
            ),
            context,
        )
        assert len(result.events) == count
        assert token not in str(result)
    queue = await broker.register_connection('filtered', api_key=token)
    await broker.subscribe(
        'filtered',
        'filtered',
        {'system:gateway'},
        {'different.event'},
        {},
        last_sequence_num=0,
    )
    assert queue.empty()
    await broker.subscribe('filtered', 'filtered2', {'*'}, set(), {})
    await broker.publish_event_message(
        replace(
            event,
            payload={
                'event_type': 'llm.dispatch.runtime.snapshot',
                'fabric_group_id': 'fabric-b',
            },
        )
    )
    assert queue.empty()


@pytest.mark.asyncio
async def test_queued_envelope_is_reauthorized_before_serialization():
    token = key(scopes=['runtime-observability'], allowed_fabrics=['fabric-a'])
    broker = GrpcEventBroker()
    queue = await broker.register_connection('connection', api_key=token)
    await broker.subscribe('connection', 'subscription', {'*'}, set(), {})
    event = EventMessage(
        id='e',
        jobid='',
        jobtag='',
        status='',
        timestamp=1,
        api_key='system:gateway',
        event='engine.event',
        payload={
            'event_type': 'llm.dispatch.runtime.snapshot',
            'fabric_group_id': 'fabric-a',
        },
        source='gateway://control-plane',
    )
    await broker.publish_event_message(event)
    envelope = queue.get_nowait()
    assert broker.can_deliver_envelope('connection', envelope)
    APIKeyManager._keys[token]['enabled'] = False
    assert not broker.can_deliver_envelope('connection', envelope)
