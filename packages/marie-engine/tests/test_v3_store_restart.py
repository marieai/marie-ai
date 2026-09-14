"""Retained-AOF restart boundaries on explicitly owned standalone test stores."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import traceback
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from marie.engine.completion_contract import CompletionCallParams
from marie.engine.llm_queue.config import LlmQueueConfig
from marie.engine.llm_queue.producer import V3Producer
from marie.engine.llm_queue.store import (
    AdmissionConflict,
    ProducerDead,
    StoreUnavailable,
)
from test_request_dispatcher import dispatcher_for, eventually, http_endpoint, store

from tools.stress.llm_v3_aimock_e2e import fixture_action, inspect_owned_fixture


def owned_fixture(store) -> dict:
    manifest = json.loads(Path(os.environ['MARIE_LLM_QUEUE_TEST_STORES']).read_text())
    matches = [entry for entry in manifest.values() if entry['url'] == store.test_url]
    assert len(matches) == 1
    fixture = dict(
        matches[0], bindings={'6379/tcp': str(urlsplit(store.test_url).port)}
    )
    inspect_owned_fixture(fixture, running=True)
    return fixture


async def wait_for_store(store) -> None:
    def available() -> bool:
        try:
            return bool(store._read("ping"))
        except StoreUnavailable:
            return False

    await eventually(available)


def save_evidence(fixture: dict, evidence: dict) -> None:
    directory = os.environ.get('MARIE_LLM_TASK6_EVIDENCE_DIR')
    if directory:
        path = (
            Path(directory) / f"store-restart-{fixture['service']}-{uuid4().hex}.json"
        )
        path.write_text(json.dumps(evidence, indent=2) + '\n')


async def restore(fixture: dict, evidence: dict) -> None:
    state = await asyncio.to_thread(inspect_owned_fixture, fixture)
    if not state['running']:
        evidence['lifecycle'].append(
            await asyncio.to_thread(fixture_action, fixture, 'start')
        )


async def cleanup_restart(
    fixture: dict, evidence: dict, original: BaseException | None, cleanup_steps: list
) -> None:
    async def verify_identity() -> None:
        evidence['final_identity'] = await asyncio.to_thread(
            inspect_owned_fixture, fixture, running=True
        )
        if (
            evidence['lifecycle']
            and evidence['final_identity']['mounts']
            != evidence['lifecycle'][0]['before']['mounts']
        ):
            raise ValueError('Owned store retained volume identity mismatch')

    primary = original
    for label, action in [
        ('restore owned store', lambda: restore(fixture, evidence)),
        *cleanup_steps,
        ('verify final identity', verify_identity),
        ('save evidence', lambda: save_evidence(fixture, evidence)),
    ]:
        try:
            result = action()
            if inspect.isawaitable(result):
                await result
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note(
                    label
                    + ':\n'
                    + ''.join(traceback.format_exception(error, chain=False))
                )
    if original is None and primary is not None:
        raise primary


@pytest.mark.parametrize('expires', [False, True], ids=['live-lease', 'expired-lease'])
async def test_store_restart_preserves_original_unsent_authority(
    store, http_endpoint, monkeypatch, expires
):
    fixture = owned_fixture(store)
    url, received, _, _ = http_endpoint
    lease = 2 if expires else 60
    batch_timeout = 25 if expires else 90
    runtime = dispatcher_for(store, url, owner_lease_ms=lease * 1000, drain_seconds=0.2)
    producer = V3Producer(
        config=LlmQueueConfig(
            enabled=True,
            queue_url=store.test_url,
            pool_id='pool',
            fabric_group_id=store.keys.fabric_id,
            queue_contract_version='v3',
            producer_ttl_seconds=lease,
            producer_refresh_interval_seconds=0.25,
        )
    )
    # Hold only the unsent ready-head boundary; all ownership, polling and I/O stay real.
    dispatch = False
    ready_head = store.ready_head

    def gated_head(*args, **kwargs):
        return ready_head(*args, **kwargs) if dispatch else None

    monkeypatch.setattr(store, 'ready_head', gated_head)
    evidence = {
        'scenario': 'expired-lease' if expires else 'live-lease',
        'lifecycle': [],
    }
    task, original = None, None
    delivered = []
    await runtime.start()
    try:
        await eventually(lambda: runtime._owner_ready)
        task = asyncio.create_task(
            asyncio.to_thread(
                producer.execute,
                calls=[CompletionCallParams(model='unsent', messages=[])],
                batch_request_id='original',
                batch_timeout=batch_timeout,
                on_result=lambda *args: delivered.append(args),
            )
        )
        await eventually(lambda: store.usage()['active_items'] == 1)
        attempt = next(iter(producer._pending))
        before = store.metadata(attempt)
        owner = runtime.owner
        evidence.update(
            attempt_id=attempt,
            producer_id=producer.producer_id,
            deadline_ms=before.expires_at_ms,
            server_before_ms=store.server_time_ms(),
            producer_ttl_before_ms=store.client.pttl(
                store.keys.alive(producer.producer_id)
            ),
            owner_ttl_before_ms=store.client.pttl(store.keys.owner),
        )
        evidence['deadline_budget_before_ms'] = (
            evidence['deadline_ms'] - evidence['server_before_ms']
        )
        assert evidence['producer_ttl_before_ms'] > lease * 1000 - 1000
        assert evidence['owner_ttl_before_ms'] > lease * 1000 / 2
        assert evidence['deadline_budget_before_ms'] > batch_timeout * 1000 - 1000
        assert received == [] and delivered == []
        evidence['lifecycle'].append(
            await asyncio.to_thread(fixture_action, fixture, 'stop')
        )
        dispatch = True
        # Explicit failed start, read and cleanup cannot authorize a side effect.
        for operation in (
            lambda: store.authorize_start(owner, attempt, claim_id='not-transmitted'),
            lambda: store.read_result(producer.producer_id, attempt),
            lambda: store.expire_producer(producer.producer_id),
        ):
            with pytest.raises(StoreUnavailable):
                await asyncio.to_thread(operation)
        await asyncio.sleep(3 if expires else 0.4)
        assert received == [] and delivered == [] and not task.done()
        evidence['outage_provider_acceptances'] = len(received)
        evidence['outage_deliveries'] = len(delivered)
        evidence['lifecycle'].append(
            await asyncio.to_thread(fixture_action, fixture, 'start')
        )
        await wait_for_store(store)
        evidence['server_after_ms'] = store.server_time_ms()
        outage_ms = evidence['server_after_ms'] - evidence['server_before_ms']
        evidence['observed_outage_ms'] = outage_ms
        if expires:
            assert outage_ms > max(
                evidence['producer_ttl_before_ms'], evidence['owner_ttl_before_ms']
            )
            assert store.client.pttl(store.keys.alive(producer.producer_id)) == -2
            with pytest.raises(ProducerDead):
                await asyncio.wait_for(task, 8)
            await eventually(lambda: store.usage()['records'] == 0)
            replacement = store.create_producer(lease_ms=5000)
            try:
                assert store.read_result(replacement, attempt) is None
                assert not store.client.smembers(store.keys.members(replacement))
                assert received == [] and delivered == []
                evidence['replacement_members'] = 0
            finally:
                store.close_producer(replacement)
        else:
            assert outage_ms < min(
                evidence['producer_ttl_before_ms'],
                evidence['owner_ttl_before_ms'],
                evidence['deadline_budget_before_ms'],
            )
            result = await asyncio.wait_for(task, 18)
            assert len(result) == 1 and result[0].error is None
            assert len(received) == len(delivered) == 1
            after = store.metadata(attempt)
            assert after.producer_id == before.producer_id == producer.producer_id
            assert after.expires_at_ms == before.expires_at_ms
            replacement = store.create_producer(lease_ms=5000)
            try:
                with pytest.raises(AdmissionConflict):
                    store.read_result(replacement, attempt)
            finally:
                store.close_producer(replacement)
            evidence['result_state'] = after.state
        evidence['provider_acceptances'] = len(received)
        evidence['deliveries'] = len(delivered)
    except BaseException as error:
        original = error
        raise
    finally:
        await cleanup_restart(
            fixture,
            evidence,
            original,
            [
                ('close producer', lambda: asyncio.to_thread(producer.close)),
                (
                    'join producer task',
                    lambda: (
                        asyncio.gather(task, return_exceptions=True)
                        if task is not None
                        else None
                    ),
                ),
                ('stop dispatcher', runtime.stop),
            ],
        )


async def test_store_restart_retains_transmitted_reservation_after_dispatcher_expiry(
    store, http_endpoint
):
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint
    from test_request_dispatcher import admit_request

    fixture = owned_fixture(store)
    url, received, releases, _ = http_endpoint
    releases['model'] = asyncio.Event()
    client = EndpointClient(RegisteredEndpoint('endpoint', url, allow_loopback=True))
    transport = None
    producer_lease_ms = 60000
    request_deadline_ms = 90000
    transport_budget_ms = 90000
    owner = store.acquire_owner('before-restart', lease_ms=2000)
    store.configure_route(
        owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=1,
        execution_bytes=100000,
    )
    producer = store.create_producer(lease_ms=producer_lease_ms)
    req = admit_request(store, producer_id=producer, deadline_ms=request_deadline_ms)
    claim = 'transmitted-before-restart'
    store.claim(owner, req.attempt_id, pool_id='pool', claim_id=claim)
    started = store.authorize_start(owner, req.attempt_id, claim_id=claim)
    assert started.disposition == 'started'
    before = store.metadata(req.attempt_id)
    server_before_ms = store.server_time_ms()
    evidence = {
        'scenario': 'transmitted-reservation',
        'lifecycle': [],
        'attempt_id': req.attempt_id,
        'producer_id': producer,
        'deadline_ms': before.expires_at_ms,
        'server_before_ms': server_before_ms,
        'producer_ttl_before_ms': store.client.pttl(store.keys.alive(producer)),
        'owner_ttl_before_ms': store.client.pttl(store.keys.owner),
        'deadline_budget_before_ms': before.expires_at_ms - server_before_ms,
        'transport_budget_ms': transport_budget_ms,
    }
    assert evidence['producer_ttl_before_ms'] > producer_lease_ms - 1000
    assert evidence['owner_ttl_before_ms'] > 1000
    assert evidence['deadline_budget_before_ms'] > request_deadline_ms - 1000
    original = None
    try:
        transport = asyncio.create_task(
            client.execute(req.call, timeout_seconds=transport_budget_ms / 1000)
        )
        await eventually(lambda: len(received) == 1)
        assert not transport.done()
        evidence['lifecycle'].append(
            await asyncio.to_thread(fixture_action, fixture, 'stop')
        )
        with pytest.raises(StoreUnavailable):
            await asyncio.to_thread(store.recover_claim, owner, req.attempt_id)
        await asyncio.sleep(3)
        evidence['lifecycle'].append(
            await asyncio.to_thread(fixture_action, fixture, 'start')
        )
        await wait_for_store(store)
        evidence['server_after_ms'] = store.server_time_ms()
        outage_ms = evidence['server_after_ms'] - evidence['server_before_ms']
        evidence['observed_outage_ms'] = outage_ms
        assert outage_ms > evidence['owner_ttl_before_ms']
        assert outage_ms < min(
            evidence['producer_ttl_before_ms'],
            evidence['deadline_budget_before_ms'],
            evidence['transport_budget_ms'],
        )
        assert store.client.pttl(store.keys.alive(producer)) > 0
        assert store.client.pttl(store.keys.owner) == -2
        replacement_owner = store.acquire_owner('after-restart', lease_ms=15000)
        assert replacement_owner.generation > owner.generation
        recovered = store.recover_claim(replacement_owner, req.attempt_id)
        assert recovered.disposition == 'outcome_unknown'
        retained = store.metadata(req.attempt_id)
        assert retained.expires_at_ms == before.expires_at_ms
        assert retained.execution_seq == before.execution_seq == started.execution_seq
        assert store.usage()['reserved_items'] == 1
        # Expiring producer content must retain the unresolved physical reservation.
        store.close_producer(producer)
        store.expire_producer(producer)
        replacement = store.create_producer(lease_ms=5000)
        with pytest.raises((AdmissionConflict, ProducerDead)):
            store.read_result(replacement, req.attempt_id)
        await asyncio.sleep(
            0.4
        )  # Beyond configured 200ms uncertainty, still no settlement proof.
        assert store.usage()['reserved_items'] == 1
        assert store.ready_head('pool') is None
        assert len(received) == 1 and not transport.done()
        evidence.update(
            recovery_disposition=recovered.disposition,
            reserved_items_after_uncertainty=store.usage()['reserved_items'],
            execution_seq=retained.execution_seq,
            uncertainty_until=retained.uncertainty_until,
            provider_acceptances=len(received),
        )
        store.close_producer(replacement)
    except BaseException as error:
        original = error
        raise
    finally:
        await cleanup_restart(
            fixture,
            evidence,
            original,
            [
                ('release held provider', releases['model'].set),
                (
                    'join provider task',
                    lambda: (
                        asyncio.gather(transport, return_exceptions=True)
                        if transport is not None
                        else None
                    ),
                ),
                ('close endpoint client', client.close),
            ],
        )


def test_fixture_mutation_rejects_missing_ownership_metadata():
    with pytest.raises(ValueError, match='metadata'):
        fixture_action({'container_id': 'unowned'}, 'stop')


@pytest.mark.parametrize(
    'has_original', [True, False], ids=['original-failure', 'cleanup-only']
)
async def test_restart_cleanup_preserves_primary_through_multiple_failures(
    monkeypatch, has_original
):
    import sys

    module = sys.modules[__name__]
    actions = []
    restart_error = RuntimeError('docker start failed with fixture detail')
    initial = AssertionError('original scenario assertion') if has_original else None

    def inspect_fixture(fixture, *, running=None):
        actions.append('restore-inspect' if running is None else 'final-identity')
        if running is True:
            raise OSError('identity verification unavailable with detail')
        return {'running': False}

    def failed_start(fixture, action):
        actions.append(action)
        raise restart_error

    def close_resource():
        actions.append('close-resource')
        raise ValueError('resource close failed with detail')

    async def stop_runtime():
        actions.append('stop-runtime')
        raise TimeoutError('runtime stop timed out with detail')

    def write_evidence(fixture, evidence):
        actions.append('save-evidence')
        raise OSError('evidence disk full with detail')

    monkeypatch.setattr(module, 'inspect_owned_fixture', inspect_fixture)
    monkeypatch.setattr(module, 'fixture_action', failed_start)
    monkeypatch.setattr(module, 'save_evidence', write_evidence)
    primary = initial or restart_error
    with pytest.raises(type(primary)) as raised:
        try:
            if initial is not None:
                raise initial
        finally:
            await cleanup_restart(
                {},
                {'lifecycle': []},
                initial,
                [('close-resource', close_resource), ('stop-runtime', stop_runtime)],
            )
    assert raised.value is primary
    assert actions == [
        'restore-inspect',
        'start',
        'close-resource',
        'stop-runtime',
        'final-identity',
        'save-evidence',
    ]
    notes = '\n'.join(raised.value.__notes__)
    for detail in (
        'resource close failed with detail',
        'runtime stop timed out with detail',
        'identity verification unavailable with detail',
        'evidence disk full with detail',
    ):
        assert detail in notes
    if initial is not None:
        assert 'RuntimeError: docker start failed with fixture detail' in notes
    assert 'Traceback (most recent call last)' in notes
