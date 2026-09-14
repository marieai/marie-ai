"""Process death qualification with real producer/store/dispatch/HTTP boundaries."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

from test_request_dispatcher import eventually, http_endpoint, store
from test_v3_producer import calls, processor_for
from v3_qualification_evidence import record_evidence


def events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


async def start_worker(store, path, count):
    config = {
        'url': store.test_url,
        'fabric': store.keys.fabric_id,
        'events': str(path),
        'count': count,
    }
    return await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).with_name('v3_worker_fixture.py')),
        json.dumps(config),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


async def start_dispatcher(
    store, path, endpoint, *, barrier='before_fetch', trace=False
):
    from dataclasses import asdict

    config = {
        'mode': 'dispatcher',
        'url': store.test_url,
        'fabric': store.keys.fabric_id,
        'events': str(path),
        'endpoint': endpoint,
        'limits': asdict(store.limits),
        'barrier': barrier,
        'trace': trace,
    }
    return await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).with_name('v3_worker_fixture.py')),
        json.dumps(config),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def kill_and_reap(child):
    child.kill()
    stdout, stderr = await child.communicate()
    assert child.returncode == -9
    return stdout.decode(), stderr.decode()


async def test_killed_worker_loses_every_state_and_replacement_inherits_nothing(
    store, monkeypatch, tmp_path
):
    from marie.engine.llm_queue.endpoint import RegisteredEndpoint
    from marie.engine.llm_queue.request_dispatcher import (
        DispatchLane,
        RequestDispatcher,
    )
    from marie.engine.llm_queue.store import StoreUnavailable

    active = set()
    slow = asyncio.Event()
    sent = []

    async def handle(reader, writer):
        active.add(asyncio.current_task())
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            headers = dict(
                line.split(': ', 1)
                for line in header.decode().split('\r\n')
                if ': ' in line
            )
            body = json.loads(await reader.readexactly(int(headers['Content-Length'])))
            sent.append(body['model'])
            if body['model'] == 'item-1':
                server.close()
                await slow.wait()
            payload = json.dumps(
                {'choices': [{'message': {'content': body['model']}}]}
            ).encode()
            writer.write(
                b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '
                + str(len(payload)).encode()
                + b'\r\n\r\n'
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    runtime = RequestDispatcher(
        store=store,
        endpoints=[
            RegisteredEndpoint(
                'endpoint',
                f'http://127.0.0.1:{port}/v1',
                allow_loopback=True,
                execution_limit=2,
            )
        ],
        lanes=[DispatchLane('pool', 'endpoint', execution_limit=2)],
        poll_seconds=0.01,
        retry_min_ms=3000,
        retry_max_ms=3000,
        circuit_open_ms=500,
    )
    original_claim = runtime.store.claim
    claimed = []
    hold_claims = True

    def claim(*args, **kwargs):
        if hold_claims and len(claimed) >= 3:
            raise StoreUnavailable('owned fixture pauses later claims')
        reply = original_claim(*args, **kwargs)
        if reply.disposition == 'claimed':
            claimed.append(args[1])
        return reply

    monkeypatch.setattr(runtime.store, 'claim', claim)
    path = tmp_path / 'worker.jsonl'
    child = replacement = None
    live_task = None
    live = processor_for(store, batch_timeout=15)
    await runtime.start()
    try:
        await eventually(lambda: store.resolve_route('pool'))
        child = await start_worker(store, path, 4)
        await eventually(
            lambda: any(e['event'] == 'session' for e in events(path)), seconds=15
        )
        session = next(e['producer'] for e in events(path) if e['event'] == 'session')

        def states():
            ids = store.client.smembers(store.keys.members(session))
            return {
                store.metadata(i.decode() if isinstance(i, bytes) else i).state
                for i in ids
            }

        await eventually(
            lambda: states() == {'succeeded', 'executing', 'delayed', 'ready'},
            seconds=10,
        )
        before_death = {'states': sorted(states()), 'usage': store.usage()}
        old_ids = [
            i.decode() if isinstance(i, bytes) else i
            for i in store.client.smembers(store.keys.members(session))
        ]
        assert len([e for e in events(path) if e['event'] == 'output']) == 1
        live_task = asyncio.create_task(
            asyncio.to_thread(
                live.batch_generate_calls, calls=calls(1), request_id='other-node'
            )
        )
        await eventually(
            lambda: (
                live._queued_executor is not None
                and live._queued_executor.producer_id is not None
            )
        )
        live_session = live._queued_executor.producer_id
        assert live_session != session and not live_task.done()
        killed_at = time.monotonic()
        child.kill()
        await child.wait()
        assert child.returncode == -9
        assert os.getpid() != child.pid
        await eventually(
            lambda: not store.client.exists(store.keys.alive(session)), seconds=3
        )
        await eventually(
            lambda: all(
                not store.client.hexists(store.keys.request(ident), 'payload')
                and not store.client.hexists(store.keys.request(ident), 'result')
                for ident in old_ids
            ),
            seconds=4,
        )
        assert store.usage()['active_items'] == 1
        assert store.client.exists(store.keys.alive(live_session))
        assert not live_task.done()
        for ident in old_ids:
            record = store.client.hgetall(store.keys.request(ident))
            assert b'payload' not in record and b'result' not in record
        cleanup_seconds = time.monotonic() - killed_at
        assert cleanup_seconds < 5
        assert store.usage()['reserved_items'] == 1
        # A different live producer uses the remaining endpoint slot.
        server = await asyncio.start_server(handle, '127.0.0.1', port)
        hold_claims = False
        assert await live_task == ['item-0']
        new_path = tmp_path / 'replacement.jsonl'
        replacement = await start_worker(store, new_path, 1)
        await asyncio.wait_for(replacement.wait(), timeout=15)
        new_events = events(new_path)
        new_session = next(e['producer'] for e in new_events if e['event'] == 'session')
        assert new_session != session
        assert [e['task'] for e in new_events if e['event'] == 'output'] == [
            'same-node_task_0'
        ]
        assert next(e['results'] for e in new_events if e['event'] == 'complete') == [
            'item-0'
        ]
        slow.set()
        await eventually(lambda: not active)
        for ident in old_ids:
            record = store.client.hgetall(store.keys.request(ident))
            assert b'payload' not in record and b'result' not in record
        assert sent.count('item-1') == 1
        record_evidence(
            'worker-death',
            store,
            {
                'old_session': session,
                'other_live_session_at_death': live_session,
                'dead_session_payload_result_fields_after_cleanup': 0,
                'new_session': new_session,
                'old_attempts': old_ids,
                'child_returncode': child.returncode,
                'parent_pid': os.getpid(),
                'child_pid': child.pid,
                'before_death': before_death,
                'cleanup_seconds': cleanup_seconds,
                'old_events': events(path),
                'new_events': new_events,
                'sent': sent,
                'final_usage': store.usage(),
                'fixture_timing': {
                    'retry_ms': 3000,
                    'circuit_open_ms': 500,
                    'live_batch_deadline_seconds': 15,
                },
                'fault': 'actual store claim I/O temporarily unavailable after three successful claims',
                'boundary': 'BatchProcessor worker subprocess SIGKILL, no Flow runtime wrapper in this fixture',
            },
        )
    finally:
        slow.set()
        for proc in (child, replacement):
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
        await asyncio.to_thread(live.close)
        if live_task is not None:
            await asyncio.gather(live_task, return_exceptions=True)
        await runtime.stop()
        server.close()
        await server.wait_closed()
        for task in list(active):
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


async def test_killed_dispatcher_recovers_only_unsent_claim_for_live_producer(
    store, http_endpoint, tmp_path
):
    from test_request_dispatcher import dispatcher_for

    url, received, _, _ = http_endpoint
    path = tmp_path / 'dispatcher.jsonl'
    child = await start_dispatcher(store, path, url)
    producer = processor_for(store, batch_timeout=15)
    runtime = dispatcher_for(store, url, owner_lease_ms=500)
    try:
        await eventually(lambda: store.resolve_route('pool'), seconds=15)
        task = asyncio.create_task(
            asyncio.to_thread(producer.batch_generate_calls, calls=calls(1))
        )
        await eventually(lambda: bool(events(path)), seconds=10)
        ident = events(path)[0]['attempt']
        assert store.metadata(ident).state == 'claimed'
        assert store.metadata(ident).execution_seq == 0
        session = producer._queued_executor.producer_id
        child.kill()
        await child.wait()
        assert child.returncode == -9
        assert store.client.exists(store.keys.alive(session))
        await runtime.start()
        assert await task == ['item-0']
        assert len(received) == 1
        assert store.metadata(ident).execution_seq == 1
        assert runtime.owner.generation > 1
        record_evidence(
            'dispatcher-death',
            store,
            {
                'attempt': ident,
                'producer': session,
                'child_returncode': child.returncode,
                'recovered_owner_generation': runtime.owner.generation,
                'execution_seq': store.metadata(ident).execution_seq,
                'provider_count': len(received),
                'boundary': 'actual dispatcher subprocess killed after claim and before fetch/start',
            },
        )
    finally:
        if child.returncode is None:
            await kill_and_reap(child)
        await asyncio.to_thread(producer.close)
        await runtime.stop()


async def test_killed_dispatcher_after_response_keeps_unknown_reservation(
    store, http_endpoint, tmp_path
):
    from dataclasses import replace
    from uuid import uuid4

    from marie.engine.llm_queue.store import RequestStore
    from test_request_dispatcher import dispatcher_for

    store_url = store.test_url
    store = RequestStore(
        store_url,
        fabric_id='uncertainty-' + uuid4().hex,
        version='v3',
        limits=replace(store.limits, remote_uncertainty_ms=2000),
    )
    store.test_url = store_url
    url, received, _, _ = http_endpoint
    path = tmp_path / 'before-finish.jsonl'
    child = await start_dispatcher(store, path, url, barrier='before_finish')
    producer = processor_for(store, batch_timeout=15)
    replacement = dispatcher_for(store, url, owner_lease_ms=500)
    delivered = []
    task = None
    try:
        await eventually(lambda: store.resolve_route('pool'), seconds=15)
        task = asyncio.create_task(
            asyncio.to_thread(
                producer.batch_generate_calls,
                calls=calls(1),
                request_id='post-response',
                on_result=lambda *args: delivered.append(args),
            )
        )
        await eventually(
            lambda: any(e['event'] == 'before_finish' for e in events(path)),
            seconds=15,
        )
        barrier = next(e for e in events(path) if e['event'] == 'before_finish')
        before = store.metadata(barrier['attempt'])
        assert len(received) == 1
        assert before.state == 'executing'
        assert before.claim_id == barrier['claim_id']
        assert before.execution_seq == barrier['execution_seq'] == 1
        assert before.expires_at_ms == barrier['expires_at_ms']
        assert store.read_result(before.producer_id, before.attempt_id) is None
        assert store.client.exists(store.keys.alive(before.producer_id))
        assert not task.done() and delivered == []

        stdout, stderr = await kill_and_reap(child)
        recoveries = []
        original_recover = replacement.store.recover_claim

        def observe_recovery(owner, attempt):
            reply = original_recover(owner, attempt)
            if attempt == before.attempt_id:
                observation = {
                    'server_time_ms': replacement.store.server_time_ms(),
                    'disposition': reply.disposition,
                }
                if not recoveries or (
                    observation['server_time_ms'] >= before.uncertainty_until
                    and recoveries[-1]['server_time_ms'] < before.uncertainty_until
                ):
                    recoveries.append(observation)
            return reply

        replacement.store.recover_claim = observe_recovery
        await replacement.start()
        await eventually(lambda: replacement._owner_ready, seconds=15)
        recovered = store.metadata(before.attempt_id)
        assert recovered.state == 'outcome_unknown'
        assert recovered.attempt_id == before.attempt_id
        assert recovered.claim_id == before.claim_id
        assert recovered.execution_seq == before.execution_seq
        assert recovered.expires_at_ms == before.expires_at_ms
        assert store.read_result(before.producer_id, before.attempt_id) is None
        assert store.client.exists(store.keys.alive(before.producer_id))
        assert not task.done() and delivered == []
        assert len(received) == 1
        assert store.usage()['reserved_items'] == 1
        assert store.route_status('pool')['reserved_items'] == 1
        assert int(store.endpoint_status('endpoint')['reserved_items']) == 1
        assert recoveries[0]['server_time_ms'] < recovered.uncertainty_until
        await eventually(
            lambda: store.server_time_ms() >= recovered.uncertainty_until,
            seconds=4,
        )
        await eventually(
            lambda: any(
                observation['server_time_ms'] >= recovered.uncertainty_until
                for observation in recoveries[1:]
            ),
            seconds=4,
        )
        post_uncertainty = next(
            observation
            for observation in recoveries[1:]
            if observation['server_time_ms'] >= recovered.uncertainty_until
        )
        still_unknown = store.metadata(before.attempt_id)
        assert post_uncertainty['disposition'] == 'outcome_unknown'
        assert post_uncertainty['server_time_ms'] < before.expires_at_ms
        assert still_unknown.state == 'outcome_unknown'
        assert still_unknown.attempt_id == before.attempt_id
        assert still_unknown.claim_id == before.claim_id
        assert still_unknown.execution_seq == before.execution_seq
        assert still_unknown.expires_at_ms == before.expires_at_ms
        assert store.read_result(before.producer_id, before.attempt_id) is None
        assert store.client.exists(store.keys.alive(before.producer_id))
        assert not task.done() and delivered == []
        assert len(received) == 1
        assert store.usage()['reserved_items'] == 1
        assert store.route_status('pool')['reserved_items'] == 1
        assert int(store.endpoint_status('endpoint')['reserved_items']) == 1

        settled = store.settle_remote(
            replacement.owner,
            before.attempt_id,
            claim_id=before.claim_id,
            execution_seq=before.execution_seq,
            evidence='remote_completed',
        )
        assert settled.disposition == 'settled'
        await eventually(lambda: store.usage()['reserved_items'] == 0)
        assert store.route_status('pool')['reserved_items'] == 0
        assert int(store.endpoint_status('endpoint')['reserved_items']) == 0
        assert store.read_result(before.producer_id, before.attempt_id) is None
        assert len(received) == 1 and delivered == []
        print(
            json.dumps(
                {
                    'scenario': 'post-response-pre-finish-kill',
                    'barrier': barrier,
                    'recovered_state': recovered.state,
                    'uncertainty_until': recovered.uncertainty_until,
                    'recovery_observations': recoveries,
                    'post_uncertainty_state': still_unknown.state,
                    'settle_disposition': settled.disposition,
                    'provider_count': len(received),
                    'child_stdout': stdout,
                    'child_stderr': stderr,
                },
                sort_keys=True,
            )
        )
    finally:
        if child.returncode is None:
            await kill_and_reap(child)
        await replacement.stop()
        await asyncio.to_thread(producer.close)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        keys = list(store.client.scan_iter(match=store.keys.prefix + '*'))
        if keys:
            store.client.delete(*keys)
        store.close()


async def test_killed_dispatcher_after_durable_finish_delivers_once(
    store, http_endpoint, tmp_path
):
    from test_request_dispatcher import dispatcher_for

    url, received, _, _ = http_endpoint
    path = tmp_path / 'after-finish.jsonl'
    child = await start_dispatcher(store, path, url, barrier='after_finish')
    producer = processor_for(store, batch_timeout=15)
    replacement = dispatcher_for(store, url, owner_lease_ms=500)
    delivered = []
    task = None
    try:
        await eventually(lambda: store.resolve_route('pool'), seconds=15)
        task = asyncio.create_task(
            asyncio.to_thread(
                producer.batch_generate_calls,
                calls=calls(1),
                request_id='lost-finish-ack',
                on_result=lambda *args: delivered.append(args),
            )
        )
        await eventually(
            lambda: any(e['event'] == 'after_finish' for e in events(path)),
            seconds=15,
        )
        barrier = next(e for e in events(path) if e['event'] == 'after_finish')
        committed = store.metadata(barrier['attempt'])
        assert barrier['reply']['disposition'] == 'finished'
        assert committed.state == 'succeeded'
        assert committed.claim_id == barrier['claim_id']
        assert committed.execution_seq == barrier['execution_seq'] == 1
        assert committed.expires_at_ms == barrier['expires_at_ms']
        assert len(received) == 1

        stdout, stderr = await kill_and_reap(child)
        await replacement.start()
        await eventually(lambda: replacement._owner_ready, seconds=15)
        first_read = store.read_result(committed.producer_id, committed.attempt_id)
        second_read = store.read_result(committed.producer_id, committed.attempt_id)
        assert first_read == second_read
        assert first_read['choices'][0]['message']['content'] == 'item-0'
        assert await task == ['item-0']
        assert delivered == [('lost-finish-ack_task_0', 'item-0')]
        assert len(received) == 1
        assert store.metadata(committed.attempt_id).state == 'succeeded'
        assert store.usage()['reserved_items'] == 0
        assert store.route_status('pool')['reserved_items'] == 0
        assert int(store.endpoint_status('endpoint')['reserved_items']) == 0
        print(
            json.dumps(
                {
                    'scenario': 'durable-finish-lost-ack-kill',
                    'barrier': barrier,
                    'rereads_equal': first_read == second_read,
                    'deliveries': delivered,
                    'provider_count': len(received),
                    'child_stdout': stdout,
                    'child_stderr': stderr,
                },
                sort_keys=True,
            )
        )
    finally:
        if child.returncode is None:
            await kill_and_reap(child)
        await replacement.stop()
        await asyncio.to_thread(producer.close)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
