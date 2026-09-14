"""UTF-8 transport limits and canonical JSON result reservations."""

import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import pytest
from test_request_dispatcher import admit_request, store


@pytest.mark.parametrize('allowance', [32 * 1024, 256 * 1024])
@pytest.mark.parametrize('character', ['x', '界', '😀'])
@pytest.mark.parametrize('over', [False, True])
def test_result_allowance_counts_canonical_encoded_bytes(
    store, allowance, character, over
):
    from marie.engine.llm_queue.store import RequestStore

    limited = RequestStore(
        store.test_url,
        fabric_id='sizes-' + uuid4().hex,
        version='v3',
        limits=replace(store.limits, result_allowance=allowance),
    )
    try:
        owner = limited.acquire_owner('sizes', lease_ms=5000)
        limited.configure_route(
            owner,
            'pool',
            'endpoint',
            revision='r1',
            execution_limit=1,
            execution_bytes=1_000_000,
        )
        req = admit_request(limited)
        limited.claim(owner, req.attempt_id, pool_id='pool', claim_id='claim')
        seq = limited.authorize_start(
            owner, req.attempt_id, claim_id='claim'
        ).execution_seq
        empty_bytes = len(json.dumps({'answer': ''}, separators=(',', ':')).encode())
        unit = len(json.dumps(character).encode()) - 2
        count = (allowance - empty_bytes) // unit + int(over)
        response = {'answer': character * count}
        canonical_size = len(
            json.dumps(response, separators=(',', ':'), sort_keys=True).encode()
        )
        assert (canonical_size > allowance) == over
        limited.finish(
            owner, req.attempt_id, claim_id='claim', execution_seq=seq, result=response
        )
        result = limited.read_result(req.producer_id, req.attempt_id)
        assert result == ({'error': 'result_too_large'} if over else response)
        assert limited.usage()['reserved_items'] == 0
        assert not limited.client.hexists(
            limited.keys.request(req.attempt_id), 'payload'
        )
    finally:
        keys = list(limited.client.scan_iter(match=limited.keys.prefix + '*'))
        if keys:
            limited.client.delete(*keys)
        limited.close()


@pytest.mark.parametrize('over', [False, True])
async def test_transport_limit_counts_utf8_bytes(over):
    from marie.engine.completion_contract import CompletionCallParams
    from marie.engine.llm_queue.endpoint import EndpointClient, RegisteredEndpoint

    cap = 1024 * 1024
    framing = len(
        json.dumps({'answer': ''}, separators=(',', ':'), ensure_ascii=False).encode()
    )
    body = json.dumps(
        {'answer': '界' * ((cap - framing) // 3 + int(over))},
        separators=(',', ':'),
        ensure_ascii=False,
    ).encode()
    assert (len(body) > cap) == over
    active = set()

    async def handle(reader, writer):
        active.add(asyncio.current_task())
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            size = int(header.decode().split('Content-Length: ')[1].split('\r\n')[0])
            await reader.readexactly(size)
            writer.write(
                b'HTTP/1.1 200 OK\r\nContent-Length: '
                + str(len(body)).encode()
                + b'\r\nConnection: close\r\n\r\n'
                + body
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            active.discard(asyncio.current_task())

    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    client = EndpointClient(
        RegisteredEndpoint(
            'size',
            f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}',
            allow_loopback=True,
        )
    )
    try:
        result = await client.execute(
            CompletionCallParams(model='size', messages=[]), timeout_seconds=5
        )
        assert result.category == ('response_too_large' if over else '')
        assert result.remote_settled == (not over)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*tuple(active), return_exceptions=True)
