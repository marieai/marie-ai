import asyncio
from unittest.mock import AsyncMock

import pytest
from marie.engine.llm_queue import endpoint


@pytest.mark.asyncio
async def test_tcp_probe_is_point_in_time_and_sends_no_request():
    received = []

    async def accept(reader, writer):
        received.append(await reader.read())
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(accept, '127.0.0.1', 0)
    port = server.sockets[0].getsockname()[1]
    target = endpoint.RegisteredEndpoint(
        'probe', f'http://127.0.0.1:{port}/v1', allow_loopback=True
    )
    try:
        result = await endpoint.probe_endpoint_connection(target)
        assert result['reachable'] is True
        assert result['category'] == 'tcp_connected'
        assert result['latency_ms'] >= 0
        for _ in range(20):
            if received:
                break
            await asyncio.sleep(0.01)
        assert received == [b'']
    finally:
        server.close()
        await server.wait_closed()
    assert (await endpoint.probe_endpoint_connection(target))['reachable'] is False


@pytest.mark.asyncio
async def test_probe_respects_resolved_address_policy_and_does_not_connect(monkeypatch):
    target = endpoint.RegisteredEndpoint('probe', 'https://inference.example/v1')
    backend_connect = AsyncMock()
    monkeypatch.setattr(endpoint.AnyIOBackend, 'connect_tcp', backend_connect)
    monkeypatch.setattr(
        asyncio.get_running_loop(),
        'getaddrinfo',
        AsyncMock(return_value=[(2, 1, 6, '', ('127.0.0.1', 443))]),
    )
    result = await endpoint.probe_endpoint_connection(target)
    assert result['reachable'] is False
    assert result['category'] == 'address_not_allowed'
    backend_connect.assert_not_called()


@pytest.mark.asyncio
async def test_probe_timeout_is_bounded_and_contains_no_transport_error(monkeypatch):
    target = endpoint.RegisteredEndpoint('probe', 'https://inference.example/v1')
    monkeypatch.setattr(
        endpoint._PinnedNetworkBackend,
        'connect_tcp',
        AsyncMock(side_effect=TimeoutError('private-url-and-key')),
    )
    result = await endpoint.probe_endpoint_connection(target)
    assert result['category'] == 'timeout'
    assert 'private' not in str(result)
