import asyncio
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session


@pytest.mark.asyncio
async def test_wire_error_flag_and_lifespan_cleanup(monkeypatch, tmp_path):
    from marie.mcp.server import server
    from marie.mcp.server.clients.marie_client import MarieClientError

    loops = []

    class Client:
        api_key = 'test-key'

        async def __aenter__(self):
            loops.append(asyncio.get_running_loop())
            return self

        async def __aexit__(self, *args):
            loops.append(asyncio.get_running_loop())

        async def health_check(self):
            raise MarieClientError('gateway unavailable')

    monkeypatch.setattr(server, 'MarieClient', Client)
    monkeypatch.setattr(server.Config, 'OUTPUT_DIR', tmp_path)
    mcp = server.create_server()
    async with create_connected_server_and_client_session(mcp._mcp_server) as session:
        tools = await session.list_tools()
        assert len(tools.tools) == 9
        response = await session.call_tool('health_check', {})
        assert response.isError is True
        assert 'gateway unavailable' in response.content[0].text
    assert len(loops) == 2 and loops[0] is loops[1]


@pytest.mark.asyncio
async def test_real_http_client_closes_in_server_loop(monkeypatch, tmp_path):
    from marie.mcp.server import server
    from marie.mcp.server.clients.marie_client import MarieClient

    async def gateway(reader, writer):
        try:
            await reader.readuntil(b'\r\n\r\n')
            body = b'{"status":"OK"}'
            writer.write(
                b'HTTP/1.1 200 OK\r\nContent-Length: '
                + str(len(body)).encode()
                + b'\r\n\r\n'
                + body
            )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

    http_server = await asyncio.start_server(gateway, '127.0.0.1', 0)
    port = http_server.sockets[0].getsockname()[1]
    clients = []

    def create_client():
        client = MarieClient(base_url=f'http://127.0.0.1:{port}', api_key='test-key')
        clients.append(client)
        return client

    monkeypatch.setattr(server, 'MarieClient', create_client)
    monkeypatch.setattr(server.Config, 'OUTPUT_DIR', tmp_path)
    async with http_server:
        mcp = server.create_server()
        async with create_connected_server_and_client_session(
            mcp._mcp_server
        ) as session:
            response = await session.call_tool('health_check', {})
            assert not response.isError
        assert clients[0]._client.is_closed


@pytest.mark.asyncio
async def test_stdio_initialize_list_and_error_call(tmp_path):
    source_root = Path(__file__).parents[1] / 'src'
    # Avoid reading workstation dotenv files in the subprocess.
    runner = 'import dotenv; dotenv.load_dotenv=lambda *a,**k:False; from marie.mcp.server.server import main; main()'
    params = StdioServerParameters(
        command=sys.executable,
        args=['-c', runner],
        cwd=tmp_path,
        env={
            'PYTHONPATH': str(source_root),
            'MARIE_API_KEY': 'synthetic-test-key',
            'MARIE_OUTPUT_DIR': str(tmp_path / 'outputs'),
            'MARIE_RETRY_MIN_WAIT': '0',
            'MARIE_RETRY_MAX_WAIT': '0',
        },
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=10)
        ) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {tool.name for tool in tools.tools} == {
                'extract_document_ocr',
                'extract_document_data',
                'get_job_status',
                'list_jobs',
                'get_job_results',
                'get_deployments',
                'get_capacity',
                'get_debug_info',
                'health_check',
            }
            response = await session.call_tool(
                'extract_document_ocr',
                {'file_path': str(tmp_path / 'missing.pdf'), 'ref_id': 'missing.pdf'},
            )
            assert response.isError is True
            assert 'does not exist' in response.content[0].text


@pytest.mark.asyncio
async def test_reused_server_resolves_current_lifespan_client(monkeypatch, tmp_path):
    from marie.mcp.server import server

    clients = []

    class Client:
        api_key = 'test-key'
        closed = False

        async def __aenter__(self):
            clients.append(self)
            return self

        async def __aexit__(self, *args):
            self.closed = True

        async def health_check(self):
            assert not self.closed, 'using closed client'
            return {'status': 'OK', 'session': len(clients)}

    monkeypatch.setattr(server, 'MarieClient', Client)
    monkeypatch.setattr(server.Config, 'OUTPUT_DIR', tmp_path)
    mcp = server.create_server()
    for session_number in (1, 2):
        async with create_connected_server_and_client_session(
            mcp._mcp_server
        ) as session:
            response = await session.call_tool('health_check', {})
            assert not response.isError
            assert json.loads(response.content[0].text)['session'] == session_number
    assert len(clients) == 2 and all(client.closed for client in clients)
