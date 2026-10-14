import asyncio
import json
import threading
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError


@pytest.mark.asyncio
async def test_package_startup_imports():
    from marie.mcp.server.server import create_server

    assert create_server().name == 'marie-mcp'


@pytest.mark.asyncio
async def test_mutating_tools_are_not_advertised(mock_marie_client, monkeypatch):
    from marie.mcp.server.tools.job_management import register_job_tools

    server = FastMCP('test')
    monkeypatch.setattr(server, 'get_context', lambda: None)
    register_job_tools(server, mock_marie_client)
    names = {tool.name for tool in await server.list_tools()}
    assert names == {'get_job_status', 'list_jobs', 'get_job_results'}


@pytest.mark.asyncio
async def test_tool_errors_and_context_are_awaited(mock_marie_client, monkeypatch):
    from marie.mcp.server.tools.job_management import register_job_tools

    server = FastMCP('test')
    monkeypatch.setattr(server, 'get_context', lambda: None)
    register_job_tools(server, mock_marie_client)
    mock_marie_client.get_job_status.side_effect = RuntimeError('unavailable')
    ctx = AsyncMock()
    tool = server._tool_manager.get_tool('get_job_status')
    with pytest.raises(ToolError, match='unavailable'):
        await tool.fn(job_id='test', ctx=ctx)
    ctx.info.assert_awaited_once()
    ctx.error.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['extract_document_ocr', 'extract_document_data'])
async def test_upload_does_not_block_and_preserves_project_id(
    mock_marie_client, monkeypatch, tmp_path, name
):
    from marie.mcp.server.tools import document_processing as tools

    entered = threading.Event()
    release = threading.Event()
    loop_thread = threading.get_ident()

    def upload(*args):
        assert threading.get_ident() != loop_thread
        entered.set()
        assert release.wait(3)
        return True

    monkeypatch.setattr(tools, 'upload_to_s3', upload)
    path = tmp_path / 'Invoice.PDF'
    path.write_bytes(b'%PDF')
    server = FastMCP('test')
    monkeypatch.setattr(server, 'get_context', lambda: None)
    tools.register_document_tools(server, mock_marie_client)
    args = {'file_path': str(path), 'ref_id': 'Invoice.PDF', 'ref_type': 'Invoice'}
    if name == 'extract_document_data':
        args['template_id'] = '117183'
    task = asyncio.create_task(server.call_tool(name, args))
    try:
        for _ in range(100):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
    finally:
        release.set()
    await task
    metadata = mock_marie_client.submit_job.call_args.kwargs['metadata']
    assert metadata['project_id'] == mock_marie_client.api_key
    assert metadata['uri'].endswith('/invoice/invoice/Invoice.PDF')


@pytest.mark.asyncio
async def test_configured_upload_limit(mock_marie_client, monkeypatch, tmp_path):
    from marie.mcp.server.config import Config
    from marie.mcp.server.tools.document_processing import register_document_tools

    monkeypatch.setattr(Config, 'MAX_FILE_SIZE_MB', 1)
    path = tmp_path / 'large.pdf'
    with path.open('wb') as f:
        f.truncate(1024 * 1024 + 1)
    server = FastMCP('test')
    monkeypatch.setattr(server, 'get_context', lambda: None)
    register_document_tools(server, mock_marie_client)
    with pytest.raises(ToolError, match='exceeds'):
        await server.call_tool(
            'extract_document_ocr', {'file_path': str(path), 'ref_id': 'large.pdf'}
        )
    mock_marie_client.submit_job.assert_not_called()


@pytest.mark.asyncio
async def test_result_download_uses_canonical_path_and_output_root(
    mock_marie_client, monkeypatch, tmp_path
):
    from marie.mcp.server.config import Config
    from marie.mcp.server.tools import job_management as tools

    monkeypatch.setattr(Config, 'OUTPUT_DIR', tmp_path)
    seen = []
    loop_thread = threading.get_ident()

    def exists(path):
        assert threading.get_ident() != loop_thread
        seen.append(path)
        return True

    monkeypatch.setattr(tools, 's3_exists', exists)

    def download(uri, destination):
        assert threading.get_ident() != loop_thread
        Path(destination).write_text('{}')
        return True

    monkeypatch.setattr(tools, 'download_from_s3', download)
    server = FastMCP('test')
    monkeypatch.setattr(server, 'get_context', lambda: None)
    tools.register_job_tools(server, mock_marie_client)
    result = await server._tool_manager.get_tool('get_job_results').run(
        {
            'ref_id': 'Invoice.PDF',
            'ref_type': 'Invoice',
            'output_path': 'results/invoice.json',
        }
    )
    assert seen[0].endswith('/invoice/invoice/Invoice.PDF.meta.json')
    assert json.loads(result)['output_path'] == str(tmp_path / 'results/invoice.json')
    assert (tmp_path / 'results/invoice.json').read_text() == '{}'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'destination', ['../escape.json', '/tmp/mcp-escape.json', 'link/escape.json']
)
async def test_output_escape_rejected_before_s3(
    mock_marie_client, monkeypatch, tmp_path, destination
):
    from marie.mcp.server.config import Config
    from marie.mcp.server.tools import job_management as tools

    root = tmp_path / 'outputs'
    root.mkdir()
    (root / 'link').symlink_to(tmp_path, target_is_directory=True)
    monkeypatch.setattr(Config, 'OUTPUT_DIR', root)
    monkeypatch.setattr(
        tools, 's3_exists', lambda _: pytest.fail('S3 called before validation')
    )
    server = FastMCP('test')
    monkeypatch.setattr(server, 'get_context', lambda: None)
    tools.register_job_tools(server, mock_marie_client)
    with pytest.raises(ToolError, match='output'):
        await server.call_tool(
            'get_job_results',
            {
                'ref_id': 'Invoice.PDF',
                'ref_type': 'Invoice',
                'output_path': destination,
            },
        )
