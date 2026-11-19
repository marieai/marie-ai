import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples import document_extract
from tests.unit.serve.runtimes.gateway.test_marie_gateway_submission import (
    build_gateway,
)


@pytest.fixture
def invoke_server_factory():
    running = []

    def start():
        state = {
            'requests': [],
            'status': 200,
            'headers': {},
            'response': {'parameters': {'status': 'ok', 'job_id': 'job-1'}},
        }

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                payload = json.loads(
                    self.rfile.read(int(self.headers['Content-Length']))
                )
                state['requests'].append((self.path, dict(self.headers), payload))
                response = json.dumps(state['response']).encode()
                self.send_response(state['status'])
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(response)))
                for key, value in state['headers'].items():
                    self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, *_args) -> None:
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        running.append((server, thread))
        state['url'] = f'http://127.0.0.1:{server.server_port}'
        return state

    try:
        yield start
    finally:
        for server, thread in running:
            server.shutdown()
            server.server_close()
            thread.join()


@pytest.fixture
def invoke_server(invoke_server_factory):
    return invoke_server_factory()


def submit(server):
    return document_extract.process_extract(
        uri='s3://input-bucket/documents/scan%201.tif',
        api_base_url=server['url'],
        api_key='test-auth',
        project_id='project-1',
        pipeline='default',
    )


@pytest.mark.asyncio
async def test_example_submits_a_valid_gateway_extraction_job(
    invoke_server, monkeypatch
):
    assert submit(invoke_server)['job_id'] == 'job-1'
    path, headers, payload = invoke_server['requests'][0]
    assert path == '/api/v1/invoke'
    assert headers['Authorization'] == 'Bearer test-auth'
    command = payload['parameters']['invoke_action']
    assert command['api_key'] == 'test-auth'
    assert (command['command'], command['action'], command['name']) == (
        'job',
        'submit',
        'extract',
    )
    metadata = command['metadata']
    assert metadata['project_id'] == 'project-1'
    assert metadata['uri'] == 's3://input-bucket/documents/scan%201.tif'
    assert metadata['ref_id'] == 'scan%201.tif'
    assert metadata['ref_type'] == 'extract'
    assert metadata['planner'] == 'extract'
    assert metadata['run_params']['parse_mode'] == 'auto'
    assert metadata['features'] == [{'type': 'pipeline', 'name': 'default'}]

    from marie.serve.runtimes.servers import marie_gateway

    monkeypatch.setattr(
        marie_gateway.APIKeyManager, 'is_valid', lambda key: key == 'test-auth'
    )
    gateway, submitted = build_gateway()
    responses = await gateway.decode_request(
        SimpleNamespace(parameters=payload['parameters'])
    )
    response = await anext(responses)
    assert response.parameters['status'] == 'ok'
    assert len(submitted) == 1
    assert submitted[0].data['metadata']['uri'] == metadata['uri']
    assert submitted[0].data['metadata']['run_params'] == {'parse_mode': 'auto'}


@pytest.mark.parametrize(
    ('parse_mode', 'expected_route'),
    [(None, 'plugin'), ('auto', 'plugin'), ('ocr', 'ocr')],
)
def test_pdf_submission_respects_format_routing(
    invoke_server: dict, parse_mode: str | None, expected_route: str
) -> None:
    from marie.extraction import FormatRouter

    kwargs = {} if parse_mode is None else {'parse_mode': parse_mode}
    document_extract.process_extract(
        uri='s3://input-bucket/scan.pdf',
        api_base_url=invoke_server['url'],
        api_key='test-auth',
        project_id='project-1',
        **kwargs,
    )
    metadata = invoke_server['requests'][0][2]['parameters']['invoke_action'][
        'metadata'
    ]
    router = FormatRouter()
    router.ingest_capabilities(
        {
            'schema_version': '1.0',
            'plugin_version': '0.2.0',
            'ready': True,
            'formats': [
                {
                    'canonical_format': 'pdf',
                    'aliases': [],
                    'extensions': ['pdf'],
                    'mime_types': ['application/pdf'],
                    'intents': ['semantic'],
                    'result_kinds': ['semantic_document'],
                    'providers': ['test-provider'],
                }
            ],
        }
    )
    assert (
        router.route('pdf', metadata['run_params']['parse_mode'], ocr_supported=True)
        == expected_route
    )


@pytest.mark.parametrize(
    'parameters',
    [
        {'status': 'error', 'msg': 'Queue unavailable'},
        {'status': 'ok'},
    ],
)
def test_http_success_without_an_accepted_job_is_an_error(invoke_server, parameters):
    invoke_server['response'] = {'parameters': parameters}
    with pytest.raises(RuntimeError):
        submit(invoke_server)


def test_submission_does_not_forward_credentials_to_a_redirect(invoke_server_factory):
    origin = invoke_server_factory()
    destination = invoke_server_factory()
    origin['status'] = 307
    origin['headers']['Location'] = f'{destination["url"]}/api/v1/invoke'

    with pytest.raises(RuntimeError, match='redirect'):
        submit(origin)

    assert len(origin['requests']) == 1
    assert destination['requests'] == []


@pytest.mark.parametrize(
    ('uri', 'filename'),
    [
        ('s3://input-bucket/documents/scan#1.tif', 'scan#1.tif'),
        ('s3://input-bucket/documents/scan?1.tif', 'scan?1.tif'),
        ('s3://input-bucket/documents/scan%201.tif', 'scan%201.tif'),
        ('https://files.example/scan%201.tif?download=1#page', 'scan 1.tif'),
    ],
)
def test_document_reference_preserves_storage_filename(invoke_server, uri, filename):
    document_extract.process_extract(
        uri, invoke_server['url'], 'test-auth', 'project-1'
    )
    metadata = invoke_server['requests'][0][2]['parameters']['invoke_action'][
        'metadata'
    ]
    assert metadata['uri'] == uri
    assert metadata['ref_id'] == filename


def test_shell_example_runs_from_another_directory(invoke_server, tmp_path):
    # Keep the obsolete curl script from contacting a real gateway in the RED run.
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    fake_curl = fake_bin / 'curl'
    fake_curl.write_text('#!/bin/sh\nexit 99\n')
    fake_curl.chmod(0o755)
    script = Path(__file__).resolve().parents[3] / 'examples/srcFile-001.sh'
    env = {
        **os.environ,
        'PATH': f'{fake_bin}:{os.environ.get("PATH", "")}',
        'MARIE_PYTHON': sys.executable,
        'MARIE_API_BASE_URL': invoke_server['url'],
        'MARIE_API_KEY': 'test-auth',
        'MARIE_PROJECT_ID': 'project-1',
    }
    result = subprocess.run(
        ['bash', str(script), '--uri', 's3://input-bucket/scan.tif'],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['job_id'] == 'job-1'
    assert invoke_server['requests'][0][0] == '/api/v1/invoke'
    assert 'test-auth' not in result.stdout


@pytest.mark.parametrize('parse_mode', ['auto', 'ocr'])
def test_default_config_uses_the_existing_batch_project_identity(
    invoke_server, monkeypatch, tmp_path, capsys, parse_mode
):
    for key in ('MARIE_API_BASE_URL', 'MARIE_API_KEY', 'MARIE_PROJECT_ID'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        document_extract, '__file__', str(tmp_path / 'document_extract.py')
    )
    (tmp_path / 'config.dev.json').write_text(
        json.dumps(
            {
                'api_base_url': invoke_server['url'],
                'api_key': 'config-auth',
                'default_queue_id': 'config-queue',
                'storage': {},
                'queue': {},
            }
        )
    )

    assert (
        document_extract.main(
            ['--uri', 's3://input-bucket/scan.tif', '--parse-mode', parse_mode]
        )
        == 0
    )

    command = invoke_server['requests'][0][2]['parameters']['invoke_action']
    assert command['api_key'] == 'config-auth'
    assert command['metadata']['project_id'] == 'config-auth'
    assert command['metadata']['queue_id'] == 'config-queue'
    assert command['metadata']['run_params']['parse_mode'] == parse_mode
    assert json.loads(capsys.readouterr().out)['job_id'] == 'job-1'


def test_missing_local_input_reports_a_file_error_from_another_directory(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'api_key': 'test-auth', 'project_id': 'project-1'}))
    script = Path(__file__).resolve().parents[3] / 'examples/document_extract.py'
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            '--config',
            str(config),
            '--input',
            str(tmp_path / 'missing.tif'),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert 'Input file not found' in result.stderr


def test_failed_local_upload_is_not_reported_as_a_storage_uri(monkeypatch, tmp_path):
    from marie import storage
    from marie.storage import StorageManager

    source = tmp_path / 'scan.tif'
    source.write_bytes(b'test input')
    monkeypatch.setenv('MARIE_S3_BUCKET', 'before-test')
    monkeypatch.setattr(storage, 'S3StorageHandler', lambda **_kwargs: object())
    monkeypatch.setattr(StorageManager, 'register_handler', lambda *_args: None)
    monkeypatch.setattr(StorageManager, 'ensure_connection', lambda *_args: True)
    monkeypatch.setattr(StorageManager, 'write', lambda *_args, **_kwargs: False)
    with pytest.raises(RuntimeError, match='upload'):
        document_extract.upload_input(
            source, {'S3_STORAGE_BUCKET_NAME': 'input-bucket'}
        )


def test_bucket_scoped_upload_does_not_require_listing_all_buckets(
    monkeypatch, tmp_path
):
    from marie import storage
    from marie.storage import StorageManager

    source = tmp_path / 'scan.tif'
    source.write_bytes(b'test input')
    monkeypatch.setenv('MARIE_S3_BUCKET', 'before-test')
    monkeypatch.setenv('S3_STORAGE_BUCKET_NAME', 'input-bucket')
    monkeypatch.setattr(storage, 'S3StorageHandler', lambda **_kwargs: object())
    monkeypatch.setattr(StorageManager, 'register_handler', lambda *_args: None)

    def denied_bucket_listing(*_args):
        raise PermissionError('ListAllMyBuckets is not permitted')

    monkeypatch.setattr(StorageManager, 'ensure_connection', denied_bucket_listing)
    monkeypatch.setattr(StorageManager, 'write', lambda *_args, **_kwargs: True)

    assert (
        document_extract.upload_input(source, {})
        == 's3://input-bucket/extract/scan/scan.tif'
    )
