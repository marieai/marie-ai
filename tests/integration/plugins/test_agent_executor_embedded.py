"""Exercise AgentExecutor routing through the real embedded plugin daemon."""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import zipfile
from pathlib import Path

import pytest
from docarray import DocList

from marie.api import AssetKeyDoc
from marie.executor.agent import AgentExecutor, AgentPluginResponse
from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor

REPO_ROOT = Path(__file__).resolve().parents[3]
DAEMON_DIR = REPO_ROOT / 'packages' / 'marie-plugin-daemon'
FIXTURE_DIR = DAEMON_DIR / 'testdata' / 'fixture-plugin'
PACKAGE = 'marie/fixture-agent'


def _daemon_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    return f'127.0.0.1:{port}'


def _package_fixture(destination: Path) -> None:
    with zipfile.ZipFile(destination, 'w') as archive:
        for name in ('marie-extension.yaml', 'main.py', 'requirements.txt'):
            archive.write(FIXTURE_DIR / name, arcname=name)


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.timeout(180)
async def test_gateway_payload_runs_fixture_agent_through_embedded_daemon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which('go') is None:
        pytest.skip('Go is required to build marie-plugin-daemon')
    if shutil.which('uv') is None:
        pytest.skip('uv is required to bootstrap the plugin environment')

    daemon_bin = tmp_path / 'marie-plugin-daemon'
    subprocess.run(
        ['go', 'build', '-o', str(daemon_bin), './cmd/server'],
        cwd=DAEMON_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    plugin_archive = tmp_path / 'fixture-agent.zip'
    _package_fixture(plugin_archive)

    monkeypatch.delenv('MARIE_PLUGIN_DAEMON_URL', raising=False)
    monkeypatch.setenv('MARIE_PLUGIN_DAEMON_BIN', str(daemon_bin))
    monkeypatch.setenv('MARIE_PLUGIN_STORAGE_ROOT', str(tmp_path / 'storage'))
    monkeypatch.setenv('MARIE_PLUGIN_DAEMON_LOG_LEVEL', 'ERROR')

    task_id = '00000000-0000-0000-0000-000000000101'
    dag_id = '00000000-0000-0000-0000-000000000201'
    attempt_id = '00000000-0000-0000-0000-000000000301'
    source = tmp_path / 'source.json'
    source.write_text('{}')
    request = {
        'agent_ref': 'fixture.echo',
        'input': {'finding': 'missing value'},
        'artifacts': {'schema_uri': 's3://bucket/schema.json'},
        'idempotency_key': 'effect-1',
    }
    job_info = JobInfo(
        status=JobStatus.PENDING,
        entrypoint='agent_executor://agent/run',
        metadata={
            'metadata': {
                'uri': source.as_uri(),
                'ref_id': 'fixture-source',
                'ref_type': 'agent-test',
                'op_params': request,
            },
            'dag_id': dag_id,
            'node_task_id': task_id,
            'run_attempt_id': attempt_id,
        },
    )
    distributor = GatewayJobDistributor(
        deployment_nodes={
            'agent_executor': [{'endpoint': '/agent/run'}],
        }
    )
    assert distributor._resolve_endpoint(task_id, job_info.entrypoint) == (
        'agent_executor',
        '/agent/run',
    )
    parameters, asset_doc = await distributor._build_payload(task_id, job_info)

    executor = AgentExecutor(
        enable_conversation_store=False,
        plugins=[
            {
                'package': PACKAGE,
                'path': str(plugin_archive),
                'actions': ['run'],
                'timeout_s': 30,
            }
        ],
        agent_routes={
            'fixture.echo': {
                'package': PACKAGE,
                'action': 'run',
            }
        },
        plugin_daemon_addr=_daemon_address(),
    )
    try:
        docs = await executor.agent_run_endpoint(
            DocList[AssetKeyDoc]([asset_doc]),
            parameters=parameters,
        )
    finally:
        executor.close()

    response = AgentPluginResponse.model_validate(json.loads(docs[0].text))
    echo = response.result['echo']
    assert echo['agent_ref'] == request['agent_ref']
    assert echo['input'] == request['input']
    assert echo['artifacts'] == request['artifacts']
    assert echo['idempotency_key'] == request['idempotency_key']
    assert echo['action'] == 'run'
    assert echo['execution'] == {
        'dag_id': dag_id,
        'task_id': task_id,
        'attempt': attempt_id,
        'job_id': task_id,
        'request_id': task_id,
        'trace_id': response.trace_id,
    }
    assert response.request_id == task_id
    assert [frame['type'] for frame in response.frames] == ['stream', 'end']
