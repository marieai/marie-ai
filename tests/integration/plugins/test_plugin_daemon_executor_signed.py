"""Verify executor requests against the daemon's real signature verifier."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import secrets
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml
from docarray import DocList
from docarray.documents import TextDoc

from marie.executor.extensions.plugin_daemon_executor import MariePluginDaemonExecutor
from marie.job.common import JobInfo, JobStatus
from marie.job.gateway_job_distributor import GatewayJobDistributor
from marie.plugin_daemon import build_invocation_envelope, sign_envelope
from marie.plugins.agent_tool import PluginToolSpec
from marie.plugins.embedded import inspect_archive

DAEMON_DIR = Path(__file__).resolve().parents[3] / 'packages' / 'marie-plugin-daemon'


@pytest.fixture(scope='module')
def daemon_binary(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if shutil.which('go') is None or shutil.which('uv') is None:
        pytest.skip('Go and uv are required for the real daemon test')
    binary = tmp_path_factory.mktemp('signed-plugin-daemon') / 'marie-plugin-daemon'
    subprocess.run(
        ['go', 'build', '-o', str(binary), './cmd/server'],
        cwd=DAEMON_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize('mode', ['sidecar', 'child', 'helm'])
@pytest.mark.timeout(60)
def test_scheduled_executor_invokes_secured_daemon(
    mode: str,
    daemon_binary: Path,
    tmp_path: Path,
    unused_tcp_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addr = f'127.0.0.1:{unused_tcp_port}'
    url = f'http://{addr}'
    key_id = 'executor-integration-test'
    secret = secrets.token_urlsafe(32)
    env = {
        **os.environ,
        'MARIE_PLUGIN_DAEMON_URL': url,
        'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID': key_id,
        'MARIE_PLUGIN_DAEMON_SIGNING_SECRET': secret,
        'MARIE_PLUGIN_DAEMON_DEV_INSECURE': 'false',
        'MARIE_PLUGIN_STORAGE_ROOT': str(tmp_path / 'storage'),
        'MARIE_PLUGIN_DAEMON_LOG_LEVEL': 'ERROR',
    }
    daemon = None
    executor = None
    try:
        if mode in ('sidecar', 'helm'):
            daemon = subprocess.Popen([str(daemon_binary), '--addr', addr], env=env)
            if mode == 'helm':
                chart = DAEMON_DIR.parents[1] / 'deploy/helm/charts/marie'
                values = tmp_path / 'executor.yaml'
                values.write_text(
                    yaml.safe_dump(
                        {
                            'executor': {
                                'pools': [
                                    {
                                        'name': 'plugins',
                                        'enabled': True,
                                        'replicas': 1,
                                        'configFile': '/etc/marie/plugin-executor.yml',
                                        'pluginRuntime': {
                                            'serviceName': 'scheduled-plugins',
                                            'signingSecret': 'execution-signing',
                                        },
                                        'resources': {},
                                    }
                                ]
                            }
                        }
                    )
                )
                output = subprocess.run(
                    [
                        'helm',
                        'template',
                        'proof',
                        str(chart),
                        '--namespace',
                        'child-proof',
                        '-f',
                        str(values),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                )
                worker = next(
                    doc
                    for doc in yaml.safe_load_all(output.stdout)
                    if doc
                    and doc['kind'] == 'Deployment'
                    and doc['metadata'].get('labels', {}).get('marie.ai/pool')
                    == 'plugins'
                )
                fixture_secret = {'key-id': key_id, 'secret': secret}
                client_env = {}
                for entry in worker['spec']['template']['spec']['containers'][0]['env']:
                    if not entry['name'].startswith('MARIE_PLUGIN_DAEMON_'):
                        continue
                    if 'value' in entry:
                        client_env[entry['name']] = entry['value']
                    else:
                        ref = entry['valueFrom']['secretKeyRef']
                        assert ref['name'] == 'execution-signing'
                        client_env[entry['name']] = fixture_secret[ref['key']]
                assert (
                    client_env['MARIE_PLUGIN_DAEMON_URL']
                    == 'http://scheduled-plugins.child-proof.svc:8099'
                )
                # Replace only cluster transport; signing references come from the real chart.
                client_env['MARIE_PLUGIN_DAEMON_URL'] = url
                executor = MariePluginDaemonExecutor(start=False, env=client_env)
            else:
                monkeypatch.setenv('MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID', key_id)
                monkeypatch.setenv('MARIE_PLUGIN_DAEMON_SIGNING_SECRET', secret)
                executor = MariePluginDaemonExecutor(daemon_url=url, start=False)
        else:
            monkeypatch.setenv('MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID', 'parent-key')
            monkeypatch.setenv(
                'MARIE_PLUGIN_DAEMON_SIGNING_SECRET', secrets.token_urlsafe(32)
            )
            del env['MARIE_PLUGIN_DAEMON_URL']
            executor = MariePluginDaemonExecutor(
                daemon_bin=str(daemon_binary), daemon_addr=addr, env=env
            )

        deadline = time.monotonic() + 5
        while True:
            try:
                with urlopen(f'{url}/health', timeout=0.2):
                    break
            except (OSError, URLError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)

        archive = tmp_path / 'plugin.zip'
        with zipfile.ZipFile(archive, 'w') as package:
            for name in ('marie-extension.yaml', 'main.py', 'requirements.txt'):
                package.write(DAEMON_DIR / 'testdata' / 'fixture-plugin' / name, name)
        package_ref, digest = inspect_archive(str(archive))
        spec = PluginToolSpec(
            tool_ref='echo',
            package_ref=package_ref,
            package_digest=digest,
            package_id='fixture-package',
        )
        install_envelope = sign_envelope(
            build_invocation_envelope(
                spec, payload={}, organization_id='test-org', workspace_id='test-ws'
            ),
            key_id=key_id,
            secret=secret,
        )
        request = Request(
            f'{url}/v1/plugins/install',
            data=archive.read_bytes(),
            headers={'X-Marie-Envelope': json.dumps(install_envelope)},
        )
        with urlopen(request, timeout=20) as response:
            assert response.status == 200

        source = tmp_path / 'source.json'
        source.write_text('{}')
        job = JobInfo(
            status=JobStatus.PENDING,
            entrypoint='plugin_daemon_executor://execute',
            metadata={
                'metadata': {
                    'uri': source.as_uri(),
                    'op_params': {
                        'plugin': {
                            'tool_ref': 'echo',
                            'package_ref': package_ref,
                            'package_digest': digest,
                            'package_id': 'fixture-package',
                        },
                        'organization_id': 'test-org',
                        'workspace_id': 'test-ws',
                        'url': 'https://example.com/signed?name=caf\u00e9&limit=2',
                    },
                }
            },
        )
        parameters, _ = asyncio.run(
            GatewayJobDistributor()._build_payload('signed-job', job)
        )
        docs = asyncio.run(
            executor.connector_invoke(DocList[TextDoc]([]), parameters=parameters)
        )
        frames = [json.loads(doc.text) for doc in docs]
        assert [frame['type'] for frame in frames] == ['stream', 'end'], frames
        assert (
            frames[0]['data']['echo']['url']
            == 'https://example.com/signed?name=caf\u00e9&limit=2'
        )
        assert frames[0]['requestId'] == 'signed-job'

        wrong_version = copy.deepcopy(parameters)
        wrong_version['payload']['op_params']['plugin']['package_digest'] = (
            'sha256:' + '0' * 64
        )
        docs = asyncio.run(
            executor.connector_invoke(DocList[TextDoc]([]), parameters=wrong_version)
        )
        assert len(docs) == 1
        assert json.loads(docs[0].text)['error']['message'] == (
            'daemon invoke returned HTTP 409'
        )

        wrong_key = MariePluginDaemonExecutor(
            daemon_url=url,
            start=False,
            env={
                **env,
                'MARIE_PLUGIN_DAEMON_SIGNING_SECRET': secrets.token_urlsafe(32),
            },
        )
        try:
            docs = asyncio.run(
                wrong_key.connector_invoke(DocList[TextDoc]([]), parameters=parameters)
            )
            assert (
                json.loads(docs[0].text)['error']['message']
                == 'daemon invoke returned HTTP 401'
            )
        finally:
            wrong_key.shutdown()
    finally:
        if executor is not None:
            executor.shutdown()
        if daemon is not None:
            daemon.terminate()
            try:
                daemon.wait(timeout=15)
            except subprocess.TimeoutExpired:
                daemon.kill()
                daemon.wait(timeout=5)
