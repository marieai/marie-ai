"""Saved Studio binding through worker admission, real daemon and SDK handler."""

import json
import os
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pytest
import yaml

from marie.plugin_daemon import (
    PluginInvocationSpec,
    build_invocation_envelope,
    sign_envelope,
)
from marie.plugins.embedded import inspect_archive

from .test_plugin_daemon_executor_signed import daemon_binary as daemon_binary

ROOT = Path(__file__).resolve().parents[3]
STUDIO = Path(os.environ.get('MARIE_STUDIO_TEST_ROOT', ROOT.parent / 'marie-studio'))


@pytest.mark.timeout(90)
def test_saved_studio_sources_execute_real_datasource_results(
    daemon_binary, tmp_path, unused_tcp_port
):
    if not (STUDIO / 'packages/api/package.json').is_file():
        pytest.skip('Set MARIE_STUDIO_TEST_ROOT to the companion Studio checkout')
    manifest = {
        'apiVersion': 'marie.ai/v1alpha1',
        'kind': 'ExtensionPackage',
        'metadata': {
            'id': 'ext.test.datasource',
            'name': 'datasource',
            'version': '0.1.0',
        },
        'runtime': {
            'type': 'python_source',
            'language': 'python',
            'version': '3.12',
            'entrypoint': 'main',
        },
        'providers': [
            {
                'ref': 'providers/wiki',
                'type': 'datasource_provider',
                'credentials': [{'key': 'token', 'type': 'secret', 'required': True}],
            }
        ],
        'datasources': [
            {
                'ref': 'datasources/engineering/wiki',
                'name': 'Wiki',
                'providerRef': 'providers/wiki',
                'implementation': {'source': 'main.py'},
                'configurationSchema': {
                    'parameters': [{'key': 'space', 'type': 'string', 'required': True}]
                },
                'outputSchema': {
                    'type': 'object',
                    'required': ['id'],
                    'properties': {'id': {'type': 'string'}},
                },
            }
        ],
    }
    archive = tmp_path / 'datasource.zip'
    with zipfile.ZipFile(archive, 'w') as package:
        package.writestr('marie-extension.yaml', yaml.safe_dump(manifest))
        package.writestr(
            'requirements.txt', 'pydantic>=2\nPyYAML>=6\njsonschema>=4.23,<5\n'
        )
        package.writestr(
            'main.py',
            '''from pathlib import Path
import yaml
from marie.extension.manifest import ExtensionPackage
from marie.extension.datasource import DatasourceResult
from marie_plugins.runtime import run
from marie_plugins.runtime.datasource import dispatch_datasource

definition = ExtensionPackage.model_validate(yaml.safe_load(Path('marie-extension.yaml').read_text())).datasources[0]
def fetch(request):
    return DatasourceResult(records=[{'id': request.source_id, 'dataset': request.configuration['space'],
                                  'count': 0, 'active': False, 'token_echo': request.credentials['token']}])
run(lambda request: dispatch_datasource(request, definition, fetch))
''',
        )
        for source_root, prefix in [
            (ROOT / 'packages/marie-extension/src/marie/extension', 'marie/extension'),
            (
                ROOT
                / 'packages/marie-plugin-daemon/python_runtime/marie_plugins/runtime',
                'marie_plugins/runtime',
            ),
        ]:
            for path in source_root.glob('*.py'):
                package.write(path, f'{prefix}/{path.name}')
    package_ref, digest = inspect_archive(str(archive))
    address = f'127.0.0.1:{unused_tcp_port}'
    url = f'http://{address}'
    env = {
        **os.environ,
        'MARIE_PLUGIN_STORAGE_ROOT': str(tmp_path / 'storage'),
        'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID': 'test',
        'MARIE_PLUGIN_DAEMON_SIGNING_SECRET': 'test-only-secret',
        'MARIE_PLUGIN_DAEMON_DEV_INSECURE': 'false',
    }
    daemon = subprocess.Popen(
        [str(daemon_binary), '--addr', address],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with urlopen(f'{url}/health', timeout=0.2):
                    break
            except (OSError, URLError):
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.02)
        spec = PluginInvocationSpec(
            action_type='stub',
            action_id='runtime/install',
            package_id='package-1',
            install_id='install-1',
            package_ref=package_ref,
            package_digest=digest,
        )
        envelope = sign_envelope(
            build_invocation_envelope(
                spec,
                payload={},
                organization_id='11111111-1111-4111-8111-111111111111',
                workspace_id='22222222-2222-4222-8222-222222222222',
            ),
            key_id='test',
            secret='test-only-secret',
        )
        with urlopen(
            Request(
                f'{url}/v1/plugins/install',
                data=archive.read_bytes(),
                headers={'X-Marie-Envelope': json.dumps(envelope)},
            ),
            timeout=30,
        ) as response:
            assert response.status == 200
        worker = tmp_path / 'worker.py'
        worker.write_text(
            '''import asyncio, contextlib, json, os, sys
from docarray import DocList
from docarray.documents import TextDoc
from marie.executor.extensions.plugin_daemon_executor import MariePluginDaemonExecutor
envelope = json.load(sys.stdin)
with contextlib.redirect_stdout(sys.stderr):
    executor = MariePluginDaemonExecutor(daemon_url=os.environ['MARIE_DATASOURCE_TEST_DAEMON_URL'], start=False,
        env={'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID': 'test', 'MARIE_PLUGIN_DAEMON_SIGNING_SECRET': 'test-only-secret'})
    docs = asyncio.run(executor.invoke_plugin(DocList[TextDoc](), parameters={'invocation_envelope': envelope}))
print(json.dumps([json.loads(doc.text) for doc in docs]))
'''
        )
        result = subprocess.run(
            [
                'pnpm',
                '--filter',
                'api',
                'exec',
                'vitest',
                'run',
                'src/server/services/extensions/extension-datasource-invocation.service.test.ts',
                '-t',
                'installed datasource handler',
            ],
            cwd=STUDIO,
            env={
                **env,
                'PYTHONPATH': str(ROOT),
                'MARIE_DATASOURCE_TEST_WORKER': str(worker),
                'MARIE_DATASOURCE_TEST_DAEMON_URL': url,
                'MARIE_DATASOURCE_TEST_PYTHON': os.sys.executable,
                'MARIE_DATASOURCE_TEST_DIGEST': digest,
            },
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert '1 passed' in result.stdout
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)
