"""Real installed datasource -> scheduled worker -> registered KB references."""

import asyncio
import json
import os
import subprocess
import zipfile
from hashlib import sha256
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen
from uuid import uuid4

import pytest
import yaml
from docarray import DocList
from docarray.documents import TextDoc

from marie.executor.extensions.plugin_daemon_executor import MariePluginDaemonExecutor
from marie.plugin_daemon import (
    PluginInvocationSpec,
    build_invocation_envelope,
    sign_envelope,
)
from marie.plugins.embedded import inspect_archive
from tests.integration.plugins.test_plugin_daemon_executor_signed import (
    daemon_binary as daemon_binary,
)
from tests.unit.executor.extensions.test_datasource_sync import sync as sync

pytestmark = pytest.mark.skipif(
    not os.environ.get('MARIE_KB_TEST_PORT'), reason='Requires disposable KB PostgreSQL'
)
ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.timeout(90)
async def test_signed_installed_handler_registers_two_saved_bindings(
    sync, daemon_binary, tmp_path, unused_tcp_port
):
    pool, ids, _, old_worker, parameters, _, _ = sync
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
        'datasources': [
            {
                'ref': 'datasources/wiki',
                'name': 'Wiki',
                'indexingAdapter': 'document',
                'implementation': {'source': 'main.py'},
                'configurationSchema': {
                    'parameters': [{'key': 'space', 'type': 'string', 'required': True}]
                },
                'outputSchema': {'type': 'object', 'required': ['id']},
            }
        ],
    }
    archive = tmp_path / 'installed-datasource.zip'
    with zipfile.ZipFile(archive, 'w') as package:
        package.writestr('marie-extension.yaml', yaml.safe_dump(manifest))
        package.writestr(
            'requirements.txt', 'pydantic>=2\nPyYAML>=6\njsonschema>=4.23,<5\n'
        )
        package.writestr(
            'main.py',
            '''from pathlib import Path
import hashlib, yaml
from marie.extension.manifest import ExtensionPackage
from marie.extension.datasource import DatasourceResult
from marie_plugins.runtime import run
from marie_plugins.runtime.datasource import dispatch_datasource
definition = ExtensionPackage.model_validate(yaml.safe_load(Path('marie-extension.yaml').read_text())).datasources[0]
def fetch(request):
    uri = f's3://saved/tenants/{request.tenant_id}/kb-indexes/{request.index_id}/sources/{request.source_id}/wiki.md'
    return DatasourceResult(records=[{'id':'wiki:42','uri':uri,'input_ref':{'kind':'text','uri':uri,
        'media_type':'text/markdown','sha256':hashlib.sha256(request.configuration['space'].encode()).hexdigest()}}])
run(lambda request: dispatch_datasource(request,definition,fetch))
''',
        )
        for source, prefix in [
            (ROOT / 'packages/marie-extension/src/marie/extension', 'marie/extension'),
            (
                ROOT
                / 'packages/marie-plugin-daemon/python_runtime/marie_plugins/runtime',
                'marie_plugins/runtime',
            ),
        ]:
            for file in source.glob('*.py'):
                package.write(file, f'{prefix}/{file.name}')
    root = Path(old_worker._env['MARIE_EXTENSION_PACKAGE_STORE'])
    digest = sha256(archive.read_bytes()).hexdigest()
    (root / str(ids['organization']) / f'{digest}.zip').write_bytes(
        archive.read_bytes()
    )
    await pool.execute(
        'UPDATE marie_studio.extension_packages SET source_digest=$1', digest
    )
    package_ref, package_digest = inspect_archive(str(archive))
    address = f'127.0.0.1:{unused_tcp_port}'
    url = f'http://{address}'
    env = {
        **os.environ,
        'MARIE_PLUGIN_STORAGE_ROOT': str(tmp_path / 'daemon-storage'),
        'MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID': 'test',
        'MARIE_PLUGIN_DAEMON_SIGNING_SECRET': 'test-only',
        'MARIE_PLUGIN_DAEMON_DEV_INSECURE': 'false',
    }
    daemon = subprocess.Popen(
        [str(daemon_binary), '--addr', address],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for attempt in range(100):
            try:
                with urlopen(f'{url}/health', timeout=0.2):
                    break
            except (OSError, URLError):
                if attempt == 99:
                    raise
                await asyncio.sleep(0.02)
        spec = PluginInvocationSpec(
            action_type='stub',
            action_id='runtime/install',
            install_id=str(ids['install']),
            package_id=str(ids['package']),
            package_ref=package_ref,
            package_digest=package_digest,
        )
        envelope = sign_envelope(
            build_invocation_envelope(
                spec,
                payload={},
                organization_id=str(ids['organization']),
                workspace_id=str(ids['workspace']),
            ),
            key_id='test',
            secret='test-only',
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
        worker = MariePluginDaemonExecutor(
            storage={'s3': {'enabled': False, 'bucket_name': 'saved'}},
            daemon_url=url,
            start=False,
            env={**old_worker._env},
        )
        worker._datasource_pool = pool
        result = await worker.invoke_plugin(DocList[TextDoc](), parameters)
        assert json.loads(result[0].text)['data'] == {'records_admitted': 1}
        other = uuid4()
        other_job = uuid4()
        await pool.execute(
            '''INSERT INTO marie_studio.kb_source SELECT $1,index_id,source_type,status,
          extension_install_id,datasource_ref,'{"space":"OPS"}',metadata,created_by,error_message,updated_at
          FROM marie_studio.kb_source WHERE id=$2''',
            other,
            ids['source'],
        )
        await pool.execute(
            '''INSERT INTO marie_scheduler.job SELECT $1,$2,
          jsonb_set(data,'{metadata,source_id}',to_jsonb($3::text)),state,run_attempt_id,run_owner,
          run_lease_expires_at,now() FROM marie_scheduler.job WHERE id=$4''',
            other_job,
            uuid4(),
            str(other),
            ids['job'],
        )
        await worker.invoke_plugin(
            DocList[TextDoc](), {**parameters, 'job_id': str(other_job)}
        )
        rows = await pool.fetch(
            'SELECT source_id,item_id,input_ref FROM marie_scheduler.kb_item_state'
        )
        assert len(rows) == 2
        assert {row['item_id'] for row in rows} == {'wiki:42'}
        assert {row['source_id'] for row in rows} == {str(ids['source']), str(other)}
        assert len({row['input_ref']['sha256'] for row in rows}) == 2
    finally:
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)
