"""Resolve and invoke Studio tool records through the agent execution endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import subprocess
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from docarray import DocList
from docarray.documents import TextDoc
from marie.agent.backends.base import AgentResult, AgentStatus
from marie.agent.streaming import StreamChunk

from marie.executor.agent import AgentExecutor
from marie.plugin_daemon import build_invocation_envelope, sign_envelope
from marie.plugins.agent_tool import PluginToolSpec
from marie.plugins.embedded import inspect_archive

from .test_plugin_daemon_executor_signed import DAEMON_DIR
from .test_plugin_daemon_executor_signed import daemon_binary as daemon_binary


@pytest.mark.parametrize(
    'endpoint',
    ['execute_endpoint', 'chat_endpoint', 'chat_stream_endpoint', 'coordinated_chat'],
)
@pytest.mark.timeout(60)
async def test_agent_binds_dropped_tool_per_request(
    endpoint: str,
    daemon_binary: Path,
    tmp_path: Path,
    unused_tcp_port: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    address = f'127.0.0.1:{unused_tcp_port}'
    url = f'http://{address}'
    key_id, secret = 'dropped-tool-test', secrets.token_urlsafe(32)
    monkeypatch.setenv('MARIE_PLUGIN_DAEMON_URL', url)
    monkeypatch.setenv('MARIE_PLUGIN_DAEMON_SIGNING_KEY_ID', key_id)
    monkeypatch.setenv('MARIE_PLUGIN_DAEMON_SIGNING_SECRET', secret)
    daemon = subprocess.Popen(
        [str(daemon_binary), '--addr', address],
        env={**os.environ, 'MARIE_PLUGIN_STORAGE_ROOT': str(tmp_path / 'storage')},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    executor = None
    original_tasks = asyncio.all_tasks()
    try:
        deadline = time.monotonic() + 5
        while True:
            try:
                with urlopen(f'{url}/health', timeout=0.2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                await asyncio.sleep(0.02)
        archive = tmp_path / 'tool.zip'
        with zipfile.ZipFile(archive, 'w') as package:
            for name in ('main.py', 'marie-extension.yaml', 'requirements.txt'):
                content = (DAEMON_DIR / 'testdata/fixture-plugin' / name).read_bytes()
                if name == 'main.py':
                    content = content.replace(
                        b'import json\n', b'import json\nfrom pathlib import Path\n', 1
                    )
                    content = content.replace(
                        b'"echo": payload, "event": event',
                        b'"echo": payload, "event": event, "tenant": Path.cwd().parent.name',
                    )
                package.writestr(name, content)
        package_ref, digest = inspect_archive(str(archive))
        dropped = {
            'id': 'fixture-tool',
            'name': 'Fixture Echo',
            'slug': 'fixture_echo',
            'toolType': 'function',
            'source': 'extension_tool',
            'runtimeExecution': 'disabled',
            'installId': 'fixture-install',
            'providerId': 'fixture-provider',
            'packageId': 'fixture-package',
            'toolRef': 'echo',
            'providerRef': 'fixture',
            'packageRef': package_ref,
            'packageDigest': digest,
            'packageTrustLevel': 'community',
            'inputSchema': {
                'type': 'object',
                'properties': {'url': {'type': 'string'}},
                'required': ['url'],
            },
        }
        for tenant in ['one', 'two']:
            envelope = sign_envelope(
                build_invocation_envelope(
                    PluginToolSpec.model_validate(dropped),
                    payload={},
                    organization_id=f'org-{tenant}',
                    workspace_id=f'ws-{tenant}',
                ),
                key_id=key_id,
                secret=secret,
            )
            with urlopen(
                Request(
                    f'{url}/v1/plugins/install',
                    data=archive.read_bytes(),
                    headers={'X-Marie-Envelope': json.dumps(envelope)},
                ),
                timeout=20,
            ) as response:
                assert response.status == 200

        ready = asyncio.Event()

        class ToolCallingBackend:
            calls = 0

            async def run(self, *, tools: dict, **kwargs: object) -> AgentResult:
                tool = tools['fixture_echo']
                assert tool.metadata.get_parameters_dict()['required'] == ['url']
                self.calls += 1
                if self.calls == 2:
                    ready.set()
                await asyncio.wait_for(ready.wait(), timeout=3)
                output = await tool.acall(url='https://example.com')
                return AgentResult(output=output.content, status=AgentStatus.COMPLETED)

            async def run_stream(self, **kwargs: object):
                result = await self.run(**kwargs)
                yield StreamChunk(content=result.output_text)
                yield result

        executor = AgentExecutor(tools=[dropped], enable_conversation_store=False)
        backend = ToolCallingBackend()
        monkeypatch.setattr(executor, '_create_backend', lambda: backend)
        if endpoint == 'coordinated_chat':

            async def coordinated(
                docs: DocList[TextDoc], parameters: dict
            ) -> DocList[TextDoc]:
                result = await backend.run(tools=executor._tools)
                return DocList[TextDoc]([TextDoc(text=result.output_text)])

            monkeypatch.setattr(executor, '_should_use_coordination', lambda: True)
            monkeypatch.setattr(executor, '_coordinated_chat', coordinated)
            endpoint = 'chat_endpoint'
        responses = await asyncio.gather(
            *[
                getattr(executor, endpoint)(
                    DocList[TextDoc]([TextDoc(text='Use the tool')]),
                    parameters={
                        'organization_id': f'org-{tenant}',
                        'workspace_id': f'ws-{tenant}',
                        'user_id': f'user-{tenant}',
                    },
                )
                for tenant in ['one', 'two']
            ]
        )
        for tenant, docs in zip(['one', 'two'], responses):
            frames = json.loads(docs[0].text)
            assert [frame['type'] for frame in frames] == ['stream', 'end']
            assert frames[0]['data']['tenant'] == f'org-{tenant}__ws-{tenant}'
            assert frames[0]['data']['echo']['user_id'] == f'user-{tenant}'
            assert frames[0]['data']['echo']['provider'] == 'fixture'
            assert frames[0]['data']['echo']['tool_parameters'] == {
                'url': 'https://example.com'
            }
    finally:
        remaining = asyncio.all_tasks() - original_tasks
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
        if executor is not None:
            executor.close()
        daemon.terminate()
        try:
            daemon.wait(timeout=5)
        except subprocess.TimeoutExpired:
            daemon.kill()
            daemon.wait(timeout=5)
