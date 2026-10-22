import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
import yaml

from marie.runtime import Flow

ROOT = Path(__file__).resolve().parents[3]
CHART = ROOT / 'deploy/helm/charts/marie'
PROBE = ROOT / 'tools/sandbox/mock_executor_smoke.py'


@pytest.fixture
def isolated_etcd_port() -> Iterator[int]:
    if not shutil.which('docker'):
        pytest.skip('Docker is required for isolated worker state')
    name = f'marie-mock-etcd-{uuid.uuid4().hex}'
    try:
        subprocess.run(
            [
                'docker',
                'run',
                '--detach',
                '--runtime=runc',
                '--name',
                name,
                '--publish',
                '127.0.0.1::2379',
                'quay.io/coreos/etcd:v3.7.0',
                '/usr/local/bin/etcd',
                '--data-dir=/tmp/etcd',
                '--listen-client-urls=http://0.0.0.0:2379',
                '--advertise-client-urls=http://127.0.0.1:2379',
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        address = subprocess.run(
            ['docker', 'port', name, '2379/tcp'],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        port = int(address.rsplit(':', 1)[1])
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(
                    f'http://127.0.0.1:{port}/health', timeout=1
                ) as response:
                    if response.status == 200:
                        break
            except (OSError, urllib.error.URLError):
                pass
            if time.monotonic() >= deadline:
                pytest.fail('Disposable etcd did not become healthy')
            time.sleep(0.1)
        yield port
    finally:
        subprocess.run(
            ['docker', 'rm', '--force', name],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )


@pytest.mark.timeout(120)
def test_rendered_mock_flow_handles_requests_and_recovers_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    port_generator: Callable[[], int],
    isolated_etcd_port: int,
) -> None:
    assert PROBE.is_file(), 'setup smoke command must validate actual results'
    rendered = subprocess.run(
        [
            'helm',
            'template',
            'mock-smoke',
            str(CHART),
            '--namespace',
            'mock-smoke',
            '-f',
            str(CHART / 'values-sandbox-mock.yaml'),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    configmap = next(
        doc
        for doc in yaml.safe_load_all(rendered.stdout)
        if doc
        and doc['kind'] == 'ConfigMap'
        and doc['metadata']['name'] == 'mock-smoke-executor-config'
    )
    flow = yaml.safe_load(configmap['data']['marie-mock-sandbox.yml'])
    # Workers use etcd for status even with service discovery disabled.
    flow['with']['discovery'] = False
    del flow['with']['kv_store_kwargs']
    flow['with']['discovery_host'] = '127.0.0.1'
    flow['with']['discovery_port'] = isolated_etcd_port
    for key in ('ETCD_ENDPOINTS', 'ETCD_CA_CERT', 'ETCD_CERT_KEY', 'ETCD_CERT_CERT'):
        monkeypatch.delenv(key, raising=False)
    flow['with']['host'] = '127.0.0.1'
    flow['with']['workspace'] = str(tmp_path / 'workspace')
    port = port_generator()
    monkeypatch.setenv('MARIE_PORT', str(port))
    monkeypatch.setenv('MARIE_DISABLE_CUDA', 'true')
    monkeypatch.setenv('JINA_LOG_LEVEL', 'ERROR')
    config = tmp_path / 'flow.yml'
    config.write_text(yaml.safe_dump(flow, sort_keys=False))

    with Flow.load_config(str(config)):
        for fail in (False, True, False):
            result = subprocess.run(
                [
                    sys.executable,
                    str(PROBE),
                    '--address',
                    f'127.0.0.1:{port}',
                    *(['--force-fail'] if fail else []),
                ],
                capture_output=True,
                text=True,
                timeout=25,
            )
            report = json.loads(result.stdout)
            if fail:
                assert result.returncode == 1, result.stderr
                assert report['ok'] is False
                assert report['error'] == 'mock_result_missing_or_invalid'
            else:
                assert result.returncode == 0, result.stderr
                assert report['ok'] is True
                assert report['processed_docs'] == 1
                assert report['executor'].startswith('mock_executor_a')
