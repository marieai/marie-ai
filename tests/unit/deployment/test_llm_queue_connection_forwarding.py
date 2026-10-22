"""Behavioral checks for LLM queue connection forwarding in deployment artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[3]
_CHART = _ROOT / 'deploy' / 'helm' / 'charts' / 'marie'
_COMPOSE_SERVICES = {
    _ROOT / 'Dockerfiles' / 'docker-compose.gateway.yml': ['marie-gateway'],
    _ROOT / 'Dockerfiles' / 'docker-compose.allinone.yml': ['marie-gateway'],
    _ROOT / 'Dockerfiles' / 'docker-compose.extract.yml': ['marie-extract-executor'],
    _ROOT / 'Dockerfiles' / 'docker-compose.g5-annotators.yml': [
        'annotator-g5-llm',
        'annotator-g5-table',
        'annotator-g5-parser',
        'annotator-g5-embeddings',
        'annotator-g5-table-parser',
    ],
}


def test_programmatic_aimock_uses_direct_host_networking_by_default() -> None:
    environment = os.environ.copy()
    environment.update(AIMOCK_PORT='4010', AIMOCK_ADMIN_PORT='4011')
    environment.pop('AIMOCK_HOST', None)
    result = subprocess.run(
        [
            'docker',
            'compose',
            '--env-file',
            '/dev/null',
            '-f',
            str(_ROOT / 'Dockerfiles/docker-compose.mock-llm-programmatic.yml'),
            'config',
            '--format',
            'json',
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    configuration = json.loads(result.stdout)
    service = configuration['services']['aimock-programmatic']
    assert service['network_mode'] == 'host'
    assert not service.get('ports')
    assert not service.get('networks')
    assert not configuration.get('networks')
    assert service['environment']['AIMOCK_HOST'] == '127.0.0.1'


def _queue_pool_values() -> dict[str, Any]:
    """Return one stateless and one stateful executor pool for render coverage."""
    values = yaml.safe_load((_CHART / 'values.yaml').read_text())
    pools = deepcopy(values['executor']['pools'])
    pools[0]['stateful'] = False
    pools[1]['stateful'] = True
    return {'executor': {'pools': pools}}


def _render(tmp_path: Path, values: dict[str, Any]) -> list[dict[str, Any]]:
    values_file = tmp_path / 'values.yaml'
    values_file.write_text(yaml.safe_dump(values))
    result = subprocess.run(
        ['helm', 'template', 'queue-test', str(_CHART), '-f', str(values_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(f'helm template failed:\n{result.stderr}')
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def _render_values_file(values_file: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ['helm', 'template', 'queue-production', str(_CHART), '-f', str(values_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.fail(f'helm template failed:\n{result.stderr}')
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def _render_failure(tmp_path: Path, values: dict[str, Any]) -> str:
    values_file = tmp_path / 'values.yaml'
    values_file.write_text(yaml.safe_dump(values))
    result = subprocess.run(
        ['helm', 'template', 'queue-test', str(_CHART), '-f', str(values_file)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    return result.stderr


def _queue_workloads(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    workloads = []
    for document in documents:
        if document.get('kind') not in {'Deployment', 'StatefulSet'}:
            continue
        labels = document.get('metadata', {}).get('labels', {})
        if labels.get('app.kubernetes.io/component') in {'server', 'executor'}:
            workloads.append(document)
    return workloads


def _queue_environment(workload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    container = workload['spec']['template']['spec']['containers'][0]
    return {
        variable['name']: variable
        for variable in container.get('env', [])
        if variable['name'].startswith('LLM_QUEUE_')
    }


def _workload_environment(workload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    container = workload['spec']['template']['spec']['containers'][0]
    return {variable['name']: variable for variable in container.get('env', [])}


def _assert_queue_workloads(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    workloads = _queue_workloads(documents)
    kinds = {workload['kind'] for workload in workloads}
    assert kinds == {'Deployment', 'StatefulSet'}
    assert any(
        workload['metadata']['name'].endswith('-server') for workload in workloads
    )
    assert sum(workload['kind'] == 'Deployment' for workload in workloads) >= 2
    return workloads


def _has_bundled_valkey(documents: list[dict[str, Any]]) -> bool:
    return any(
        document.get('kind') == 'StatefulSet'
        and document.get('metadata', {}).get('name', '').endswith('-valkey')
        for document in documents
    )


def test_helm_external_queue_url_wins_without_bundled_valkey(tmp_path: Path) -> None:
    documents = _render(
        tmp_path,
        {
            'global': {
                'valkey': {'enabled': False},
                'llmQueue': {'enabled': True, 'url': 'redis://external-queue:6380/5'},
            },
            'valkey': {'enabled': False},
            **_queue_pool_values(),
        },
    )

    assert not _has_bundled_valkey(documents)
    for workload in _assert_queue_workloads(documents):
        environment = _queue_environment(workload)
        assert environment['LLM_QUEUE_URL']['value'] == 'redis://external-queue:6380/5'
        assert (
            environment['LLM_QUEUE_VALKEY_URL']['value']
            == 'redis://external-queue:6380/5'
        )


def test_helm_external_queue_secret_wins_without_bundled_valkey(tmp_path: Path) -> None:
    documents = _render(
        tmp_path,
        {
            'global': {
                'valkey': {'enabled': False},
                'llmQueue': {
                    'enabled': True,
                    'existingSecret': 'queue-connection',
                    'existingSecretUrlKey': 'connection-url',
                },
            },
            'valkey': {'enabled': False},
            **_queue_pool_values(),
        },
    )

    assert not _has_bundled_valkey(documents)
    expected_ref = {'name': 'queue-connection', 'key': 'connection-url'}
    for workload in _assert_queue_workloads(documents):
        environment = _queue_environment(workload)
        assert environment['LLM_QUEUE_URL']['valueFrom']['secretKeyRef'] == expected_ref
        assert (
            environment['LLM_QUEUE_VALKEY_URL']['valueFrom']['secretKeyRef']
            == expected_ref
        )


def test_helm_bundled_valkey_connection_sets_canonical_and_legacy_urls(
    tmp_path: Path,
) -> None:
    documents = _render(
        tmp_path,
        {
            'global': {'llmQueue': {'enabled': True}},
            **_queue_pool_values(),
        },
    )

    assert _has_bundled_valkey(documents)
    for workload in _assert_queue_workloads(documents):
        environment = _queue_environment(workload)
        assert (
            environment['LLM_QUEUE_URL']['value'] == 'redis://queue-test-valkey:6379/0'
        )
        assert (
            environment['LLM_QUEUE_VALKEY_URL']['value']
            == 'redis://queue-test-valkey:6379/0'
        )


def test_helm_queue_disabled_without_valkey_has_no_connection_url(
    tmp_path: Path,
) -> None:
    documents = _render(
        tmp_path,
        {
            'global': {
                'valkey': {'enabled': False},
                'llmQueue': {'enabled': False},
            },
            'valkey': {'enabled': False},
            **_queue_pool_values(),
        },
    )

    for workload in _assert_queue_workloads(documents):
        environment = _queue_environment(workload)
        assert environment['LLM_QUEUE_ENABLED']['value'] == 'false'
        assert 'LLM_QUEUE_URL' not in environment
        assert 'LLM_QUEUE_VALKEY_URL' not in environment


def test_helm_rejects_enabled_queue_without_connection_or_bundled_valkey(
    tmp_path: Path,
) -> None:
    error = _render_failure(
        tmp_path,
        {
            'global': {
                'valkey': {'enabled': False},
                'llmQueue': {'enabled': True, 'url': '   '},
            },
            'valkey': {'enabled': False},
            **_queue_pool_values(),
        },
    )

    assert 'global.llmQueue.url or global.llmQueue.existingSecret' in error
    assert 'redis://' not in error


def test_helm_rejects_queue_when_bundled_valkey_dependency_is_disabled(
    tmp_path: Path,
) -> None:
    error = _render_failure(
        tmp_path,
        {
            'global': {'llmQueue': {'enabled': True}},
            'valkey': {'enabled': False},
            **_queue_pool_values(),
        },
    )

    assert 'global.llmQueue.url or global.llmQueue.existingSecret' in error
    assert 'redis://' not in error


def test_helm_production_values_keep_explicit_legacy_valkey_connection() -> None:
    documents = _render_values_file(_CHART / 'values-production.yaml')

    assert not _has_bundled_valkey(documents)
    server = next(
        workload
        for workload in _queue_workloads(documents)
        if workload['metadata']['name'].endswith('-server')
    )
    environment = _queue_environment(server)
    expected_url = 'rediss://:$(VALKEY_PASSWORD)@valkey.example.internal:6379/0'
    assert environment['LLM_QUEUE_URL']['value'] == expected_url
    assert environment['LLM_QUEUE_VALKEY_URL']['value'] == expected_url
    all_environment = _workload_environment(server)
    assert all_environment['VALKEY_PASSWORD']['valueFrom']['secretKeyRef'] == {
        'name': 'marie-valkey',
        'key': 'password',
    }


def _minimal_compose_file(tmp_path: Path, source: Path, services: list[str]) -> Path:
    source_config = yaml.safe_load(source.read_text())
    minimal_services = {
        service: {
            'image': 'busybox:latest',
            'environment': source_config['services'][service]['environment'],
        }
        for service in services
    }
    compose_file = tmp_path / source.name
    compose_file.write_text(yaml.safe_dump({'services': minimal_services}))
    return compose_file


def _render_compose(compose_file: Path, environment: dict[str, str]) -> dict[str, Any]:
    command_environment = {'PATH': os.environ['PATH'], **environment}
    result = subprocess.run(
        [
            'docker',
            'compose',
            '--env-file',
            '/dev/null',
            '-f',
            str(compose_file),
            'config',
            '--format',
            'json',
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=command_environment,
    )
    if result.returncode != 0:
        pytest.fail(f'docker compose config failed:\n{result.stderr}')
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ('environment', 'expected'),
    [
        ({}, 'redis://localhost:6379/0'),
        (
            {'LLM_QUEUE_VALKEY_URL': 'redis://legacy-queue:6380/1'},
            'redis://legacy-queue:6380/1',
        ),
        (
            {'LLM_QUEUE_URL': 'redis://canonical-queue:6380/2'},
            'redis://canonical-queue:6380/2',
        ),
        (
            {
                'LLM_QUEUE_URL': 'redis://canonical-queue:6380/3',
                'LLM_QUEUE_VALKEY_URL': 'redis://legacy-queue:6380/4',
            },
            'redis://canonical-queue:6380/3',
        ),
        (
            {
                'LLM_QUEUE_URL': '',
                'LLM_QUEUE_VALKEY_URL': 'redis://legacy-queue:6380/5',
            },
            'redis://legacy-queue:6380/5',
        ),
        (
            {'LLM_QUEUE_URL': '', 'LLM_QUEUE_VALKEY_URL': ''},
            'redis://localhost:6379/0',
        ),
    ],
)
def test_compose_queue_connection_precedence(
    tmp_path: Path, environment: dict[str, str], expected: str
) -> None:
    for source, services in _COMPOSE_SERVICES.items():
        compose_file = _minimal_compose_file(tmp_path, source, services)
        config = _render_compose(compose_file, environment)
        for service in services:
            rendered_environment = config['services'][service]['environment']
            assert rendered_environment['LLM_QUEUE_URL'] == expected
            assert rendered_environment['LLM_QUEUE_VALKEY_URL'] == expected


def test_run_gateway_forwards_the_canonical_connection_to_legacy_images(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    docker_log = tmp_path / 'docker-arguments.txt'
    docker = fake_bin / 'docker'
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$FAKE_DOCKER_LOG"\n')
    docker.chmod(0o755)
    logger = fake_bin / 'logger'
    logger.write_text('#!/bin/sh\ncat >/dev/null\n')
    logger.chmod(0o755)

    result = subprocess.run(
        ['bash', 'run-gateway.sh'],
        cwd=_ROOT / 'docker-scripts',
        capture_output=True,
        text=True,
        timeout=60,
        env={
            'PATH': f'{fake_bin}:{os.environ["PATH"]}',
            'FAKE_DOCKER_LOG': str(docker_log),
            'LLM_QUEUE_URL': 'redis://canonical-queue:6380/9',
            'LLM_QUEUE_VALKEY_URL': 'redis://legacy-queue:6380/9',
        },
    )

    assert result.returncode == 0, result.stderr
    arguments = docker_log.read_text().splitlines()
    assert not any(
        value.startswith('LLM_QUEUE_CONTRACT_VERSION=') for value in arguments
    )
    assert 'LLM_QUEUE_URL=redis://canonical-queue:6380/9' in arguments
    assert 'LLM_QUEUE_VALKEY_URL=redis://canonical-queue:6380/9' in arguments


def test_bootstrap_hides_queue_connection_contents() -> None:
    queue_url = 'redis://dummy-user:dummy-password@queue.example:6380/9'
    result = subprocess.run(
        [
            'bash',
            '-c',
            (
                "source <(sed '/^main \"\\$@\"$/d' \"$1\"); "
                "DEPLOY_INFRASTRUCTURE=true; LLM_QUEUE_URL=\"$2\"; "
                'show_service_endpoints'
            ),
            'bash',
            str(_ROOT / 'bootstrap-marie.sh'),
            queue_url,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            'PATH': os.environ['PATH'],
            'DEPLOY_INFRASTRUCTURE': 'true',
        },
    )

    assert result.returncode == 0, result.stderr
    assert 'LLM Queue: configured (LLM_QUEUE_URL)' in result.stdout
    assert queue_url not in result.stdout
    assert 'dummy-user' not in result.stdout
    assert 'dummy-password' not in result.stdout


@pytest.mark.parametrize('store', ['valkey', 'redis'])
def test_durable_queue_round_trip_uses_configured_brand_neutral_url(store: str) -> None:
    stores_file = os.environ.get('MARIE_LLM_QUEUE_TEST_STORES')
    if not stores_file:
        pytest.skip('isolated queue store manifest required')

    from marie.engine.completion_contract import (
        CompletionCallParams,
        QueuedCompletionEnvelope,
    )
    from marie.engine.llm_queue.config import LlmQueueRuntimeConfig
    from marie.engine.llm_queue.store import RequestStore

    url = json.loads(Path(stores_file).read_text())[store]['url']
    fabric_id = 'deployment-' + uuid.uuid4().hex
    config = LlmQueueRuntimeConfig(
        enabled=True, queue_url=url, fabric_group_id=fabric_id
    )
    client = RequestStore(config.queue_url, fabric_id=fabric_id)
    owner = client.acquire_owner('deployment-test', lease_ms=10_000)
    client.configure_route(
        owner,
        'pool',
        'endpoint',
        revision='r1',
        execution_limit=1,
        execution_bytes=100_000,
    )
    producer_id = client.create_producer(lease_ms=10_000)
    request = QueuedCompletionEnvelope(
        fabric_group_id=fabric_id,
        producer_id=producer_id,
        attempt_id=uuid.uuid4().hex,
        pool_id='pool',
        endpoint_id='endpoint',
        config_revision='r1',
        logical_batch_id='deployment-batch',
        logical_task_id='deployment-task',
        item_index=0,
        expires_at_ms=client.server_time_ms() + 5000,
        call=CompletionCallParams(
            model='queue-round-trip-test',
            messages=[{'role': 'user', 'content': 'round trip'}],
        ),
    )
    claim_id = uuid.uuid4().hex
    try:
        assert client.admit(request).disposition == 'admitted'
        assert (
            client.claim(
                owner, request.attempt_id, pool_id='pool', claim_id=claim_id
            ).disposition
            == 'claimed'
        )
        received = client.fetch_payload(owner, request.attempt_id, claim_id=claim_id)
        assert received['schema_version'] == 1
        assert received['call']['messages'] == [
            {'role': 'user', 'content': 'round trip'}
        ]
        started = client.authorize_start(owner, request.attempt_id, claim_id=claim_id)
        assert started.disposition == 'started'
        assert (
            client.finish(
                owner,
                request.attempt_id,
                claim_id=claim_id,
                execution_seq=started.execution_seq,
                result={'choices': [{'message': {'content': 'complete'}}]},
            ).disposition
            == 'finished'
        )
        result = client.read_result(producer_id, request.attempt_id)
        assert result == {'choices': [{'message': {'content': 'complete'}}]}
        assert client.ack_result(producer_id, request.attempt_id).disposition == 'acked'
    finally:
        client.release_owner(owner)
        keys = list(client.client.scan_iter(match=client.keys.prefix + '*'))
        if keys:
            client.client.delete(*keys)
        client.close()


def test_helm_enabled_queue_has_no_contract_selector(tmp_path: Path) -> None:
    documents = _render(
        tmp_path,
        {'global': {'llmQueue': {'enabled': True}}, **_queue_pool_values()},
    )
    for workload in _assert_queue_workloads(documents):
        environment = _queue_environment(workload)
        assert environment['LLM_QUEUE_ENABLED']['value'] == 'true'
        assert 'LLM_QUEUE_CONTRACT_VERSION' not in environment


@pytest.mark.parametrize('enabled', ['true', 'false'])
def test_compose_queue_enablement_has_no_contract_selector(
    tmp_path: Path, enabled: str
) -> None:
    for source, services in _COMPOSE_SERVICES.items():
        config = _render_compose(
            _minimal_compose_file(tmp_path, source, services),
            {'LLM_QUEUE_ENABLED': enabled},
        )
        for service in services:
            environment = config['services'][service]['environment']
            assert environment['LLM_QUEUE_ENABLED'] == enabled
            assert 'LLM_QUEUE_CONTRACT_VERSION' not in environment
