"""Behavioral checks for LLM queue connection forwarding in deployment artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import time
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
    _ROOT
    / 'Dockerfiles'
    / 'docker-compose.g5-annotators.yml': [
        'annotator-g5-llm',
        'annotator-g5-table',
        'annotator-g5-parser',
        'annotator-g5-embeddings',
        'annotator-g5-table-parser',
    ],
}


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
    assert 'LLM_QUEUE_CONTRACT_VERSION=v2' in arguments
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
def test_v2_queue_round_trip_uses_configured_brand_neutral_url(store: str) -> None:
    stores_file = os.environ.get('MARIE_LLM_QUEUE_TEST_STORES')
    if not stores_file:
        pytest.skip(
            'set MARIE_LLM_QUEUE_TEST_STORES to run against isolated queue stores'
        )

    stores = json.loads(Path(stores_file).read_text())
    url = stores[store]['url']

    from marie.engine.completion_contract import (
        COMPLETION_QUEUE_CONTRACT_VERSION,
        CompletionReplyEnvelope,
        QueuedCompletionEnvelope,
        build_completion_call,
    )
    from marie.engine.llm_queue.config import LlmQueueConfig
    from marie.engine.llm_queue.queue_io import StoreListQueueClient

    suffix = uuid.uuid4().hex
    pool_id = f'task2-{store}-{suffix}'
    producer_id = f'task2-{store}-{suffix}'
    config = LlmQueueConfig(enabled=True, queue_url=url, pool_id=pool_id)
    client = StoreListQueueClient(config.queue_url)
    request = QueuedCompletionEnvelope(
        request_id=f'request-{suffix}',
        producer_id=producer_id,
        pool_id=pool_id,
        submitted_at=time.time(),
        call=build_completion_call(
            model='queue-round-trip-test',
            messages=[{'role': 'user', 'content': 'round trip'}],
            stream=False,
        ),
    )

    try:
        client.push_request(request)
        received_request = client.pop_request(pool_id, timeout=1)
        assert received_request is not None
        assert (
            received_request.contract_version
            == COMPLETION_QUEUE_CONTRACT_VERSION
            == 'v2'
        )
        assert received_request.call.messages == [
            {'role': 'user', 'content': 'round trip'}
        ]

        client.push_reply(
            CompletionReplyEnvelope(
                request_id=request.request_id,
                producer_id=producer_id,
                pool_id=pool_id,
                status='ok',
                completed_at=time.time(),
                completion={'choices': [{'message': {'content': 'complete'}}]},
            ),
            ttl_seconds=60,
        )
        received_reply = client.pop_reply(producer_id, timeout=1)
        assert received_reply is not None
        assert received_reply.contract_version == 'v2'
        assert received_reply.completion == {
            'choices': [{'message': {'content': 'complete'}}]
        }
    finally:
        client.close()


@pytest.mark.parametrize('version', ['v2', 'v3'])
def test_helm_forwards_explicit_contract_version(tmp_path: Path, version: str) -> None:
    documents = _render(
        tmp_path,
        {
            'global': {'llmQueue': {'enabled': True, 'contractVersion': version}},
            **_queue_pool_values(),
        },
    )
    for workload in _assert_queue_workloads(documents):
        assert (
            _queue_environment(workload)['LLM_QUEUE_CONTRACT_VERSION']['value']
            == version
        )


def test_helm_rejects_unknown_contract_version(tmp_path: Path) -> None:
    error = _render_failure(
        tmp_path, {'global': {'llmQueue': {'enabled': True, 'contractVersion': 'v4'}}}
    )
    assert 'global.llmQueue.contractVersion must be v2 or v3' in error


@pytest.mark.parametrize(
    'environment,expected', [({}, 'v2'), ({'LLM_QUEUE_CONTRACT_VERSION': 'v3'}, 'v3')]
)
def test_compose_forwards_contract_selection(
    tmp_path: Path, environment: dict[str, str], expected: str
) -> None:
    for source, services in _COMPOSE_SERVICES.items():
        config = _render_compose(
            _minimal_compose_file(tmp_path, source, services), environment
        )
        for service in services:
            assert (
                config['services'][service]['environment']['LLM_QUEUE_CONTRACT_VERSION']
                == expected
            )
