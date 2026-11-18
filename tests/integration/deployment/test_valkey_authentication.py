"""Password authentication through the shipped Compose deployment."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
from valkey import Valkey
from valkey.exceptions import AuthenticationError

ROOT = Path(__file__).resolve().parents[3]
SETUP = ROOT / 'tools/setup-valkey-auth.py'


def provision(directory: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SETUP), '--output-dir', str(directory), *arguments],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_setup_generates_private_credentials_and_client_configuration(
    tmp_path: Path,
) -> None:
    directory = tmp_path / 'material'
    result = provision(directory, '--host', 'worker.example.internal')
    assert result.returncode == 0, result.stderr
    password = (directory / 'server/password').read_text()
    assert len(password) >= 48
    assert not (password in result.stdout or password in result.stderr)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / 'server/password').stat().st_mode) == 0o600
    acl = (directory / 'server/users.acl').read_text()
    assert 'user default off' in acl
    assert (
        f'user marie on #{hashlib.sha256(password.encode()).hexdigest()} ~* &* +@all'
        in acl
    )
    connection = urlsplit((directory / 'clients/queue-url').read_text())
    assert connection.scheme == 'redis'
    assert connection.username == 'marie'
    assert connection.hostname == 'worker.example.internal'
    assert not connection.query


def test_setup_refuses_to_replace_existing_material(tmp_path: Path) -> None:
    directory = tmp_path / 'material'
    directory.mkdir()
    sentinel = directory / 'existing'
    sentinel.write_text('preserve')
    result = provision(directory)
    assert result.returncode != 0
    assert 'already exists' in result.stderr
    assert sentinel.read_text() == 'preserve'
    assert list(directory.iterdir()) == [sentinel]


def test_setup_keeps_credentials_ignored_in_any_checkout_directory(
    tmp_path: Path,
) -> None:
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    result = provision(tmp_path / 'credential-rotation')
    assert result.returncode == 0, result.stderr
    status = subprocess.run(
        ['git', '-C', str(tmp_path), 'status', '--porcelain'],
        capture_output=True,
        text=True,
        check=True,
    )
    assert status.stdout == ''


@pytest.mark.parametrize('host', ['bad\nhost', 'user@host', '../../host'])
def test_setup_rejects_invalid_connection_hosts(tmp_path: Path, host: str) -> None:
    directory = tmp_path / 'material'
    result = provision(directory, '--host', host)
    assert result.returncode != 0
    assert 'invalid Valkey host' in result.stderr
    assert not directory.exists()


@pytest.fixture(scope='module')
def authenticated_valkey(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Any]]:
    directory = tmp_path_factory.mktemp('valkey-security')
    material = directory / 'material'
    result = provision(material)
    assert result.returncode == 0, result.stderr
    environment = os.environ.copy()
    environment['VALKEY_SECURITY_DIR'] = str(material)
    rendered = subprocess.run(
        [
            'docker',
            'compose',
            '--env-file',
            '/dev/null',
            '--project-directory',
            str(ROOT),
            '-f',
            str(ROOT / 'Dockerfiles/docker-compose.valkey.yml'),
            'config',
            '--format',
            'json',
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    configuration = json.loads(rendered.stdout)
    service = configuration['services']['valkey']
    service.pop('container_name', None)
    service['ports'] = [
        {'target': 6379, 'published': '0', 'host_ip': '127.0.0.1', 'protocol': 'tcp'}
    ]
    service['healthcheck'].update(interval='1s', start_period='0s')
    configuration['networks'] = {'marie_default': {}}
    configuration['volumes'] = {'valkey_data': {}}
    compose_file = directory / 'compose.json'
    compose_file.write_text(json.dumps(configuration))
    command = [
        'docker',
        'compose',
        '--env-file',
        '/dev/null',
        '-p',
        f'valkey-security-{uuid4().hex[:10]}',
        '-f',
        str(compose_file),
    ]

    def compose(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*command, *arguments],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )

    try:
        compose('up', '-d', '--wait', '--wait-timeout', '30')
        port = int(compose('port', 'valkey', '6379').stdout.strip().rsplit(':', 1)[1])
        yield {
            'material': material,
            'port': port,
            'compose': compose,
            'compose_file': compose_file,
        }
    finally:
        compose('down', '--volumes', '--remove-orphans')


def authenticated_client(runtime: dict[str, Any], **options: Any) -> Valkey:
    parameters = {
        'host': '127.0.0.1',
        'port': runtime['port'],
        'username': 'marie',
        'password': (runtime['material'] / 'server/password').read_text(),
        'socket_timeout': 2,
        'socket_connect_timeout': 2,
        'decode_responses': True,
    }
    parameters.update(options)
    return Valkey(**parameters)


def test_authenticated_access_allows_all_keys_commands_and_lua(
    authenticated_valkey: dict[str, Any]
) -> None:
    with authenticated_client(authenticated_valkey) as client:
        assert client.ping()
        assert (
            client.eval(
                "return redis.call('SET', KEYS[1], ARGV[1])",
                1,
                'outside:queue',
                'synthetic',
            )
            == 'OK'
        )
        assert client.get('outside:queue') == 'synthetic'
        assert client.config_get('protected-mode') == {'protected-mode': 'yes'}


@pytest.mark.parametrize(
    'credentials',
    [{'username': None, 'password': None}, {'password': 'invalid-test-password'}],
)
def test_rejects_anonymous_and_wrong_credentials(
    authenticated_valkey: dict[str, Any], credentials: dict[str, Any]
) -> None:
    with authenticated_client(authenticated_valkey, **credentials) as client:
        with pytest.raises(AuthenticationError):
            client.ping()


def test_recreation_preserves_data_authentication_and_health(
    authenticated_valkey: dict[str, Any]
) -> None:
    with authenticated_client(authenticated_valkey) as client:
        client.set('recreation:sentinel', 'synthetic')
    authenticated_valkey['compose'](
        'up', '-d', '--force-recreate', '--wait', '--wait-timeout', '30'
    )
    authenticated_valkey['port'] = int(
        authenticated_valkey['compose']('port', 'valkey', '6379')
        .stdout.strip()
        .rsplit(':', 1)[1]
    )
    with authenticated_client(authenticated_valkey) as client:
        assert client.get('recreation:sentinel') == 'synthetic'
    with authenticated_client(
        authenticated_valkey, username=None, password=None
    ) as client:
        with pytest.raises(AuthenticationError):
            client.ping()
    status = json.loads(
        authenticated_valkey['compose']('ps', '--format', 'json').stdout
    )
    assert status['Health'] == 'healthy'


@pytest.mark.parametrize(
    'damage',
    ['missing-directory', 'missing-files', 'anonymous-default', 'mismatched-password'],
)
def test_invalid_credentials_fail_closed(
    authenticated_valkey: dict[str, Any], tmp_path: Path, damage: str
) -> None:
    material = tmp_path / 'missing-material'
    if damage == 'missing-files':
        material.mkdir()
    elif damage in {'anonymous-default', 'mismatched-password'}:
        result = provision(material)
        assert result.returncode == 0, result.stderr
        if damage == 'anonymous-default':
            acl = material / 'server/users.acl'
            acl.write_text(acl.read_text().replace('user default off\n', ''))
        else:
            (material / 'server/password').write_text('changed-test-password')
        material = material / 'server'
    configuration = json.loads(authenticated_valkey['compose_file'].read_text())
    service = configuration['services']['valkey']
    service['restart'] = 'no'
    for mount in service['volumes']:
        if mount['target'] == '/run/valkey-source':
            mount['source'] = str(material)
    compose_file = tmp_path / 'compose.json'
    compose_file.write_text(json.dumps(configuration))
    command = [
        'docker',
        'compose',
        '--env-file',
        '/dev/null',
        '-p',
        f'valkey-missing-{uuid4().hex[:10]}',
        '-f',
        str(compose_file),
    ]
    try:
        result = subprocess.run(
            [*command, 'up', '-d', '--wait', '--wait-timeout', '5'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode != 0
    finally:
        subprocess.run(
            [*command, 'down', '--volumes', '--remove-orphans'],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )


def test_queue_store_round_trip_over_authenticated_connection(
    authenticated_valkey: dict[str, Any]
) -> None:
    from marie.engine.completion_contract import (
        CompletionCallParams,
        QueuedCompletionEnvelope,
    )
    from marie.engine.llm_queue.store import RequestStore

    password = (authenticated_valkey['material'] / 'server/password').read_text()
    store = RequestStore(
        f'redis://marie:{password}@127.0.0.1:{authenticated_valkey["port"]}/0',
        fabric_id=f'auth-{uuid4().hex}',
    )
    try:
        owner = store.acquire_owner('dispatcher', lease_ms=10_000)
        store.configure_route(
            owner,
            'pool',
            'endpoint',
            revision='r1',
            execution_limit=2,
            execution_bytes=100_000,
        )
        producer = store.create_producer(lease_ms=10_000)
        request = QueuedCompletionEnvelope(
            schema_version=1,
            fabric_group_id=store.keys.fabric_id,
            producer_id=producer,
            attempt_id=uuid4().hex,
            pool_id='pool',
            endpoint_id='endpoint',
            config_revision='r1',
            logical_batch_id='batch',
            logical_task_id='task',
            item_index=0,
            expires_at_ms=store.server_time_ms() + 10_000,
            call=CompletionCallParams(model='synthetic', messages=[]),
        )
        assert store.admit(request).disposition == 'admitted'
        claim = uuid4().hex
        assert (
            store.claim(
                owner, request.attempt_id, pool_id='pool', claim_id=claim
            ).disposition
            == 'claimed'
        )
        assert (
            store.fetch_payload(owner, request.attempt_id, claim_id=claim)['call'][
                'model'
            ]
            == 'synthetic'
        )
        started = store.authorize_start(owner, request.attempt_id, claim_id=claim)
        assert started.disposition == 'started'
        assert (
            store.finish(
                owner,
                request.attempt_id,
                claim_id=claim,
                execution_seq=started.execution_seq,
                result={'answer': 'synthetic'},
            ).disposition
            == 'finished'
        )
        assert store.read_result(producer, request.attempt_id) == {
            'answer': 'synthetic'
        }
        assert store.ack_result(producer, request.attempt_id).disposition == 'acked'
    finally:
        store.close()
