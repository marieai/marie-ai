from __future__ import annotations

import subprocess
import time
import uuid
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[3]
IMAGE = 'clickhouse/clickhouse-server@sha256:7234d193fbeb3a375f21f0bb99040396f68bb2e82cada4f4947d7fcf9638bbf3'


class QueryFixture:
    def __init__(self, name: str, password: str, folder: Path) -> None:
        self.name = name
        self.password = password
        self.system_password = password
        self.folder = folder

    def query(
        self, sql: str, reader: bool = False, *, user: str | None = None
    ) -> subprocess.CompletedProcess[str]:
        command = [
            'docker',
            'exec',
            '-i',
            self.name,
            'clickhouse-client',
            '--multiquery', '--password', self.password if reader or user else self.system_password,
            '--user', user or ('studio_query' if reader else 'marie'),
        ]
        return subprocess.run(
            command, input=sql, text=True, capture_output=True, timeout=50
        )

    def apply(self) -> None:
        script = ROOT / 'config/clickhouse/query-explorer/bootstrap.sh'
        if not script.is_file():
            return
        result = subprocess.run(
            [
                'docker',
                'exec',
                '-e',
                'QUERY_READER_PASSWORD_FILE=/run/query-reader',
                self.name,
                '/bin/sh',
                '/query-explorer/bootstrap.sh',
            ],
            text=True,
            capture_output=True,
            timeout=50,
        )
        assert result.returncode == 0, result.stderr


@pytest.fixture(scope='module')
def clickhouse(tmp_path_factory: pytest.TempPathFactory) -> Iterator[QueryFixture]:
    folder = tmp_path_factory.mktemp('query-explorer')
    password = uuid.uuid4().hex
    (folder / 'reader').write_text(password)
    name = 'query-explorer-test-' + uuid.uuid4().hex[:12]
    subprocess.run(
        ['docker', 'volume', 'create', name], check=True, capture_output=True
    )
    subprocess.run(
        [
            'docker',
            'run',
            '--detach',
            '--rm',
            '--name',
            name,
            '--network',
            'none',
            '--memory',
            '2g',
            '--cpus',
            '2',
            '--mount',
            f'type=volume,src={name},dst=/var/lib/clickhouse',
            '--mount',
            f'type=bind,src={ROOT / "config/clickhouse/telemetry-dictionary.xml"},dst=/etc/clickhouse-server/config.d/telemetry-dictionary.xml,readonly',
            '-e', 'CLICKHOUSE_USER=marie',
            '-e', f'CLICKHOUSE_PASSWORD={password}',
            '-e', 'CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1',
            '--mount',
            f'type=bind,src={folder / "reader"},dst=/run/query-reader,readonly',
            '--mount',
            f'type=bind,src={ROOT / "config/clickhouse/query-explorer"},dst=/query-explorer,readonly',
            IMAGE,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fixture = QueryFixture(name, password, folder)
    try:
        for _ in range(120):
            if fixture.query('SELECT 1').returncode == 0:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError('Disposable ClickHouse did not start')
        for source in ['config/clickhouse/workspace-isolation/schema.sql', 'config/clickhouse/telemetry-system.sql']:
            setup = fixture.query((ROOT / source).read_text())
            assert setup.returncode == 0, setup.stderr
        fixture.apply()
        yield fixture
    finally:
        subprocess.run(
            ['docker', 'stop', '--time', '3', name], check=False, capture_output=True
        )
        subprocess.run(
            ['docker', 'volume', 'rm', name], check=False, capture_output=True
        )
