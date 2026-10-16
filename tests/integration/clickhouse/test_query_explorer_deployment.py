from __future__ import annotations

import subprocess
import time
import uuid

import pytest
from conftest import IMAGE, ROOT, QueryFixture


def test_reader_uses_explicit_definer_and_survives_repeat_upgrade(clickhouse: QueryFixture) -> None:
    before = clickhouse.query('SELECT currentUser(), count() FROM query.spans', reader=True)
    assert before.returncode == 0, before.stderr
    assert before.stdout.strip() == 'studio_query\t0'
    clickhouse.apply()
    after = clickhouse.query('SELECT currentUser(), count() FROM query.spans', reader=True)
    assert after.returncode == 0, after.stderr
    assert after.stdout == before.stdout


@pytest.mark.parametrize('sql', [
    'SELECT count() FROM otel.otel_traces',
    'SELECT count() FROM system.users',
    "SELECT dictGet('otel.model_price_dict', 'input_price', ('x', ''))",
    'CREATE TABLE query.bad (x UInt64) ENGINE=Memory',
    "INSERT INTO otel.otel_traces (TraceId) VALUES ('forbidden')",
    'SELECT 1 SETTINGS max_execution_time=60',
    'SELECT 1 SETTINGS max_result_rows=1000000',
    'SELECT 1 SETTINGS readonly=0',
    "SELECT * FROM file('forbidden', 'CSV', 'x String')",
    "SELECT * FROM url('http://127.0.0.1:1', 'CSV', 'x String')",
])
def test_reader_denies_privilege_and_limit_overrides(clickhouse: QueryFixture, sql: str) -> None:
    result = clickhouse.query(sql, reader=True)
    assert result.returncode != 0
    assert any(code in result.stderr for code in ('ACCESS_DENIED', 'READONLY', 'SETTING_CONSTRAINT_VIOLATION')), result.stderr


def test_effective_limits_and_current_manifest(clickhouse: QueryFixture) -> None:
    result = clickhouse.query("SELECT getSetting('readonly'), getSetting('max_execution_time'), getSetting('max_threads')", reader=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '1\t30\t2'
    manifest = clickhouse.query('SELECT version, length(artifact_hash) FROM query.schema_version', reader=True)
    assert manifest.returncode == 0, manifest.stderr
    assert manifest.stdout.strip() == '1\t64'


def test_existing_volume_restart_rotation_and_rollback(clickhouse: QueryFixture) -> None:
    inserted = clickhouse.query("INSERT INTO otel.otel_logs (Timestamp, Body) VALUES (now64(), 'persistent-lifecycle-fixture')")
    assert inserted.returncode == 0, inserted.stderr
    subprocess.run(['docker', 'restart', '--time', '3', clickhouse.name], check=True, capture_output=True)
    for _ in range(120):
        if clickhouse.query('SELECT 1').returncode == 0:
            break
        time.sleep(0.1)
    persisted = clickhouse.query("SELECT count() FROM query.logs WHERE Body='persistent-lifecycle-fixture'", reader=True)
    assert persisted.returncode == 0, persisted.stderr
    assert persisted.stdout.strip() == '1'
    old_password = clickhouse.password
    new_password = uuid.uuid4().hex
    (clickhouse.folder / 'reader').write_text(new_password)
    clickhouse.apply()
    assert clickhouse.query('SELECT count() FROM query.logs', reader=True).returncode != 0
    clickhouse.password = new_password
    assert clickhouse.query('SELECT count() FROM query.logs', reader=True).returncode == 0
    rollback = subprocess.run(['docker', 'exec', clickhouse.name, '/bin/sh', '/query-explorer/rollback.sh'], text=True, capture_output=True)
    assert rollback.returncode == 0, rollback.stderr
    assert clickhouse.query('SELECT count() FROM query.logs', reader=True).returncode != 0
    ingestion = clickhouse.query("INSERT INTO otel.otel_logs (Timestamp, Body) VALUES (now64(), 'ingestion-after-reader-rollback')")
    assert ingestion.returncode == 0, ingestion.stderr
    clickhouse.apply()
    restored = clickhouse.query("SELECT count() FROM query.logs WHERE Body='ingestion-after-reader-rollback'", reader=True)
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == '1'
    assert old_password != new_password


def test_local_bootstrap_generates_and_reuses_private_reader_password(clickhouse: QueryFixture) -> None:
    def apply_without_extra_parameters() -> str:
        return subprocess.run(
            ['docker', 'run', '--rm', '--network', f'container:{clickhouse.name}', '--user', '101:101',
             '--memory', '256m', '--cpus', '1', '--entrypoint', '/bin/sh',
             '--mount', f'type=bind,src={ROOT / "config/clickhouse/query-explorer"},dst=/query-explorer,readonly',
             '--mount', f'type=volume,src={clickhouse.name},dst=/var/lib/clickhouse',
             IMAGE, '/query-explorer/bootstrap.sh'],
            text=True, capture_output=True, check=True,
        ).stdout

    previous = clickhouse.password
    try:
        first_output = apply_without_extra_parameters()
        password = subprocess.run(
            ['docker', 'exec', clickhouse.name, 'cat', '/var/lib/clickhouse/query-explorer/reader-password'],
            text=True, capture_output=True, check=True,
        ).stdout.strip()
        assert len(password) == 64
        assert password not in first_output
        clickhouse.password = password
        assert clickhouse.query('SELECT count() FROM query.logs', reader=True).returncode == 0
        apply_without_extra_parameters()
        assert clickhouse.query('SELECT count() FROM query.logs', reader=True).returncode == 0
        mode = subprocess.run(
            ['docker', 'exec', clickhouse.name, 'stat', '-c', '%a', '/var/lib/clickhouse/query-explorer/reader-password'],
            text=True, capture_output=True, check=True,
        ).stdout.strip()
        assert mode == '600'
    finally:
        clickhouse.password = previous
        clickhouse.apply()
