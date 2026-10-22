from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from conftest import ROOT, QueryFixture

SPEC = importlib.util.spec_from_file_location(
    'workspace_provisioning',
    ROOT / 'config/clickhouse/workspace-isolation/provision.py',
)
assert SPEC and SPEC.loader
provision = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = provision
SPEC.loader.exec_module(provision)
ORG_A = '11111111-1111-4111-8111-111111111111'
ORG_B = '22222222-2222-4222-8222-222222222222'
WS_A1 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa1'
WS_A2 = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaa2'
WS_B1 = 'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbb1'
SCOPES = [
    provision.Scope(ORG_A, WS_A1, 1),
    provision.Scope(ORG_A, WS_A2, 1),
    provision.Scope(ORG_B, WS_B1, 1),
]


def operator(fixture: QueryFixture):
    def query(sql: str) -> str:
        result = fixture.query(sql)
        assert result.returncode == 0, result.stderr
        return result.stdout

    return query


@pytest.fixture(scope='module')
def scoped(clickhouse: QueryFixture) -> QueryFixture:
    query = operator(clickhouse)
    query((provision.DIRECTORY / 'schema.sql').read_text())
    # Useful review artifact on failures; never exports payloads or credentials.
    snapshot = provision.schema_snapshot(query)
    (clickhouse.folder / 'schema-manifest.json').write_text(
        json.dumps(snapshot, indent=2) + '\n'
    )
    assert (provision.DIRECTORY / 'schema-manifest.json').exists(), (
        f'Review schema snapshot: {clickhouse.folder}'
    )
    for scope in SCOPES:
        provision.reconcile(
            query,
            scope,
            hashlib.sha256(clickhouse.password.encode()).hexdigest(),
            activate=True,
        )
    for scope in SCOPES:
        values = f"'{scope.organization_id}', '{scope.workspace_id}', 'verified', 1"
        query(
            f"INSERT INTO telemetry.spans (organization_id, workspace_id, identity_state, identity_version, Timestamp, TraceId, SpanId, StatusCode, SpanAttributes) VALUES ({values}, now64(9), 'duplicate-trace', 'duplicate-span', 'ERROR', map('llm.model_name', 'shared-model', 'llm.token_count.prompt', '1'))"
        )
        query(
            f"INSERT INTO telemetry.logs (organization_id, workspace_id, identity_state, identity_version, Timestamp, TraceId, SpanId, SeverityText, LogAttributes) VALUES ({values}, now64(9), 'duplicate-trace', 'duplicate-span', 'ERROR', map('event.id', 'duplicate-event', 'event.source_type', 'job', 'job.id', 'duplicate-job'))"
        )
        for kind in ('gauge', 'sum', 'histogram', 'exponential_histogram', 'summary'):
            query(
                f"INSERT INTO telemetry.metrics_{kind} (organization_id, workspace_id, identity_state, identity_version, TimeUnix, MetricName) VALUES ({values}, now64(9), 'shared-metric')"
            )
        query(
            f"INSERT INTO telemetry.annotations (organization_id, workspace_id, identity_state, identity_version, id, trace_id, span_id, name) VALUES ({values}, 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', 'duplicate-trace', 'duplicate-span', 'shared-score')"
        )
    query(
        "INSERT INTO telemetry.spans (Timestamp, TraceId, SpanId, SpanAttributes) VALUES (now64(9), 'forged', 'span', map('marie.organization_id', '"
        + ORG_A
        + "', 'marie.workspace_id', '"
        + WS_A1
        + "', 'marie.identity_state', 'verified'))"
    )
    query(
        f"INSERT INTO telemetry.spans (organization_id, workspace_id, identity_state, Timestamp) VALUES ('{ORG_A}', '{WS_A1}', 'platform', now64(9))"
    )
    return clickhouse


def test_all_shipped_targets_and_views_are_isolated(scoped: QueryFixture) -> None:
    for scope in SCOPES:
        for table in provision.MANIFEST['tables']:
            result = scoped.query(
                f'SELECT organization_id, workspace_id, identity_state FROM telemetry.{table}',
                user=scope.principal,
            )
            assert result.returncode == 0, (table, result.stderr)
            assert (
                result.stdout.strip()
                == f'{scope.organization_id}\t{scope.workspace_id}\tverified'
            ), table
        for view in provision.MANIFEST['views']:
            result = scoped.query(
                f'SELECT count() FROM workspace_query.{view}', user=scope.principal
            )
            assert result.returncode == 0, (view, result.stderr)
        assert (
            scoped.query(
                'SELECT count() FROM workspace_query.spans', user=scope.principal
            ).stdout.strip()
            == '1'
        )


@pytest.mark.parametrize(
    'sql',
    [
        'SELECT count() FROM telemetry.spans WHERE 1 OR workspace_id != workspace_id',
        'SELECT count() FROM workspace_query.spans UNION ALL SELECT count() FROM telemetry.spans',
        'SELECT count() FROM telemetry.spans s JOIN telemetry.logs l ON s.TraceId = l.TraceId',
        'SELECT count() FROM (SELECT * FROM telemetry.spans) WHERE TraceId IN (SELECT TraceId FROM telemetry.logs)',
    ],
)
def test_native_sql_cannot_expand_scope(scoped: QueryFixture, sql: str) -> None:
    result = scoped.query(sql, user=SCOPES[0].principal)
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {'1'}


@pytest.mark.parametrize(
    'sql',
    [
        'SELECT * FROM otel.otel_traces',
        'SELECT * FROM workspace_query.shared_spans',
        'SELECT * FROM telemetry.capacity_ts',
        'SELECT * FROM telemetry.events_mv',
        'SELECT * FROM telemetry.model_prices',
        "SELECT dictGet('telemetry.model_price_dict', 'input_price', ('model', 'provider'))",
        'SELECT query FROM system.processes',
        "SELECT * FROM url('http://127.0.0.1:8123', 'CSV', 'x String')",
        "SELECT * FROM file('/etc/passwd', 'LineAsString')",
        'SELECT * FROM remote(\'127.0.0.1\', system.one)',
        'SELECT count() FROM telemetry.spans SETTINGS readonly = 0',
        'SELECT count() FROM telemetry.spans SETTINGS use_query_cache = 1',
        'CREATE TABLE telemetry.escape (x String) ENGINE=Memory',
        "INSERT INTO telemetry.spans (TraceId) VALUES ('escape')",
        'SET ROLE studio_query',
    ],
)
def test_privilege_and_resource_escape_is_denied(
    scoped: QueryFixture, sql: str
) -> None:
    assert scoped.query(sql, user=SCOPES[0].principal).returncode != 0, sql


def test_missing_policy_is_detected_before_activation(scoped: QueryFixture) -> None:
    scope = SCOPES[0]
    query = operator(scoped)
    provision.suspend(query, scope)
    query(
        provision.prepare_sql(
            scope, hashlib.sha256(scoped.password.encode()).hexdigest()
        )
    )
    query(f'DROP ROW POLICY {scope.principal}_logs_scope ON telemetry.logs')
    with pytest.raises(RuntimeError, match='policies'):
        provision.verify(query, scope)
    assert scoped.query('SELECT 1', user=scope.principal).returncode != 0
    provision.reconcile(
        query,
        scope,
        hashlib.sha256(scoped.password.encode()).hexdigest(),
        activate=True,
    )
    assert (
        scoped.query(
            'SELECT count() FROM telemetry.logs', user=scope.principal
        ).stdout.strip()
        == '1'
    )


def test_unknown_future_columns_and_tables_are_not_granted(
    scoped: QueryFixture,
) -> None:
    query = operator(scoped)
    query('ALTER TABLE telemetry.spans ADD COLUMN future_secret String')
    try:
        assert (
            scoped.query(
                'SELECT future_secret FROM telemetry.spans', user=SCOPES[0].principal
            ).returncode
            != 0
        )
        with pytest.raises(RuntimeError, match='schema'):
            provision.reconcile(
                query,
                SCOPES[0],
                hashlib.sha256(scoped.password.encode()).hexdigest(),
                activate=True,
            )
        assert scoped.query('SELECT 1', user=SCOPES[0].principal).returncode != 0
    finally:
        query('ALTER TABLE telemetry.spans DROP COLUMN future_secret')
        provision.reconcile(
            query,
            SCOPES[0],
            hashlib.sha256(scoped.password.encode()).hexdigest(),
            activate=True,
        )


def test_suspend_revokes_grants_not_only_new_network_connections(
    scoped: QueryFixture,
) -> None:
    scope = SCOPES[0]
    query = operator(scoped)
    try:
        provision.suspend(query, scope)
        assert (
            query(
                f"SELECT count() FROM system.grants WHERE user_name = '{scope.principal}'"
            ).strip()
            == '0'
        )
    finally:
        provision.reconcile(
            query,
            scope,
            hashlib.sha256(scoped.password.encode()).hexdigest(),
            activate=True,
        )


def test_broad_permissive_policy_cannot_override_workspace_scope(
    scoped: QueryFixture,
) -> None:
    query = operator(scoped)
    query(
        f'CREATE ROW POLICY extra_read ON telemetry.spans AS PERMISSIVE USING 1 TO {SCOPES[0].principal}'
    )
    try:
        result = scoped.query(
            'SELECT count() FROM telemetry.spans', user=SCOPES[0].principal
        )
        assert result.stdout.strip() == '1'
        with pytest.raises(RuntimeError, match='policies'):
            provision.verify(query, SCOPES[0])
    finally:
        query('DROP ROW POLICY extra_read ON telemetry.spans')


def test_role_and_resource_drift_are_not_accepted(scoped: QueryFixture) -> None:
    scope = SCOPES[0]
    query = operator(scoped)
    query(f'CREATE ROLE drift_role; GRANT drift_role TO {scope.principal}')
    try:
        with pytest.raises(RuntimeError, match='roles'):
            provision.verify(query, scope)
    finally:
        query(f'REVOKE drift_role FROM {scope.principal}; DROP ROLE drift_role')
    query(
        f'ALTER SETTINGS PROFILE {scope.principal}_limits SETTINGS readonly = 0 CONST TO {scope.principal}'
    )
    try:
        with pytest.raises(RuntimeError, match='limits'):
            provision.verify(query, scope)
    finally:
        provision.reconcile(
            query,
            scope,
            hashlib.sha256(scoped.password.encode()).hexdigest(),
            activate=True,
        )
