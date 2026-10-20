from __future__ import annotations

import hashlib
import importlib.util
import json
import sys

import pytest
from conftest import ROOT, QueryFixture

SPEC = importlib.util.spec_from_file_location(
    'organization_provisioning',
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
NIL = '00000000-0000-0000-0000-000000000000'


def operator(fixture: QueryFixture):
    def query(sql: str) -> str:
        result = fixture.query(sql)
        assert result.returncode == 0, result.stderr
        return result.stdout

    return query


def test_verified_organization_ingestion_does_not_require_workspace(
    clickhouse: QueryFixture,
) -> None:
    result = clickhouse.query(
        f"INSERT INTO telemetry.logs (organization_id, identity_state, identity_version, Timestamp, Body) "
        "VALUES ('33333333-3333-4333-8333-333333333333', 'verified', 1, now64(9), 'organization-only')"
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope='module')
def organization_scoped(clickhouse: QueryFixture) -> QueryFixture:
    query = operator(clickhouse)
    # Keep disposable-engine canonical DDL available for reviewed regeneration.
    (clickhouse.folder / 'organization-schema-manifest.json').write_text(
        json.dumps(provision.schema_snapshot(query), indent=2) + '\n'
    )
    for organization, workspace in [
        (ORG_A, ''),
        (ORG_A, WS_A1),
        (ORG_A, WS_A2),
        (ORG_B, ''),
    ]:
        values = f"'{organization}', '{workspace}', 'verified', 1"
        query(
            f"INSERT INTO telemetry.spans (organization_id, workspace_id, identity_state, identity_version, Timestamp, TraceId, SpanId, StatusCode, SpanAttributes) "
            f"VALUES ({values}, now64(9), 'duplicate-trace', 'duplicate-span', 'ERROR', map('llm.model_name', 'shared-model', 'llm.token_count.prompt', '1'))"
        )
        query(
            f"INSERT INTO telemetry.logs (organization_id, workspace_id, identity_state, identity_version, Timestamp, TraceId, SpanId, SeverityText, LogAttributes) "
            f"VALUES ({values}, now64(9), 'duplicate-trace', 'duplicate-span', 'ERROR', map('event.id', 'duplicate-event', 'event.source_type', 'job', 'job.id', 'duplicate-job'))"
        )
        for kind in ('gauge', 'sum', 'histogram', 'exponential_histogram', 'summary'):
            query(
                f"INSERT INTO telemetry.metrics_{kind} (organization_id, workspace_id, identity_state, identity_version, TimeUnix, MetricName) "
                f"VALUES ({values}, now64(9), 'shared-metric')"
            )
        query(
            f"INSERT INTO telemetry.annotations (organization_id, workspace_id, identity_state, identity_version, id, trace_id, span_id, name) "
            f"VALUES ({values}, 'cccccccc-cccc-4ccc-8ccc-cccccccccccc', 'duplicate-trace', 'duplicate-span', 'shared-score')"
        )
    query(
        f"INSERT INTO telemetry.spans (Timestamp, TraceId, SpanAttributes) VALUES (now64(9), 'forged', "
        f"map('marie.organization_id', '{ORG_A}', 'marie.identity_state', 'verified'))"
    )
    query(
        f"INSERT INTO telemetry.spans (organization_id, identity_state, Timestamp) VALUES ('{ORG_A}', 'platform', now64(9))"
    )
    for organization in (ORG_A, ORG_B):
        provision.reconcile(
            query,
            provision.Scope(organization, '', 1),
            hashlib.sha256(clickhouse.password.encode()).hexdigest(),
            activate=True,
        )
    return clickhouse


@pytest.mark.parametrize('organization,expected', [(ORG_A, 3), (ORG_B, 1)])
def test_all_targets_and_views_enforce_organization_boundary(
    organization_scoped: QueryFixture, organization: str, expected: int
) -> None:
    principal = provision.Scope(organization, '', 1).principal
    for table in provision.MANIFEST['tables']:
        result = organization_scoped.query(
            f"SELECT count(), countIf(organization_id != '{organization}' OR identity_state != 'verified') FROM telemetry.{table}",
            user=principal,
        )
        assert result.returncode == 0, (table, result.stderr)
        assert result.stdout.strip() == f'{expected}\t0', table
    for view in provision.MANIFEST['views']:
        result = organization_scoped.query(
            f"SELECT count() FROM workspace_query.{view} WHERE organization_id != '{organization}' OR identity_state != 'verified'",
            user=principal,
        )
        assert result.returncode == 0, (view, result.stderr)
        assert result.stdout.strip() == '0', view
    result = organization_scoped.query(
        'SELECT DISTINCT workspace_id FROM workspace_query.spans ORDER BY workspace_id',
        user=principal,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == (
        ['', WS_A1, WS_A2] if organization == ORG_A else ['']
    )


@pytest.mark.parametrize(
    'sql,expected',
    [
        (
            "SELECT count() FROM telemetry.spans WHERE 1 OR organization_id != organization_id",
            '3',
        ),
        (
            'SELECT count() FROM workspace_query.spans UNION ALL SELECT count() FROM telemetry.spans',
            '3\n3',
        ),
        (
            'SELECT count() FROM workspace_query.spans s JOIN telemetry.logs l ON s.TraceId = l.TraceId',
            '9',
        ),
        (
            'SELECT count() FROM telemetry.spans WHERE TraceId IN (SELECT TraceId FROM telemetry.logs)',
            '3',
        ),
    ],
)
def test_arbitrary_native_sql_cannot_expand_organization_scope(
    organization_scoped: QueryFixture, sql: str, expected: str
) -> None:
    result = organization_scoped.query(
        sql, user=provision.Scope(ORG_A, '', 1).principal
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    'organization,workspace,version',
    [
        ('', '', 1),
        ('forged', '', 1),
        (NIL, '', 1),
        (ORG_A.replace('-', ''), '', 1),
        (ORG_A, 'forged', 1),
        (ORG_A, NIL, 1),
        (ORG_A, WS_A1.upper(), 1),
        (ORG_A, '', 0),
    ],
)
def test_all_physical_targets_reject_invalid_verified_scope(
    clickhouse: QueryFixture, organization: str, workspace: str, version: int
) -> None:
    for table in provision.MANIFEST['tables']:
        result = clickhouse.query(
            f"INSERT INTO telemetry.{table} (organization_id, workspace_id, identity_state, identity_version) "
            f"VALUES ('{organization}', '{workspace}', 'verified', {version})"
        )
        assert result.returncode != 0, table
        assert 'valid_verified' in result.stderr, (table, result.stderr)


@pytest.mark.parametrize(
    'sql',
    [
        'SELECT * FROM workspace_query.shared_spans',
        'SELECT * FROM telemetry.model_prices',
        'SELECT * FROM telemetry.capacity_ts',
        "SELECT dictGet('telemetry.model_price_dict', 'input_price', ('model', 'provider'))",
        'SELECT count() FROM telemetry.spans SETTINGS readonly = 0',
        'SELECT count() FROM telemetry.spans SETTINGS use_query_cache = 1',
        'SELECT query FROM system.processes',
        "INSERT INTO telemetry.logs (Body) VALUES ('forged')",
    ],
)
def test_organization_readers_keep_private_and_resource_restrictions(
    organization_scoped: QueryFixture, sql: str
) -> None:
    assert (
        organization_scoped.query(
            sql, user=provision.Scope(ORG_A, '', 1).principal
        ).returncode
        != 0
    )


def test_permissive_policy_cannot_widen_organization_boundary(
    organization_scoped: QueryFixture,
) -> None:
    query = operator(organization_scoped)
    scope = provision.Scope(ORG_A, '', 1)
    query(
        f'CREATE ROW POLICY organization_extra_read ON telemetry.spans AS PERMISSIVE USING 1 TO {scope.principal}'
    )
    try:
        result = organization_scoped.query(
            'SELECT count() FROM telemetry.spans', user=scope.principal
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == '3'
        with pytest.raises(RuntimeError, match='policies'):
            provision.verify(query, scope)
    finally:
        query('DROP ROW POLICY organization_extra_read ON telemetry.spans')


def test_constraint_upgrade_preserves_retained_tables_and_data(
    clickhouse: QueryFixture,
) -> None:
    query = operator(clickhouse)
    database = 'organization_scope_upgrade'
    query(f'CREATE DATABASE {database}')
    timestamps = {
        'spans': 'Timestamp',
        'logs': 'Timestamp',
        'metrics_gauge': 'TimeUnix',
        'metrics_sum': 'TimeUnix',
        'metrics_histogram': 'TimeUnix',
        'metrics_exponential_histogram': 'TimeUnix',
        'metrics_summary': 'TimeUnix',
        'annotations': 'created_at',
        'model_names': 'last_seen',
        'event_counts': 'timestamp_hour',
        'error_logs': 'Timestamp',
        'error_spans': 'Timestamp',
        'events': 'Timestamp',
        'job_events': 'Timestamp',
        'spans_trace_id_ts': 'Start',
    }
    for table in provision.MANIFEST['tables']:
        timestamp = 'now()' if table in ('model_names', 'event_counts') else 'now64(9)'
        count_column = ', count' if table == 'event_counts' else ''
        count_value = ', 1' if table == 'event_counts' else ''
        query(f'CREATE TABLE {database}.{table} AS telemetry.{table}')
        # Rehearse the old mandatory-workspace contract with retained rows.
        query(
            f'ALTER TABLE {database}.{table} DROP CONSTRAINT IF EXISTS valid_verified_scope, '
            "ADD CONSTRAINT valid_verified_scope CHECK identity_state != 'verified' OR "
            "(identity_version = 1 AND organization_id != '' AND workspace_id != '')"
        )
        query(
            f'INSERT INTO {database}.{table} (organization_id, workspace_id, identity_state, identity_version, {timestamps[table]}{count_column}) '
            f"VALUES ('{ORG_A}', '{WS_A1}', 'verified', 1, {timestamp}{count_value})"
        )
        assert query(f'SELECT count() FROM {database}.{table}').strip() == '1', table
    before_tables = query(
        f"SELECT name, uuid FROM system.tables WHERE database = '{database}' ORDER BY name"
    )
    before_parts = query(
        f"SELECT table, name, rows FROM system.parts WHERE database = '{database}' AND active ORDER BY table, name"
    )
    migration = (provision.DIRECTORY / 'migrate-organization-scope.sql').read_text()
    for _ in range(2):
        query(migration.replace('telemetry.', database + '.'))
    assert (
        query(
            f"SELECT name, uuid FROM system.tables WHERE database = '{database}' ORDER BY name"
        )
        == before_tables
    )
    assert (
        query(
            f"SELECT table, name, rows FROM system.parts WHERE database = '{database}' AND active ORDER BY table, name"
        )
        == before_parts
    )
    for table in provision.MANIFEST['tables']:
        timestamp = 'now()' if table in ('model_names', 'event_counts') else 'now64(9)'
        count_column = ', count' if table == 'event_counts' else ''
        count_value = ', 1' if table == 'event_counts' else ''
        assert query(f'SELECT count() FROM {database}.{table}').strip() == '1', table
        query(
            f'INSERT INTO {database}.{table} (organization_id, workspace_id, identity_state, identity_version, {timestamps[table]}{count_column}) '
            f"VALUES ('{ORG_A}', '', 'verified', 1, {timestamp}{count_value})"
        )
        rejected = clickhouse.query(
            f'INSERT INTO {database}.{table} (organization_id, workspace_id, identity_state, identity_version) '
            f"VALUES ('{NIL}', '', 'verified', 1)"
        )
        assert rejected.returncode != 0, table
        assert 'valid_verified_scope' in rejected.stderr, rejected.stderr
