"""Prepare and verify workspace readers using an operator-owned ClickHouse client."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit
from uuid import UUID

DIRECTORY = Path(__file__).resolve().parent
MANIFEST = json.loads((DIRECTORY / 'manifest.json').read_text())
LIMITS = {
    'readonly': 1,
    'log_queries': 0,
    'max_execution_time': 30,
    'timeout_overflow_mode': 'throw',
    'max_result_rows': 20000,
    'max_result_bytes': 16777216,
    'result_overflow_mode': 'throw',
    'max_rows_to_read': 100000000,
    'max_bytes_to_read': 10737418240,
    'read_overflow_mode': 'throw',
    'max_memory_usage': 1073741824,
    'max_memory_usage_for_user': 2147483648,
    'max_threads': 2,
    'max_concurrent_queries_for_user': 4,
    'max_query_size': 65536,
    'use_query_cache': 0,
    'enable_reads_from_query_cache': 0,
    'enable_writes_to_query_cache': 0,
    'output_format_json_quote_64bit_integers': 1,
    'output_format_json_quote_decimals': 1,
    'cancel_http_readonly_queries_on_client_close': 1,
    'replace_running_query': 1,
    'replace_running_query_max_wait_ms': 5000,
}
Query = Callable[[str], str]


def canonical_uuid(value: str) -> str:
    """Validate a non-nil UUID at the control-plane boundary."""
    parsed = UUID(value)
    if not parsed.int:
        raise ValueError('Scope UUID cannot be nil')
    return str(parsed)


@dataclass(frozen=True)
class Scope:
    organization_id: str
    workspace_id: str
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, 'organization_id', canonical_uuid(self.organization_id)
        )
        object.__setattr__(self, 'workspace_id', canonical_uuid(self.workspace_id))
        if type(self.revision) is not int or not 1 <= self.revision < 2**31:
            raise ValueError('Revision must be a positive int32')

    @property
    def principal(self) -> str:
        identity = f'{self.organization_id}/{self.workspace_id}/{self.revision}'
        return 'wsq_' + hashlib.sha256(identity.encode()).hexdigest()[:32]


def literal(value: str | int) -> str:
    """Quote internal manifest constants, not caller SQL."""
    return (
        str(value) if isinstance(value, int) else "'" + value.replace("'", "''") + "'"
    )


def expected_filter(scope: Scope) -> str:
    # Same canonical expression ClickHouse stores in system.row_policies.
    return (
        f"(organization_id = '{scope.organization_id}') AND "
        f"(workspace_id = '{scope.workspace_id}') AND "
        "(identity_state = 'verified') AND (identity_version = 1)"
    )


def suspend(query: Query, scope: Scope) -> None:
    """Block new connections and stop in-flight queries before changing policies."""
    if (
        query(
            f"SELECT count() FROM system.users WHERE name = '{scope.principal}'"
        ).strip()
        != '0'
    ):
        query(f'ALTER USER IF EXISTS {scope.principal} HOST NONE')
        query(f'REVOKE ALL ON *.* FROM {scope.principal}')
        query(f'REVOKE ALL FROM {scope.principal}')
    # Queries can outlive DROP USER.
    query(f"KILL QUERY WHERE user = '{scope.principal}' SYNC")


def prepare_sql(scope: Scope, password_hash: str) -> str:
    """Render a bounded grant/policy manifest; the user stays unreachable."""
    if not re.fullmatch('[0-9a-f]{64}', password_hash):
        raise ValueError('Expected a SHA256 password hash')
    user = scope.principal
    statements = [
        f"CREATE USER IF NOT EXISTS {user} IDENTIFIED WITH sha256_hash BY '{password_hash}' HOST NONE",
        f"ALTER USER {user} IDENTIFIED WITH sha256_hash BY '{password_hash}' HOST NONE DEFAULT DATABASE workspace_query",
        f'REVOKE ALL ON *.* FROM {user}',
        f'CREATE SETTINGS PROFILE IF NOT EXISTS {user}_limits',
        f'ALTER SETTINGS PROFILE {user}_limits SETTINGS '
        + ', '.join(f'{key} = {literal(value)} CONST' for key, value in LIMITS.items())
        + f' TO {user}',
        f'CREATE QUOTA IF NOT EXISTS {user}_hourly',
        f'ALTER QUOTA {user}_hourly FOR INTERVAL 1 HOUR MAX queries = 120, execution_time = 600 TO {user}',
    ]
    for table, entry in MANIFEST['tables'].items():
        name = f'{user}_{table}'
        statements.extend(
            [
                f'CREATE ROW POLICY IF NOT EXISTS {name}_read ON telemetry.{table} AS PERMISSIVE USING 1 TO {user}',
                f'ALTER ROW POLICY {name}_read ON telemetry.{table} AS PERMISSIVE USING 1 TO {user}',
                f'CREATE ROW POLICY IF NOT EXISTS {name}_scope ON telemetry.{table} AS RESTRICTIVE USING {expected_filter(scope)} TO {user}',
                f'ALTER ROW POLICY {name}_scope ON telemetry.{table} AS RESTRICTIVE USING {expected_filter(scope)} TO {user}',
                'GRANT SELECT('
                + ', '.join(f'`{column}`' for column in entry['columns'])
                + f') ON telemetry.{table} TO {user}',
            ]
        )
    for view in MANIFEST['views']:
        statements.append(f'GRANT SELECT ON workspace_query.{view} TO {user}')
    return ';\n'.join(statements) + ';\n'


def rows(query: Query, sql: str) -> list[dict]:
    """Read bounded operator metadata without exposing it to tenant principals."""
    return [
        json.loads(line)
        for line in query(sql + ' FORMAT JSONEachRow').splitlines()
        if line
    ]


def schema_snapshot(query: Query) -> list[dict]:
    """Fingerprint every prepared table, MV and view including its expressions."""
    return rows(
        query,
        "SELECT database, name, create_table_query FROM system.tables WHERE database IN ('telemetry', 'workspace_query') ORDER BY database, name",
    )


def verify(query: Query, scope: Scope) -> str:
    """Refuse schema, role, policy, profile, quota or grant drift before activation."""
    expected = json.loads((DIRECTORY / 'schema-manifest.json').read_text())
    if schema_snapshot(query) != expected:
        raise RuntimeError('Prepared schema differs from the reviewed manifest')
    user = scope.principal
    if rows(
        query,
        f"SELECT granted_role_name FROM system.role_grants WHERE user_name = '{user}'",
    ):
        raise RuntimeError('Workspace readers cannot inherit roles')
    access = rows(
        query,
        f"SELECT access_type, database, table, column, is_partial_revoke, grant_option FROM system.grants WHERE user_name = '{user}' ORDER BY database, table, column",
    )
    actual = {
        (
            r['access_type'],
            r['database'],
            r['table'],
            r['column'],
            r['is_partial_revoke'],
            r['grant_option'],
        )
        for r in access
    }
    required = {
        ('SELECT', 'telemetry', table, column, 0, 0)
        for table, entry in MANIFEST['tables'].items()
        for column in entry['columns']
    }
    required |= {
        ('SELECT', 'workspace_query', view, None, 0, 0) for view in MANIFEST['views']
    }
    if actual != required:
        raise RuntimeError(
            'Reader grants differ from the explicit column/view manifest'
        )
    policies = rows(
        query,
        f"SELECT short_name, database, table, is_restrictive, select_filter, apply_to_all, apply_to_list, apply_to_except FROM system.row_policies WHERE has(apply_to_list, '{user}') ORDER BY table, short_name",
    )
    required_policies = []
    for table in MANIFEST['tables']:
        for suffix, restrictive, expression in [
            ('read', 0, '1'),
            ('scope', 1, expected_filter(scope)),
        ]:
            required_policies.append(
                dict(
                    short_name=f'{user}_{table}_{suffix}',
                    database='telemetry',
                    table=table,
                    is_restrictive=restrictive,
                    select_filter=expression,
                    apply_to_all=0,
                    apply_to_list=[user],
                    apply_to_except=[],
                )
            )
    if policies != sorted(
        required_policies, key=lambda r: (r['table'], r['short_name'])
    ):
        raise RuntimeError('Workspace row policies are missing or differ from scope')
    # Profiles/quotas are unique per revision. Reject additional assignments too.
    profiles = rows(
        query,
        f"SELECT name FROM system.settings_profiles WHERE has(apply_to_list, '{user}') OR apply_to_all = 1 ORDER BY name",
    )
    if profiles != [{'name': f'{user}_limits'}]:
        raise RuntimeError('Unexpected effective profile assignments')
    if rows(
        query,
        f"SELECT setting_name FROM system.settings_profile_elements WHERE user_name = '{user}'",
    ):
        raise RuntimeError('Unexpected direct reader settings')
    elements = rows(
        query,
        f"SELECT setting_name, value, writability, inherit_profile FROM system.settings_profile_elements WHERE profile_name = '{user}_limits' ORDER BY index",
    )
    if elements != [
        dict(
            setting_name=key,
            value=str(value),
            writability='CONST',
            inherit_profile=None,
        )
        for key, value in LIMITS.items()
    ]:
        raise RuntimeError('Workspace resource limits differ from the manifest')
    quotas = rows(
        query,
        f"SELECT name FROM system.quotas WHERE has(apply_to_list, '{user}') OR apply_to_all = 1 ORDER BY name",
    )
    limits = rows(
        query,
        f"SELECT toUInt32(duration) AS duration, toUInt32(max_queries) AS max_queries, max_execution_time FROM system.quota_limits WHERE quota_name = '{user}_hourly'",
    )
    if quotas != [{'name': f'{user}_hourly'}] or limits != [
        {'duration': 3600, 'max_queries': 120, 'max_execution_time': 600}
    ]:
        raise RuntimeError('Unexpected workspace quota')
    artifact = (DIRECTORY / 'schema-manifest.json').read_bytes()
    return hashlib.sha256(artifact).hexdigest()


def reconcile(
    query: Query, scope: Scope, password_hash: str, *, activate: bool = False
) -> str:
    """Suspend, apply and verify; failure never restores reachability."""
    suspend(query, scope)
    for statement in prepare_sql(scope, password_hash).split(';\n'):
        if statement.strip():
            query(statement)
    artifact_hash = verify(query, scope)
    if activate:
        query(f'ALTER USER {scope.principal} HOST ANY')
    return artifact_hash


def operator_query() -> Query:
    """Use explicit operator credentials; never use application/default fallbacks."""
    endpoint = os.environ['CLICKHOUSE_URL']
    url = urlsplit(endpoint)
    if (
        url.scheme not in ('http', 'https')
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ('', '/')
    ):
        raise ValueError('CLICKHOUSE_URL must be a plain operator server URL')
    auth = base64.b64encode(
        (
            os.environ['CLICKHOUSE_USER'] + ':' + os.environ['CLICKHOUSE_PASSWORD']
        ).encode()
    ).decode()

    def query(sql: str) -> str:
        request = urllib.request.Request(
            endpoint, data=sql.encode(), headers={'Authorization': 'Basic ' + auth}
        )
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                result = response.read(2**20 + 1)
                if len(result) > 2**20:
                    raise RuntimeError('Operator metadata response exceeded its bound')
                return result.decode()
        except urllib.error.HTTPError as error:
            # Do not include SQL, credentials or arbitrary engine payload in output.
            raise RuntimeError(
                f'ClickHouse operator request failed ({error.code})'
            ) from None

    return query


def main() -> None:
    """Run operator preparation, verification, activation or suspension."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'activate', 'verify', 'suspend'])
    parser.add_argument('--organization', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--revision', required=True, type=int)
    args = parser.parse_args()
    scope = Scope(args.organization, args.workspace, args.revision)
    query = operator_query()
    if args.action == 'suspend':
        suspend(query, scope)
        print(
            json.dumps(
                {
                    'organizationId': scope.organization_id,
                    'workspaceId': scope.workspace_id,
                    'revision': scope.revision,
                    'principal': scope.principal,
                    'state': 'suspended',
                    'drained': True,
                }
            )
        )
        return
    if args.action == 'verify':
        artifact = verify(query, scope)
    else:
        password = sys.stdin.buffer.readline(1025).rstrip(b'\r\n')
        if not 32 <= len(password) <= 1024:
            raise ValueError(
                'Provide a generated reader secret (32–1024 bytes) on stdin'
            )
        artifact = reconcile(
            query,
            scope,
            hashlib.sha256(password).hexdigest(),
            activate=args.action == 'activate',
        )
    print(
        json.dumps(
            {
                'organizationId': scope.organization_id,
                'workspaceId': scope.workspace_id,
                'revision': scope.revision,
                'principal': scope.principal,
                'schemaHash': artifact,
                'state': 'verified',
            }
        )
    )


if __name__ == '__main__':
    main()
