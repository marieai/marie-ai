from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'workspace_provisioning',
    ROOT / 'config/clickhouse/workspace-isolation/provision.py',
)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

ORG = '11111111-1111-4111-8111-111111111111'
WORKSPACE = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'


def test_scope_principal_is_canonical_and_revision_bound() -> None:
    scope = module.Scope(ORG, WORKSPACE.upper(), 1)
    assert scope.workspace_id == WORKSPACE
    assert scope.principal == module.Scope(ORG, WORKSPACE, 1).principal
    assert scope.principal != module.Scope(ORG, WORKSPACE, 2).principal
    assert 'project' not in scope.principal


def test_organization_reader_has_distinct_revision_bound_principal() -> None:
    scope = module.Scope(ORG, '', 1)
    assert scope.workspace_id == ''
    assert (
        scope.principal
        == 'orgq_' + hashlib.sha256(f'{ORG}/1'.encode()).hexdigest()[:32]
    )
    assert scope.principal != module.Scope(ORG, '', 2).principal
    assert scope.principal != module.Scope(ORG, WORKSPACE, 1).principal
    assert module.expected_filter(scope) == (
        f"(organization_id = '{ORG}') AND "
        "(identity_state = 'verified') AND (identity_version = 1)"
    )


@pytest.mark.parametrize(
    'workspace', ['invalid', '00000000-0000-0000-0000-000000000000']
)
def test_optional_workspace_still_rejects_invalid_association(workspace: str) -> None:
    with pytest.raises(ValueError):
        module.Scope(ORG, workspace, 1)


def test_organization_suspend_cli_does_not_require_workspace(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(module, 'operator_query', lambda: lambda sql: '0')
    monkeypatch.setattr(
        sys,
        'argv',
        ['provision.py', 'suspend', '--organization', ORG, '--revision', '1'],
    )
    module.main()
    output = json.loads(capsys.readouterr().out)
    assert output['workspaceId'] == ''
    assert output['principal'].startswith('orgq_')
    assert output['drained'] is True


@pytest.mark.parametrize(
    'value',
    [
        '',
        'not-a-uuid',
        "x'; GRANT ALL ON *.* TO x; --",
        '00000000-0000-0000-0000-000000000000',
    ],
)
def test_reject_invalid_scope(value: str) -> None:
    with pytest.raises(ValueError):
        module.Scope(value, WORKSPACE, 1)


@pytest.mark.parametrize('revision', [0, -1, True, '1', 2**31])
def test_reject_invalid_revision(revision: object) -> None:
    with pytest.raises(ValueError):
        module.Scope(ORG, WORKSPACE, revision)


def test_preparation_never_makes_user_reachable() -> None:
    sql = module.prepare_sql(module.Scope(ORG, WORKSPACE, 1), 'a' * 64)
    assert 'HOST NONE' in sql
    assert 'HOST ANY' not in sql
    assert 'SQL SECURITY DEFINER' not in sql
    assert 'SELECT ON telemetry.*' not in sql
    assert 'AS RESTRICTIVE' in sql


def test_reject_unhashed_password() -> None:
    with pytest.raises(ValueError):
        module.prepare_sql(
            module.Scope(ORG, WORKSPACE, 1), "password'; DROP DATABASE otel;"
        )


@pytest.mark.parametrize('present', ['0', '1'])
def test_suspension_drains_queries_even_after_user_disappears(present: str) -> None:
    scope = module.Scope(ORG, WORKSPACE, 1)
    calls: list[str] = []

    def query(sql: str) -> str:
        calls.append(sql)
        return present if sql.startswith('SELECT count()') else ''

    module.suspend(query, scope)
    assert calls[-1] == f"KILL QUERY WHERE user = '{scope.principal}' SYNC"
    if present == '0':
        assert len(calls) == 2
    else:
        assert calls[1:4] == [
            f'ALTER USER IF EXISTS {scope.principal} HOST NONE',
            f'REVOKE ALL ON *.* FROM {scope.principal}',
            f'REVOKE ALL FROM {scope.principal}',
        ]


def test_suspension_does_not_acknowledge_a_failed_drain() -> None:
    scope = module.Scope(ORG, WORKSPACE, 1)

    def query(sql: str) -> str:
        if sql.startswith('KILL QUERY'):
            raise RuntimeError('Drain failed')
        return '0'

    with pytest.raises(RuntimeError, match='Drain failed'):
        module.suspend(query, scope)


@pytest.mark.parametrize('drain_succeeds', [False, True])
def test_suspend_cli_acknowledges_only_a_completed_drain(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    drain_succeeds: bool,
) -> None:
    def query(sql: str) -> str:
        if sql.startswith('KILL QUERY') and not drain_succeeds:
            raise RuntimeError('Drain failed')
        return '0'

    monkeypatch.setattr(module, 'operator_query', lambda: query)
    monkeypatch.setattr(
        sys,
        'argv',
        [
            'provision.py',
            'suspend',
            '--organization',
            ORG,
            '--workspace',
            WORKSPACE,
            '--revision',
            '1',
        ],
    )
    if not drain_succeeds:
        with pytest.raises(RuntimeError, match='Drain failed'):
            module.main()
        assert capsys.readouterr().out == ''
        return
    module.main()
    output = capsys.readouterr().out
    assert output
    assert json.loads(output) == {
        'organizationId': ORG,
        'workspaceId': WORKSPACE,
        'revision': 1,
        'principal': module.Scope(ORG, WORKSPACE, 1).principal,
        'state': 'suspended',
        'drained': True,
    }
