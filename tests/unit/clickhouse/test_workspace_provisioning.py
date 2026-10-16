from __future__ import annotations

import importlib.util
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
