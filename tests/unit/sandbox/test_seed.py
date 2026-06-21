"""
Unit tests for marie.sandbox.seed — idempotency and record creation.

All tests use an in-memory fake-database (no real Postgres needed) built from
fake psycopg objects, following the same pattern used in
tests/unit/scheduler/repository/test_job_repository.py.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from marie.auth.api_key_manager import KeyGenerator
from marie.sandbox.seed import (
    SeedResult,
    _upsert_admin_user,
    _upsert_api_key,
    _upsert_org,
    _upsert_workspace,
    seed_defaults,
)

# ----------------------------------------------------------------- fakes ----


class _FakeCursor:
    """Cursor that executes against an in-memory table dict."""

    def __init__(self, tables: dict[str, list[dict]]):
        self._tables = tables
        self._rows: list = []

    def execute(self, sql: str, params: tuple = ()) -> None:
        sql_stripped = ' '.join(sql.split()).lower()

        if 'sandbox_organizations' in sql_stripped and 'insert into' in sql_stripped:
            name, slug = params
            existing = [r for r in self._tables['orgs'] if r['slug'] == slug]
            if not existing:
                row = {'id': uuid.uuid4(), 'name': name, 'slug': slug}
                self._tables['orgs'].append(row)
                self._rows = [(row['id'],)]
            else:
                self._rows = []  # ON CONFLICT DO NOTHING

        elif 'sandbox_organizations' in sql_stripped and 'select' in sql_stripped:
            slug = params[0]
            existing = [r for r in self._tables['orgs'] if r['slug'] == slug]
            self._rows = [(r['id'],) for r in existing]

        elif 'sandbox_workspaces' in sql_stripped and 'insert into' in sql_stripped:
            org_id, name, slug = params
            existing = [
                r
                for r in self._tables['workspaces']
                if r['org_id'] == org_id and r['slug'] == slug
            ]
            if not existing:
                row = {'id': uuid.uuid4(), 'org_id': org_id, 'name': name, 'slug': slug}
                self._tables['workspaces'].append(row)
                self._rows = [(row['id'],)]
            else:
                self._rows = []

        elif 'sandbox_workspaces' in sql_stripped and 'select' in sql_stripped:
            org_id, slug = params
            existing = [
                r
                for r in self._tables['workspaces']
                if r['org_id'] == org_id and r['slug'] == slug
            ]
            self._rows = [(r['id'],) for r in existing]

        elif 'sandbox_admin_users' in sql_stripped and 'insert into' in sql_stripped:
            org_id, username, email = params
            existing = [
                r for r in self._tables['users'] if r['username'] == username
            ]
            if not existing:
                row = {
                    'id': uuid.uuid4(),
                    'org_id': org_id,
                    'username': username,
                    'email': email,
                }
                self._tables['users'].append(row)
                self._rows = [(row['id'],)]
            else:
                self._rows = []

        elif 'sandbox_admin_users' in sql_stripped and 'select' in sql_stripped:
            username = params[0]
            existing = [r for r in self._tables['users'] if r['username'] == username]
            self._rows = [(r['id'],) for r in existing]

        elif 'sandbox_api_keys' in sql_stripped and 'insert into' in sql_stripped:
            user_id, org_id, name, api_key = params
            # ON CONFLICT (org_id, name) DO NOTHING
            existing = [
                r
                for r in self._tables['keys']
                if r['org_id'] == org_id and r['name'] == name
            ]
            if not existing:
                row = {
                    'id': uuid.uuid4(),
                    'user_id': user_id,
                    'org_id': org_id,
                    'name': name,
                    'api_key': api_key,
                }
                self._tables['keys'].append(row)
                self._rows = [(row['id'], row['api_key'])]
            else:
                self._rows = []

        elif 'sandbox_api_keys' in sql_stripped and 'select' in sql_stripped:
            # SELECT id, api_key FROM ... WHERE org_id = %s AND name = %s
            org_id, name = params
            existing = [
                r
                for r in self._tables['keys']
                if r['org_id'] == org_id and r['name'] == name
            ]
            self._rows = [(r['id'], r['api_key']) for r in existing]

        else:
            self._rows = []

    def fetchall(self) -> list:
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _FakeTransaction:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _FakeConnection:
    def __init__(self, tables: dict[str, list[dict]]):
        self._tables = tables
        self._cursor = _FakeCursor(tables)

    def cursor(self):
        return self._cursor

    def transaction(self):
        return _FakeTransaction()

    def close(self):
        pass


def _make_tables() -> dict[str, list[dict]]:
    return {'orgs': [], 'workspaces': [], 'users': [], 'keys': []}


def _fake_pool(tables: dict[str, list[dict]]):
    """Return a context-manager-compatible fake pool."""
    conn = _FakeConnection(tables)

    class _Pool:
        def connection(self):
            class _CM:
                def __enter__(self_inner):
                    return conn

                def __exit__(self_inner, *_):
                    pass

            return _CM()

        def close(self):
            pass

    return _Pool()


# ------------------------------------------------------------------ tests ---


class TestKeyGenerator:
    def test_generate_key_has_correct_prefix(self):
        key = KeyGenerator.generate_key('mas_')
        assert key.startswith('mas_')

    def test_generate_key_passes_validate(self):
        key = KeyGenerator.generate_key('mas_')
        assert KeyGenerator.validate_key(key)

    def test_mau_prefix_also_valid(self):
        key = KeyGenerator.generate_key('mau_')
        assert KeyGenerator.validate_key(key)

    def test_short_key_is_invalid(self):
        assert not KeyGenerator.validate_key('mas_tooshort')

    def test_wrong_prefix_is_invalid(self):
        key = KeyGenerator.generate_key('bad_')
        # Force correct length but wrong prefix
        padded = 'bad_' + 'x' * 54
        assert not KeyGenerator.validate_key(padded)


class TestUpsertHelpers:
    def _conn(self) -> _FakeConnection:
        return _FakeConnection(_make_tables())

    def test_upsert_org_creates_row(self):
        conn = self._conn()
        org_id = _upsert_org(conn, 'marie_scheduler', 'my-org', 'My Org')
        assert isinstance(org_id, uuid.UUID)

    def test_upsert_org_is_idempotent(self):
        conn = self._conn()
        id1 = _upsert_org(conn, 'marie_scheduler', 'my-org', 'My Org')
        id2 = _upsert_org(conn, 'marie_scheduler', 'my-org', 'My Org')
        assert id1 == id2

    def test_upsert_workspace_creates_row(self):
        conn = self._conn()
        org_id = _upsert_org(conn, 'marie_scheduler', 'org', 'Org')
        ws_id = _upsert_workspace(conn, 'marie_scheduler', org_id, 'default', 'Default')
        assert isinstance(ws_id, uuid.UUID)

    def test_upsert_workspace_is_idempotent(self):
        conn = self._conn()
        org_id = _upsert_org(conn, 'marie_scheduler', 'org', 'Org')
        ws_id1 = _upsert_workspace(conn, 'marie_scheduler', org_id, 'default', 'Default')
        ws_id2 = _upsert_workspace(conn, 'marie_scheduler', org_id, 'default', 'Default')
        assert ws_id1 == ws_id2

    def test_upsert_admin_user_creates_row(self):
        conn = self._conn()
        org_id = _upsert_org(conn, 'marie_scheduler', 'org', 'Org')
        user_id = _upsert_admin_user(
            conn, 'marie_scheduler', org_id, 'admin', 'admin@local'
        )
        assert isinstance(user_id, uuid.UUID)

    def test_upsert_admin_user_is_idempotent(self):
        conn = self._conn()
        org_id = _upsert_org(conn, 'marie_scheduler', 'org', 'Org')
        uid1 = _upsert_admin_user(
            conn, 'marie_scheduler', org_id, 'admin', 'admin@local'
        )
        uid2 = _upsert_admin_user(
            conn, 'marie_scheduler', org_id, 'admin', 'admin@local'
        )
        assert uid1 == uid2

    def test_upsert_api_key_creates_row(self):
        conn = self._conn()
        org_id = _upsert_org(conn, 'marie_scheduler', 'org', 'Org')
        user_id = _upsert_admin_user(
            conn, 'marie_scheduler', org_id, 'admin', 'admin@local'
        )
        key = KeyGenerator.generate_key('mas_')
        key_id, resolved_key = _upsert_api_key(
            conn, 'marie_scheduler', user_id, org_id, 'sandbox-admin', key
        )
        assert isinstance(key_id, uuid.UUID)
        assert resolved_key == key

    def test_upsert_api_key_is_idempotent(self):
        conn = self._conn()
        org_id = _upsert_org(conn, 'marie_scheduler', 'org', 'Org')
        user_id = _upsert_admin_user(
            conn, 'marie_scheduler', org_id, 'admin', 'admin@local'
        )
        key = KeyGenerator.generate_key('mas_')
        kid1, rkey1 = _upsert_api_key(
            conn, 'marie_scheduler', user_id, org_id, 'sandbox-admin', key
        )
        kid2, rkey2 = _upsert_api_key(
            conn, 'marie_scheduler', user_id, org_id, 'sandbox-admin', key
        )
        assert kid1 == kid2
        assert rkey1 == rkey2 == key

    def test_upsert_api_key_keeps_existing_on_name_conflict(self):
        """When a key with the same (org_id, name) exists, the existing key is
        returned — not the newly proposed one."""
        conn = self._conn()
        org_id = _upsert_org(conn, 'marie_scheduler', 'org', 'Org')
        user_id = _upsert_admin_user(
            conn, 'marie_scheduler', org_id, 'admin', 'admin@local'
        )
        first_key = KeyGenerator.generate_key('mas_')
        _upsert_api_key(conn, 'marie_scheduler', user_id, org_id, 'sandbox-admin', first_key)
        second_key = KeyGenerator.generate_key('mas_')
        _kid, resolved = _upsert_api_key(
            conn, 'marie_scheduler', user_id, org_id, 'sandbox-admin', second_key
        )
        # The FIRST key wins; the second proposed key is ignored.
        assert resolved == first_key


class TestSeedDefaults:
    """Tests that patch _build_pool so no real DB is needed."""

    def _run(
        self,
        tables: dict,
        api_key: str | None = None,
        org_slug: str = 'default-org',
    ) -> SeedResult:
        pool = _fake_pool(tables)
        with patch('marie.sandbox.seed._build_pool', return_value=pool):
            return seed_defaults(
                {'hostname': 'x', 'port': 5432, 'username': 'u', 'password': 'p', 'database': 'd'},
                org_slug=org_slug,
                api_key=api_key,
            )

    def test_creates_all_records(self):
        tables = _make_tables()
        result = self._run(tables)
        assert result.org_id
        assert result.workspace_id
        assert result.user_id
        assert result.api_key
        assert result.api_key.startswith('mas_')

    def test_result_has_correct_slugs(self):
        tables = _make_tables()
        result = self._run(tables, org_slug='my-sandbox')
        assert result.org_slug == 'my-sandbox'
        assert result.workspace_slug == 'default'
        assert result.username == 'admin'

    def test_idempotent_second_call_returns_same_ids(self):
        tables = _make_tables()
        r1 = self._run(tables)
        r2 = self._run(tables)
        assert r1.org_id == r2.org_id
        assert r1.workspace_id == r2.workspace_id
        assert r1.user_id == r2.user_id
        assert r1.api_key == r2.api_key

    def test_accepts_explicit_api_key(self):
        tables = _make_tables()
        explicit_key = KeyGenerator.generate_key('mas_')
        result = self._run(tables, api_key=explicit_key)
        assert result.api_key == explicit_key

    def test_rejects_invalid_api_key(self):
        tables = _make_tables()
        with pytest.raises(ValueError, match='invalid'):
            self._run(tables, api_key='bad_key')

    def test_generates_key_when_not_provided(self):
        tables = _make_tables()
        result = self._run(tables, api_key=None)
        assert KeyGenerator.validate_key(result.api_key)

    def test_seed_result_is_dataclass_with_expected_fields(self):
        tables = _make_tables()
        result = self._run(tables)
        assert hasattr(result, 'org_id')
        assert hasattr(result, 'workspace_id')
        assert hasattr(result, 'user_id')
        assert hasattr(result, 'api_key')
        assert hasattr(result, 'username')
        assert hasattr(result, 'org_slug')
        assert hasattr(result, 'workspace_slug')

    def test_three_consecutive_seeds_are_stable(self):
        """Re-seeding three times must produce identical state."""
        tables = _make_tables()
        r1 = self._run(tables)
        r2 = self._run(tables)
        r3 = self._run(tables)
        assert r1.org_id == r2.org_id == r3.org_id
        assert r1.api_key == r2.api_key == r3.api_key
        # Only one org, one workspace, one user, one key in the table
        assert len(tables['orgs']) == 1
        assert len(tables['workspaces']) == 1
        assert len(tables['users']) == 1
        assert len(tables['keys']) == 1
