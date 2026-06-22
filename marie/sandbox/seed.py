"""
Marie Sandbox — Wave-1 seed entrypoint.

Seeds a freshly provisioned sandbox with the Wave-1 defaults:
  - default organization
  - default workspace
  - admin user
  - admin API key

All inserts are idempotent (ON CONFLICT DO NOTHING).  The function is safe
to re-run: running it twice against the same database yields identical state.

DB interaction uses the same ``psycopg`` + ``PostgresqlMixin`` pattern that
the scheduler already uses so there is no new dependency.

Public surface
--------------
``seed_defaults(config, org_slug, admin_username, admin_email, api_key)``
    Idempotent top-level function.  Returns a :class:`SeedResult`.

CLI::

    python -m marie.sandbox seed \\
        --db-host localhost --db-port 5432 \\
        --db-name postgres --db-user postgres --db-password secret \\
        --org-slug default-org \\
        --admin-username admin \\
        --admin-email admin@sandbox.local \\
        [--api-key mas_existing...]   # optional; generated if omitted
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg_pool import ConnectionPool

from marie.auth.api_key_manager import KeyGenerator
from marie.excepts import BadConfigSource
from marie.logging_core.logger import MarieLogger

_logger = MarieLogger('marie.sandbox.seed')

_SCHEMA = 'marie_scheduler'

# ------------------------------------------------------------------ models ---


@dataclass
class SeedResult:
    """Records created (or confirmed existing) by :func:`seed_defaults`."""

    org_id: str
    org_slug: str
    workspace_id: str
    workspace_slug: str
    user_id: str
    username: str
    api_key: str


# ----------------------------------------------------------------- helpers ---


def _build_pool(config: dict[str, Any]) -> ConnectionPool:
    """Create a minimal psycopg connection pool from *config*."""
    try:
        return ConnectionPool(
            '',
            min_size=1,
            max_size=3,
            open=True,
            kwargs={
                'host': config['hostname'],
                'port': int(config['port']),
                'user': config['username'],
                'password': config['password'],
                'dbname': config['database'],
                'connect_timeout': 10,
                'options': '-c timezone=UTC',
                'application_name': 'marie-sandbox-seed',
            },
        )
    except Exception as exc:
        raise BadConfigSource(
            f'Cannot connect to database for sandbox seed: {exc}'
        ) from exc


def _exec(conn: psycopg.Connection, sql: str, params: tuple = ()) -> list:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        try:
            return cur.fetchall()
        except Exception:
            return []


# ----------------------------------------------------------------- public ---


def seed_defaults(
    config: dict[str, Any],
    *,
    org_slug: str = 'default-org',
    org_name: str = 'Default Organization',
    workspace_slug: str = 'default',
    workspace_name: str = 'Default Workspace',
    admin_username: str = 'admin',
    admin_email: str = 'admin@sandbox.local',
    api_key: str | None = None,
    schema: str = _SCHEMA,
) -> SeedResult:
    """Idempotently seed Wave-1 defaults into a sandbox database.

    Args:
        config:           psycopg connection config (hostname/port/username/
                          password/database keys, matching PostgresqlMixin).
        org_slug:         Unique slug for the default organization.
        org_name:         Display name for the organization.
        workspace_slug:   Slug for the default workspace.
        workspace_name:   Display name for the workspace.
        admin_username:   Username of the seeded admin user.
        admin_email:      E-mail of the seeded admin user.
        api_key:          API key to use.  Generated with ``mas_`` prefix if
                          omitted or ``None``.  Must satisfy
                          :meth:`~marie.auth.api_key_manager.KeyGenerator.validate_key`
                          when provided.
        schema:           PostgreSQL schema name (default: ``marie_scheduler``).

    Returns:
        :class:`SeedResult` with the IDs and key of the seeded records.

    Raises:
        :class:`~marie.excepts.BadConfigSource`: on connection failure.
        ``ValueError``: if a provided *api_key* is structurally invalid.
    """
    if api_key is None:
        api_key = KeyGenerator.generate_key('mas_')
    elif not KeyGenerator.validate_key(api_key):
        raise ValueError(
            f'Provided api_key is invalid; must be 58 chars starting with '
            f'mas_ or mau_.  Got: {api_key!r}'
        )

    _logger.info(f'Seeding sandbox defaults (org_slug={org_slug!r})')

    pool = _build_pool(config)
    try:
        with pool.connection() as conn:
            result = _run_seed(
                conn,
                schema=schema,
                org_slug=org_slug,
                org_name=org_name,
                workspace_slug=workspace_slug,
                workspace_name=workspace_name,
                admin_username=admin_username,
                admin_email=admin_email,
                api_key=api_key,
            )
        conn.close()
    finally:
        pool.close()

    _logger.info(
        f'Seed complete: org={result.org_id} workspace={result.workspace_id} '
        f'user={result.user_id}'
    )
    return result


# ----------------------------------------------------------------- internals ---


def _run_seed(
    conn: psycopg.Connection,
    *,
    schema: str,
    org_slug: str,
    org_name: str,
    workspace_slug: str,
    workspace_name: str,
    admin_username: str,
    admin_email: str,
    api_key: str,
) -> SeedResult:
    """Execute all seed inserts inside a single transaction."""
    with conn.transaction():
        # -- organization --------------------------------------------------
        org_id = _upsert_org(conn, schema, org_slug, org_name)

        # -- workspace -----------------------------------------------------
        workspace_id = _upsert_workspace(
            conn, schema, org_id, workspace_slug, workspace_name
        )

        # -- admin user ----------------------------------------------------
        user_id = _upsert_admin_user(conn, schema, org_id, admin_username, admin_email)

        # -- API key -------------------------------------------------------
        # _upsert_api_key returns the *resolved* key: either the newly created
        # one or the pre-existing key when the org+name slot is already taken.
        _key_id, resolved_api_key = _upsert_api_key(
            conn, schema, user_id, org_id, 'sandbox-admin', api_key
        )

    return SeedResult(
        org_id=str(org_id),
        org_slug=org_slug,
        workspace_id=str(workspace_id),
        workspace_slug=workspace_slug,
        user_id=str(user_id),
        username=admin_username,
        api_key=resolved_api_key,
    )


def _upsert_org(
    conn: psycopg.Connection,
    schema: str,
    slug: str,
    name: str,
) -> uuid.UUID:
    rows = _exec(
        conn,
        f"""
        INSERT INTO {schema}.sandbox_organizations (name, slug)
        VALUES (%s, %s)
        ON CONFLICT (slug) DO NOTHING
        RETURNING id
        """,
        (name, slug),
    )
    if rows:
        return rows[0][0]
    rows = _exec(
        conn,
        f'SELECT id FROM {schema}.sandbox_organizations WHERE slug = %s',
        (slug,),
    )
    return rows[0][0]


def _upsert_workspace(
    conn: psycopg.Connection,
    schema: str,
    org_id: uuid.UUID,
    slug: str,
    name: str,
) -> uuid.UUID:
    rows = _exec(
        conn,
        f"""
        INSERT INTO {schema}.sandbox_workspaces (org_id, name, slug)
        VALUES (%s, %s, %s)
        ON CONFLICT (org_id, slug) DO NOTHING
        RETURNING id
        """,
        (org_id, name, slug),
    )
    if rows:
        return rows[0][0]
    rows = _exec(
        conn,
        f'SELECT id FROM {schema}.sandbox_workspaces WHERE org_id = %s AND slug = %s',
        (org_id, slug),
    )
    return rows[0][0]


def _upsert_admin_user(
    conn: psycopg.Connection,
    schema: str,
    org_id: uuid.UUID,
    username: str,
    email: str,
) -> uuid.UUID:
    rows = _exec(
        conn,
        f"""
        INSERT INTO {schema}.sandbox_admin_users (org_id, username, email, role)
        VALUES (%s, %s, %s, 'admin')
        ON CONFLICT (username) DO NOTHING
        RETURNING id
        """,
        (org_id, username, email),
    )
    if rows:
        return rows[0][0]
    rows = _exec(
        conn,
        f'SELECT id FROM {schema}.sandbox_admin_users WHERE username = %s',
        (username,),
    )
    return rows[0][0]


def _upsert_api_key(
    conn: psycopg.Connection,
    schema: str,
    user_id: uuid.UUID,
    org_id: uuid.UUID,
    name: str,
    api_key: str,
) -> tuple[uuid.UUID, str]:
    """Insert the key; if a key with the same (org_id, name) already exists,
    keep it (DO NOTHING) and return the *existing* key value so the SeedResult
    always reflects what is actually stored in the database.

    Returns ``(key_id, resolved_api_key)`` where ``resolved_api_key`` may
    differ from the *api_key* argument when the row already existed.
    """
    rows = _exec(
        conn,
        f"""
        INSERT INTO {schema}.sandbox_api_keys (user_id, org_id, name, api_key)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (org_id, name) DO NOTHING
        RETURNING id, api_key
        """,
        (user_id, org_id, name, api_key),
    )
    if rows:
        return rows[0][0], rows[0][1]
    rows = _exec(
        conn,
        f'SELECT id, api_key FROM {schema}.sandbox_api_keys WHERE org_id = %s AND name = %s',
        (org_id, name),
    )
    return rows[0][0], rows[0][1]


# -------------------------------------------------------------------- CLI ---


def _config_from_env() -> dict[str, Any]:
    """Build a DB config dict from environment variables."""
    return {
        'hostname': os.getenv(
            'POSTGRES_HOSTNAME', os.getenv('SANDBOX_DB_HOST', 'localhost')
        ),
        'port': int(os.getenv('POSTGRES_PORT', os.getenv('SANDBOX_DB_PORT', '5432'))),
        'username': os.getenv(
            'POSTGRES_USER', os.getenv('SANDBOX_DB_USER', 'postgres')
        ),
        'password': os.getenv(
            'POSTGRES_PASSWORD', os.getenv('SANDBOX_DB_PASSWORD', '')
        ),
        'database': os.getenv('POSTGRES_DB', os.getenv('SANDBOX_DB_NAME', 'postgres')),
    }


def _cli_seed(args: 'argparse.Namespace') -> None:  # noqa: F821
    config = {
        'hostname': args.db_host,
        'port': args.db_port,
        'username': args.db_user,
        'password': args.db_password,
        'database': args.db_name,
    }
    result = seed_defaults(
        config,
        org_slug=args.org_slug,
        org_name=args.org_name,
        workspace_slug=args.workspace_slug,
        workspace_name=args.workspace_name,
        admin_username=args.admin_username,
        admin_email=args.admin_email,
        api_key=args.api_key or os.getenv('SANDBOX_ADMIN_API_KEY') or None,
        schema=args.schema,
    )
    # Print key=value pairs so the calling Job can capture them.
    import json

    output = {
        'org_id': result.org_id,
        'org_slug': result.org_slug,
        'workspace_id': result.workspace_id,
        'workspace_slug': result.workspace_slug,
        'user_id': result.user_id,
        'username': result.username,
        'api_key': result.api_key,
    }
    print(json.dumps(output))


def build_parser() -> 'argparse.ArgumentParser':  # noqa: F821
    """Return the argument parser for the seed CLI sub-command."""
    import argparse

    p = argparse.ArgumentParser(
        prog='python -m marie.sandbox seed',
        description='Wave-1 idempotent seed for a Marie sandbox.',
    )
    # DB connection
    p.add_argument('--db-host', default=os.getenv('POSTGRES_HOSTNAME', 'localhost'))
    p.add_argument(
        '--db-port', type=int, default=int(os.getenv('POSTGRES_PORT', '5432'))
    )
    p.add_argument('--db-name', default=os.getenv('POSTGRES_DB', 'postgres'))
    p.add_argument('--db-user', default=os.getenv('POSTGRES_USER', 'postgres'))
    p.add_argument('--db-password', default=os.getenv('POSTGRES_PASSWORD', ''))
    p.add_argument('--schema', default=_SCHEMA)
    # Seed identity
    p.add_argument('--org-slug', default='default-org')
    p.add_argument('--org-name', default='Default Organization')
    p.add_argument('--workspace-slug', default='default')
    p.add_argument('--workspace-name', default='Default Workspace')
    p.add_argument('--admin-username', default='admin')
    p.add_argument('--admin-email', default='admin@sandbox.local')
    p.add_argument(
        '--api-key',
        default=None,
        help=(
            'API key to seed.  If omitted, read from SANDBOX_ADMIN_API_KEY env var '
            'or generated automatically.'
        ),
    )
    return p
