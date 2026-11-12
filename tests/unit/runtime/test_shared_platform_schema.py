from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from tests.unit.runtime.test_runtime_schema import database
from tests.unit.runtime.test_runtime_schema import install as install_registry

ROOT = Path(__file__).resolve().parents[3]
DDL = ROOT / 'config/psql/schema/102_shared_platform_registry.sql'
HANDOFF = ROOT / 'tools/runtime/shared-platform-handoff.sql'
TABLES = ('organization', 'tenant', 'git_repositories')
TYPES = (
    'tenant_status',
    'repository_asset_type',
    'git_provider_type',
    'git_sync_status',
)


def install(connection):
    install_registry(connection)
    connection.execute(DDL.read_text())


def objects(connection, schema):
    return dict(
        connection.execute(
            'SELECT relname,oid FROM pg_class WHERE relnamespace=%s::regnamespace AND relname=ANY(%s)',
            (schema, list(TABLES)),
        ).fetchall()
    )


def legacy(connection):
    install(connection)
    connection.execute('CREATE SCHEMA marie_studio')
    connection.execute(
        "CREATE TYPE marie_studio.connection_visibility AS ENUM ('personal','organization','system')"
    )
    connection.execute(
        '''ALTER TABLE marie_runtime.git_repositories DROP CONSTRAINT git_repositories_scope_check;
        ALTER TABLE marie_runtime.git_repositories ALTER COLUMN visibility DROP DEFAULT;
        ALTER TABLE marie_runtime.git_repositories ALTER COLUMN visibility TYPE marie_studio.connection_visibility
          USING visibility::text::marie_studio.connection_visibility;
        ALTER TABLE marie_runtime.git_repositories ALTER COLUMN visibility SET DEFAULT 'system';
        ALTER TABLE marie_runtime.git_repositories ADD CONSTRAINT git_repositories_scope_check CHECK (
          (visibility='system' AND organization_id IS NULL) OR (visibility='organization' AND organization_id IS NOT NULL));'''
    )
    for name in TABLES:
        connection.execute(f'ALTER TABLE marie_runtime.{name} SET SCHEMA marie_studio')
    for name in TYPES:
        if name == 'repository_asset_type':
            connection.execute(
                "CREATE TYPE marie_studio.repository_asset_type AS ENUM ('prompt','query_plan','processor','workflow','webapp')"
            )
            connection.execute(
                'ALTER TABLE marie_studio.git_repositories ALTER COLUMN asset_type TYPE marie_studio.repository_asset_type USING asset_type::text::marie_studio.repository_asset_type'
            )
            connection.execute('DROP TYPE marie_runtime.repository_asset_type')
        else:
            connection.execute(
                f'ALTER TYPE marie_runtime.{name} SET SCHEMA marie_studio'
            )
    org, tenant, child, repository, owner = (uuid4() for _ in range(5))
    connection.execute(
        'INSERT INTO marie_studio.organization(id,name,slug,owner_id,settings) VALUES (%s,%s,%s,%s,%s::jsonb)',
        (org, 'Shared organization', 'shared', owner, '{"retain":true}'),
    )
    connection.execute(
        'INSERT INTO marie_studio.tenant(id,name,slug,organization_id,path) VALUES (%s,%s,%s,%s,%s)',
        (tenant, 'Root tenant', 'root', org, '/root/'),
    )
    connection.execute(
        'INSERT INTO marie_studio.tenant(id,name,slug,organization_id,parent_id,path,depth) VALUES (%s,%s,%s,%s,%s,%s,1)',
        (child, 'Child tenant', 'child', org, tenant, '/root/child/'),
    )
    connection.execute(
        "INSERT INTO marie_studio.git_repositories(id,repo_id,asset_type,name,path,organization_id,visibility,remote_provider) VALUES (%s,'shared','prompt','Shared repo','/logical/repo',%s,'organization','github')",
        (repository, org),
    )
    connection.execute(
        '''CREATE TABLE marie_studio.submissions(id uuid PRIMARY KEY,tenant_id uuid REFERENCES marie_studio.tenant(id));
        CREATE TABLE marie_studio.workspace_repositories(repository_id uuid REFERENCES marie_studio.git_repositories(id));
        CREATE TABLE marie_studio.provider_binding(provider marie_studio.git_provider_type);
        CREATE VIEW marie_studio.shared_scope AS SELECT t.id,t.organization_id,o.name FROM marie_studio.tenant t JOIN marie_studio.organization o ON o.id=t.organization_id;'''
    )
    connection.execute(
        'INSERT INTO marie_studio.submissions VALUES (%s,%s)', (uuid4(), child)
    )
    connection.execute(
        'INSERT INTO marie_studio.workspace_repositories VALUES (%s)', (repository,)
    )
    connection.execute("INSERT INTO marie_studio.provider_binding VALUES ('github')")
    return org, tenant, child, repository, owner


def transfer(connection):
    try:
        connection.execute(HANDOFF.read_text())
    except psycopg.Error:
        connection.rollback()
        raise


def test_shared_platform_installs_without_studio_and_replays(database):
    install(database)
    before = objects(database, 'marie_runtime')
    install(database)
    assert objects(database, 'marie_runtime') == before
    assert set(before) == set(TABLES)
    assert (
        database.execute("SELECT to_regnamespace('marie_studio')").fetchone()[0] is None
    )
    org, tenant, owner = uuid4(), uuid4(), uuid4()
    database.execute(
        "INSERT INTO marie_runtime.organization(id,name,slug,owner_id) VALUES (%s,'Shared','shared',%s)",
        (org, owner),
    )
    database.execute(
        "INSERT INTO marie_runtime.tenant(id,name,slug,organization_id) VALUES (%s,'Tenant','tenant',%s)",
        (tenant, org),
    )
    assert (
        database.execute(
            'SELECT organization_id FROM marie_runtime.tenant WHERE id=%s', (tenant,)
        ).fetchone()[0]
        == org
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        database.execute(
            "INSERT INTO marie_runtime.tenant(name,slug,organization_id) VALUES ('Foreign','foreign',%s)",
            (uuid4(),),
        )


def test_transfer_preserves_ids_types_incoming_foreign_keys_and_hierarchy(database):
    org, tenant, child, repository, owner = legacy(database)
    before = objects(database, 'marie_studio')
    enum_before = dict(
        database.execute(
            "SELECT typname,oid FROM pg_type WHERE typnamespace='marie_studio'::regnamespace AND typname=ANY(%s)",
            (list(TYPES),),
        ).fetchall()
    )
    transfer(database)
    install(database)
    transfer(database)
    assert objects(database, 'marie_runtime') == before
    assert objects(database, 'marie_studio') == {}
    enum_after = dict(
        database.execute(
            "SELECT typname,oid FROM pg_type WHERE typnamespace='marie_runtime'::regnamespace AND typname=ANY(%s)",
            (list(TYPES),),
        ).fetchall()
    )
    assert enum_after == enum_before
    assert (
        database.execute(
            "SELECT 'code'::marie_runtime.repository_asset_type::text"
        ).fetchone()[0]
        == 'code'
    )
    assert database.execute(
        'SELECT organization_id,parent_id,path,depth FROM marie_runtime.tenant WHERE id=%s',
        (child,),
    ).fetchone() == (org, tenant, '/root/child/', 1)
    assert database.execute(
        'SELECT settings,owner_id FROM marie_runtime.organization WHERE id=%s', (org,)
    ).fetchone() == ({'retain': True}, owner)
    assert (
        database.execute(
            'SELECT id FROM marie_studio.shared_scope WHERE id=%s', (child,)
        ).fetchone()[0]
        == child
    )
    assert (
        database.execute(
            "SELECT provider::text FROM marie_studio.provider_binding"
        ).fetchone()[0]
        == 'github'
    )
    assert (
        database.execute(
            'SELECT repository_id FROM marie_studio.workspace_repositories'
        ).fetchone()[0]
        == repository
    )
    assert (
        database.execute(
            "SELECT confrelid FROM pg_constraint WHERE conrelid='marie_studio.submissions'::regclass AND contype='f'"
        ).fetchone()[0]
        == before['tenant']
    )
    assert (
        database.execute(
            "SELECT count(*) FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid JOIN pg_type t ON t.oid=a.atttypid WHERE c.relnamespace='marie_runtime'::regnamespace AND c.relname=ANY(%s) AND t.typnamespace='marie_studio'::regnamespace",
            (list(TABLES),),
        ).fetchone()[0]
        == 0
    )


def test_bootstrap_refuses_legacy_tables_instead_of_creating_a_second_catalog(database):
    legacy(database)
    with pytest.raises(psycopg.Error, match='handoff|transfer'):
        database.execute(DDL.read_text())
    assert objects(database, 'marie_runtime') == {}
    assert len(objects(database, 'marie_studio')) == 3


def test_transfer_refuses_populated_dual_catalogs_without_changing_legacy(database):
    legacy(database)
    before = objects(database, 'marie_studio')
    for name in TABLES:
        database.execute(
            f'CREATE TABLE marie_runtime.{name} (LIKE marie_studio.{name} INCLUDING ALL)'
        )
    database.execute(
        'INSERT INTO marie_runtime.organization SELECT * FROM marie_studio.organization'
    )
    with pytest.raises(psycopg.Error, match='populated|not empty'):
        transfer(database)
    assert objects(database, 'marie_studio') == before
    assert (
        database.execute('SELECT count(*) FROM marie_runtime.organization').fetchone()[
            0
        ]
        == 1
    )


def test_transfer_can_replace_known_empty_bootstrap_tables(database):
    legacy(database)
    before = objects(database, 'marie_studio')
    for name in TABLES:
        database.execute(
            f'CREATE TABLE marie_runtime.{name} (LIKE marie_studio.{name} INCLUDING ALL)'
        )
    transfer(database)
    assert objects(database, 'marie_runtime') == before


def test_transfer_rejects_partial_legacy_catalog_and_private_dependencies(database):
    _, _, _, _, owner = legacy(database)
    before = objects(database, 'marie_studio')
    database.execute('CREATE TABLE marie_studio.private_user(id uuid PRIMARY KEY)')
    database.execute('INSERT INTO marie_studio.private_user VALUES (%s)', (owner,))
    database.execute(
        'ALTER TABLE marie_studio.organization ADD CONSTRAINT private_owner FOREIGN KEY(owner_id) REFERENCES marie_studio.private_user(id)'
    )
    with pytest.raises(psycopg.Error, match='private|Studio|dependency'):
        transfer(database)
    assert objects(database, 'marie_studio') == before
    assert objects(database, 'marie_runtime') == {}
    database.execute(
        'ALTER TABLE marie_studio.organization DROP CONSTRAINT private_owner'
    )
    database.execute(
        'ALTER TABLE marie_studio.git_repositories SET SCHEMA marie_runtime'
    )
    with pytest.raises(psycopg.Error, match='Partial|partial'):
        transfer(database)
