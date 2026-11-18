from pathlib import Path
from uuid import uuid4

import psycopg
import pytest

from tests.unit.runtime.test_runtime_schema import database
from tests.unit.runtime.test_shared_platform_schema import install as install_platform

ROOT = Path(__file__).resolve().parents[3]
DDL = ROOT / 'config/psql/schema/104_git_provider_connections.sql'
HANDOFF = ROOT / 'tools/runtime/git-provider-handoff.sql'


def install(connection):
    install_platform(connection)
    connection.execute(DDL.read_text())


def test_connections_install_without_studio_and_replay(database):
    install(database)
    oid = database.execute(
        "SELECT 'marie_runtime.git_provider_connections'::regclass::oid"
    ).fetchone()[0]
    install(database)
    assert (
        database.execute(
            "SELECT 'marie_runtime.git_provider_connections'::regclass::oid"
        ).fetchone()[0]
        == oid
    )
    assert (
        database.execute("SELECT to_regnamespace('marie_studio')").fetchone()[0] is None
    )


def test_named_connections_have_scoped_identity_and_require_shared_secret_refs(
    database,
):
    install(database)
    user, org = uuid4(), uuid4()
    database.execute(
        "INSERT INTO marie_runtime.organization(id,name,slug,owner_id) VALUES (%s,'Main','main',%s)",
        (org, user),
    )
    for key, url in [
        ('default', 'https://system.test'),
        ('research', 'https://research.test'),
    ]:
        database.execute(
            '''INSERT INTO marie_runtime.git_provider_connections
            (user_id,owner_key,connection_key,name,provider,provider_user_id,provider_username,provider_url,access_token)
            VALUES (%s,%s,%s,%s,'gitea','1','reader',%s,'test-token')''',
            (user, f'user:{user}', key, key, url),
        )
    assert (
        database.execute(
            'SELECT count(*) FROM marie_runtime.git_provider_connections'
        ).fetchone()[0]
        == 2
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        database.execute(
            '''INSERT INTO marie_runtime.git_provider_connections
            (organization_id,owner_key,connection_key,name,provider,provider_user_id,provider_username,access_token)
            VALUES (%s,%s,'shared','Shared','github','2','service','personal-token')''',
            (org, f'organization:{org}'),
        )
    database.execute(
        '''INSERT INTO marie_runtime.git_provider_connections
        (organization_id,owner_key,connection_key,name,provider,provider_user_id,provider_username,secret_ref)
        VALUES (%s,%s,'shared','Shared','github','2','service','env:GIT_READER_TOKEN')''',
        (org, f'organization:{org}'),
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        database.execute(
            '''INSERT INTO marie_runtime.git_provider_connections
            (organization_id,owner_key,connection_key,name,provider,provider_user_id,provider_username,secret_ref)
            VALUES (%s,%s,'shared','Duplicate','github','2','service','env:OTHER')''',
            (org, f'organization:{org}'),
        )


def test_registration_references_upstream_identity_without_cloning_or_duplicate_items(
    database,
):
    install(database)
    org, user, connection = uuid4(), uuid4(), uuid4()
    database.execute(
        "INSERT INTO marie_runtime.organization(id,name,slug,owner_id) VALUES (%s,'Main','main',%s)",
        (org, user),
    )
    database.execute(
        '''INSERT INTO marie_runtime.git_provider_connections
        (id,organization_id,owner_key,connection_key,name,provider,provider_user_id,provider_username,secret_ref)
        VALUES (%s,%s,%s,'shared','Shared','gitea','2','service','env:GIT_READER_TOKEN')''',
        (connection, org, f'organization:{org}'),
    )
    sql = '''INSERT INTO marie_runtime.git_repositories
        (repo_id,asset_type,name,path,organization_id,visibility,provider_connection_id,remote_repository_id)
        VALUES (%s,'code','Repo','',%s,'organization',%s,'repo-42')'''
    database.execute(sql, ('first', org, connection))
    with pytest.raises(psycopg.errors.UniqueViolation):
        database.execute(sql, ('second', org, connection))


@pytest.mark.parametrize('access_token', ['oauth-access', ''])
def test_handoff_preserves_oauth_accounts_ids_and_publish_references(database, access_token):
    install_platform(database)
    user, connection = uuid4(), uuid4()
    database.execute('''CREATE SCHEMA marie_studio;
        CREATE TABLE marie_studio.git_provider_connections (
            id uuid PRIMARY KEY,user_id uuid NOT NULL,provider marie_runtime.git_provider_type NOT NULL,
            provider_user_id varchar(100) NOT NULL,provider_username varchar(255) NOT NULL,
            display_name varchar(255),avatar_url varchar(2000),provider_url varchar(500),
            access_token text NOT NULL,refresh_token text,scopes text[] NOT NULL DEFAULT '{}',
            is_valid boolean NOT NULL DEFAULT true,last_validated_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT git_provider_connections_user_id_provider_key UNIQUE(user_id,provider));
        CREATE TABLE marie_studio.publish_target(connection_id uuid REFERENCES marie_studio.git_provider_connections(id));''')
    database.execute(
        '''INSERT INTO marie_studio.git_provider_connections
        (id,user_id,provider,provider_user_id,provider_username,access_token,refresh_token,is_valid)
        VALUES (%s,%s,'gitea','1','reader',%s,'oauth-refresh',false)''',
        (connection, user, access_token),
    )
    database.execute(
        'INSERT INTO marie_studio.publish_target VALUES (%s)', (connection,)
    )
    old_oid = database.execute(
        "SELECT 'marie_studio.git_provider_connections'::regclass::oid"
    ).fetchone()[0]
    database.execute(HANDOFF.read_text())
    database.execute(HANDOFF.read_text())
    assert (
        database.execute(
            "SELECT 'marie_runtime.git_provider_connections'::regclass::oid"
        ).fetchone()[0]
        == old_oid
    )
    assert database.execute(
        'SELECT id,user_id,owner_key,connection_key,access_token,refresh_token,is_valid FROM marie_runtime.git_provider_connections'
    ).fetchone() == (
        connection,
        user,
        f'user:{user}',
        'default',
        access_token,
        'oauth-refresh',
        False,
    )
    assert (
        database.execute(
            'SELECT connection_id FROM marie_studio.publish_target'
        ).fetchone()[0]
        == connection
    )
    database.execute(
        'SELECT connection_id FROM marie_studio.publish_target JOIN marie_runtime.git_provider_connections c ON c.id=connection_id'
    )
