BEGIN;

DO $handoff$
BEGIN
    IF to_regclass('marie_studio.git_provider_connections') IS NOT NULL THEN
        IF to_regclass('marie_runtime.git_provider_connections') IS NOT NULL THEN
            RAISE EXCEPTION 'Both provider registries exist. Stop the API and gateway and reconcile the duplicate registry before this handoff.';
        END IF;
        ALTER TABLE marie_studio.git_provider_connections SET SCHEMA marie_runtime;
        ALTER TABLE marie_runtime.git_provider_connections
            ADD COLUMN organization_id UUID REFERENCES marie_runtime.organization(id) ON DELETE CASCADE,
            ADD COLUMN owner_key VARCHAR(100),
            ADD COLUMN connection_key VARCHAR(100) NOT NULL DEFAULT 'default',
            ADD COLUMN name VARCHAR(255),
            ADD COLUMN secret_ref TEXT,
            ADD COLUMN is_enabled BOOLEAN NOT NULL DEFAULT true,
            ADD COLUMN discovery_config JSONB,
            ADD COLUMN discovered_dag_id UUID;
        UPDATE marie_runtime.git_provider_connections
            SET owner_key='user:' || user_id::text,
                name=provider::text || ' · ' || provider_username;
        ALTER TABLE marie_runtime.git_provider_connections
            ALTER COLUMN owner_key SET NOT NULL,
            ALTER COLUMN name SET NOT NULL,
            ALTER COLUMN user_id DROP NOT NULL,
            ALTER COLUMN access_token DROP NOT NULL,
            ALTER COLUMN provider_user_id SET DEFAULT '',
            ALTER COLUMN provider_username SET DEFAULT '';
        ALTER TABLE marie_runtime.git_provider_connections
            DROP CONSTRAINT IF EXISTS git_provider_connections_user_id_provider_key;
        DROP INDEX IF EXISTS marie_runtime.git_provider_connections_user_id_provider_key;
        UPDATE marie_runtime.git_provider_connections SET is_enabled=false,is_valid=false
            WHERE NULLIF(access_token,'') IS NULL AND NULLIF(secret_ref,'') IS NULL;
        ALTER TABLE marie_runtime.git_provider_connections
            ADD CONSTRAINT git_provider_connections_owner_check CHECK (
                (user_id IS NOT NULL AND organization_id IS NULL AND owner_key='user:' || user_id::text)
                OR (user_id IS NULL AND organization_id IS NOT NULL AND owner_key='organization:' || organization_id::text)
            ),
            ADD CONSTRAINT git_provider_connections_credential_check CHECK (
                NOT is_enabled OR
                (organization_id IS NOT NULL AND NULLIF(secret_ref,'') IS NOT NULL AND access_token IS NULL AND refresh_token IS NULL)
                OR (user_id IS NOT NULL AND (NULLIF(access_token,'') IS NOT NULL OR NULLIF(secret_ref,'') IS NOT NULL))
            );
    END IF;
END;
$handoff$;

CREATE TABLE IF NOT EXISTS marie_runtime.git_provider_connections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID,
    organization_id UUID REFERENCES marie_runtime.organization(id) ON DELETE CASCADE,
    owner_key VARCHAR(100) NOT NULL,
    connection_key VARCHAR(100) NOT NULL DEFAULT 'default',
    name VARCHAR(255) NOT NULL,
    provider marie_runtime.git_provider_type NOT NULL,
    provider_user_id VARCHAR(100) NOT NULL DEFAULT '',
    provider_username VARCHAR(255) NOT NULL DEFAULT '',
    display_name VARCHAR(255),
    avatar_url VARCHAR(2000),
    provider_url VARCHAR(500),
    access_token TEXT,
    refresh_token TEXT,
    secret_ref TEXT,
    scopes TEXT[] NOT NULL DEFAULT '{}',
    is_valid BOOLEAN NOT NULL DEFAULT true,
    is_enabled BOOLEAN NOT NULL DEFAULT true,
    last_validated_at TIMESTAMPTZ,
    discovery_config JSONB,
    discovered_dag_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT git_provider_connections_owner_check CHECK (
        (user_id IS NOT NULL AND organization_id IS NULL AND owner_key='user:' || user_id::text)
        OR (user_id IS NULL AND organization_id IS NOT NULL AND owner_key='organization:' || organization_id::text)
    ),
    CONSTRAINT git_provider_connections_credential_check CHECK (
        NOT is_enabled OR
        (organization_id IS NOT NULL AND NULLIF(secret_ref,'') IS NOT NULL AND access_token IS NULL AND refresh_token IS NULL)
        OR (user_id IS NOT NULL AND (NULLIF(access_token,'') IS NOT NULL OR NULLIF(secret_ref,'') IS NOT NULL))
    )
);
CREATE UNIQUE INDEX IF NOT EXISTS git_provider_connections_identity_key
    ON marie_runtime.git_provider_connections(owner_key,provider,connection_key);
CREATE INDEX IF NOT EXISTS git_provider_connections_user_idx ON marie_runtime.git_provider_connections(user_id);
CREATE INDEX IF NOT EXISTS git_provider_connections_organization_idx ON marie_runtime.git_provider_connections(organization_id);

ALTER TABLE marie_runtime.git_repositories
    ADD COLUMN IF NOT EXISTS provider_connection_id UUID REFERENCES marie_runtime.git_provider_connections(id),
    ADD COLUMN IF NOT EXISTS remote_repository_id VARCHAR(255);
CREATE UNIQUE INDEX IF NOT EXISTS git_repositories_upstream_identity_key
    ON marie_runtime.git_repositories(organization_id,provider_connection_id,remote_repository_id,asset_type)
    WHERE organization_id IS NOT NULL AND provider_connection_id IS NOT NULL AND remote_repository_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS marie_runtime.git_repository_discovery (
    connection_id UUID NOT NULL REFERENCES marie_runtime.git_provider_connections(id) ON DELETE CASCADE,
    dag_id UUID NOT NULL,
    connection_fingerprint VARCHAR(64) NOT NULL,
    repositories JSONB NOT NULL CHECK(jsonb_typeof(repositories)='array'),
    requested_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY(connection_id,dag_id)
);
CREATE INDEX IF NOT EXISTS git_repository_discovery_completed_idx
    ON marie_runtime.git_repository_discovery(connection_id,completed_at DESC);

DO $$
BEGIN
    IF to_regclass('marie_studio.system_configuration') IS NOT NULL THEN
        UPDATE marie_runtime.git_provider_connections c
        SET provider_url=config.value #>> '{}'
        FROM marie_studio.system_configuration config
        WHERE c.provider='gitea' AND c.user_id IS NOT NULL AND c.connection_key='default' AND c.provider_url IS NULL
          AND config.category='gitea' AND config.key='url' AND NOT config.is_secret
          AND jsonb_typeof(config.value)='string';
    END IF;
END $$;

COMMIT;
