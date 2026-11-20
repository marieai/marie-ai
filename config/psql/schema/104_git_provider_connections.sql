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
